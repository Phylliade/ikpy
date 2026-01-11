# coding= utf8
"""
.. module:: jax_backend
This module implements JAX-based forward and inverse kinematics.
JAX enables automatic differentiation and JIT compilation for faster computation.
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

from ikpy.utils import jax_geometry as geom


# Check if JAX is available
JAX_AVAILABLE = True


def extract_chain_parameters(chain):
    """
    Extract chain parameters into JAX-compatible arrays.

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The kinematic chain

    Returns
    -------
    dict
        Dictionary containing all link parameters as JAX arrays
    """
    n_links = len(chain.links)

    # Initialize arrays
    origin_translations = []
    origin_orientations = []
    rotation_axes = []
    translation_axes = []
    joint_types = []  # 0=origin, 1=revolute, 2=prismatic, 3=fixed

    for link in chain.links:
        if link.__class__.__name__ == "OriginLink":
            origin_translations.append([0.0, 0.0, 0.0])
            origin_orientations.append([0.0, 0.0, 0.0])
            rotation_axes.append([0.0, 0.0, 1.0])
            translation_axes.append([0.0, 0.0, 0.0])
            joint_types.append(0)  # Origin link
        elif link.__class__.__name__ == "URDFLink":
            origin_translations.append(link.origin_translation.tolist())
            origin_orientations.append(link.origin_orientation.tolist())

            if link.has_rotation and link.rotation is not None:
                rotation_axes.append(link.rotation.tolist())
            else:
                rotation_axes.append([0.0, 0.0, 1.0])

            if link.has_translation and link.translation is not None:
                translation_axes.append(link.translation.tolist())
            else:
                translation_axes.append([0.0, 0.0, 0.0])

            if link.joint_type == "revolute":
                joint_types.append(1)
            elif link.joint_type == "prismatic":
                joint_types.append(2)
            else:  # fixed
                joint_types.append(3)
        elif link.__class__.__name__ == "DHLink":
            # DH parameters - will need special handling
            origin_translations.append([0.0, 0.0, 0.0])
            origin_orientations.append([0.0, 0.0, 0.0])
            rotation_axes.append([0.0, 0.0, 1.0])
            translation_axes.append([0.0, 0.0, 0.0])
            joint_types.append(4)  # DH link
        else:
            raise ValueError(f"Unknown link type: {link.__class__.__name__}")

    return {
        'origin_translations': jnp.array(origin_translations),
        'origin_orientations': jnp.array(origin_orientations),
        'rotation_axes': jnp.array(rotation_axes),
        'translation_axes': jnp.array(translation_axes),
        'joint_types': jnp.array(joint_types, dtype=jnp.int32),
        'n_links': n_links
    }


def compute_single_link_matrix(origin_translation, origin_orientation, rotation_axis,
                               translation_axis, joint_type, joint_param):
    """
    Compute a single link's transformation matrix.

    Parameters
    ----------
    origin_translation: jnp.ndarray (3,)
    origin_orientation: jnp.ndarray (3,)
    rotation_axis: jnp.ndarray (3,)
    translation_axis: jnp.ndarray (3,)
    joint_type: int
        0=origin, 1=revolute, 2=prismatic, 3=fixed
    joint_param: float
        The joint parameter (angle for revolute, displacement for prismatic)

    Returns
    -------
    jnp.ndarray (4, 4)
        The transformation matrix
    """
    # Origin link - identity
    def origin_link(_):
        return jnp.eye(4)

    # Revolute joint
    def revolute_link(theta):
        return geom.compute_link_frame_matrix_revolute(
            origin_translation, origin_orientation, rotation_axis, theta
        )

    # Prismatic joint
    def prismatic_link(mu):
        return geom.compute_link_frame_matrix_prismatic(
            origin_translation, origin_orientation, translation_axis, mu
        )

    # Fixed joint
    def fixed_link(_):
        return geom.compute_link_frame_matrix_fixed(origin_translation, origin_orientation)

    # Use lax.switch for conditional selection (JIT-compatible)
    return jax.lax.switch(
        joint_type,
        [origin_link, revolute_link, prismatic_link, fixed_link],
        joint_param
    )


def forward_kinematics_jax(joints, chain_params):
    """
    Compute forward kinematics using JAX.

    Parameters
    ----------
    joints: jnp.ndarray
        Joint parameters (n_links,)
    chain_params: dict
        Chain parameters extracted by extract_chain_parameters

    Returns
    -------
    jnp.ndarray (4, 4)
        The end-effector transformation matrix
    """
    frame_matrix = jnp.eye(4)

    def body_fn(i, frame_matrix):
        link_matrix = compute_single_link_matrix(
            chain_params['origin_translations'][i],
            chain_params['origin_orientations'][i],
            chain_params['rotation_axes'][i],
            chain_params['translation_axes'][i],
            chain_params['joint_types'][i],
            joints[i]
        )
        return jnp.dot(frame_matrix, link_matrix)

    frame_matrix = jax.lax.fori_loop(0, chain_params['n_links'], body_fn, frame_matrix)
    return frame_matrix


def forward_kinematics_full_jax(joints, chain_params):
    """
    Compute forward kinematics for all links using JAX.

    Parameters
    ----------
    joints: jnp.ndarray
        Joint parameters (n_links,)
    chain_params: dict
        Chain parameters extracted by extract_chain_parameters

    Returns
    -------
    jnp.ndarray (n_links, 4, 4)
        Transformation matrices for all links
    """
    def scan_fn(frame_matrix, link_idx):
        link_matrix = compute_single_link_matrix(
            chain_params['origin_translations'][link_idx],
            chain_params['origin_orientations'][link_idx],
            chain_params['rotation_axes'][link_idx],
            chain_params['translation_axes'][link_idx],
            chain_params['joint_types'][link_idx],
            joints[link_idx]
        )
        new_frame = jnp.dot(frame_matrix, link_matrix)
        return new_frame, new_frame

    _, all_frames = jax.lax.scan(scan_fn, jnp.eye(4), jnp.arange(chain_params['n_links']))
    return all_frames


def inverse_kinematics_jax(chain, target_frame, starting_nodes_angles,
                           regularization_parameter=None, max_iter=100,
                           orientation_mode=None, no_position=False,
                           optimizer="L-BFGS-B", learning_rate=0.01, tol=1e-6):
    """
    Compute inverse kinematics using JAX's automatic differentiation.

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The kinematic chain
    target_frame: np.ndarray (4, 4)
        The target pose
    starting_nodes_angles: np.ndarray
        Initial joint angles
    regularization_parameter: float
        Regularization coefficient
    max_iter: int
        Maximum number of iterations
    orientation_mode: str
        One of None, "X", "Y", "Z", "all"
    no_position: bool
        If True, don't optimize position
    optimizer: str
        Optimizer to use: "L-BFGS-B", "gradient_descent", or "adam"
    learning_rate: float
        Learning rate for gradient-based optimizers
    tol: float
        Tolerance for convergence

    Returns
    -------
    np.ndarray
        Optimal joint angles
    """
    # Extract chain parameters
    chain_params = extract_chain_parameters(chain)

    # Determine dtype based on JAX configuration
    # Use float64 if x64 mode is enabled, otherwise float32
    try:
        from jax import config
        use_float64 = config.jax_enable_x64
    except (ImportError, AttributeError):
        use_float64 = False

    dtype = jnp.float64 if use_float64 else jnp.float32

    # Convert to JAX arrays with consistent dtype
    target_frame_jax = jnp.array(target_frame, dtype=dtype)
    starting_nodes_angles_jax = jnp.array(starting_nodes_angles, dtype=dtype)
    active_links_mask = jnp.array(chain.active_links_mask)

    # Get bounds for active joints
    bounds_list = [link.bounds for link in chain.links]
    lower_bounds = []
    upper_bounds = []
    for i, (mask, bounds) in enumerate(zip(chain.active_links_mask, bounds_list)):
        if mask:
            lower_bounds.append(bounds[0] if np.isfinite(bounds[0]) else -np.pi * 2)
            upper_bounds.append(bounds[1] if np.isfinite(bounds[1]) else np.pi * 2)
    lower_bounds = jnp.array(lower_bounds, dtype=dtype)
    upper_bounds = jnp.array(upper_bounds, dtype=dtype)

    # Extract active joints from full joints
    def active_from_full(joints):
        return joints[active_links_mask]

    # More efficient version using index_update
    active_indices = jnp.where(active_links_mask)[0]

    def active_to_full_v2(active_joints, initial_position):
        return initial_position.at[active_indices].set(active_joints)

    # Target position
    target_position = target_frame_jax[:3, 3]

    # Define loss function
    def compute_loss(active_joints):
        # Ensure consistent dtype
        active_joints = active_joints.astype(dtype)

        # Convert to full joints
        full_joints = active_to_full_v2(active_joints, starting_nodes_angles_jax)

        # Compute FK
        fk = forward_kinematics_jax(full_joints, chain_params)

        # Position error
        if not no_position:
            position_error = fk[:3, 3] - target_position
            loss = jnp.sum(position_error ** 2)
        else:
            loss = jnp.array(0.0, dtype=dtype)

        # Orientation error
        if orientation_mode == "X":
            orientation_error = fk[:3, 0] - target_frame_jax[:3, 0]
            loss = loss + jnp.sum(orientation_error ** 2)
        elif orientation_mode == "Y":
            orientation_error = fk[:3, 1] - target_frame_jax[:3, 1]
            loss = loss + jnp.sum(orientation_error ** 2)
        elif orientation_mode == "Z":
            orientation_error = fk[:3, 2] - target_frame_jax[:3, 2]
            loss = loss + jnp.sum(orientation_error ** 2)
        elif orientation_mode == "all":
            orientation_error = (fk[:3, :3] - target_frame_jax[:3, :3]).ravel()
            loss = loss + jnp.sum(orientation_error ** 2)

        # Regularization
        if regularization_parameter is not None:
            reg_term = regularization_parameter * jnp.sum(
                (active_joints - active_from_full(starting_nodes_angles_jax)) ** 2
            )
            loss = loss + reg_term

        return loss

    # Compute gradient
    grad_loss = jax.grad(compute_loss)

    # Initial active joints
    x0 = active_from_full(starting_nodes_angles_jax)

    if optimizer == "L-BFGS-B":
        # Use scipy optimization through JAX value_and_grad
        from jax.scipy.optimize import minimize as jax_scipy_minimize

        # Use BFGS from JAX (no bounds support, so we clip manually)
        def loss_clipped(x):
            x = x.astype(dtype)
            x_clipped = jnp.clip(x, lower_bounds, upper_bounds)
            return compute_loss(x_clipped)

        result = jax_scipy_minimize(loss_clipped, x0, method='BFGS',
                                    options={'maxiter': max_iter})
        optimal_active = jnp.clip(result.x, lower_bounds, upper_bounds)

    elif optimizer == "gradient_descent":
        # Simple gradient descent with bounds
        x = x0
        for _ in range(max_iter):
            g = grad_loss(x)
            x = x - learning_rate * g
            x = jnp.clip(x, lower_bounds, upper_bounds)

            # Check convergence
            if jnp.max(jnp.abs(g)) < tol:
                break
        optimal_active = x

    elif optimizer == "adam":
        # Adam optimizer
        x = x0
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        beta1, beta2 = 0.9, 0.999
        eps = jnp.array(1e-8, dtype=dtype)

        for t in range(1, max_iter + 1):
            g = grad_loss(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            x = jnp.clip(x, lower_bounds, upper_bounds)

            # Check convergence
            if jnp.max(jnp.abs(g)) < tol:
                break
        optimal_active = x

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Convert back to full joints
    result = active_to_full_v2(optimal_active, starting_nodes_angles_jax)

    # Convert to numpy for compatibility
    return np.array(result)


def inverse_kinematics_scipy_with_jax_grad(chain, target_frame, starting_nodes_angles,
                                           regularization_parameter=None, max_iter=None,
                                           orientation_mode=None, no_position=False,
                                           optimizer="least_squares"):
    """
    Compute inverse kinematics using scipy but with JAX-computed gradients.
    This provides faster gradient computation while maintaining scipy's robust optimizers.

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The kinematic chain
    target_frame: np.ndarray (4, 4)
        The target pose
    starting_nodes_angles: np.ndarray
        Initial joint angles
    regularization_parameter: float
        Regularization coefficient
    max_iter: int
        Maximum iterations (deprecated)
    orientation_mode: str
        One of None, "X", "Y", "Z", "all"
    no_position: bool
        If True, don't optimize position
    optimizer: str
        "least_squares" or "scalar"

    Returns
    -------
    np.ndarray
        Optimal joint angles
    """
    import scipy.optimize

    # Extract chain parameters
    chain_params = extract_chain_parameters(chain)

    # Convert to JAX arrays
    target_frame_jax = jnp.array(target_frame)
    starting_nodes_angles_jax = jnp.array(starting_nodes_angles, dtype=jnp.float64)
    active_links_mask = jnp.array(chain.active_links_mask)

    # Get bounds
    bounds_list = [link.bounds for link in chain.links]
    active_bounds = []
    for mask, bounds in zip(chain.active_links_mask, bounds_list):
        if mask:
            active_bounds.append(bounds)
    active_bounds = np.array(active_bounds)

    # Active indices
    active_indices = jnp.where(active_links_mask)[0]

    def active_to_full(active_joints, initial_position):
        full = initial_position.copy()
        return full.at[active_indices].set(active_joints)

    def active_from_full(joints):
        return joints[active_links_mask]

    # Target
    target_position = target_frame_jax[:3, 3]

    # Define residual function for least_squares
    ORIENTATION_COEFF = 1.0

    def compute_residuals(active_joints):
        active_joints = jnp.array(active_joints)
        full_joints = active_to_full(active_joints, starting_nodes_angles_jax)
        fk = forward_kinematics_jax(full_joints, chain_params)

        errors = []

        if not no_position:
            position_error = fk[:3, 3] - target_position
            errors.append(position_error)

        if orientation_mode == "X":
            orientation_error = ORIENTATION_COEFF * (fk[:3, 0] - target_frame_jax[:3, 0])
            errors.append(orientation_error)
        elif orientation_mode == "Y":
            orientation_error = ORIENTATION_COEFF * (fk[:3, 1] - target_frame_jax[:3, 1])
            errors.append(orientation_error)
        elif orientation_mode == "Z":
            orientation_error = ORIENTATION_COEFF * (fk[:3, 2] - target_frame_jax[:3, 2])
            errors.append(orientation_error)
        elif orientation_mode == "all":
            orientation_error = ORIENTATION_COEFF * (fk[:3, :3] - target_frame_jax[:3, :3]).ravel()
            errors.append(orientation_error)

        return jnp.concatenate(errors) if len(errors) > 1 else errors[0]

    # Wrap for numpy compatibility
    def residuals_np(x):
        return np.array(compute_residuals(x))

    # Add regularization if needed
    if regularization_parameter is not None:
        starting_active = active_from_full(starting_nodes_angles_jax)

        def residuals_with_reg(x):
            base_residuals = compute_residuals(jnp.array(x))
            reg = regularization_parameter * jnp.linalg.norm(jnp.array(x) - starting_active)
            return np.array(base_residuals + reg)

        optimize_fn = residuals_with_reg
    else:
        optimize_fn = residuals_np

    # Compute Jacobian using JAX
    jac_fn = jax.jacfwd(compute_residuals)

    def jacobian_np(x):
        return np.array(jac_fn(jnp.array(x)))

    # Initial point
    x0 = np.array(active_from_full(starting_nodes_angles_jax))

    if optimizer == "least_squares":
        # Unzip bounds
        bounds_unzipped = np.moveaxis(active_bounds, -1, 0)
        result = scipy.optimize.least_squares(
            optimize_fn, x0, jac=jacobian_np, bounds=bounds_unzipped
        )
    elif optimizer == "scalar":
        def scalar_loss(x):
            return np.linalg.norm(optimize_fn(x))
        result = scipy.optimize.minimize(scalar_loss, x0, bounds=active_bounds)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Convert back to full joints
    full_result = np.array(active_to_full(jnp.array(result.x), starting_nodes_angles_jax))
    return full_result


class JaxKinematicsCache:
    """
    Cache for JAX kinematics computations.
    Stores JIT-compiled functions for a specific chain configuration.
    """

    def __init__(self, chain):
        """
        Initialize the cache for a chain.

        Parameters
        ----------
        chain: ikpy.chain.Chain
            The kinematic chain
        """
        self.chain = chain
        self.chain_params = extract_chain_parameters(chain)
        self.active_links_mask = jnp.array(chain.active_links_mask)
        self.active_indices = jnp.where(self.active_links_mask)[0]

        # JIT compile the forward kinematics
        self._fk_jit = jit(partial(forward_kinematics_jax, chain_params=self.chain_params))
        self._fk_full_jit = jit(partial(forward_kinematics_full_jax, chain_params=self.chain_params))

        # Store bounds
        bounds_list = [link.bounds for link in chain.links]
        lower_bounds = []
        upper_bounds = []
        for mask, bounds in zip(chain.active_links_mask, bounds_list):
            if mask:
                lower_bounds.append(bounds[0] if np.isfinite(bounds[0]) else -np.pi * 2)
                upper_bounds.append(bounds[1] if np.isfinite(bounds[1]) else np.pi * 2)
        self.lower_bounds = jnp.array(lower_bounds)
        self.upper_bounds = jnp.array(upper_bounds)

    def forward_kinematics(self, joints, full_kinematics=False):
        """
        Compute forward kinematics.

        Parameters
        ----------
        joints: array-like
            Joint parameters
        full_kinematics: bool
            If True, return all intermediate frames

        Returns
        -------
        np.ndarray
            Transformation matrix or list of matrices
        """
        joints = jnp.array(joints)

        if full_kinematics:
            result = self._fk_full_jit(joints)
            return [np.array(result[i]) for i in range(self.chain_params['n_links'])]
        else:
            return np.array(self._fk_jit(joints))

    def active_to_full(self, active_joints, initial_position):
        """Convert active joints to full joint array"""
        initial_position = jnp.array(initial_position)
        return initial_position.at[self.active_indices].set(active_joints)

    def active_from_full(self, joints):
        """Extract active joints from full joint array"""
        return jnp.array(joints)[self.active_links_mask]

    def inverse_kinematics(self, target_frame, initial_position=None,
                           orientation_mode=None, no_position=False,
                           regularization_parameter=None, max_iter=100,
                           optimizer="L-BFGS-B", learning_rate=0.01, tol=1e-6):
        """
        Compute inverse kinematics using JAX optimization.

        Parameters
        ----------
        target_frame: np.ndarray (4, 4)
            Target pose
        initial_position: np.ndarray
            Initial joint positions
        orientation_mode: str
            Orientation constraint mode
        no_position: bool
            Disable position optimization
        regularization_parameter: float
            Regularization strength
        max_iter: int
            Maximum iterations
        optimizer: str
            Optimizer type
        learning_rate: float
            Learning rate for gradient-based optimizers
        tol: float
            Convergence tolerance

        Returns
        -------
        np.ndarray
            Optimal joint positions
        """
        if initial_position is None:
            initial_position = np.zeros(len(self.chain.links))

        return inverse_kinematics_jax(
            self.chain, target_frame, initial_position,
            regularization_parameter=regularization_parameter,
            max_iter=max_iter,
            orientation_mode=orientation_mode,
            no_position=no_position,
            optimizer=optimizer,
            learning_rate=learning_rate,
            tol=tol
        )
