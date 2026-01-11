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


class JaxKinematicsCache:
    """
    Cache for JAX kinematics computations.
    Stores AOT-compiled functions for a specific chain configuration.
    
    Uses JAX's lower/compile API for explicit ahead-of-time compilation,
    ensuring compilation happens exactly once at cache creation time.
    """

    def __init__(self, chain, precompile=True):
        """
        Initialize the cache for a chain.

        Parameters
        ----------
        chain: ikpy.chain.Chain
            The kinematic chain
        precompile: bool
            If True, compile JAX functions immediately using AOT compilation (slower init, faster first call).
            If False, compile lazily on first call (faster init, slower first call).
        """
        self.chain = chain
        self.chain_params = extract_chain_parameters(chain)
        self.active_links_mask = jnp.array(chain.active_links_mask)
        self.active_indices = jnp.where(self.active_links_mask)[0]
        self.n_active = int(jnp.sum(self.active_links_mask))
        self._precompile = precompile

        # Determine dtype based on JAX configuration
        try:
            from jax import config
            self._use_float64 = config.jax_enable_x64
        except (ImportError, AttributeError):
            self._use_float64 = False
        self._dtype = jnp.float64 if self._use_float64 else jnp.float32

        # Store bounds
        bounds_list = [link.bounds for link in chain.links]
        lower_bounds = []
        upper_bounds = []
        for mask, bounds in zip(chain.active_links_mask, bounds_list):
            if mask:
                lower_bounds.append(bounds[0] if np.isfinite(bounds[0]) else -np.pi * 2)
                upper_bounds.append(bounds[1] if np.isfinite(bounds[1]) else np.pi * 2)
        self.lower_bounds = jnp.array(lower_bounds, dtype=self._dtype)
        self.upper_bounds = jnp.array(upper_bounds, dtype=self._dtype)

        # Create FK functions with partial application
        self._fk_fn = partial(forward_kinematics_jax, chain_params=self.chain_params)
        self._fk_full_fn = partial(forward_kinematics_full_jax, chain_params=self.chain_params)

        # Dummy inputs for AOT compilation
        self._dummy_joints = jnp.zeros(self.chain_params['n_links'], dtype=self._dtype)

        if precompile:
            # AOT compilation using lower/compile
            self._fk_compiled = jax.jit(self._fk_fn).lower(self._dummy_joints).compile()
            self._fk_full_compiled = jax.jit(self._fk_full_fn).lower(self._dummy_joints).compile()
            
            # Compile IK functions for all orientation modes
            self._compile_ik_functions()
        else:
            # Lazy JIT compilation (fallback)
            self._fk_compiled = None
            self._fk_full_compiled = None
            self._fk_jit = jax.jit(self._fk_fn)
            self._fk_full_jit = jax.jit(self._fk_full_fn)
            self._ik_compiled = {}
            self._ik_residuals = {}
            self._ik_jacobian = {}

    def _create_loss_function(self, orientation_mode, no_position):
        """
        Create a loss function for a specific orientation mode.
        
        Parameters
        ----------
        orientation_mode: str or None
            One of None, "X", "Y", "Z", "all"
        no_position: bool
            If True, don't optimize position
            
        Returns
        -------
        callable
            Loss function that takes (active_joints, target_frame, initial_position, reg_param)
        """
        chain_params = self.chain_params
        active_indices = self.active_indices
        dtype = self._dtype

        def compute_loss(active_joints, target_frame, initial_position, reg_param):
            # Convert to full joints
            full_joints = initial_position.at[active_indices].set(active_joints)
            
            # Compute FK
            fk = forward_kinematics_jax(full_joints, chain_params)
            
            # Position error
            if not no_position:
                position_error = fk[:3, 3] - target_frame[:3, 3]
                loss = jnp.sum(position_error ** 2)
            else:
                loss = jnp.array(0.0, dtype=dtype)
            
            # Orientation error based on mode
            if orientation_mode == "X":
                orientation_error = fk[:3, 0] - target_frame[:3, 0]
                loss = loss + jnp.sum(orientation_error ** 2)
            elif orientation_mode == "Y":
                orientation_error = fk[:3, 1] - target_frame[:3, 1]
                loss = loss + jnp.sum(orientation_error ** 2)
            elif orientation_mode == "Z":
                orientation_error = fk[:3, 2] - target_frame[:3, 2]
                loss = loss + jnp.sum(orientation_error ** 2)
            elif orientation_mode == "all":
                orientation_error = (fk[:3, :3] - target_frame[:3, :3]).ravel()
                loss = loss + jnp.sum(orientation_error ** 2)
            
            # Regularization (reg_param is traced, so it can vary at runtime)
            # We regularize against initial position
            initial_active = initial_position[active_indices]
            reg_term = reg_param * jnp.sum((active_joints - initial_active) ** 2)
            loss = loss + reg_term
            
            return loss
        
        return compute_loss

    def _create_residual_function(self, orientation_mode, no_position):
        """
        Create a residual function for scipy least_squares.
        
        Parameters
        ----------
        orientation_mode: str or None
            One of None, "X", "Y", "Z", "all"
        no_position: bool
            If True, don't optimize position
            
        Returns
        -------
        callable
            Residual function that takes (active_joints, target_frame, initial_position)
            and returns a vector of residuals
        """
        chain_params = self.chain_params
        active_indices = self.active_indices

        def compute_residuals(active_joints, target_frame, initial_position):
            # Convert to full joints
            full_joints = initial_position.at[active_indices].set(active_joints)
            
            # Compute FK
            fk = forward_kinematics_jax(full_joints, chain_params)
            
            residuals = []
            
            # Position error
            if not no_position:
                position_error = fk[:3, 3] - target_frame[:3, 3]
                residuals.append(position_error)
            
            # Orientation error based on mode
            if orientation_mode == "X":
                orientation_error = fk[:3, 0] - target_frame[:3, 0]
                residuals.append(orientation_error)
            elif orientation_mode == "Y":
                orientation_error = fk[:3, 1] - target_frame[:3, 1]
                residuals.append(orientation_error)
            elif orientation_mode == "Z":
                orientation_error = fk[:3, 2] - target_frame[:3, 2]
                residuals.append(orientation_error)
            elif orientation_mode == "all":
                orientation_error = (fk[:3, :3] - target_frame[:3, :3]).ravel()
                residuals.append(orientation_error)
            
            return jnp.concatenate(residuals) if len(residuals) > 1 else residuals[0]
        
        return compute_residuals

    def _compile_ik_functions(self):
        """
        Pre-compile IK loss and gradient functions for all orientation mode combinations.
        This ensures compilation happens only once at cache creation time using AOT compilation.
        """
        # Dummy inputs for compilation
        dummy_active = jnp.zeros(self.n_active, dtype=self._dtype)
        dummy_target = jnp.eye(4, dtype=self._dtype)
        dummy_initial = jnp.zeros(self.chain_params['n_links'], dtype=self._dtype)
        dummy_reg = jnp.array(0.0, dtype=self._dtype)
        
        self._ik_compiled = {}  # AOT compiled value_and_grad (for adam/gradient_descent)
        self._ik_residuals = {}  # AOT compiled residuals (for scipy)
        self._ik_jacobian = {}   # AOT compiled jacobian (for scipy)
        
        # Compile for each orientation mode and position flag combination
        for orient_mode in [None, "X", "Y", "Z", "all"]:
            for no_pos in [False, True]:
                # Skip invalid combination (no position and no orientation)
                if no_pos and orient_mode is None:
                    continue
                    
                key = (orient_mode, no_pos)
                
                # Compile value_and_grad for adam/gradient_descent
                loss_fn = self._create_loss_function(orient_mode, no_pos)
                value_and_grad_fn = jax.value_and_grad(loss_fn)
                lowered = jax.jit(value_and_grad_fn).lower(
                    dummy_active, dummy_target, dummy_initial, dummy_reg
                )
                self._ik_compiled[key] = lowered.compile()
                
                # Compile residuals and jacobian for scipy least_squares
                residual_fn = self._create_residual_function(orient_mode, no_pos)
                jacobian_fn = jax.jacfwd(residual_fn)  # Jacobian via forward-mode autodiff
                
                # AOT compile both
                self._ik_residuals[key] = jax.jit(residual_fn).lower(
                    dummy_active, dummy_target, dummy_initial
                ).compile()
                self._ik_jacobian[key] = jax.jit(jacobian_fn).lower(
                    dummy_active, dummy_target, dummy_initial
                ).compile()

    def forward_kinematics(self, joints, full_kinematics=False):
        """
        Compute forward kinematics using pre-compiled functions.

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
        joints = jnp.array(joints, dtype=self._dtype)

        if full_kinematics:
            if self._fk_full_compiled is not None:
                result = self._fk_full_compiled(joints)
            else:
                result = self._fk_full_jit(joints)
            return [np.array(result[i]) for i in range(self.chain_params['n_links'])]
        else:
            if self._fk_compiled is not None:
                return np.array(self._fk_compiled(joints))
            else:
                return np.array(self._fk_jit(joints))

    def active_to_full(self, active_joints, initial_position):
        """Convert active joints to full joint array"""
        initial_position = jnp.array(initial_position, dtype=self._dtype)
        return initial_position.at[self.active_indices].set(active_joints)

    def active_from_full(self, joints):
        """Extract active joints from full joint array"""
        return jnp.array(joints, dtype=self._dtype)[self.active_links_mask]

    def inverse_kinematics(self, target_frame, initial_position=None,
                           orientation_mode=None, no_position=False,
                           regularization_parameter=None, max_iter=100,
                           optimizer="scipy", learning_rate=0.05, tol=1e-6):
        """
        Compute inverse kinematics using pre-compiled JAX functions.

        Parameters
        ----------
        target_frame: np.ndarray (4, 4)
            Target pose
        initial_position: np.ndarray
            Initial joint positions
        orientation_mode: str
            Orientation constraint mode: None, "X", "Y", "Z", or "all"
        no_position: bool
            Disable position optimization
        regularization_parameter: float
            Regularization strength (default: 0.0)
        max_iter: int
            Maximum iterations (only for adam/gradient_descent)
        optimizer: str
            Optimizer type: "scipy" (default), "adam", or "gradient_descent"
            - "scipy": scipy least_squares with JAX jacobian (fast & precise)
            - "adam": Adam optimizer (good for batched optimization)
            - "gradient_descent": simple gradient descent
        learning_rate: float
            Learning rate for adam/gradient_descent (default: 0.05)
        tol: float
            Convergence tolerance

        Returns
        -------
        np.ndarray
            Optimal joint positions
        """
        import scipy.optimize
        
        if initial_position is None:
            initial_position = np.zeros(len(self.chain.links))

        # Convert inputs to JAX arrays
        target_frame_jax = jnp.array(target_frame, dtype=self._dtype)
        initial_position_jax = jnp.array(initial_position, dtype=self._dtype)
        reg_param = jnp.array(regularization_parameter or 0.0, dtype=self._dtype)
        
        # Get the pre-compiled functions for this configuration
        key = (orientation_mode, no_position)
        
        # Initial active joints
        x0 = self.active_from_full(initial_position_jax)
        
        if optimizer == "scipy":
            # Use scipy least_squares with JAX-computed jacobian
            residual_fn = self._ik_residuals.get(key)
            jacobian_fn = self._ik_jacobian.get(key)
            
            if residual_fn is None or jacobian_fn is None:
                # Fallback: compile on the fly
                res_fn = self._create_residual_function(orientation_mode, no_position)
                residual_fn = jax.jit(res_fn)
                jacobian_fn = jax.jit(jax.jacfwd(res_fn))
            
            # Wrapper functions for scipy (convert JAX arrays to numpy)
            def residuals_np(x):
                return np.array(residual_fn(
                    jnp.array(x, dtype=self._dtype),
                    target_frame_jax,
                    initial_position_jax
                ))
            
            def jacobian_np(x):
                return np.array(jacobian_fn(
                    jnp.array(x, dtype=self._dtype),
                    target_frame_jax,
                    initial_position_jax
                ))
            
            # Bounds for scipy
            bounds = (np.array(self.lower_bounds), np.array(self.upper_bounds))
            
            # Run scipy least_squares with JAX jacobian
            result = scipy.optimize.least_squares(
                residuals_np,
                np.array(x0),
                jac=jacobian_np,
                bounds=bounds,
                ftol=tol,
                xtol=tol
            )
            optimal_active = jnp.array(result.x, dtype=self._dtype)
        
        elif optimizer == "gradient_descent":
            # Simple gradient descent with bounds
            compiled_fn = self._ik_compiled.get(key)
            if compiled_fn is None:
                loss_fn = self._create_loss_function(orientation_mode, no_position)
                compiled_fn = jax.jit(jax.value_and_grad(loss_fn))
            
            x = x0
            for _ in range(max_iter):
                _, g = compiled_fn(x, target_frame_jax, initial_position_jax, reg_param)
                x = x - learning_rate * g
                x = jnp.clip(x, self.lower_bounds, self.upper_bounds)
                
                # Check convergence
                if jnp.max(jnp.abs(g)) < tol:
                    break
            optimal_active = x
            
        elif optimizer == "adam":
            # Adam optimizer
            compiled_fn = self._ik_compiled.get(key)
            if compiled_fn is None:
                loss_fn = self._create_loss_function(orientation_mode, no_position)
                compiled_fn = jax.jit(jax.value_and_grad(loss_fn))
            
            x = x0
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            beta1, beta2 = 0.9, 0.999
            eps = jnp.array(1e-8, dtype=self._dtype)
            
            for t in range(1, max_iter + 1):
                _, g = compiled_fn(x, target_frame_jax, initial_position_jax, reg_param)
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g ** 2
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                x = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                x = jnp.clip(x, self.lower_bounds, self.upper_bounds)
                
                # Check convergence
                if jnp.max(jnp.abs(g)) < tol:
                    break
            optimal_active = x
            
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Choose 'scipy', 'adam', or 'gradient_descent'.")
        
        # Convert back to full joints
        result = self.active_to_full(optimal_active, initial_position_jax)
        
        # Convert to numpy for compatibility
        return np.array(result)
