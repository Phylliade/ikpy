# coding= utf8
import warnings

import scipy.optimize
import numpy as np
from . import logs

# Check for JAX availability
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


ORIENTATION_COEFF = 1.


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None, orientation_mode=None, no_position=False, optimizer="least_squares", optimizer_kwargs=None):
    """
    Computes the inverse kinematic on the specified target

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The chain used for the Inverse kinematics.
    target_frame: numpy.array
        The desired target.
    starting_nodes_angles: numpy.array
        The initial pose of your chain.
    regularization_parameter: float
        The coefficient of the regularization.
    max_iter: int
        Deprecated and ignored: capping the iterations of the optimizer can harm the quality of the
        IK results. Use `optimizer_kwargs` to give the optimizer a stopping criterion of your own.
    orientation_mode: str
        Orientation to target. Choices:
        * None: No orientation
        * "X": Target the X axis
        * "Y": Target the Y axis
        * "Z": Target the Z axis
        * "all": Target the three axes
    no_position: bool
        Do not optimize against position
    optimizer: str
        The optimizer to use. Choices:
        * "least_squares": Use scipy.optimize.least_squares (the default)
        * "scalar": Use scipy.optimize.minimize (the default prior to IKPy 3.3)
    optimizer_kwargs: dict
        Optional keyword arguments forwarded as-is to the underlying SciPy optimizer, to trade
        accuracy for speed. The accepted keys are the ones of the optimizer in use:
        * "least_squares": :func:`scipy.optimize.least_squares`, e.g. {"max_nfev": 10, "ftol": 1e-4}
        * "scalar": :func:`scipy.optimize.minimize`, e.g. {"options": {"maxiter": 10}, "tol": 1e-4}
        The bounds are derived from the chain itself, so they cannot be overridden here.
    """
    if optimizer not in ["least_squares", "scalar"]:
        raise ValueError("Unknown solver: {}".format(optimizer))

    # Begin with the position
    target = target_frame[:3, -1]

    # Initial function call when optimizing
    def optimize_basis(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        fk = chain.forward_kinematics(y)

        return fk

    # Compute error to target
    def optimize_target_function(fk):
        target_error = (fk[:3, -1] - target)

        # We need to return the fk, it will be used in a later function
        # This way, we don't have to recompute it
        return target_error

    if orientation_mode is None:
        if no_position:
            raise ValueError("Unable to optimize against neither position or orientation")

        else:
            def optimize_function(x):
                fk = optimize_basis(x)
                target_error = optimize_target_function(fk)
                return target_error
    else:
        # Only get the first orientation vector
        if orientation_mode == "X":
            target_orientation = target_frame[:3, 0]

            def get_orientation(fk):
                return fk[:3, 0]

        elif orientation_mode == "Y":
            target_orientation = target_frame[:3, 1]

            def get_orientation(fk):
                return fk[:3, 1]

        elif orientation_mode == "Z":
            target_orientation = target_frame[:3, 2]

            def get_orientation(fk):
                return fk[:3, 2]

        elif orientation_mode == "all":
            target_orientation = target_frame[:3, :3]

            def get_orientation(fk):
                return fk[:3, :3]
        else:
            raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

        if not no_position:
            def optimize_function(x):
                # Note: This function casts x into a np.float64 array, to have good precision in the computation of the gradients
                fk = optimize_basis(x)

                target_error = optimize_target_function(fk)
                orientation_error = (get_orientation(fk) - target_orientation).ravel()

                # Put more pressure on optimizing the distance to target, to avoid being stuck in a local minimum where the orientation is perfectly reached, but the target is nowhere to be reached
                total_error = np.concatenate([target_error, ORIENTATION_COEFF * orientation_error])

                return total_error
        else:
            def optimize_function(x):
                fk = optimize_basis(x)

                orientation_error = (get_orientation(fk) - target_orientation).ravel()
                total_error = orientation_error

                return total_error

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")

    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - chain.active_from_full(starting_nodes_angles))
            return optimize_function(x) + regularization_parameter * regularization
    else:
        optimize_total = optimize_function

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    # real_bounds = real_bounds[chain.first_active_joint:]
    real_bounds = chain.active_from_full(real_bounds)

    logs.logger.info("Bounds: {}".format(real_bounds))

    if max_iter is not None:
        warnings.warn(
            "max_iter is not used anymore in the IK, and using it as a parameter will raise an exception in the "
            "future: capping the iterations can harm the quality of the results. Pass a stopping criterion to the "
            "optimizer instead, with optimizer_kwargs={'max_nfev': 10} for the \"least_squares\" optimizer or "
            "optimizer_kwargs={'options': {'maxiter': 10}} for the \"scalar\" one.",
            DeprecationWarning,
            stacklevel=2)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    elif "bounds" in optimizer_kwargs:
        # The bounds come from the limits of the links, so letting them through would silently
        # detach the solution from the chain it is supposed to describe
        raise ValueError("The bounds are derived from the chain and cannot be overridden in optimizer_kwargs")

    # least squares optimization
    if optimizer == "scalar":
        def optimize_scalar(x):
            return np.linalg.norm(optimize_total(x))
        res = scipy.optimize.minimize(
            optimize_scalar, chain.active_from_full(starting_nodes_angles), bounds=real_bounds, **optimizer_kwargs)
    elif optimizer == "least_squares":
        # We need to unzip the bounds
        real_bounds = np.moveaxis(real_bounds, -1, 0)
        res = scipy.optimize.least_squares(
            optimize_total, chain.active_from_full(starting_nodes_angles), bounds=real_bounds, **optimizer_kwargs)

    if res.status != -1:
        logs.logger.info("Inverse kinematic optimisation OK, termination status: {}".format(res.status))
    else:
        logs.logger.warning("Inverse kinematic optimisation returned an error: termination status: {}".format(res.status))

    return chain.active_to_full(res.x, starting_nodes_angles)
