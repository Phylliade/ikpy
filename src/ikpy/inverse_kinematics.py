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


def _refuse_duplicate(options, keys, argument):
    """Refuse to let a portable argument and its raw SciPy equivalent both set the same knob"""
    clash = [key for key in keys if key in options]
    if clash:
        raise ValueError(
            "{} and optimizer_kwargs={} both set the same option. Pass only one of them.".format(
                argument, {key: options[key] for key in clash}))


def resolve_optimizer_options(optimizer, optimizer_budget=None, tol=None, optimizer_kwargs=None):
    """
    Translate the portable optimizer arguments into the keys of the optimizer in use.

    `optimizer_budget` and `tol` mean the same thing whatever the optimizer or the backend, and
    this is where that promise is kept. Everything else travels through `optimizer_kwargs`, which
    is passed to SciPy untouched.

    Parameters
    ----------
    optimizer: str
        The optimizer the options are destined for: "least_squares" or "scalar".
    optimizer_budget: int
        Approximate number of evaluations the optimizer is allowed.
    tol: float
        Convergence tolerance.
    optimizer_kwargs: dict
        Raw keyword arguments for the SciPy optimizer.

    Returns
    -------
    dict
        The keyword arguments to give to the SciPy optimizer.
    """
    options = dict(optimizer_kwargs) if optimizer_kwargs is not None else {}

    if "bounds" in options:
        # The bounds come from the limits of the links, so letting them through would silently
        # detach the solution from the chain it is supposed to describe
        raise ValueError("The bounds are derived from the chain and cannot be overridden in optimizer_kwargs")

    if optimizer == "scalar":
        if optimizer_budget is not None:
            nested = dict(options.get("options") or {})
            _refuse_duplicate(nested, ["maxfun", "maxiter"], "optimizer_budget")
            nested["maxfun"] = optimizer_budget
            options["options"] = nested
        if tol is not None:
            _refuse_duplicate(options, ["tol"], "tol")
            options["tol"] = tol
    else:
        if optimizer_budget is not None:
            _refuse_duplicate(options, ["max_nfev"], "optimizer_budget")
            options["max_nfev"] = optimizer_budget
        if tol is not None:
            _refuse_duplicate(options, ["ftol", "xtol"], "tol")
            options["ftol"] = tol
            options["xtol"] = tol

    return options


def strictly_inside_bounds(x0, bounds, margin=1e-3):
    """Return a starting point strictly inside the bounds, for the trust-region optimizers of SciPy

    Parameters
    ----------
    x0: numpy.array
        The starting point, one value per active link
    bounds: numpy.array
        The bounds of the active links, of shape (n_links, 2)
    margin: float
        The distance kept from each finite bound, shrunk when the range of a link is narrower

    Returns
    -------
    numpy.array
        The starting point, moved inside the bounds where needed

    Notes
    -----
    `scipy.optimize.least_squares` only nudges a starting point lying on a bound by about 1e-10, and sets its
    initial trust radius to the norm of the (scaled) starting point. A chain starting with every joint at zero,
    one of them on a bound such as a prismatic joint with `lower="0"`, therefore gets a trust radius of 1e-10:
    the first step is negligible, the relative decrease of the cost falls under `ftol`, and the optimizer stops
    on its starting point. Keeping a real margin from the bounds avoids that. A starting point outside the
    bounds is clipped instead of being refused by SciPy.
    """
    x0 = np.array(x0, dtype=float)
    lower, upper = np.asarray(bounds, dtype=float).T
    finite_range = np.where(np.isfinite(lower) & np.isfinite(upper), upper - lower, np.inf)
    margins = np.minimum(margin, finite_range / 4)
    low = np.where(np.isfinite(lower), lower + margins, -np.inf)
    high = np.where(np.isfinite(upper), upper - margins, np.inf)
    outside = (x0 < lower) | (x0 > upper)
    if np.any(outside):
        warnings.warn("The initial position is outside the bounds of the links {}: it is clipped to the bounds".format(np.flatnonzero(outside).tolist()))
    return np.clip(x0, low, high)


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None, orientation_mode=None, no_position=False, optimizer="least_squares", optimizer_budget=None, tol=None, optimizer_kwargs=None):
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
        Deprecated and ignored: it never named a real stopping criterion. Use `optimizer_budget`.
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
    optimizer_budget: int
        Approximate number of evaluations the optimizer is allowed, to trade accuracy for speed.
        It means the same thing for every optimizer and every backend. It is a budget rather than
        a hard cap: an optimizer finishing a gradient estimation can overshoot it slightly.
    tol: float
        Convergence tolerance. Like `optimizer_budget`, it means the same thing everywhere.
    optimizer_kwargs: dict
        Escape hatch for the options that have no portable equivalent, forwarded as-is to the
        SciPy optimizer in use: :func:`scipy.optimize.least_squares` for "least_squares",
        :func:`scipy.optimize.minimize` for "scalar". Prefer `optimizer_budget` and `tol` when
        they cover your need. The bounds are derived from the chain, so they cannot be set here,
        and neither can an option a portable argument is already setting.
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
            "future. Use optimizer_budget instead, which bounds the work of the optimizer the same way for every "
            "optimizer and every backend.",
            DeprecationWarning,
            stacklevel=2)

    optimizer_options = resolve_optimizer_options(optimizer, optimizer_budget, tol, optimizer_kwargs)

    # The starting point must be strictly inside the bounds, see strictly_inside_bounds
    x0 = strictly_inside_bounds(chain.active_from_full(starting_nodes_angles), real_bounds)

    # least squares optimization
    if optimizer == "scalar":
        def optimize_scalar(x):
            return np.linalg.norm(optimize_total(x))
        res = scipy.optimize.minimize(optimize_scalar, x0, bounds=real_bounds, **optimizer_options)
    elif optimizer == "least_squares":
        # We need to unzip the bounds
        real_bounds = np.moveaxis(real_bounds, -1, 0)
        res = scipy.optimize.least_squares(optimize_total, x0, bounds=real_bounds, **optimizer_options)

    if res.status != -1:
        logs.logger.info("Inverse kinematic optimisation OK, termination status: {}".format(res.status))
    else:
        logs.logger.warning("Inverse kinematic optimisation returned an error: termination status: {}".format(res.status))

    return chain.active_to_full(res.x, starting_nodes_angles)
