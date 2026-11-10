# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""
import numpy as np
import json
import os
from typing import List
import warnings

# IKPY imports
from .urdf import URDF
from .mjcf import MJCF
from .bvh import BVH
from . import inverse_kinematics as ik
from . import link as link_lib
from . import logs

# Check for JAX availability
try:
    from . import jax_backend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _is_bool(value):
    return isinstance(value, (bool, np.bool_))


def _parse_links_mask(active_links_mask, links):
    """Split an active_links_mask into the active links and the values of the held ones

    Returns
    -------
    numpy.array, numpy.array
        The boolean mask of the active links, and the value of every held link (NaN for the other links)
    """
    active = np.zeros(len(links), dtype=bool)
    fixed_values = np.full(len(links), np.nan)
    legacy = []
    for index, value in enumerate(active_links_mask):
        if _is_bool(value):
            if value:
                active[index] = True
            else:
                legacy.append(index)
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            fixed_values[index] = float(value)
        else:
            raise ValueError("The active_links_mask takes True for an active link or a number for a held one, got {!r} for link {} (index: {})".format(value, links[index].name, index))
    if not active.any() and not np.isnan(fixed_values).all():
        raise ValueError("The active_links_mask {} has no active link: an active link is True, a number holds the link at that value (1 is not True)".format(list(active_links_mask)))
    if legacy:
        warnings.warn("False in the active_links_mask is deprecated (links {}): give the value at which to hold the link instead, e.g. 0. With False, the value is read from the starting point of the inverse kinematics".format([links[index].name for index in legacy]), FutureWarning, stacklevel=3)
    for index in np.flatnonzero(~np.isnan(fixed_values)):
        lower, upper = links[index].bounds
        if not lower <= fixed_values[index] <= upper:
            warnings.warn("Link {} (index: {}) is held at {}, outside of its bounds {}".format(links[index].name, index, fixed_values[index], links[index].bounds))
    return active, fixed_values


def _is_unbounded_prismatic(link):
    return link.joint_type == "prismatic" and not np.all(np.isfinite(link.bounds))


def _default_links_mask(links):
    """The active_links_mask of a chain created without one: every link active, except the fixed links, held at 0,
    and the prismatic links without a bound, held at 0 clipped to their bounds

    Returns
    -------
    numpy.array, numpy.array
        The boolean mask of the active links, and the value of every held link (NaN for the other links)
    """
    held = np.array([link.joint_type == "fixed" or _is_unbounded_prismatic(link) for link in links], dtype=bool)
    unbounded = [link.name for link in links if _is_unbounded_prismatic(link)]
    if unbounded:
        warnings.warn("The prismatic links {} have no bound on one side or both, so they are held at 0, clipped to their bounds: give them bounds to move them".format(unbounded), stacklevel=4)
    values = [float(np.clip(0.0, *link.bounds)) if _is_unbounded_prismatic(link) else 0.0 for link in links]
    return ~held, np.where(held, values, np.nan)


class Chain:
    """The base Chain class

    Parameters
    ----------
    links: list[ikpy.link.Link]
        List of the links of the chain
    active_links_mask: list
        One element per link: ``True`` for an active link, moved by the inverse kinematics, or a number for an
        inactive link, held at that value. An active prismatic link must have finite bounds. Defaults to every
        link active, except the fixed links and the prismatic links without a bound, held at 0 (clipped to their
        bounds); the resulting
        mask is :attr:`links_mask`, which can be edited and given back For example ``[0, 0.3, True, True, True, 0]`` moves the links 2 to 4
        and holds the second one at 0.3. Defaults to every link active. ``False`` (an inactive link whose value is
        read from the starting point) is deprecated: give the value instead
    name: str
        The name of the Chain
    urdf_metadata
        Technical attribute
    jax_precompile: bool
        If True (default), JAX functions are compiled when the JAX cache is first accessed.
        If False, compilation is deferred to the first JAX call (faster init, slower first call).
    """
    def __init__(self, links, active_links_mask=None, name="chain", urdf_metadata=None, jax_precompile=True, **kwargs):
        self.name = name
        self.links = links
        self._length = sum([link.length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link.length == 0:
                link.axis_length = self.links[index - 1].axis_length
        # Optional argument
        self._urdf_metadata = urdf_metadata

        # JAX backend cache (lazily initialized)
        self._jax_cache = None
        self._jax_precompile = jax_precompile

        # If the active_links_mask is not given, set it to True for every link
        if active_links_mask is not None:
            if len(active_links_mask) != len(self.links):
                raise ValueError("Your active links mask length of {} is different from the number of your links, which is {}".format(len(active_links_mask), len(self.links)))
            self.active_links_mask, self.fixed_values = _parse_links_mask(active_links_mask, self.links)
        else:
            self.active_links_mask, self.fixed_values = _default_links_mask(self.links)

        # The last link is usually the end effector, which is not meant to be an optimization variable. Only warn,
        # without overriding: it can also be a genuine joint, e.g. in a chain without a tip link. As for mimic
        # joints below, the default mask stays silent, and a fixed last link is already reported by the next check.
        if active_links_mask is not None and self.active_links_mask[-1] and self.links[-1].joint_type != "fixed":
            warnings.warn("The last link {} (index: {}) is set as active in the active_links_mask. The last link is usually the end effector, which is not meant to be active: check that this is intended".format(self.links[-1].name, len(self.links) - 1))

        # Check that none of the active links are fixed
        for link_index, (link_active, link) in enumerate(zip(self.active_links_mask, self.links)):
            if link.joint_type == "fixed" and link_active:
                warnings.warn("Link {} (index: {}) is of type 'fixed' but set as active in the active_links_mask. In practice, this fixed link doesn't provide any transformation so is as it were inactive".format(link.name, link_index))

        # Mimic joints are dependent: a mimicking joint is not a degree of freedom of the robot, its value is
        # derived from its master joint (see _resolve_mimic_dependencies). So it is never active, i.e. never an
        # optimization variable, whether its master is in the chain or not. With the default mask this is
        # silent; it is only worth a warning when the user explicitly asked for the joint to be active.
        self.mimic_dependencies, orphan_mimics = self._resolve_mimic_dependencies()
        for index in [dependency[0] for dependency in self.mimic_dependencies] + orphan_mimics:
            if self.active_links_mask[index]:
                if active_links_mask is not None:
                    warnings.warn("Link {} (index: {}) mimics joint {} but is set as active in the active_links_mask: a mimicking joint is never active, its value is derived from its master joint. Overriding and setting to False".format(self.links[index].name, index, self.links[index].mimic_joint))
                self.active_links_mask[index] = False
            elif not np.isnan(self.fixed_values[index]):
                warnings.warn("Link {} (index: {}) mimics joint {} but is held at {} in the active_links_mask: a mimicking joint takes the value derived from its master joint. Ignoring the given value".format(self.links[index].name, index, self.links[index].mimic_joint, self.fixed_values[index]))
                self.fixed_values[index] = np.nan
        self._tighten_mimic_master_bounds()

        # A prismatic link without a bound is either the position of the robot in the world (e.g. the root of a
        # BVH skeleton) or a missing limit: it is not meant to be moved by the inverse kinematics, and leaves no
        # range to draw starting points from. Checked after the bounds of the mimic masters are tightened.
        unbounded = [link.name for link, active in zip(self.links, self.active_links_mask) if active and _is_unbounded_prismatic(link)]
        if unbounded:
            raise ValueError("The prismatic links {} are active without a bound on one side or both: give them bounds, or hold them in the active_links_mask with a number, e.g. 0".format(unbounded))

    def _resolve_mimic_dependencies(self):
        """Resolve the mimic relations of the chain's links

        Returns
        -------
        dependencies: list of (index, master_index, multiplier, offset)
            The joint at `index` takes the value
            `multiplier * joints[master_index] + offset`. The entries are
            ordered so that a master always comes before the joints
            mimicking it, which also handles a joint mimicking another
            mimicking joint.
        orphans: list of int
            The indices of the mimicking joints whose master is not part of
            this chain (the fingers of a gripper are typically siblings, so
            a chain to one finger never contains the other). Their value
            cannot be derived: it is the one given in the joints vector,
            like for any inactive link. A warning says so.
        """
        index_by_name = {}
        for index, link in enumerate(self.links):
            index_by_name.setdefault(link.name, index)

        relations = {}
        orphans = []
        for index, link in enumerate(self.links):
            master_name = link.mimic_joint
            if master_name is None:
                continue
            if master_name == link.name:
                raise ValueError("Joint {} mimics itself".format(link.name))
            if master_name not in index_by_name:
                warnings.warn("Joint {} mimics joint {}, which is not part of this chain: its value cannot be derived, it is kept inactive and takes the value given in the joints vector".format(link.name, master_name))
                orphans.append(index)
                continue
            relations[index] = (index_by_name[master_name], link.mimic_multiplier, link.mimic_offset)

        ordered = []
        resolved = set()
        pending = dict(relations)
        while pending:
            progressed = False
            for index in list(pending):
                master_index, multiplier, offset = pending[index]
                if master_index not in relations or master_index in resolved:
                    ordered.append((index, master_index, multiplier, offset))
                    resolved.add(index)
                    del pending[index]
                    progressed = True
            if not progressed:
                raise ValueError("Circular mimic relation between joints: {}".format([self.links[index].name for index in pending]))
        return ordered, orphans

    def _tighten_mimic_master_bounds(self):
        """Narrow the bounds of each master joint so that the joints mimicking it stay within their own limits

        The optimizer only sees the master, so the limits of a mimicking joint would otherwise be ignored:
        with ``value = multiplier * master + offset`` they translate into ``master`` within
        ``[(lower - offset) / multiplier, (upper - offset) / multiplier]`` (swapped for a negative multiplier).
        The dependencies are walked from the last to the first, so that the limits of a joint mimicking
        another mimicking joint reach the root master.
        """
        for (index, master_index, multiplier, offset) in reversed(self.mimic_dependencies):
            if multiplier == 0:
                # The mimicking joint is constant, it says nothing about its master
                continue
            lower, upper = self.links[index].bounds
            lower, upper = (lower - offset) / multiplier, (upper - offset) / multiplier
            if multiplier < 0:
                lower, upper = upper, lower
            master = self.links[master_index]
            new_bounds = (max(master.bounds[0], lower), min(master.bounds[1], upper))
            if new_bounds[0] > new_bounds[1]:
                warnings.warn("The limits of joint {} and of joint {}, which mimics it, cannot both be satisfied: the limits of {} are ignored".format(master.name, self.links[index].name, self.links[index].name))
                continue
            if new_bounds != tuple(master.bounds):
                logs.logger.info("Bounds of joint {} narrowed to {} by the limits of joint {}, which mimics it".format(master.name, new_bounds, self.links[index].name))
                master.bounds = new_bounds

    def _derive_mimic_joints(self, joints):
        """Set each mimicking joint to ``multiplier * master + offset``, in place, and return the joints"""
        for (index, master_index, multiplier, offset) in self.mimic_dependencies:
            joints[index] = multiplier * joints[master_index] + offset
        return joints

    def __repr__(self):
        return "Kinematic chain name={} links={} active_links={}".format(self.name, [link.name for link in self.links], self.active_links_mask)

    def __len__(self):
        return len(self.links)

    @property
    def jax_cache(self):
        """
        Get or initialize the JAX kinematics cache.

        Returns
        -------
        JaxKinematicsCache
            The JAX cache for this chain

        Raises
        ------
        ImportError
            If JAX is not installed
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Install it with: pip install jax jaxlib")

        if self._jax_cache is None:
            self._jax_cache = jax_backend.JaxKinematicsCache(self, precompile=self._jax_precompile)

        return self._jax_cache

    def forward_kinematics(self, joints: List, full_kinematics=False, backend: str = "numpy"):
        """Returns the transformation matrix of the forward kinematics

        Parameters
        ----------
        joints: list
            The list of the positions of each joint. Note : Inactive joints must be in the list. The value
            given for a joint mimicking another joint of the chain is ignored: it is derived from its master,
            see :attr:`mimic_dependencies`.
        full_kinematics: bool
            Return the transformation matrices of each joint
        backend: str
            The computational backend to use: "numpy" (default) or "jax".
            The JAX backend provides JIT compilation and automatic differentiation.

        Returns
        -------
        frame_matrix:
            The transformation matrix
        """
        if len(self.links) != len(joints):
            raise ValueError("Your joints vector length is {} but you have {} links".format(len(joints), len(self.links)))

        # Use JAX backend if requested
        if backend == "jax":
            return self.jax_cache.forward_kinematics(joints, full_kinematics=full_kinematics)

        # Default numpy implementation
        if self.mimic_dependencies:
            # The dependent (mimic) joints take the value derived from their master, whatever was given for them
            joints = self._derive_mimic_joints(list(joints))
        frame_matrix = np.eye(4)

        if full_kinematics:
            frame_matrixes = []

        for index, (link, joint_parameters) in enumerate(zip(self.links, joints)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            # FIXME: The casting to array is a loss of time
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix(joint_parameters)))
            if full_kinematics:
                # rotation_axe = np.dot(frame_matrix, link.rotation)
                frame_matrixes.append(frame_matrix)

        # Return the matrix, or matrixes
        if full_kinematics:
            return frame_matrixes
        else:
            return frame_matrix

    @staticmethod
    def _target_frame(target_position, target_orientation, orientation_mode):
        """Build the 4x4 target frame of inverse_kinematics. Returns (frame, no_position)"""
        frame_target = np.eye(4)

        # Compute orientation
        if orientation_mode is not None:
            if orientation_mode == "X":
                frame_target[:3, 0] = target_orientation
            elif orientation_mode == "Y":
                frame_target[:3, 1] = target_orientation
            elif orientation_mode == "Z":
                frame_target[:3, 2] = target_orientation
            elif orientation_mode == "all":
                frame_target[:3, :3] = target_orientation
            else:
                raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

        # Compute target
        if target_position is None:
            no_position = True
        else:
            no_position = False
            frame_target[:3, 3] = target_position

        return frame_target, no_position

    def inverse_kinematics(self, target_position=None, target_orientation=None, orientation_mode=None, backend: str = "numpy", **kwargs):
        """

        Parameters
        ----------
        target_position: np.ndarray
            Vector of shape (3,): the target point
        target_orientation: np.ndarray
            Vector of shape (3,): the target orientation
        orientation_mode: str
            Orientation to target. Choices:
            * None: No orientation
            * "X": Target the X axis
            * "Y": Target the Y axis
            * "Z": Target the Z axis
            * "all": Target the entire frame (e.g. the three axes) (not currently supported)
        backend: str
            The computational backend to use: "numpy" (default) or "jax".
            The JAX backend uses automatic differentiation for gradient computation.
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        frame_target, no_position = self._target_frame(target_position, target_orientation, orientation_mode)
        return self.inverse_kinematics_frame(target=frame_target, orientation_mode=orientation_mode, no_position=no_position, backend=backend, **kwargs)

    def sample_initial_positions(self, n_samples, seed=None, around=None, radius=None):
        """Draw starting points for the inverse kinematics, inside the joint limits

        See :func:`ikpy.inverse_kinematics.sample_initial_positions`. The result can be given as the
        `initial_position` of :meth:`inverse_kinematics`, which then keeps the best of the solutions reached
        from every point::

            starts = chain.sample_initial_positions(10, seed=0)

        `around` and `radius` are deprecated: give the values of the held links in the active_links_mask, and
        the starting points to keep as `initial_position` (a list of configurations, or a dict with ``points``)
        """
        return ik.sample_initial_positions(self, n_samples, seed=seed, around=around, radius=radius)

    @property
    def links_mask(self):
        """The active_links_mask of the chain, as given to the constructor: True for an active link, the value of
        a held link, or False for the deprecated inactive links that read their value from the starting point.

        To change it, edit it and create the chain again with it::

            mask = chain.links_mask
            mask[4] = 0
            chain = Chain.from_urdf_file(urdf_file, active_links_mask=mask)
        """
        return [True if active else (False if np.isnan(value) else float(value))
                for active, value in zip(self.active_links_mask, self.fixed_values)]

    def _parse_starts(self, starts, name="initial_position"):
        """Read one configuration or several, mirroring the active_links_mask

        An active link takes a number, its starting value. A held link takes False, its value then comes from
        the active_links_mask; a number is accepted too, with a warning, but the value of the mask is used.

        Returns
        -------
        numpy.array
            The configurations, of shape (n_links,) or (n_starts, n_links), with the values of the held links
        """
        n_links = len(self.links)
        if isinstance(starts, np.ndarray) and starts.dtype.kind in "fiu":
            values = starts.astype(float)
            given_false = np.zeros(values.shape, dtype=bool)
        else:
            # Through object, not to lose False among numbers: np.array([False, 0.1]) is [0., 0.1]
            raw = np.asarray(starts, dtype=object)
            if raw.ndim not in (1, 2) or raw.shape[-1] != n_links:
                values = raw
            else:
                if any(_is_bool(value) and value for value in raw.flat):
                    raise ValueError("{} takes a number for an active link and False for a held one, not True".format(name))
                given_false = np.vectorize(_is_bool, otypes=[bool])(raw)
                try:
                    values = np.where(given_false, 0.0, raw).astype(float)
                except (TypeError, ValueError):
                    raise ValueError("{} must contain numbers, or False for the held links, got {!r}".format(name, starts))
        if values.ndim not in (1, 2) or values.shape[-1] != n_links:
            raise ValueError("{} must be a configuration of shape (n_links,), an int, a dict, or an array of starting points of shape (n_starts, n_links); got an array of shape {} for {} links".format(name, np.shape(values), n_links))
        false_on_active = given_false[..., self.active_links_mask]
        if false_on_active.any():
            names = [link.name for link, active in zip(self.links, self.active_links_mask) if active]
            raise ValueError("{} gives False for the active links {}: they need a starting value".format(name, sorted({names[index] for index in np.argwhere(false_on_active)[:, -1]})))
        held = ~np.isnan(self.fixed_values)
        if held.any():
            given_number = ~given_false[..., held]
            if given_number.any():
                held_names = [link.name for link, is_held in zip(self.links, held) if is_held]
                warnings.warn("{} gives numbers for the held links {}: they take the values of the active_links_mask, give False for them".format(name, sorted({held_names[index] for index in np.argwhere(given_number)[:, -1]})), stacklevel=4)
            values[..., held] = self.fixed_values[held]
        return values

    def _as_mirror(self, configurations):
        """Configurations with False for the held links, to be given back as initial_position without warning"""
        mirror = np.array(configurations, dtype=object)
        mirror[..., ~np.isnan(self.fixed_values)] = False
        return mirror

    def _resolve_initial_position(self, initial_position):
        """Turn the initial_position argument of the inverse kinematics into starting points

        Returns
        -------
        numpy.array
            A single starting point, of shape (n_links,), or several, of shape (n_starts, n_links)
        """
        if initial_position is None:
            held = ~np.isnan(self.fixed_values)
            return np.where(held, np.nan_to_num(self.fixed_values), 0.0)
        if isinstance(initial_position, (int, np.integer)) and not _is_bool(initial_position):
            return self.sample_initial_positions(int(initial_position))
        if isinstance(initial_position, dict):
            spec = dict(initial_position)
            unknown = set(spec) - {"points", "n", "seed", "around", "radius"}
            if unknown:
                raise ValueError("Unknown keys in initial_position: {}. Expected points, n, seed".format(sorted(unknown)))
            points = spec.pop("points", None)
            if points is None or len(points) == 0:
                points = np.zeros((0, len(self.links)))
            else:
                points = self._parse_starts(points, name="initial_position['points']")
                if points.ndim != 2:
                    raise ValueError("initial_position['points'] must be an array of shape (n_points, n_links), got {} for {} links".format(points.shape, len(self.links)))
            n_samples = spec.pop("n", 0)
            sampled = self.sample_initial_positions(n_samples, **spec) if n_samples > 0 else np.zeros((0, len(self.links)))
            starts = np.concatenate([points, sampled])
            if len(starts) == 0:
                raise ValueError("initial_position gives no starting point: add n (points to draw) or points (given ones)")
            return starts
        return self._parse_starts(initial_position)

    def _inverse_kinematics_solutions_frame(self, target, initial_position=10, distinct_tol=1e-2, max_error=None, backend: str = "numpy", **kwargs):
        """Computes the distinct solutions of the inverse kinematics on a frame, reached from several starting points

        The optimizer is local: it converges to the solution closest to its starting point. Running it from
        several starting points spread inside the joint limits finds the other solutions of a target (elbow up
        and elbow down, wrist flipped...), and the best one when a single start lands in a poor local minimum.

        This is the engine of the multi-start mode of :meth:`inverse_kinematics_frame`, which returns the first
        row of its result. It is kept out of the main API, but it is meant to be called when all the solutions
        are wanted, to pick one on a criterion of your own (closest to the current configuration, elbow up...)::

            frame = chain.forward_kinematics(joints)
            solutions = chain._inverse_kinematics_solutions_frame(frame, initial_position=20, max_error=1e-3, orientation_mode="all")

        Parameters
        ----------
        target: numpy.array
            The 4x4 target frame
        initial_position: numpy.array, int or dict
            The starting points, in the forms accepted by :meth:`inverse_kinematics_frame`: a configuration
            (a single start), an int, a dict, or an array of shape (n_starts, n_links). Defaults to 10 points
            drawn anywhere in the joint limits
        distinct_tol: float
            Two solutions whose active joints differ by less than this (infinity norm) are the same solution
        max_error: float
            Optional : drop the solutions whose error against the target (the norm of the residual, in meters
            for the position part) is above this value, typically the starting points that ended in a local
            minimum away from the target. By default every distinct local minimum is returned, so an
            unreachable target still gives its best approximations
        backend: str
            The computational backend to use: "numpy" (default) or "jax"
        kwargs: See :meth:`inverse_kinematics_frame`

        Returns
        -------
        numpy.array
            The distinct solutions, of shape (n_solutions, n_links), sorted by increasing error against the
            target. The first row is the best one.
        """
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")
        starts = np.atleast_2d(self._resolve_initial_position(initial_position))
        orientation_mode = kwargs.get("orientation_mode")
        no_position = kwargs.get("no_position", False)

        solutions = []
        errors = []
        for start in starts:
            solution = self.inverse_kinematics_frame(target, initial_position=self._as_mirror(start), backend=backend, **kwargs)
            solutions.append(np.array(solution, dtype=float))
            errors.append(np.linalg.norm(ik.target_error(self, solution, target, orientation_mode, no_position)))

        # Keep the distinct solutions, best first
        distinct = []
        for index in np.argsort(errors, kind="stable"):
            if max_error is not None and errors[index] > max_error:
                break
            candidate = solutions[index]
            if all(np.max(np.abs(self.active_from_full(candidate - kept))) > distinct_tol for kept in distinct):
                distinct.append(candidate)
        return np.array(distinct).reshape((len(distinct), len(self.links)))

    def inverse_kinematics_trajectory(self, target_positions=None, target_orientations=None, orientation_mode=None, initial_position=None, backend: str = "numpy", **kwargs):
        """Computes the inverse kinematics along a sequence of targets, each solution starting from the previous one

        Same conventions as :meth:`inverse_kinematics`, with one target per step. See
        :meth:`inverse_kinematics_trajectory_frame` for the behaviour and the returned value.

        Parameters
        ----------
        target_positions: numpy.array
            The target positions, of shape (n_steps, 3), or None to target orientations only
        target_orientations: numpy.array
            The target orientations, one per step: of shape (n_steps, 3) for the "X", "Y" and "Z" modes,
            (n_steps, 3, 3) for the "all" mode, or None
        orientation_mode: str
            See :meth:`inverse_kinematics`
        initial_position: numpy.array, int or dict
            Optional : where the first step starts, as in :meth:`inverse_kinematics_frame`. Defaults to 0 for
            each joint
        backend: str
            The computational backend to use: "numpy" (default) or "jax"
        kwargs: See :meth:`inverse_kinematics_trajectory_frame`
        """
        if target_positions is None and target_orientations is None:
            raise ValueError("Give target_positions, target_orientations or both")
        n_steps = len(target_positions) if target_positions is not None else len(target_orientations)
        if target_positions is not None and len(target_positions) != n_steps:
            raise ValueError("target_positions and target_orientations must have the same length")
        if target_orientations is not None and len(target_orientations) != n_steps:
            raise ValueError("target_positions and target_orientations must have the same length")

        frames = []
        no_position = False
        for step in range(n_steps):
            position = None if target_positions is None else target_positions[step]
            orientation = None if target_orientations is None else target_orientations[step]
            frame, no_position = self._target_frame(position, orientation, orientation_mode)
            frames.append(frame)
        return self.inverse_kinematics_trajectory_frame(frames, initial_position=initial_position, orientation_mode=orientation_mode, no_position=no_position, backend=backend, **kwargs)

    def inverse_kinematics_trajectory_frame(self, targets, initial_position=None, backend: str = "numpy", **kwargs):
        """Computes the inverse kinematics along a sequence of target frames, each solution starting from the previous one

        Solving each target from the solution of the previous one keeps the motion continuous: the optimizer is
        local, so it returns the configuration closest to where the chain already is, instead of jumping to
        another branch of solutions. To move between two frames, interpolate them first with
        :func:`ikpy.utils.geometry.interpolate_frames`.

        Parameters
        ----------
        targets: numpy.array
            The target frames, of shape (n_steps, 4, 4)
        initial_position: numpy.array, int or dict
            Optional : where the first step starts, as in :meth:`inverse_kinematics_frame`. Defaults to 0 for
            each joint. An int or a dict runs the multi-start on the first step, which picks the branch of
            solutions the trajectory then follows
        backend: str
            The computational backend to use: "numpy" (default) or "jax"
        kwargs: See :meth:`inverse_kinematics_frame`. `regularization_parameter` pulls every step towards the
            previous solution, for smoother joint motion at the price of accuracy.

        Returns
        -------
        numpy.array
            The configurations, of shape (n_steps, n_links), one row per target. A target out of reach gets
            the closest configuration, as :meth:`inverse_kinematics_frame` does; check it with
            :meth:`forward_kinematics` if needed.
        """
        targets = np.asarray(targets, dtype=float)
        if targets.ndim != 3 or targets.shape[1:] != (4, 4):
            raise ValueError("targets must be an array of 4x4 transformation matrices, of shape (n_steps, 4, 4)")
        if len(targets) == 0:
            return np.zeros((0, len(self.links)))
        trajectory = np.zeros((len(targets), len(self.links)))
        previous = initial_position
        for step, target in enumerate(targets):
            trajectory[step] = self.inverse_kinematics_frame(target, initial_position=previous, backend=backend, **kwargs)
            previous = self._as_mirror(trajectory[step])
        return trajectory

    def inverse_kinematics_frame(self, target, initial_position=None, backend: str = "numpy", **kwargs):
        """Computes the inverse kinematic on the specified target

        Parameters
        ----------
        target: numpy.array
            The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        initial_position: numpy.array, int or dict
            Where the optimizer starts. Defaults to 0 for each joint. The optimizer is local and converges to
            the solution closest to its starting point, so this argument also drives the multi-start mode:

            * a configuration, of shape (n_links,): a single start, the usual case. It mirrors the
              active_links_mask: a number for each active link, False for each held link, which takes the
              value of the mask (a number is accepted, with a warning, but the mask's value is used);
            * an int: that many starting points drawn anywhere in the joint limits, the best solution is
              returned;
            * a dict combining given points and drawn ones: ``points``, a list of configurations, plus ``n``
              points drawn by :meth:`sample_initial_positions` with ``seed``. For example
              ``{"points": [current_joints], "n": 9, "seed": 0}`` starts from the current configuration and 9
              drawn points. ``n`` defaults to 0;
            * a list of configurations, of shape (n_starts, n_links): the starting points themselves.

            To get every solution rather than the best one, see :meth:`_inverse_kinematics_solutions_frame`.
        backend: str
            The computational backend to use: "numpy" (default) or "jax".
            The JAX backend uses scipy least_squares with analytical Jacobian via autodiff.
            For JAX, additional kwargs are supported:
            * tol: float (default: 1e-6)
            * use_analytical_jacobian: bool (default: True)
            * scipy_method: 'trf' (default), 'dogbox', or 'lm'
            * scipy_x_scale: 'jac' (default) for auto-scaling
            * scipy_loss: 'linear' (default), 'soft_l1', 'huber', etc.
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")

        starts = self._resolve_initial_position(initial_position)
        if starts.ndim == 2:
            # Multi-start: the best of the solutions reached from every starting point
            return self._inverse_kinematics_solutions_frame(target, self._as_mirror(starts), backend=backend, **kwargs)[0]
        initial_position = starts

        # Use JAX backend if requested - delegate to jax_cache for pre-compiled functions
        if backend == "jax":
            # Extract JAX-specific kwargs
            jax_kwargs = {}
            jax_specific_keys = ['tol', 'optimizer_budget', 'optimizer_kwargs',
                                 'use_analytical_jacobian', 'scipy_method',
                                 'scipy_x_scale', 'scipy_loss', 'scipy_gtol',
                                 'scipy_max_nfev', 'scipy_tr_solver', 'scipy_tr_options',
                                 'scipy_verbose']
            for key in jax_specific_keys:
                if key in kwargs:
                    jax_kwargs[key] = kwargs.pop(key)

            # Map common kwargs
            if 'orientation_mode' in kwargs:
                jax_kwargs['orientation_mode'] = kwargs.pop('orientation_mode')
            if 'no_position' in kwargs:
                jax_kwargs['no_position'] = kwargs.pop('no_position')

            if kwargs:
                # Anything left over was destined for the numpy optimizer and used to be dropped
                # here without a word, which is exactly how max_iter went unnoticed for years.
                # This will raise in the next major version.
                warnings.warn(
                    "Ignored by the \"jax\" backend: {}. Use optimizer_budget and tol, which both "
                    "backends understand. Passing them will raise in a future version.".format(
                        ", ".join(sorted(kwargs))),
                    UserWarning,
                    stacklevel=2)

            # Use the jax_cache which has pre-compiled functions
            return self.jax_cache.inverse_kinematics(
                target, initial_position, **jax_kwargs
            )

        # Default numpy/scipy implementation
        return ik.inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)

    def plot(self, joints, ax, target=None, show=False):
        """Plots the Chain using Matplotlib

        Parameters
        ----------
        joints: list
            The list of the positions of each joint
        ax: matplotlib.axes.Axes
            A matplotlib axes
        target: numpy.array
            An optional target
        show: bool
            Display the axe. Defaults to False
        """
        from ikpy.utils import plot

        if ax is None:
            # If ax is not given, create one
            _, ax = plot.init_3d_figure()
        plot.plot_chain(self, joints, ax, name=self.name)

        # Plot the goal position
        if target is not None:
            plot.plot_target(target, ax)
        if show:
            plot.show_figure()

    @property
    def _json_path(self):
        """Path where the JSON is expected to be"""
        return os.path.dirname(self._urdf_metadata["urdf_file"]) + "/" + self.name + ".json"

    @classmethod
    def from_json_file(cls, json_file):
        """
        Load a chain serialized in the JSON format.
        This is basically a URDF file with some metadata

        Parameters
        ----------
        json_file: str
            Path to the json file serializing the robot

        Returns
        -------

        """
        with open(json_file, "r") as fd:
            chain_config = json.load(fd)

        # The path where is stored the URDF file
        chain_basedir = os.path.dirname(json_file)

        # Get the different attributes
        urdf_file = chain_basedir + "/" + chain_config["urdf_file"]
        active_links_mask = chain_config.get("active_links_mask")
        if active_links_mask == "":
            active_links_mask = None
        name = chain_config["name"]

        if "base" in chain_config or "tip" in chain_config:
            # Format v2: the chain is selected with base and tip
            return cls.from_urdf_file(
                urdf_file=urdf_file,
                base=chain_config.get("base"),
                tip=chain_config.get("tip"),
                tip_offset=chain_config.get("tip_offset"),
                active_links_mask=active_links_mask,
                name=name
            )

        # Format v1: the chain is selected with the deprecated base_elements. This is our own file format, so no
        # deprecation warning for the user: the chain is simply loaded the way it was saved
        elements = chain_config["elements"]
        if elements == "":
            elements = None
        last_link_vector = chain_config["last_link_vector"]
        if last_link_vector == "":
            last_link_vector = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return cls.from_urdf_file(
                urdf_file=urdf_file,
                base_elements=elements,
                active_links_mask=active_links_mask,
                last_link_vector=last_link_vector,
                name=name
            )

    def to_json_file(self, force=False):
        """
        Serialize the chain into a json that will be saved next to the original URDF, with the name of the chain

        Parameters
        ----------
        force: bool
            Overwrite if existing

        Returns
        -------
        str:
            Path of the exported JSON

        """
        metadata = self._urdf_metadata
        if "base_elements" in metadata:
            # The chain was selected with the deprecated base_elements: keep the v1 format
            chain_dict = {
                "elements": metadata["base_elements"],
                "urdf_file": os.path.basename(metadata["urdf_file"]),
                "active_links_mask": self.links_mask,
                "last_link_vector": metadata["last_link_vector"],
                "name": self.name,
                "version": "v1"
            }
        else:
            tip_offset = metadata.get("tip_offset")
            chain_dict = {
                "base": metadata.get("base"),
                "tip": metadata.get("tip"),
                "tip_offset": None if tip_offset is None else [float(x) for x in tip_offset],
                "urdf_file": os.path.basename(metadata["urdf_file"]),
                "active_links_mask": self.links_mask,
                "name": self.name,
                "version": "v2"
            }

        if os.path.exists(self._json_path) and not force:
            raise OSError("File {} exists".format(self._json_path))

        # And create the json file
        with open(self._json_path, "w") as fd:
            json.dump(chain_dict, fd, indent=2)

        return self._json_path

    @classmethod
    def from_urdf_file(cls, urdf_file, *, base=None, tip=None, tip_offset=None, active_links_mask=None, name="chain", symbolic=True, jax_precompile=True, base_elements=None, last_link_vector=None, base_element_type=None):
        """Creates a chain from an URDF file

        A URDF describes a tree of links connected by joints; the chain is the path from `base` to `tip`, which
        is unique, so the two names are all it takes to select it.

        Parameters
        ----------
        urdf_file: str
            The path of the URDF file
        base: str or dict
            The name of the link or joint where the chain starts. Defaults to the root link of the file. The
            chain starts with the joints under a base link, or with a base joint itself. When a link and a joint
            share a name, the name means the link; ``{"name": ..., "type": "joint"}`` or
            ``{"name": ..., "type": "link"}`` says which one is meant.
        tip: str or dict
            The name of the link or joint where the chain ends, as a string or a dict like base; the path from
            base to tip is found automatically. Defaults to the only leaf under base; when there are several,
            the error lists them to choose from.
        tip_offset: numpy.array
            Optional : a translation added at the end of the chain, in the frame of its last joint, as a fixed
            link named ``last_joint``. Typically the tool center point.
        active_links_mask: list[bool]
            A list of boolean indicating whether or not the corresponding link is active
        name: str
            The name of the Chain
        symbolic: bool
            Use symbolic computations
        jax_precompile: bool
            If True (default), JAX functions are compiled when the JAX cache is first accessed.
            If False, compilation is deferred to the first JAX call.
        base_elements: list of strings
            Deprecated, use base and tip. An ordered list of link and joint names to follow, alternating links
            and joints; when the list is exhausted, the parser follows the first child of each element down to
            a leaf.
        last_link_vector: numpy.array
            Deprecated, use tip_offset, which means the same thing.
        base_element_type: str
            Deprecated. The type of the first element in ``base_elements``, "link" or "joint".

        Example
        -------
        >>> from ikpy.chain import Chain
        >>> chain = Chain.from_urdf_file("baxter.urdf", base="base", tip="left_gripper", tip_offset=[0, 0, 0.1])
        >>> chain = Chain.from_urdf_file("baxter.urdf", base={"name": "left_hand", "type": "joint"}, tip="left_gripper")

        Note
        ----
        IKPY works with links, whereas URDF works with joints and links. The mapping is currently misleading:

        * URDF joints = IKPY links
        * URDF links are not used by IKPY. They are thrown away when parsing
        """
        links = URDF.get_urdf_parameters(urdf_file, base=base, tip=tip, tip_offset=tip_offset, symbolic=symbolic, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type)

        if base_elements is not None or base_element_type is not None:
            urdf_metadata = {
                "base_elements": base_elements,
                "urdf_file": urdf_file,
                "last_link_vector": last_link_vector if last_link_vector is not None else tip_offset
            }
        else:
            urdf_metadata = {
                "base": base,
                "tip": tip,
                "tip_offset": tip_offset if tip_offset is not None else last_link_vector,
                "urdf_file": urdf_file
            }

        # Add an origin link at the beginning
        chain = cls([link_lib.OriginLink()] + links, active_links_mask=active_links_mask, name=name, urdf_metadata=urdf_metadata, jax_precompile=jax_precompile)

        # Save some useful metadata
        # FIXME: We have attributes specific to objects created in this style, not great...
        chain.urdf_file = urdf_file
        chain.base = base
        chain.tip = tip
        chain.base_elements = base_elements

        return chain

    def active_to_full(self, active_joints, initial_position):
        full_joints = np.array(initial_position, copy=True, dtype=np.float64)
        np.place(full_joints, self.active_links_mask, active_joints)
        # The forward kinematics derives the dependent (mimic) joints by itself; doing it here too makes the
        # returned configuration readable as it is, with the right value in every slot
        return self._derive_mimic_joints(full_joints)

    def active_from_full(self, joints):
        return np.compress(self.active_links_mask, joints, axis=0)

    @classmethod
    def from_mjcf_file(cls, mjcf_file, *, base=None, tip=None, tip_offset=None, active_links_mask=None, name="chain", symbolic=True, base_elements=None, last_link_vector=None):
        """Creates a chain from a MuJoCo MJCF file

        MJCF nests bodies inside each other; the chain is the path from the `base` body to the `tip` body.

        Parameters
        ----------
        mjcf_file: str
            The path of the MJCF file
        base: str
            The name of the body where the chain starts. Defaults to the only body under ``worldbody``; when
            there are several, the error lists them.
        tip: str
            The name of the body where the chain ends; the path from base to tip is found automatically.
            Defaults to the only leaf under base; when there are several, the error lists them to choose from.
        tip_offset: numpy.array
            Optional: a translation added at the end of the chain, in the frame of its last joint (the
            end-effector offset)
        active_links_mask: list[bool]
            A list of boolean indicating whether or not the corresponding link is active
        name: str
            The name of the Chain
        symbolic: bool
            Use symbolic computations
        base_elements: list of strings
            Deprecated, use base and tip. An ordered list of body names to follow; when the list is exhausted,
            the parser follows the first child of each body down to a leaf.
        last_link_vector: numpy.array
            Deprecated, use tip_offset.

        Returns
        -------
        Chain
            The kinematic chain

        Example
        -------
        >>> from ikpy.chain import Chain
        >>> chain = Chain.from_mjcf_file("ur5e.xml", base="base", tip="wrist_3_link")
        >>> chain.forward_kinematics([0] * len(chain.links))
        """
        links = MJCF.get_mjcf_parameters(mjcf_file, base=base, tip=tip, tip_offset=tip_offset, symbolic=symbolic, base_elements=base_elements, last_link_vector=last_link_vector)
        mjcf_metadata = {
            "base": base,
            "tip": tip,
            "tip_offset": tip_offset if tip_offset is not None else last_link_vector,
            "base_elements": base_elements,
            "mjcf_file": mjcf_file
        }
        # Add an origin link at the beginning
        chain = cls([link_lib.OriginLink()] + links, active_links_mask=active_links_mask, name=name, urdf_metadata=mjcf_metadata)

        # Save some useful metadata
        chain.mjcf_file = mjcf_file
        chain.base = base
        chain.tip = tip
        chain.base_elements = base_elements

        return chain

    @classmethod
    def from_bvh_file(cls, bvh_file, *, base=None, tip=None, tip_offset=None, active_links_mask=None, name="chain", symbolic=True, scale=1.0, base_elements=None, last_link_vector=None):
        """Creates a chain from a BVH (Biovision Hierarchy) motion capture file

        Each channel of a BVH joint becomes a link of the chain, named ``<joint>_<channel>`` (for example
        ``Hips_Zrotation``), so the chain reproduces exactly the rotation order of the file. The motion of the
        file can be replayed on the chain with :func:`ikpy.bvh.BVH.get_bvh_motion`.

        Parameters
        ----------
        bvh_file: str
            The path of the BVH file
        base: str
            The name of the joint where the chain starts. Defaults to the root of the skeleton.
        tip: str
            The name of the joint where the chain ends; the path from base to tip is found automatically.
            Defaults to the only leaf under base; when there are several, the error lists them to choose from.
        tip_offset: numpy.array
            Optional: the translation of the tip of the chain, in the frame of its last joint. Defaults to the
            ``End Site`` of the last joint when it has one.
        active_links_mask: list[bool]
            A list of boolean indicating whether or not the corresponding link is active. Note that the position
            channels of the root (``Hips_Xposition``...) are prismatic links: you may want to deactivate them for
            the inverse kinematics.
        name: str
            The name of the Chain
        symbolic: bool
            Use symbolic computations
        scale: float
            A factor applied to all the lengths of the file (offsets, end sites and position channels), to
            convert them to meters: BVH files do not specify their unit, and are often in centimeters or inches.
        base_elements: list of strings
            Deprecated, use base and tip.
        last_link_vector: numpy.array
            Deprecated, use tip_offset.

        Returns
        -------
        Chain
            The kinematic chain

        Example
        -------
        >>> from ikpy.chain import Chain
        >>> from ikpy.bvh import BVH
        >>> chain = Chain.from_bvh_file("walk.bvh", base="Hips", tip="LeftHand", scale=0.01)
        >>> trajectory, frame_time = BVH.get_bvh_motion("walk.bvh", base="Hips", tip="LeftHand", scale=0.01)
        >>> hand_frames = [chain.forward_kinematics(joints) for joints in trajectory]
        """
        links = BVH.get_bvh_parameters(bvh_file, base=base, tip=tip, tip_offset=tip_offset, symbolic=symbolic, scale=scale, base_elements=base_elements, last_link_vector=last_link_vector)
        bvh_metadata = {
            "base": base,
            "tip": tip,
            "tip_offset": tip_offset if tip_offset is not None else last_link_vector,
            "base_elements": base_elements,
            "bvh_file": bvh_file,
            "scale": scale
        }
        chain = cls([link_lib.OriginLink()] + links, active_links_mask=active_links_mask, name=name, urdf_metadata=bvh_metadata)

        # Save some useful metadata
        chain.bvh_file = bvh_file
        chain.base = base
        chain.tip = tip
        chain.base_elements = base_elements
        chain.scale = scale

        return chain

    @classmethod
    def concat(cls, chain1, chain2):
        """Concatenate two chains"""
        return cls(links=chain1.links + chain2.links, active_links_mask=chain1.links_mask + chain2.links_mask)
