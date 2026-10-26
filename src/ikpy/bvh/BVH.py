# coding= utf8
"""
This module contains the functions used to parse BVH (Biovision Hierarchy) motion capture files.

A BVH file has two sections:

* ``HIERARCHY`` describes a skeleton: a tree of joints, each with an ``OFFSET`` (its position in the frame of its
  parent) and ``CHANNELS`` (the degrees of freedom animated by the file, as rotations about the local axes and,
  usually for the root only, translations). Leaves end with an ``End Site`` that gives the position of the tip.
* ``MOTION`` gives, for each frame, the value of every channel: angles in degrees, translations in the unit of
  the file.

The transformation of a joint is ``T(position) . R1 . R2 . R3`` where ``position`` is its ``OFFSET``, replaced by
the value of its position channels when it has some, and each ``Ri`` is the rotation of one of its rotation
channels, applied in the order of the file. IKPy represents this with one link per channel: the first link of a
joint carries the offset, each rotation channel is a revolute link about its axis, and each position channel is
a prismatic link along its axis whose value is the displacement from the offset. A chain with all its joint
values at zero is therefore the rest pose of the skeleton, and the resulting chain follows the exact semantics
of the file, whatever the rotation order of the joints.

For more information on the format, see: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
"""

import numpy as np
import warnings

# Ikpy imports
from ikpy import link as lib_link
from ikpy import logs


_AXES = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}


class BVHJoint:
    """A joint of a BVH skeleton

    Attributes
    ----------
    name: str
        The name of the joint
    offset: numpy.array
        The position of the joint in the frame of its parent, in the unit of the file
    channels: list[str]
        The channels of the joint, in the order of the file. Each one is ``Xposition``, ``Yposition``,
        ``Zposition``, ``Xrotation``, ``Yrotation`` or ``Zrotation``
    parent: BVHJoint
        The parent joint, or None for a root
    children: list[BVHJoint]
        The child joints, in the order of the file
    end_site: numpy.array
        The offset of the ``End Site`` of the joint if it has one, None otherwise
    channel_index: int
        The index of the first channel of the joint in a frame of the ``MOTION`` section
    """

    def __init__(self, name, offset, channels, parent=None, channel_index=0):
        self.name = name
        self.offset = np.array(offset, dtype=float)
        self.channels = list(channels)
        self.parent = parent
        self.children = []
        self.end_site = None
        self.channel_index = channel_index

    def __repr__(self):
        return "BVHJoint name={} offset={} channels={} children={}".format(
            self.name, self.offset.tolist(), self.channels, [child.name for child in self.children])

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def iter_joints(self):
        """Iterate over this joint and all its descendants, depth first, in the order of the file"""
        yield self
        for child in self.children:
            yield from child.iter_joints()

    def find(self, name):
        """Return the descendant (or self) with the given name, None if there is none"""
        for joint in self.iter_joints():
            if joint.name == name:
                return joint
        return None

    def path_to(self, descendant):
        """Return the list of joints from self to the given descendant, both included"""
        path = []
        joint = descendant
        while joint is not None and joint is not self:
            path.append(joint)
            joint = joint.parent
        if joint is None:
            raise ValueError("Joint {} is not a descendant of joint {}".format(descendant.name, self.name))
        path.append(self)
        return path[::-1]


class BVHSkeleton:
    """The content of a BVH file

    Attributes
    ----------
    roots: list[BVHJoint]
        The root joints of the file (BVH files usually have a single ``ROOT``)
    n_channels: int
        The number of channels of a frame
    motion: numpy.array
        The ``MOTION`` section, of shape (n_frames, n_channels): angles in degrees, translations in the unit of the
        file. Empty if the file has no motion section
    frame_time: float
        The duration of a frame, in seconds, None if the file has no motion section
    """

    def __init__(self, roots, n_channels, motion, frame_time):
        self.roots = roots
        self.n_channels = n_channels
        self.motion = motion
        self.frame_time = frame_time

    @property
    def n_frames(self):
        return self.motion.shape[0]

    def iter_joints(self):
        for root in self.roots:
            yield from root.iter_joints()

    def find(self, name):
        """Return the joint with the given name, None if there is none"""
        for root in self.roots:
            joint = root.find(name)
            if joint is not None:
                return joint
        return None

    def __repr__(self):
        return "BVHSkeleton roots={} joints={} channels={} frames={}".format(
            [root.name for root in self.roots], sum(1 for _ in self.iter_joints()), self.n_channels, self.n_frames)


class _Tokens:
    """A cursor over the whitespace-separated tokens of a file"""

    def __init__(self, text):
        self.tokens = text.split()
        self.position = 0

    def __len__(self):
        return len(self.tokens) - self.position

    def peek(self):
        if self.position >= len(self.tokens):
            raise ValueError("Unexpected end of BVH file")
        return self.tokens[self.position]

    def next(self):
        token = self.peek()
        self.position += 1
        return token

    def expect(self, expected):
        token = self.next()
        if token != expected:
            raise ValueError("Malformed BVH file: expected '{}', got '{}'".format(expected, token))
        return token

    def floats(self, n):
        try:
            return [float(self.next()) for _ in range(n)]
        except ValueError as error:
            raise ValueError("Malformed BVH file: expected {} numbers ({})".format(n, error))


def _parse_joint(tokens, name, parent, channel_index):
    """Parse the body of a joint, starting at its opening brace. Returns the joint and the next channel index"""
    tokens.expect("{")
    joint = BVHJoint(name, [0, 0, 0], [], parent=parent, channel_index=channel_index)
    while True:
        token = tokens.next()
        if token == "OFFSET":
            joint.offset = np.array(tokens.floats(3))
        elif token == "CHANNELS":
            n_channels = int(tokens.next())
            joint.channels = []
            for _ in range(n_channels):
                channel = tokens.next()
                axis, kind = channel[0].upper(), channel[1:].lower()
                if axis not in _AXES or kind not in ("rotation", "position"):
                    raise ValueError("Unknown BVH channel: {} (joint {})".format(channel, name))
                joint.channels.append(axis + kind)
            joint.channel_index = channel_index
            channel_index += n_channels
        elif token == "JOINT":
            child, channel_index = _parse_joint(tokens, tokens.next(), joint, channel_index)
            joint.children.append(child)
        elif token == "End":
            tokens.expect("Site")
            tokens.expect("{")
            tokens.expect("OFFSET")
            joint.end_site = np.array(tokens.floats(3))
            tokens.expect("}")
        elif token == "}":
            return joint, channel_index
        else:
            raise ValueError("Malformed BVH file: unexpected token '{}' in joint {}".format(token, name))


def parse_bvh(bvh_file):
    """Parse a BVH file

    Parameters
    ----------
    bvh_file: str
        The path of the BVH file

    Returns
    -------
    BVHSkeleton
        The skeleton, with the motion data of the file
    """
    with open(bvh_file, "r") as fd:
        tokens = _Tokens(fd.read())

    tokens.expect("HIERARCHY")
    roots = []
    channel_index = 0
    while len(tokens) > 0 and tokens.peek() != "MOTION":
        tokens.expect("ROOT")
        root, channel_index = _parse_joint(tokens, tokens.next(), None, channel_index)
        roots.append(root)
    if len(roots) == 0:
        raise ValueError("Malformed BVH file: no ROOT joint")
    n_channels = channel_index

    motion = np.zeros((0, n_channels))
    frame_time = None
    if len(tokens) > 0:
        tokens.expect("MOTION")
        tokens.expect("Frames:")
        n_frames = int(tokens.next())
        tokens.expect("Frame")
        tokens.expect("Time:")
        frame_time = float(tokens.next())
        values = np.array(tokens.floats(len(tokens)))
        if n_channels > 0:
            available_frames = len(values) // n_channels
            if available_frames * n_channels != len(values) or available_frames != n_frames:
                warnings.warn("BVH file {} announces {} frames of {} channels but contains {} values: using {} frames".format(
                    bvh_file, n_frames, n_channels, len(values), available_frames))
            motion = values[:available_frames * n_channels].reshape((available_frames, n_channels))

    skeleton = BVHSkeleton(roots, n_channels, motion, frame_time)
    logs.logger.debug("Parsed BVH file {}: {}".format(bvh_file, skeleton))
    return skeleton


def get_joint_names(bvh_file):
    """Return the names of all the joints of a BVH file, in the order of the file

    Parameters
    ----------
    bvh_file: str
        The path of the BVH file

    Returns
    -------
    list[str]
    """
    return [joint.name for joint in parse_bvh(bvh_file).iter_joints()]


def _resolve_path(skeleton, base_elements):
    """Return the list of joints forming the chain described by base_elements, see get_bvf_parameters"""
    if base_elements is None or len(base_elements) == 0:
        path = [skeleton.roots[0]]
        stop_at_last = False
    else:
        base_elements = list(base_elements)
        start = skeleton.find(base_elements[0])
        if start is None:
            raise ValueError("Joint '{}' not found in the BVH file. Available joints: {}".format(
                base_elements[0], [joint.name for joint in skeleton.iter_joints()]))
        path = [start]
        for name in base_elements[1:]:
            target = path[-1].find(name)
            if target is None or target is path[-1]:
                raise ValueError("Joint '{}' is not a descendant of joint '{}' in the BVH file".format(name, path[-1].name))
            path.extend(path[-1].path_to(target)[1:])
        stop_at_last = len(base_elements) > 1

    if not stop_at_last:
        # Follow the first child down to a leaf
        while not path[-1].is_leaf:
            path.append(path[-1].children[0])

    logs.logger.debug("BVH chain: {}".format([joint.name for joint in path]))
    return path


def _chain_links(skeleton, base_elements, last_link_vector, scale):
    """Describe the links of a chain: a list of (name, origin_translation, axis, joint_type, channel_index, rest_value)

    rest_value is the value of the channel in the rest pose: the offset component for a position channel, 0 otherwise
    """
    specs = []
    path = _resolve_path(skeleton, base_elements)
    for joint in path:
        origin = joint.offset * scale
        if len(joint.channels) == 0:
            specs.append((joint.name, origin, None, "fixed", None, 0.0))
            continue
        for k, channel in enumerate(joint.channels):
            axis = _AXES[channel[0]]
            if channel.endswith("rotation"):
                joint_type, rest_value = "revolute", 0.0
            else:
                # Position channels are absolute: they replace the offset
                joint_type, rest_value = "prismatic", joint.offset["XYZ".index(channel[0])]
            specs.append(("{}_{}".format(joint.name, channel), origin if k == 0 else np.zeros(3), axis, joint_type, joint.channel_index + k, rest_value))

    last_joint = path[-1]
    if last_link_vector is not None:
        specs.append(("last_joint", np.array(last_link_vector, dtype=float), None, "fixed", None, 0.0))
    elif last_joint.end_site is not None:
        specs.append(("{}_end_site".format(last_joint.name), last_joint.end_site * scale, None, "fixed", None, 0.0))
    return specs


def get_bvh_parameters(bvh_file, base_elements=None, last_link_vector=None, symbolic=True, scale=1.0):
    """Return the links of the chain described by a BVH file

    Parameters
    ----------
    bvh_file: str
        The path of the BVH file
    base_elements: list[str]
        The joints of the chain, from its start to its end, as an ordered list of joint names. The joints between
        two consecutive names are filled in automatically, so only the branching points need to be given
        (for example ``["Hips", "LeftHand"]``). The chain ends at the last name. If a single name is given, or
        None, the chain starts at that joint (or at the root) and follows the first child of each joint down to
        a leaf.
    last_link_vector: numpy.array
        Optional: the translation of the tip of the chain, in the frame of its last joint. Defaults to the
        ``End Site`` of the last joint when it has one.
    symbolic: bool
        Use symbolic matrices
    scale: float
        A factor applied to all the lengths of the file (offsets, end sites and position channels), to convert
        them to meters: BVH files do not specify their unit, and are often in centimeters or inches.

    Returns
    -------
    list[ikpy.link.URDFLink]
        One link per channel of each joint of the chain (the first link of a joint carries its offset), named
        ``<joint>_<channel>`` (for example ``Hips_Zrotation``), plus a fixed link for the tip when there is one.
        Rotation channels are revolute links (angles in radians), position channels are prismatic links whose
        value is the displacement from the offset of the joint: all-zero joint values give the rest pose.
    """
    skeleton = parse_bvh(bvh_file)
    links = []
    for name, origin, axis, joint_type, _, _ in _chain_links(skeleton, base_elements, last_link_vector, scale):
        links.append(lib_link.URDFLink(
            name=name,
            origin_translation=origin,
            origin_orientation=[0, 0, 0],
            rotation=axis if joint_type == "revolute" else None,
            translation=axis if joint_type == "prismatic" else None,
            use_symbolic_matrix=symbolic,
            joint_type=joint_type
        ))
    return links


def get_bvh_motion(bvh_file, base_elements=None, last_link_vector=None, scale=1.0):
    """Return the motion of a BVH file as joint values of the chain built with the same parameters

    Parameters
    ----------
    bvh_file: str
        The path of the BVH file
    base_elements: list[str]
        The same list as the one given to :func:`get_bvh_parameters` or :meth:`ikpy.chain.Chain.from_bvh_file`
    last_link_vector: numpy.array
        The same vector as the one given to the chain
    scale: float
        The same factor as the one given to the chain

    Returns
    -------
    trajectory: numpy.array
        An array of shape (n_frames, n_links) where each row can be given to ``Chain.forward_kinematics``:
        rotation channels in radians, position channels as scaled displacements from the offset of their joint,
        zeros for the origin link and the tip
    frame_time: float
        The duration of a frame, in seconds
    """
    skeleton = parse_bvh(bvh_file)
    specs = _chain_links(skeleton, base_elements, last_link_vector, scale)
    # One leading column for the origin link of the chain
    trajectory = np.zeros((skeleton.n_frames, len(specs) + 1))
    for column, (_, _, _, joint_type, channel_index, rest_value) in enumerate(specs, start=1):
        if channel_index is None:
            continue
        values = skeleton.motion[:, channel_index]
        trajectory[:, column] = np.deg2rad(values) if joint_type == "revolute" else (values - rest_value) * scale
    return trajectory, skeleton.frame_time
