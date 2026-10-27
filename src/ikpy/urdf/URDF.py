# coding= utf8
"""
This module contains the main functions used to parse URDF files.
"""

import xml.etree.ElementTree as ET
import json
import numpy as np
import itertools
import warnings

# Ikpy imports
from ikpy import link as lib_link
from ikpy import logs
from ikpy import selection


def _find_next_joint(root, current_link, next_joint_name):
    """
    Find the next joint in the URDF tree

    Parameters
    ----------
    root
    current_link: xml.etree.ElementTree
        The current URDF link
    next_joint_name: str
        Optional : The name of the next joint. If not provided, find it automatically as the first child of the link.

    Returns
    -------
    has_next: bool
        True if the next joint has been found. Otherwise, the next joint will be None
    next_joint: Optional[Element]
        The next joint
    """
    # Find the joint attached to the link
    has_next = False
    next_joint = None
    search_by_name = (next_joint_name is not None)
    current_link_name = None

    if not search_by_name:
        # If no next joint is provided, find it automatically
        current_link_name = current_link.attrib["name"]

    # FIXME: Use a filter expression directly in the findall statement, instead of filtering by hand
    for joint in root.findall("joint"):
        # Only use joints and links defined at the root.
        # There may be other joints and links elsewhere, but they are just metadata
        if joint is not None:
            # Iterate through all joints to find the good one
            if search_by_name:
                # Find the joint given its name
                if joint.attrib["name"] == next_joint_name:
                    has_next = True
                    next_joint = joint
                    break

            else:
                # Find the first joint whose parent is the current_link
                # FIXME: We are not sending a warning when we have two children for the same link
                # Even if this is not possible, we should ensure something coherent
                if joint.find("parent").attrib["link"] == current_link_name:
                    has_next = True
                    next_joint = joint
                    break

    if search_by_name and not has_next:
        available = [j.attrib["name"] for j in root.findall("joint")]
        raise ValueError("Error: joint '{}' given but not found in the URDF. Available joints: {}".format(next_joint_name, available))

    return has_next, next_joint


def _find_next_link(root, current_joint, next_link_name):
    """
    Find the next link in the URDF tree

    Parameters
    ----------
    root
    current_joint: xml.etree.ElementTree
        The current URDF joint
    next_link_name: str
        Optional : The name of the next link. If not provided, find it automatically as the first child of the joint.

    Returns
    -------
    has_next: bool
        True if the next link has been found. Otherwise, the next link will be None
    next_link: Optional[Element]
        The next link
    """
    has_next = False
    next_link = None

    given_next_link = (next_link_name is not None)

    # If no next link, find it automatically
    if next_link_name is None:
        # If the name of the next link is not provided, find it
        next_link_name = current_joint.find("child").attrib["link"]

    # FIXME: Directly find the link using a regex and a filter
    for urdf_link in root.findall("link"):
        if urdf_link.attrib["name"] == next_link_name:
            next_link = urdf_link
            has_next = True

    if given_next_link and not has_next:
        available = [lk.attrib["name"] for lk in root.findall("link")]
        raise ValueError("Error: link '{}' given but not found in the URDF. Available links: {}. If your robot has no 'base_link', pass the name of its root link with base_elements=[...]".format(next_link_name, available))

    return has_next, next_link


def _find_parent_link(root, joint_name):
    """Find the first link which is the parent of the given joint"""
    try:
        parent_link = next(joint.find("parent").attrib["link"]
                           for joint in root.iter("joint")
                           if joint.attrib["name"] == joint_name)
    except StopIteration:
        raise ValueError("Unable to locate the parent link")

    return parent_link


def get_chain_from_joints(urdf_file, joints):
    """
    Return a complete URDF chain (e.g. links + joints) from a list of joints.
    This function is notafbly used by PyPot, which considers only joints, but needs to create the chain in order to use them as the `base_elements` of the `get_urdf_parameters`

    Parameters
    ----------
    urdf_file: str
    joints: list[str]

    Returns
    -------
    list[Link]
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    links = [_find_parent_link(root, j) for j in joints]

    # Merge and interleave the `links` and `chain`
    # Two ways to do this:
    # First
    # chain = list(itertools.chain(*list(zip(links, joints))))

    # Second:
    iters = [iter(links), iter(joints)]
    chain = []
    for it in itertools.cycle(iters):
        try:
            item = next(it)
        except StopIteration:
            break
        chain.append(item)

    return chain


def _parse_urdf(urdf_file):
    """Parse a URDF file, and refuse a xacro file with a clear message"""
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("Expected a URDF file (root element 'robot'), got '{}' in {}".format(root.tag, urdf_file))
    for element in root.iter():
        unexpanded = "xacro" in element.tag or any("$(" in value or "${" in value for value in element.attrib.values())
        if unexpanded:
            raise ValueError(
                "{} is a xacro file, not a URDF file: it contains macros or expressions that must be expanded "
                "first, with `xacro {} > robot.urdf` (ROS) or `python -m xacro`".format(urdf_file, urdf_file))
    return root


def _urdf_tree(root):
    """Index the links and joints of a URDF

    Links and joints are two namespaces (a joint and its child link often share a name), so the nodes of the tree
    are keyed by ("link", name) and ("joint", name).

    Returns
    -------
    links: dict
        name -> link element
    joints: dict
        name -> joint element
    parent_of: dict
        node -> parent node: the parent link of a joint, the joint leading to a link
    children_of: dict
        node -> child nodes: the joints under a link, the child link of a joint
    """
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {}
    parent_of = {}
    children_of = {("link", name): [] for name in links}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joints[name] = joint
        node = ("joint", name)
        parent = ("link", joint.find("parent").attrib["link"])
        child = ("link", joint.find("child").attrib["link"])
        parent_of[node] = parent
        parent_of[child] = node
        children_of.setdefault(parent, []).append(node)
        children_of[node] = [child]
    return links, joints, parent_of, children_of


def _urdf_node(element, links, joints):
    """The node of a base or tip argument

    A string names a link or a joint; when a link and a joint share a name, the name means the link. A dict
    ``{"name": ..., "type": "link" | "joint"}`` says which of the two is meant.
    """
    if isinstance(element, dict):
        if set(element) != {"name", "type"} or element["type"] not in ("link", "joint"):
            raise ValueError('A URDF element given as a dict must be {{"name": ..., "type": "link"}} or {{"name": ..., "type": "joint"}}, got {}'.format(element))
        kind, name = element["type"], element["name"]
        namespace = links if kind == "link" else joints
        if name not in namespace:
            raise ValueError("'{}' is not a {} of the URDF. {}s: {}".format(name, kind, kind.capitalize(), list(namespace)))
        return (kind, name)
    if element in links:
        return ("link", element)
    if element in joints:
        return ("joint", element)
    raise ValueError("'{}' is neither a link nor a joint of the URDF. Links: {}. Joints: {}".format(element, list(links), list(joints)))


def _resolve_urdf_chain(root, base, tip):
    """Return the joint elements of the chain from base to tip

    base and tip are the names of a link or of a joint; when a joint and a link share a name, the link is meant.
    The chain starts with the joints under a base link, or with a base joint itself; it ends with the joint
    leading to a tip link, or with a tip joint itself.
    """
    links, joints, parent_of, children_of = _urdf_tree(root)
    display = lambda node: node[1]  # noqa: E731

    if base is None:
        roots = [node for node in children_of if node[0] == "link" and node not in parent_of]
        base_node = selection.unique_root(roots, "links", display)
    else:
        base_node = _urdf_node(base, links, joints)

    if tip is None:
        tip_node = selection.unique_leaf(children_of, base_node, "links", display)
    else:
        tip_node = _urdf_node(tip, links, joints)

    path = selection.path_between(parent_of, base_node, tip_node, display)
    chain_joints = [joints[name] for kind, name in path if kind == "joint"]
    logs.logger.debug("URDF chain from {} to {}: {}".format(display(base_node), display(tip_node), [joint.attrib["name"] for joint in chain_joints]))
    return chain_joints


def _legacy_urdf_chain(root, base_elements, base_element_type):
    """The deprecated traversal: follow the given names, then the first child of each element down to a leaf"""
    base_elements = list(base_elements)
    if base_elements == []:
        raise ValueError("base_elements can't be the empty list []")

    joints = []
    links = []
    has_next = True
    current_joint = None
    current_link = None

    # Initialize the tree traversal
    if base_element_type == "link":
        # The first element is a link, so its (virtual) parent should be a joint
        node_type = "joint"
    elif base_element_type == "joint":
        # The same as before, but swap link and joint
        node_type = "link"
    else:
        raise ValueError("Unknown type: {}".format(base_element_type))

    # Parcours récursif de la structure de la chain
    while has_next:
        if len(base_elements) != 0:
            next_element = base_elements.pop(0)
        else:
            next_element = None

        if node_type == "link":
            # Current element is a link, find child joint
            (has_next, current_joint) = _find_next_joint(root, current_link, next_element)
            node_type = "joint"
            if has_next:
                joints.append(current_joint)
                logs.logger.debug("Next element: joint {}".format(current_joint.attrib["name"]))

        elif node_type == "joint":
            # Current element is a joint, find child link
            (has_next, current_link) = _find_next_link(root, current_joint, next_element)
            node_type = "link"
            if has_next:
                links.append(current_link)
                logs.logger.debug("Next element: link {}".format(current_link.attrib["name"]))

    return joints


def _joint_to_link(joint, symbolic):
    """Build the IKPy link of a URDF joint, with the conventions of ROS (urdfdom and KDL)"""
    name = joint.attrib["name"]
    origin_translation = [0, 0, 0]
    origin_orientation = [0, 0, 0]
    rotation = None
    translation = None
    bounds = [-np.inf, np.inf]

    origin = joint.find("origin")
    if origin is not None:
        if "xyz" in origin.attrib.keys():
            origin_translation = [float(x) for x in origin.attrib["xyz"].split()]
        if "rpy" in origin.attrib.keys():
            origin_orientation = [float(x) for x in origin.attrib["rpy"].split()]

    joint_type = joint.attrib["type"]
    if joint_type in ("floating", "planar"):
        # Like KDL: these joints have no single joint value, the chain goes through them as through a fixed joint
        warnings.warn("Joint {} is of type '{}', which has several degrees of freedom: it is treated as a fixed joint, as KDL does".format(name, joint_type))
        joint_type = "fixed"
    if joint_type not in ["revolute", "continuous", "prismatic", "fixed"]:
        raise ValueError("Unknown joint type: {}".format(joint_type))

    axis = joint.find("axis")
    if axis is not None:
        if "xyz" in axis.attrib:
            axis_xyz = axis.attrib["xyz"]
        else:
            # The xyz attribute is required by the URDF specification; fall back on the
            # same default as a missing axis element, but let the user know
            warnings.warn("Joint {} has an 'axis' element without an 'xyz' attribute. Defaulting to the URDF default axis (1, 0, 0)".format(name))
            axis_xyz = "1 0 0"
        axis_vector = np.array([float(x) for x in axis_xyz.split()])
        if joint_type != "fixed":
            norm = np.linalg.norm(axis_vector)
            if norm == 0:
                raise ValueError("Joint {} has a zero axis".format(name))
            if not np.isclose(norm, 1.0):
                # Like KDL: the axis is a direction, its length is meaningless
                warnings.warn("Joint {} has a non-unit axis {}: it is normalized".format(name, axis_vector.tolist()))
                axis_vector = axis_vector / norm
        if joint_type in ("revolute", "continuous"):
            rotation = axis_vector.tolist()
            translation = None
        elif joint_type == "prismatic":
            rotation = None
            translation = axis_vector.tolist()
        elif joint_type == "fixed":
            warnings.warn("Joint {} is of type: fixed, but has an 'axis' attribute defined. This is not in the URDF spec and thus this axis is ignored".format(name))
        else:
            raise ValueError("Unknown joint type with an axis: {}, {}".format(joint_type, axis))
    elif joint_type in ("revolute", "continuous"):
        # No axis element at all: the URDF specification defaults the
        # joint axis to (1, 0, 0)
        rotation = [1, 0, 0]
        translation = None
    elif joint_type == "prismatic":
        translation = [1, 0, 0]
        rotation = None

    limit = joint.find("limit")
    # Continuous joints have no limits in the URDF specification, so
    # their bounds always stay unbounded, even if a limit element is
    # present in the file.
    if limit is not None and joint_type != "continuous":
        if "lower" in limit.attrib:
            bounds[0] = float(limit.attrib["lower"])
        if "upper" in limit.attrib:
            bounds[1] = float(limit.attrib["upper"])

    return lib_link.URDFLink(
        name=name,
        bounds=tuple(bounds),
        origin_translation=origin_translation,
        origin_orientation=origin_orientation,
        rotation=rotation,
        translation=translation,
        use_symbolic_matrix=symbolic,
        joint_type=joint_type
    )


def get_urdf_parameters(urdf_file, base=None, tip=None, tip_offset=None, symbolic=True, base_elements=None, last_link_vector=None, base_element_type=None):
    """
    Returns translated parameters from the given URDF file.
    Parse the URDF joints into IKPY links, throw away the URDF links.

    Parameters
    ----------
    urdf_file: str
        The path of the URDF file
    base: str or dict
        The name of the link or joint where the chain starts. Defaults to the root link of the file. The chain
        starts with the joints under a link, or with the joint itself. When a link and a joint share a name, the
        name means the link; ``{"name": ..., "type": "joint"}`` or ``{"name": ..., "type": "link"}`` says which one is meant.
    tip: str or dict
        The name of the link or joint where the chain ends, as a string or a dict like base. The path from base
        to tip is found automatically. Defaults to the only leaf under base; when there are several, the error
        lists them.
    tip_offset: numpy.array
        Optional : a translation added at the end of the chain, in the frame of its last joint, as a fixed
        link named ``last_joint``
    symbolic: bool
        Use symbolic matrix computations
    base_elements: list of strings
        Deprecated, use base and tip. An ordered list of link and joint names to follow; when the list is
        exhausted, the parser follows the first child of each element down to a leaf.
    last_link_vector: numpy.array
        Deprecated, use tip_offset.
    base_element_type: str
        Deprecated. The type of the first element in ``base_elements``, "link" or "joint".

    Returns
    -------
    list[ikpy.link.URDFLink]
    """
    base, tip, tip_offset, legacy = selection.resolve_selection(base, tip, tip_offset, base_elements, last_link_vector, base_element_type)
    root = _parse_urdf(urdf_file)

    if legacy is not None:
        legacy_elements, legacy_type = legacy
        if legacy_elements is None:
            legacy_elements = ["base_link"]
        joints = _legacy_urdf_chain(root, legacy_elements, legacy_type)
    else:
        joints = _resolve_urdf_chain(root, base, tip)

    parameters = [_joint_to_link(joint, symbolic) for joint in joints]

    # Add the tip offset
    if tip_offset is not None:
        # The last link doesn't provide a rotation
        parameters.append(lib_link.URDFLink(
            origin_translation=tip_offset,
            origin_orientation=[0, 0, 0],
            rotation=None,
            translation=None,
            name="last_joint",
            use_symbolic_matrix=symbolic,
            joint_type="fixed"
        ))

    return parameters


def _get_motor_parameters(json_file):
    """Returns a dictionary with joints as keys, and a description (dict) of each joint as value"""
    with open(json_file) as motor_fd:
        global_config = json.load(motor_fd)

    motors = global_config["motors"]
    # Returned dict
    motor_config = {}

    # Add motor to the config
    for motor in motors:
        motor_config[motor] = motors[motor]

    return motor_config


def _convert_angle_to_pypot(angle, joint, **kwargs):
    """Converts an angle to a PyPot-compatible format"""
    angle_deg = (angle * 180 / np.pi)

    if joint["orientation-convention"] == "indirect":
        angle_deg = -1 * angle_deg

    # UGLY
    if joint["name"].startswith("l_shoulder_x"):
        angle_deg = -1 * angle_deg

    angle_pypot = angle_deg - joint["offset"]

    return angle_pypot


def _convert_angle_from_pypot(angle, joint, **kwargs):
    """Converts an angle to a PyPot-compatible format"""
    angle_internal = angle + joint["offset"]

    if joint["orientation-convention"] == "indirect":
        angle_internal = -1 * angle_internal

    # UGLY
    if joint["name"].startswith("l_shoulder_x"):
        angle_internal = -1 * angle_internal

    angle_internal = (angle_internal / 180 * np.pi)

    return angle_internal


def _convert_angle_limit(angle, joint, **kwargs):
    """Converts the limit angle of the PyPot JSON file to the internal format"""
    angle_pypot = angle

    # No need to take care of orientation
    if joint["orientation"] == "indirect":
        angle_pypot = 1 * angle_pypot

    # angle_pypot = angle_pypot + offset

    return angle_pypot * np.pi / 180
