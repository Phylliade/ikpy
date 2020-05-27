# coding= utf8
"""
This module contains the main functions used to parse URDF files.
"""

import xml.etree.ElementTree as ET
import json
import numpy as np
import itertools

# Ikpy imports
from ikpy import link as lib_link
from ikpy import logs


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
    """
    # Find the joint attached to the link
    has_next = False
    next_joint = None
    search_by_name = True
    current_link_name = None

    if next_joint_name is None:
        # If no next joint is provided, find it automatically
        search_by_name = False
        current_link_name = current_link.attrib["name"]

    for joint in root.findall("joint"):
        # Only use joints and links defined at the root.
        # There may be other joints and links elsewhere, but there are just metadata
        if joint is not None:
            # Iterate through all joints to find the good one
            if search_by_name:
                # Find the joint given its name
                if joint.attrib["name"] == next_joint_name:
                    has_next = True
                    next_joint = joint
            else:
                # Find the first joint whose parent is the current_link
                # FIXME: We are not sending a warning when we have two children for the same link
                # Even if this is not possible, we should ensure something coherent
                if joint.find("parent").attrib["link"] == current_link_name:
                    has_next = True
                    next_joint = joint
                    break

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
    """
    has_next = False
    next_link = None

    # If no next link, find it automatically
    if next_link_name is None:
        # If the name of the next link is not provided, find it
        next_link_name = current_joint.find("child").attrib["link"]

    for urdf_link in root.findall("link"):
        if urdf_link.attrib["name"] == next_link_name:
            next_link = urdf_link
            has_next = True
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
    This function is notably used by PyPot, which considers only joints, but needs to create the chain in order to use them as the `base_elements` of the `get_urdf_parameters`

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
    # chain = list(itertools.chain(*list(zip(links, joints))))

    iters = [iter(links), iter(joints)]
    chain = []
    for it in itertools.cycle(iters):
        try:
            item = next(it)
        except StopIteration:
            break
        chain.append(item)

    return chain


def get_urdf_parameters(urdf_file, base_elements=None, last_link_vector=None, base_element_type="link", symbolic=True):
    """
    Returns translated parameters from the given URDF file.
    Parse the URDF joints into IKPY links, throw away the URDF links.

    Parameters
    ----------
    urdf_file: str
        The path of the URDF file
    base_elements: list of strings
        List of the links beginning the chain
    last_link_vector: numpy.array
        Optional : The translation vector of the tip.
    base_element_type: str
    symbolic: bool

    Returns
    -------
    list[ikpy.link.URDFLink]
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    base_elements = list(base_elements)
    if base_elements is None:
        base_elements = ["base_link"]
    elif base_elements is []:
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

    parameters = []

    # Save the joints in the good format
    for joint in joints:
        translation = [0, 0, 0]
        orientation = [0, 0, 0]
        rotation = [1, 0, 0]
        bounds = [None, None]

        origin = joint.find("origin")
        if origin is not None:
            if origin.attrib["xyz"]:
                translation = [float(x) for x in origin.attrib["xyz"].split()]
            if origin.attrib["rpy"]:
                orientation = [float(x) for x in origin.attrib["rpy"].split()]

        axis = joint.find("axis")
        if axis is not None:
            rotation = [float(x) for x in axis.attrib["xyz"].split()]

        limit = joint.find("limit")
        if limit is not None:
            if limit.attrib["lower"]:
                bounds[0] = float(limit.attrib["lower"])
            if limit.attrib["upper"]:
                bounds[1] = float(limit.attrib["upper"])

        parameters.append(lib_link.URDFLink(
            name=joint.attrib["name"],
            bounds=tuple(bounds),
            translation_vector=translation,
            orientation=orientation,
            rotation=rotation,
            use_symbolic_matrix=symbolic
        ))

    # Add last_link_vector to parameters
    if last_link_vector is not None:
        # The last link doesn't provide a rotation
        parameters.append(lib_link.URDFLink(
            translation_vector=last_link_vector,
            orientation=[0, 0, 0],
            rotation=None,
            name="last_joint",
            use_symbolic_matrix=symbolic
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
