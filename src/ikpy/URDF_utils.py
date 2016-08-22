# coding= utf8
"""
.. module:: URDF_utils
This module contains helper functions used to parse URDF.
"""

import xml.etree.ElementTree as ET
import json
import numpy as np
import itertools

from . import link as lib_link


def find_next_joint(root, current_link, next_joint_name):
    """Find the next joint in the URDF tree

    :param xml.etree.ElementTree current_link: The current URDF link
    :param string next_joint_name: Optional : The name of the next joint
    """
    # Trouver le joint attaché
    has_next = False
    next_joint = None
    search_by_name = True

    if next_joint_name is None:
        # If no next joint is provided, find it automatically
        search_by_name = False
        current_link_name = current_link.attrib["name"]

    for joint in root.iter("joint"):
        # Iterate through all joints to find the good one
        if(search_by_name):
            # Find the joint given its name
            if joint.attrib["name"] == next_joint_name:
                has_next = True
                next_joint = joint
        else:
            # Find the joint which parent is the current_link
            if joint.find("parent").attrib["link"] == current_link_name:
                has_next = True
                next_joint = joint

    return(has_next, next_joint)


def find_next_link(root, current_joint, next_link_name):
    """Find the next link in the URDF tree

    :param xml.etree.ElementTree current_joint: The current URDF joint
    :param string next_link_name: Optional : The name of the next link
    """
    has_next = False
    next_link = None
    # If no next link, find it automaticly
    if next_link_name is None:
        # If the name of the next link is not provided, find it
        next_link_name = current_joint.find("child").attrib["link"]

    for urdf_link in root.iter("link"):
        if urdf_link.attrib["name"] == next_link_name:
            next_link = urdf_link
            has_next = True
    return(has_next, next_link)


def find_parent_link(root, joint_name):
    return next(joint.find("parent").attrib["link"]
                for joint in root.iter("joint")
                if joint.attrib["name"] == joint_name)


def get_chain_from_joints(urdf_file, joints):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    links = [find_parent_link(root, j) for j in joints]

    iters = [iter(links), iter(joints)]
    chain = list(next(it) for it in itertools.cycle(iters))

    return chain


def get_urdf_parameters(urdf_file, base_elements=["base_link"], last_link_vector=None, base_element_type="link"):
    """Returns translated parameters from the given URDF file

   :param urdf_file: The path of the URDF file
   :type urdf_file: string
   :param base_elements: List of the links beginning the chain
   :type base_elements: list of strings
   :param last_link_vector: Optional : The translation vector of the tip.
   :type last_link_vector: numpy.array
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    base_elements = list(base_elements)
    if base_elements == []:
        raise ValueError("base_elements can't be the empty list []")

    joints = []
    links = []
    has_next = True
    current_joint = None
    current_link = None

    if base_element_type == "link":
        # The first element is a link, so its (virtual) parent should be a joint
        node_type = "joint"
    elif base_element_type == "joint":
        # The same as before, but swap link and joint
        node_type = "link"

    # Parcours récursif de la structure de la chain
    while(has_next):
        if base_elements != []:
            next_element = base_elements.pop(0)
        else:
            next_element = None

        if node_type == "link":
            # Current element is a link, find child joint
            (has_next, current_joint) = find_next_joint(root, current_link, next_element)
            node_type = "joint"
            if(has_next):
                joints.append(current_joint)

        elif node_type == "joint":
            # Current element is a joint, find child link
            (has_next, current_link) = find_next_link(root, current_joint, next_element)
            node_type = "link"
            if(has_next):
                links.append(current_link)

    parameters = []

    # Save the joints in the good format
    for joint in joints:
        translation = joint.find("origin").attrib["xyz"].split()
        orientation = joint.find("origin").attrib["rpy"].split()
        rotation = joint.find("axis").attrib['xyz'].split()
        parameters.append(lib_link.URDFLink(
            translation_vector=[float(translation[0]), float(translation[1]), float(translation[2])],
            orientation=[float(orientation[0]), float(orientation[1]), float(orientation[2])],
            rotation=[float(rotation[0]), float(rotation[1]), float(rotation[2])],
            name=joint.attrib["name"]
        ))

    # Add last_link_vector to parameters
    if last_link_vector is not None:
        parameters.append(lib_link.URDFLink(
            translation_vector=last_link_vector,
            orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            name="last_joint"
        ))

    return(parameters)


def _get_motor_parameters(json_file):
    """Returns a dictionnary with joints as keys, and a description (dict) of each joint as value"""
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
    angle_deg = (angle * 180 / (np.pi))

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

    angle_internal = (angle_internal / 180 * (np.pi))

    return angle_internal


def _convert_angle_limit(angle, joint, **kwargs):
    """Converts the limit angle of the PyPot JSON file to the internal format"""
    angle_pypot = angle

    # No need to take care of orientation
    if joint["orientation"] == "indirect":
        angle_pypot = 1 * angle_pypot

    # angle_pypot = angle_pypot + offset

    return angle_pypot * (np.pi) / 180
