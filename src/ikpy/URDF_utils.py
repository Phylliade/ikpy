# coding= utf8
"""
.. module:: URDF_utils
"""

from . import link as lib_link
import xml.etree.ElementTree as ET
import json
import numpy as np


def find_next_joint(root, current_link, next_joints):
    """Find the next joint in the URDF tree"""
    # Trouver le joint attaché
    has_next = False
    next_joint = None

    if next_joints == []:
        # If no next joints, find it automaticly
        search_by_name = False
        current_link_name = current_link.attrib["name"]
    else:
        # If a next joint is provided, use it
        next_joint_name = next_joints.pop(0)
        search_by_name = True

    for joint in root.iter("joint"):
        if(search_by_name):
            if joint.attrib["name"] == next_joint_name:
                has_next = True
                next_joint = joint
        else:
            if joint.find("parent").attrib["link"] == current_link_name:
                has_next = True
                next_joint = joint

    return(has_next, next_joint)


def find_next_link(root, current_joint, next_links):
    """Find the next link in the URDF tree"""
    has_next = False
    next_link = None
    # If no next link, find it automaticly
    if next_links == []:
        next_link_name = current_joint.find("child").attrib["link"]
    else:
        # If a next link is provided, use it
        next_link_name = next_links.pop(0)

    for urdf_link in root.iter("link"):
        if urdf_link.attrib["name"] == next_link_name:
            next_link = urdf_link
            has_next = True
    return(has_next, next_link)


def get_urdf_parameters(urdf_file, base_elements=["base_link"], last_link_vector=None, base_elements_type="joint"):
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

    base_joints = []
    base_links = []
    for index, element in enumerate(base_elements):
        if index % 2 == 0:
            base_links.append(element)
        else:
            base_joints.append(element)

    joints = []
    links = []
    has_next = True
    current_joint = None
    current_link = None

    node_type = "joint"

    # Parcours récursif de la structure du bras
    while(has_next):
        if node_type == "link":
            # Current element is a link, find child joint
            (has_next, current_joint) = find_next_joint(
                root, current_link, base_joints)
            node_type = "joint"
            if(has_next):
                joints.append(current_joint)

        elif node_type == "joint":
            # Current element is a joint, find child link
            (has_next, current_link) = find_next_link(
                root, current_joint, base_links)
            node_type = "link"
            if(has_next):
                links.append(current_link)

    parameters = []

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


def get_motor_parameters(json_file):
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


def convert_angle_to_pypot(angle, joint, **kwargs):
    """Converts an angle to a PyPot-compatible format"""
    angle_deg = (angle * 180 / (np.pi))

    if joint["orientation-convention"] == "indirect":
        angle_deg = -1 * angle_deg

    # UGLY
    if joint["name"].startswith("l_shoulder_x"):
        angle_deg = -1 * angle_deg

    angle_pypot = angle_deg - joint["offset"]

    return angle_pypot


def convert_angle_from_pypot(angle, joint, **kwargs):
    """Converts an angle to a PyPot-compatible format"""
    angle_internal = angle + joint["offset"]

    if joint["orientation-convention"] == "indirect":
        angle_internal = -1 * angle_internal

    # UGLY
    if joint["name"].startswith("l_shoulder_x"):
        angle_internal = -1 * angle_internal

    angle_internal = (angle_internal / 180 * (np.pi))

    return angle_internal


def convert_angle_limit(angle, joint, **kwargs):
    """Converts the limit angle of the PyPot JSON file to the internal format"""
    angle_pypot = angle

    # No need to take care of orientation
    if joint["orientation"] == "indirect":
        angle_pypot = 1 * angle_pypot

    # angle_pypot = angle_pypot + offset

    return angle_pypot * (np.pi) / 180
