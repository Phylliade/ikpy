"""
.. module:: robot_utils
"""

from . import forward_kinematics
import xml.etree.ElementTree as ET


def robot_from_urdf_parameters(urdf_params):
    """Converts URDF-formated parameters to compatible parameters"""
    robot_params = []
    for (rot, trans) in urdf_params:
        euler_angles = forward_kinematics.euler_from_unit_vector(*rot)
        robot_params.append((euler_angles[0], euler_angles[1], trans))
    return robot_params


def find_next_joint(root, current_link, next_joints):
    """Find the next joint in the URDF tree"""
    # Trouver le joint attaché
    has_next = False
    next_joint = None
    print(current_link, next_joints)

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
    print("Find next link : ", current_joint, next_links)
    # If no next link, find it automaticly
    if next_links == []:
        next_link_name = current_joint.find("child").attrib["link"]
    else:
        # If a next link is provided, use it
        next_link_name = next_links.pop(0)
        print("Next Link : ", next_link_name)

    for link in root.iter("link"):
        if link.attrib["name"] == next_link_name:
            next_link = link
            has_next = True
    return(has_next, next_link)


def get_urdf_parameters(urdf_file, base_elements=["base_link"], last_link_vector=None, base_elements_type="joint"):
    """Returns translated parameters from the given URDF file"""
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
            (has_next, current_joint) = find_next_joint(root, current_link, base_joints)
            node_type = "joint"
            if(has_next):
                joints.append(current_joint)

        elif node_type == "joint":
            # Current element is a joint, find child link
            (has_next, current_link) = find_next_link(root, current_joint, base_links)
            node_type = "link"
            if(has_next):
                links.append(current_link)

    parameters = []

    for joint in joints:
        translation = joint.find("origin").attrib["xyz"].split()
        orientation = joint.find("origin").attrib["rpy"].split()
        rotation = joint.find("axis").attrib['xyz'].split()
        parameters.append((
            [float(translation[0]), float(translation[1]), float(translation[2])],
            [float(orientation[0]), float(orientation[1]), float(orientation[2])],
            [float(rotation[0]), float(rotation[1]), float(rotation[2])]
        ))

    # Descriptive chain of the robot
    chain = []
    for link in links:
        chain.append(link.attrib["name"])

    print(chain)

    # Add last_link_vector to parameters
    if last_link_vector is not None:
        parameters.append((
            last_link_vector,
            [0, 0, 0],
            [0, 0, 0]
        ))

    return(parameters)
