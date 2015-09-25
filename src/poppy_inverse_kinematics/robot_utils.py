from . import forward_kinematics
import xml.etree.ElementTree as ET


def robot_from_urdf_parameters(urdf_params):
    robot_params = []
    for (rot, trans) in urdf_params:
        euler_angles = forward_kinematics.euler_from_unit_vector(*rot)
        robot_params.append((euler_angles[0], euler_angles[1], trans))
    return robot_params


def find_next_joint(root, current_link):
    # Trouver le joint attaché
    has_next = False
    next_joint = 0

    for joint in root.iter("joint"):
        if joint.find("parent").attrib["link"] == current_link.attrib["name"]:
            has_next = True
            next_joint = joint
    return(has_next, next_joint)


def find_next_link(root, current_joint):
    # Trouver le next_link
    has_next = False
    next_link = 0
    for link in root.iter("link"):
        if link.attrib["name"] == current_joint.find("child").attrib["link"]:
            next_link = link
            has_next = True
    return(has_next, next_link)


def get_urdf_parameters(urdf_file, base_link_name):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Récupération du 1er link
    for link in root.iter('link'):
        if link.attrib["name"] == base_link_name:
            base_link = link

    has_next = True
    current_link = base_link
    node_type = "link"
    joints = []
    links = []
    while(has_next):
        if node_type == "link":
            links.append(link)
            (has_next, current_joint) = find_next_joint(root, current_link)
            node_type = "joint"
        elif node_type == "joint":
            joints.append(current_joint)
            (has_next, current_link) = find_next_link(root, current_joint)

            node_type = "link"

    parameters = []
    for joint in joints:
        translation = joint.find("origin").attrib["xyz"].split()
        rotation = joint.find("origin").attrib["rpy"].split()
        parameters.append(([float(rotation[0]), float(rotation[1]), float(rotation[2])], (float(translation[0]), float(translation[1]), float(translation[2]))))

    return(parameters)
