# coding= utf8
"""
This module contains the main functions used to parse MJCF (MuJoCo XML) files.

MJCF is MuJoCo's native XML format for defining robots and environments.
Unlike URDF which has a flat structure with separate joints and links,
MJCF uses a hierarchical structure where bodies are nested inside each other.

For more information, see: https://mujoco.readthedocs.io/en/latest/modeling.html
"""

import xml.etree.ElementTree as ET
import numpy as np
import warnings

# Ikpy imports
from ikpy import link as lib_link
from ikpy import logs
from . import utils as mjcf_utils


def _get_compiler_settings(root):
    """
    Extract compiler settings from the MJCF file.

    Parameters
    ----------
    root: xml.etree.ElementTree.Element
        The root element of the MJCF XML

    Returns
    -------
    dict
        Compiler settings including angle unit
    """
    settings = {
        "angle": "degree",  # Default is degree in MJCF
        "eulerseq": "xyz"   # Default euler sequence
    }

    compiler = root.find("compiler")
    if compiler is not None:
        if "angle" in compiler.attrib:
            settings["angle"] = compiler.attrib["angle"]
        if "eulerseq" in compiler.attrib:
            settings["eulerseq"] = compiler.attrib["eulerseq"]

    return settings


def _get_default_class(root, class_name):
    """
    Get the default settings for a given class.

    Parameters
    ----------
    root: xml.etree.ElementTree.Element
        The root element of the MJCF XML
    class_name: str
        The name of the default class to find

    Returns
    -------
    dict
        Default settings for joints in this class
    """
    defaults = {
        "axis": [0, 1, 0],  # Default axis in MJCF
        "range": [-np.inf, np.inf]
    }

    default_section = root.find("default")
    if default_section is None:
        return defaults

    # Search for the class recursively
    def find_class(element, target_class):
        if element.tag == "default":
            element_class = element.get("class", "main" if element.getparent() is None else None)
            if element_class == target_class:
                # Get joint defaults
                joint_elem = element.find("joint")
                if joint_elem is not None:
                    if "axis" in joint_elem.attrib:
                        defaults["axis"] = [float(x) for x in joint_elem.attrib["axis"].split()]
                    if "range" in joint_elem.attrib:
                        defaults["range"] = [float(x) for x in joint_elem.attrib["range"].split()]
                return defaults

            # Search children
            for child in element:
                if child.tag == "default":
                    result = find_class(child, target_class)
                    if result:
                        return result
        return None

    # Try to find the class
    for default_elem in default_section.iter("default"):
        default_class = default_elem.get("class")
        if default_class == class_name:
            joint_elem = default_elem.find("joint")
            if joint_elem is not None:
                if "axis" in joint_elem.attrib:
                    defaults["axis"] = [float(x) for x in joint_elem.attrib["axis"].split()]
                if "range" in joint_elem.attrib:
                    defaults["range"] = [float(x) for x in joint_elem.attrib["range"].split()]
            break

    return defaults


def _parse_body_transform(body_elem, compiler_settings):
    """
    Parse the position and orientation of a body element.

    Parameters
    ----------
    body_elem: xml.etree.ElementTree.Element
        The body element
    compiler_settings: dict
        Compiler settings (angle unit, etc.)

    Returns
    -------
    tuple
        (translation, orientation_rpy) where orientation is in radians
    """
    # Parse position
    pos = [0, 0, 0]
    if "pos" in body_elem.attrib:
        pos = [float(x) for x in body_elem.attrib["pos"].split()]

    # Parse orientation - MJCF supports multiple formats
    orientation_rpy = [0, 0, 0]

    if "quat" in body_elem.attrib:
        quat = [float(x) for x in body_elem.attrib["quat"].split()]
        orientation_rpy = mjcf_utils.quat_to_rpy(quat)

    elif "axisangle" in body_elem.attrib:
        axisangle = [float(x) for x in body_elem.attrib["axisangle"].split()]
        angle = axisangle[3]
        if compiler_settings["angle"] == "degree":
            angle = np.radians(angle)
        orientation_rpy = mjcf_utils.axisangle_to_rpy(axisangle[:3], angle)

    elif "euler" in body_elem.attrib:
        euler = [float(x) for x in body_elem.attrib["euler"].split()]
        if compiler_settings["angle"] == "degree":
            euler = [np.radians(e) for e in euler]
        orientation_rpy = mjcf_utils.euler_to_rpy(euler, compiler_settings["eulerseq"])

    elif "xyaxes" in body_elem.attrib:
        xyaxes = [float(x) for x in body_elem.attrib["xyaxes"].split()]
        orientation_rpy = mjcf_utils.xyaxes_to_rpy(xyaxes)

    elif "zaxis" in body_elem.attrib:
        zaxis = [float(x) for x in body_elem.attrib["zaxis"].split()]
        orientation_rpy = mjcf_utils.zaxis_to_rpy(zaxis)

    return pos, orientation_rpy


def _parse_joint(joint_elem, compiler_settings, default_settings):
    """
    Parse a joint element.

    Parameters
    ----------
    joint_elem: xml.etree.ElementTree.Element
        The joint element
    compiler_settings: dict
        Compiler settings
    default_settings: dict
        Default settings for this joint's class

    Returns
    -------
    dict
        Joint parameters
    """
    joint_info = {
        "name": joint_elem.get("name", "unnamed_joint"),
        "type": joint_elem.get("type", "hinge"),  # Default is hinge (revolute)
        "axis": default_settings["axis"].copy(),
        "range": default_settings["range"].copy(),
        "pos": [0, 0, 0]
    }

    # Parse joint-specific position (offset within the body)
    if "pos" in joint_elem.attrib:
        joint_info["pos"] = [float(x) for x in joint_elem.attrib["pos"].split()]

    # Parse axis
    if "axis" in joint_elem.attrib:
        joint_info["axis"] = [float(x) for x in joint_elem.attrib["axis"].split()]

    # Parse range (limits)
    if "range" in joint_elem.attrib:
        range_vals = [float(x) for x in joint_elem.attrib["range"].split()]
        if compiler_settings["angle"] == "degree" and joint_info["type"] == "hinge":
            range_vals = [np.radians(r) for r in range_vals]
        joint_info["range"] = range_vals

    # Check limited attribute
    if "limited" in joint_elem.attrib:
        if joint_elem.attrib["limited"].lower() not in ["true", "1"]:
            joint_info["range"] = [-np.inf, np.inf]

    return joint_info


def _traverse_body_tree(root, body_elem, compiler_settings, chain_path, base_elements, current_depth=0):
    """
    Recursively traverse the body tree and extract kinematic chain.

    Parameters
    ----------
    root: xml.etree.ElementTree.Element
        Root element for accessing defaults
    body_elem: xml.etree.ElementTree.Element
        Current body element
    compiler_settings: dict
        Compiler settings
    chain_path: list
        List to accumulate chain elements
    base_elements: list or None
        List of body names to follow, or None to follow first child
    current_depth: int
        Current recursion depth

    Returns
    -------
    bool
        True if we should continue traversing
    """
    body_name = body_elem.get("name", f"body_{current_depth}")
    logs.logger.debug(f"Processing body: {body_name}")

    # Check if we should process this body based on base_elements
    if base_elements is not None and len(base_elements) > 0:
        if body_name != base_elements[0]:
            return False
        base_elements = base_elements[1:]

    # Get childclass for this body
    childclass = body_elem.get("childclass")

    # Parse body transform
    body_pos, body_orientation = _parse_body_transform(body_elem, compiler_settings)

    # Find joints in this body
    joints = body_elem.findall("joint")

    if len(joints) == 0:
        # No joint means this body is welded to parent - create a fixed link
        chain_path.append({
            "name": body_name,
            "type": "fixed",
            "origin_translation": body_pos,
            "origin_orientation": body_orientation,
            "axis": None,
            "bounds": (-np.inf, np.inf)
        })
    else:
        # Process each joint
        for i, joint_elem in enumerate(joints):
            # Get default class for this joint
            joint_class = joint_elem.get("class", childclass)
            default_settings = _get_default_class(root, joint_class) if joint_class else {
                "axis": [0, 1, 0],
                "range": [-np.inf, np.inf]
            }

            joint_info = _parse_joint(joint_elem, compiler_settings, default_settings)

            # For the first joint, include the body transform
            # For subsequent joints, they share the same body frame
            if i == 0:
                origin_translation = [body_pos[j] + joint_info["pos"][j] for j in range(3)]
                origin_orientation = body_orientation
            else:
                origin_translation = joint_info["pos"]
                origin_orientation = [0, 0, 0]

            # Map MJCF joint types to IKPy joint types
            if joint_info["type"] in ["hinge", "revolute"]:
                ikpy_type = "revolute"
                rotation_axis = joint_info["axis"]
                translation_axis = None
            elif joint_info["type"] in ["slide", "prismatic"]:
                ikpy_type = "prismatic"
                rotation_axis = None
                translation_axis = joint_info["axis"]
            elif joint_info["type"] == "ball":
                warnings.warn(f"Ball joint '{joint_info['name']}' not fully supported, treating as fixed")
                ikpy_type = "fixed"
                rotation_axis = None
                translation_axis = None
            elif joint_info["type"] == "free":
                warnings.warn(f"Free joint '{joint_info['name']}' not supported, treating as fixed")
                ikpy_type = "fixed"
                rotation_axis = None
                translation_axis = None
            else:
                ikpy_type = "fixed"
                rotation_axis = None
                translation_axis = None

            chain_path.append({
                "name": joint_info["name"],
                "type": ikpy_type,
                "origin_translation": origin_translation,
                "origin_orientation": origin_orientation,
                "rotation": rotation_axis,
                "translation": translation_axis,
                "bounds": tuple(joint_info["range"])
            })

    # Find child bodies and traverse
    child_bodies = body_elem.findall("body")

    if len(child_bodies) == 0:
        return True

    # Determine which child to follow
    if base_elements is not None and len(base_elements) > 0:
        # Follow the specified path
        for child in child_bodies:
            if child.get("name") == base_elements[0]:
                _traverse_body_tree(root, child, compiler_settings, chain_path, base_elements, current_depth + 1)
                return True
        # If specified body not found, stop
        return True
    else:
        # Follow first child (default behavior)
        _traverse_body_tree(root, child_bodies[0], compiler_settings, chain_path, None, current_depth + 1)

    return True


def get_mjcf_parameters(mjcf_file, base_elements=None, last_link_vector=None, symbolic=True):
    """
    Returns translated parameters from the given MJCF file.
    Parse the MJCF bodies and joints into IKPY links.

    Parameters
    ----------
    mjcf_file: str
        The path of the MJCF file
    base_elements: list of strings
        An ordered list of body names that defines the path to traverse in the MJCF tree.
        When the list is exhausted or empty, the parser will automatically follow the first child.
        If None, starts from worldbody and follows first children.
    last_link_vector: numpy.array
        Optional: The translation vector of the tip (end-effector offset)
    symbolic: bool
        Use symbolic matrix computations

    Returns
    -------
    list[ikpy.link.URDFLink]
        List of links that can be used to create a Chain

    Example
    -------
    >>> # Parse UR5e robot
    >>> links = get_mjcf_parameters("ur5e.xml", base_elements=["base", "shoulder_link"])
    """
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    # Verify this is an MJCF file
    if root.tag != "mujoco":
        raise ValueError(f"Expected MJCF file (root element 'mujoco'), got '{root.tag}'")

    # Get compiler settings
    compiler_settings = _get_compiler_settings(root)
    logs.logger.debug(f"Compiler settings: {compiler_settings}")

    # Find worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No 'worldbody' element found in MJCF file")

    # Prepare base_elements
    if base_elements is not None:
        base_elements = list(base_elements)

    # Find starting point
    chain_path = []

    if base_elements is not None and len(base_elements) > 0:
        # Find the first body that matches
        start_body = None
        for body in worldbody.iter("body"):
            if body.get("name") == base_elements[0]:
                start_body = body
                base_elements = base_elements[1:]
                break

        if start_body is None:
            raise ValueError(f"Starting body '{base_elements[0]}' not found in MJCF")

        _traverse_body_tree(root, start_body, compiler_settings, chain_path, base_elements)
    else:
        # Start from first body in worldbody
        first_body = worldbody.find("body")
        if first_body is None:
            raise ValueError("No body found in worldbody")

        _traverse_body_tree(root, first_body, compiler_settings, chain_path, None)

    # Convert chain_path to URDFLink objects
    parameters = []
    for link_info in chain_path:
        parameters.append(lib_link.URDFLink(
            name=link_info["name"],
            bounds=link_info.get("bounds", (-np.inf, np.inf)),
            origin_translation=link_info["origin_translation"],
            origin_orientation=link_info["origin_orientation"],
            rotation=link_info.get("rotation"),
            translation=link_info.get("translation"),
            use_symbolic_matrix=symbolic,
            joint_type=link_info["type"]
        ))

    # Add last_link_vector if provided
    if last_link_vector is not None:
        parameters.append(lib_link.URDFLink(
            origin_translation=last_link_vector,
            origin_orientation=[0, 0, 0],
            rotation=None,
            translation=None,
            name="last_joint",
            use_symbolic_matrix=symbolic,
            joint_type="fixed"
        ))

    return parameters


def get_body_names(mjcf_file):
    """
    Get all body names from an MJCF file.

    Parameters
    ----------
    mjcf_file: str
        Path to the MJCF file

    Returns
    -------
    list[str]
        List of body names
    """
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        return []

    names = []
    for body in worldbody.iter("body"):
        name = body.get("name")
        if name:
            names.append(name)

    return names


def get_joint_names(mjcf_file):
    """
    Get all joint names from an MJCF file.

    Parameters
    ----------
    mjcf_file: str
        Path to the MJCF file

    Returns
    -------
    list[str]
        List of joint names
    """
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        return []

    names = []
    for joint in worldbody.iter("joint"):
        name = joint.get("name")
        if name:
            names.append(name)

    return names
