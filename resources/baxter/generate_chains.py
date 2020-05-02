from graphviz import Digraph
import numpy as np
import json

# IKPy imports
from ikpy import chain
from ikpy.utils import plot
from ikpy.urdf.utils import get_urdf_tree

# Generate the pdf
dot, urdf_tree = get_urdf_tree("./baxter.urdf", out_image_path="./baxter")


########################## Left arm ##########################

baxter_left_arm_links = ["base",
                         "torso",
                         "left_arm_mount",
                         "left_upper_shoulder",
                         "left_lower_shoulder",
                         "left_upper_elbow",
                         "left_lower_elbow",
                         "left_upper_forearm",
                         "left_lower_forearm",
                         "left_wrist",
                         "left_hand",
                         "left_gripper_base",
                         "left_gripper"
                         ]

baxter_left_arm_joints = ["torso_t0",
                          "left_torso_arm_mount",
                          "left_s0",
                          "left_s1",
                          "left_e0",
                          "left_e1",
                          "left_w0",
                          "left_w1",
                          "left_w2",
                          "left_hand",
                          "left_gripper_base",
                          "left_endpoint"]

baxter_left_arm_elements = [x for pair in zip(baxter_left_arm_links, baxter_left_arm_joints) for x in pair] + ["left_gripper"]
# Remove the gripper, it's weird
# baxter_left_arm_elements = [x for pair in zip(baxter_left_arm_links, baxter_left_arm_joints) for x in pair][:-3]

baxter_left_arm_chain = chain.Chain.from_urdf_file(
    "./baxter.urdf",
    base_elements=baxter_left_arm_elements,
    last_link_vector=[0, 0.18, 0],
    active_links_mask=3 * [False] + 11 * [True],
    symbolic=False,
    name="baxter_left_arm")
baxter_left_arm_chain.to_json_file(force=True)

############################## Right arm ##############################

baxter_right_arm_elements = [x.replace("left", "right") for x in baxter_left_arm_elements]

baxter_right_arm_chain = chain.Chain.from_urdf_file(
    "./baxter.urdf",
    base_elements=baxter_right_arm_elements,
    last_link_vector=[0, 0.18, 0],
    active_links_mask=3 * [False] + 11 * [True],
    symbolic=False,
    name="baxter_right_arm")
baxter_right_arm_chain.to_json_file(force=True)

############################ Head ####################################

baxter_head_elements = ["base",
                        "torso_t0",
                        "torso",
                        "head_pan",
                        "head",
                        "head_nod",
                        "screen",
                        "display_joint",
                        "display"]

baxter_head_chain = chain.Chain.from_urdf_file(
    "./baxter.urdf",
    base_elements=baxter_head_elements,
    last_link_vector=[0, 0.18, 0],
    symbolic=False,
    name="baxter_head")
baxter_head_chain.to_json_file(force=True)

########################## Pedestal ###############################

baxter_pedestal_elements = ["base",
                            "torso_t0",
                            "torso",
                            "pedestal_fixed",
                            "pedestal"]

baxter_pedestal_chain = chain.Chain.from_urdf_file(
    "./baxter.urdf",
    base_elements=baxter_pedestal_elements,
    last_link_vector=[0, 0.18, 0],
    symbolic=False,
    name="baxter_pedestal")
baxter_pedestal_chain.to_json_file(force=True)
