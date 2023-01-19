# IKPy imports
from ikpy import chain
from ikpy.urdf.utils import get_urdf_tree

# Generate the pdf
dot, urdf_tree = get_urdf_tree("./pepper.urdf", out_image_path="./pepper", root_element="base_link")


########################## Left arm ##########################

pepper_left_arm_links = ["base_link",
                         "torso",
                         "LShoulder",
                         "LBicep",
                         "LElbow",
                         "LForeArm",
                         "l_wrist"]

pepper_left_arm_joints = ["base_link_fixedjoint",
                          "LShoulderPitch",
                          "LShoulderRoll",
                          "LElbowYaw",
                          "LElbowRoll",
                          "LWristYaw",
                          "LHand"]

pepper_left_arm_elements = [x for pair in zip(pepper_left_arm_links, pepper_left_arm_joints) for x in pair] + ["l_gripper"]
# Remove the gripper, it's weird
# pepper_left_arm_elements = [x for pair in zip(pepper_left_arm_links, pepper_left_arm_joints) for x in pair][:-3]

pepper_left_arm_chain = chain.Chain.from_urdf_file(
    "./pepper.urdf",
    base_elements=pepper_left_arm_elements,
    last_link_vector=[0, 0.0, 0],
    active_links_mask = 2*[False] + 4 * [True] + 3 * [False],
    symbolic=False,
    name="pepper_left_arm")
pepper_left_arm_chain.to_json_file(force=True)

############################## Right arm ##############################

pepper_right_arm_elements = [x.replace("L", "R") for x in pepper_left_arm_elements]

pepper_right_arm_chain = chain.Chain.from_urdf_file(
    "./pepper.urdf",
    base_elements=pepper_right_arm_elements,
    last_link_vector=[0, 0.0, 0],
    active_links_mask = 2*[False] + 4 * [True] + 3 * [False],
    symbolic=False,
    name="pepper_right_arm")
pepper_right_arm_chain.to_json_file(force=True)

############################ Head ####################################

pepper_head_links = ["base_link",
                         "torso",
                         "Neck"]

pepper_head_joints = ["base_link_fixedjoint",
                          "HeadYaw",
                          "HeadPitch"]

pepper_head_elements = [x for pair in zip(pepper_head_links, pepper_head_joints) for x in pair] + ["Head"]

pepper_head_chain = chain.Chain.from_urdf_file(
    "./pepper.urdf",
    base_elements=pepper_head_elements,
    last_link_vector=[0, 0.0, 0],
    symbolic=False,
    name="pepper_head",
    active_links_mask=[False, True, True, False, False, False, False]
)
pepper_head_chain.to_json_file(force=True)

########################## Pedestal ###############################

pepper_pedestal_elements = ["base_link",
                            "base_link_fixedjoint",
                            "torso",
                            "HipRoll",
                            "Hip",
                            "HipPitch",
                            "Pelvis",
                            "KneePitch",
                            "Tibia",
                            "base_footprint_joint",
                            "base_footprint"]

pepper_pedestal_chain = chain.Chain.from_urdf_file(
    "./pepper.urdf",
    base_elements=pepper_pedestal_elements,
    last_link_vector=[0, 0.0, 0],
    symbolic=False,
    name="pepper_legs",
    active_links_mask=[False, False, True, True, True, False, False])
print(pepper_pedestal_chain.to_json_file(force=True))
