# IKPy imports
from ikpy import chain


########################## Simple prismatic robot ###############################

prismatic_robot_chain = chain.Chain.from_urdf_file(
    "./prismatic_robot.URDF",
    base_elements=[
            "baseLink", "joint_baseLink_childA", "childA"
    ],
    last_link_vector=[0, 1, 0],
    active_links_mask=[False, True, False],
    name="prismatic_robot",
)

prismatic_robot_chain.to_json_file(force=True)
print("Saved chain: ", prismatic_robot_chain)

########################## Prismatic mixed robot ###############################

prismatic_mixed_robot_elements = [
    "base_link",
    "base_link_base_slider_joint",
    "base_slider",
    "linear_actuator",
    "robot_base",
    "arm2",
    "link_2",
    "arm3",
    "link_3",
    "arm4",
    "link_4",
    "arm5",
    "link_5",
]

prismatic_mixed_robot_chain = chain.Chain.from_urdf_file(
    "./prismatic_mixed_robot.URDF",
    base_elements=prismatic_mixed_robot_elements,
    last_link_vector=[0, 0.18, 0],
    symbolic=False,
    name="prismatic_mixed_robot",
    active_links_mask=2 * [False] + [True] * 5 + [False])

prismatic_mixed_robot_chain.to_json_file(force=True)
print("Saved chain: ", prismatic_mixed_robot_chain)
