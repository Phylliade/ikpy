import os
import matplotlib.pyplot as plt
import numpy as np

# ikpy imports
from ikpy import chain
from ikpy.utils import plot
from ikpy.urdf import URDF
from ikpy.urdf.utils import get_urdf_tree


def test_urdf_chain(resources_path, interactive):
    """Test that we can open chain from a URDF file"""
    chain1 = chain.Chain.from_urdf_file(
        os.path.join(resources_path, "poppy_torso/poppy_torso.URDF"),
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ])

    joints = [0] * len(chain1.links)
    joints[-4] = 0
    fig, ax = plot.init_3d_figure()
    chain1.plot(joints, ax)
    plt.savefig("out/chain1.png")
    if interactive:
        plt.show()


def test_urdf_parser(poppy_torso_urdf):
    """Test the correctness of a URDF parser"""
    base_elements = [
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ]
    last_link_vector = [0, 0.18, 0]

    links = URDF.get_urdf_parameters(poppy_torso_urdf, base_elements=base_elements, last_link_vector=last_link_vector)

    assert len(links) == len(base_elements)


def test_plot_urdf_tree(baxter_urdf):
    dot = get_urdf_tree(baxter_urdf, root_element="base", out_image_path="./out/baxter")


def test_chain_from_joints(poppy_ergo_urdf):
    joint_list = ["m1", "m2", "m3", "m4", "m5", "m6"]
    elements_list = URDF.get_chain_from_joints(poppy_ergo_urdf, joints=joint_list)
    assert len(elements_list) == 2 * len(joint_list)
    assert elements_list[1::2] == joint_list

    # And also a manual test in the end, can be removed in the future
    # Note: The last link is not included here, by design
    assert elements_list == [
        "base_link",
        "m1",
        "U_shape",
        "m2",
        "module_1",
        "m3",
        "base_module_2",
        "m4",
        "tip_module_1",
        "m5",
        "tip_middle",
        "m6",
        "tip"
    ][:-1]


def test_prismatic_joints(prismatic_robot_urdf):
    """Test prismatic joints"""
    chain1 = chain.Chain.from_urdf_file(
        prismatic_robot_urdf,
        base_elements=[
            "baseLink", "joint_baseLink_childA", "childA"],
        last_link_vector=[0, 1, 0])

    initial_kinematics = [0.15] * len(chain1)
    fk = chain1.forward_kinematics(initial_kinematics)
    ik = chain1.inverse_kinematics_frame(fk)

    np.testing.assert_almost_equal(fk, chain1.forward_kinematics(ik))
