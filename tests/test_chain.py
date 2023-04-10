import unittest
import numpy as np
import sys
import matplotlib.pyplot as plt

# IKPy imports
from ikpy import chain
from ikpy.utils import plot


def test_chain():
    fig, ax = plot.init_3d_figure()

    torso_right_arm = chain.Chain.from_urdf_file(
        "../resources/poppy_torso/poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ])
    torso_left_arm = chain.Chain.from_urdf_file(
        "../resources/poppy_torso/poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "l_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ])

    # Plot right arm
    joints = [0] * len(torso_right_arm.links)
    torso_left_arm.plot(joints, ax)
    # Plot left arm
    joints = [0] * len(torso_left_arm.links)
    torso_right_arm.plot(joints, ax)
    plt.savefig("out/torso.png")


def test_ik(torso_right_arm):
    fig, ax = plot.init_3d_figure()

    # Objectives
    target = [0.1, -0.2, 0.1]
    joints = [0] * len(torso_right_arm.links)
    joints[-4] = 0
    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    ik = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints)

    torso_right_arm.plot(ik, ax, target=target)

    np.testing.assert_almost_equal(
        torso_right_arm.forward_kinematics(ik)[:3, 3], target, decimal=3)


def test_ik_optimization(torso_right_arm):
    """Tests the IK optimization-based method"""
    # Objectives
    target = [0.1, -0.2, 0.1]
    joints = [1] * len(torso_right_arm.links)
    joints[-4] = 0
    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    args = {"max_iter": 3}
    ik = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints, **args)
    # Check whether the results are almost equal
    np.testing.assert_almost_equal(
        torso_right_arm.forward_kinematics(ik)[:3, 3], target, decimal=3)

    # Check using the scalar optimizer
    ik = torso_right_arm.inverse_kinematics_frame(
    frame_target, initial_position=joints, optimizer="scalar")
    # Check whether the results are almost equal
    np.testing.assert_almost_equal(
        torso_right_arm.forward_kinematics(ik)[:3, 3], target, decimal=3)


def test_chain_serialization(torso_right_arm):

    chain_json_path = torso_right_arm.to_json_file(force=True)
    chain.Chain.from_json_file(chain_json_path)

