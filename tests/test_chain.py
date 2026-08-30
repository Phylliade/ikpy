import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def _target_error(chain, ik, target):
    """How far the pose reached by the IK ended up from the target"""
    return np.linalg.norm(chain.forward_kinematics(ik)[:3, 3] - target)


def test_ik_optimizer_kwargs_reach_the_least_squares_optimizer(torso_right_arm):
    """A budget given to the optimizer has to be honoured, not just accepted"""
    target = [0.1, -0.2, 0.1]
    joints = [1] * len(torso_right_arm.links)
    joints[-4] = 0
    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    converged = torso_right_arm.inverse_kinematics_frame(frame_target, initial_position=joints)
    starved = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints, optimizer_kwargs={"max_nfev": 1})

    # Stopping the optimizer after a single evaluation cannot reach the target the full run does,
    # so a worse result is what proves the argument went through to SciPy
    assert _target_error(torso_right_arm, starved, target) > _target_error(torso_right_arm, converged, target)


def test_ik_optimizer_kwargs_reach_the_scalar_optimizer(torso_right_arm):
    """The scalar optimizer takes its own keys, nested under "options" the way SciPy wants them"""
    target = [0.1, -0.2, 0.1]
    joints = [1] * len(torso_right_arm.links)
    joints[-4] = 0
    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    converged = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints, optimizer="scalar")
    starved = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints, optimizer="scalar",
        optimizer_kwargs={"options": {"maxiter": 1}})

    assert _target_error(torso_right_arm, starved, target) > _target_error(torso_right_arm, converged, target)


def test_ik_optimizer_kwargs_refuse_to_override_the_bounds(torso_right_arm):
    """The bounds describe the limits of the links, so they are not the caller's to replace"""
    frame_target = np.eye(4)
    frame_target[:3, 3] = [0.1, -0.2, 0.1]

    with pytest.raises(ValueError):
        torso_right_arm.inverse_kinematics_frame(
            frame_target, optimizer_kwargs={"bounds": (-1, 1)})


def test_max_iter_is_deprecated(torso_right_arm):
    """max_iter is still accepted and still ignored, but it no longer goes by unnoticed"""
    frame_target = np.eye(4)
    frame_target[:3, 3] = [0.1, -0.2, 0.1]

    with pytest.warns(DeprecationWarning, match="optimizer_kwargs"):
        torso_right_arm.inverse_kinematics_frame(frame_target, max_iter=3)
