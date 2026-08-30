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

    ik = torso_right_arm.inverse_kinematics_frame(
        frame_target, initial_position=joints)
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


def _converged_and_starved(chain, budget_kwargs, **kwargs):
    """Solve the same target twice: once to convergence, once on a starvation budget"""
    target = [0.1, -0.2, 0.1]
    joints = [1] * len(chain.links)
    joints[-4] = 0
    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    converged = chain.inverse_kinematics_frame(frame_target, initial_position=joints, **kwargs)
    starved = chain.inverse_kinematics_frame(
        frame_target, initial_position=joints, **kwargs, **budget_kwargs)
    return target, converged, starved


def test_optimizer_budget_reaches_the_least_squares_optimizer(torso_right_arm):
    """A budget given to the optimizer has to be honoured, not just accepted"""
    target, converged, starved = _converged_and_starved(torso_right_arm, {"optimizer_budget": 1})

    # Stopping the optimizer after a single evaluation cannot reach the target the full run does,
    # so a worse result is what proves the argument went through to SciPy
    assert _target_error(torso_right_arm, starved, target) > _target_error(torso_right_arm, converged, target)


def test_optimizer_budget_reaches_the_scalar_optimizer(torso_right_arm):
    """The same argument, spelled the same way, has to work for the other optimizer too"""
    target, converged, starved = _converged_and_starved(
        torso_right_arm, {"optimizer_budget": 1}, optimizer="scalar")

    assert _target_error(torso_right_arm, starved, target) > _target_error(torso_right_arm, converged, target)


def test_optimizer_kwargs_remain_an_escape_hatch(torso_right_arm):
    """The raw SciPy keys stay reachable for the options that have no portable name"""
    target, converged, starved = _converged_and_starved(
        torso_right_arm, {"optimizer_kwargs": {"max_nfev": 1}})

    assert _target_error(torso_right_arm, starved, target) > _target_error(torso_right_arm, converged, target)


def test_tol_reaches_both_optimizers(torso_right_arm):
    """tol means the same thing whichever optimizer is in use"""
    for optimizer in ("least_squares", "scalar"):
        target, converged, sloppy = _converged_and_starved(
            torso_right_arm, {"tol": 1e-1}, optimizer=optimizer)

        assert _target_error(torso_right_arm, sloppy, target) > _target_error(torso_right_arm, converged, target)


def test_optimizer_kwargs_refuse_to_override_the_bounds(torso_right_arm):
    """The bounds describe the limits of the links, so they are not the caller's to replace"""
    frame_target = np.eye(4)
    frame_target[:3, 3] = [0.1, -0.2, 0.1]

    with pytest.raises(ValueError):
        torso_right_arm.inverse_kinematics_frame(
            frame_target, optimizer_kwargs={"bounds": (-1, 1)})


@pytest.mark.parametrize("optimizer, clash", [
    ("least_squares", {"max_nfev": 5}),
    ("scalar", {"options": {"maxiter": 5}}),
])
def test_optimizer_budget_refuses_to_be_set_twice(torso_right_arm, optimizer, clash):
    """Setting the same knob through both doors is a mistake, not a precedence puzzle"""
    frame_target = np.eye(4)
    frame_target[:3, 3] = [0.1, -0.2, 0.1]

    with pytest.raises(ValueError, match="optimizer_budget"):
        torso_right_arm.inverse_kinematics_frame(
            frame_target, optimizer=optimizer, optimizer_budget=5, optimizer_kwargs=clash)


def test_max_iter_is_deprecated(torso_right_arm):
    """max_iter is still accepted and still ignored, but it no longer goes by unnoticed"""
    frame_target = np.eye(4)
    frame_target[:3, 3] = [0.1, -0.2, 0.1]

    with pytest.warns(DeprecationWarning, match="optimizer_budget"):
        torso_right_arm.inverse_kinematics_frame(frame_target, max_iter=3)
