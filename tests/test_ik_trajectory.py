import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.utils import geometry


@pytest.fixture
def ergo(resources_path):
    return chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")


def test_interpolate_frames_endpoints_and_midpoint():
    start = geometry.to_transformation_matrix([0, 0, 0], np.eye(3))
    end = geometry.to_transformation_matrix([1, 2, 3], geometry.rz_matrix(np.pi / 2))
    frames = geometry.interpolate_frames(start, end, 5)
    assert frames.shape == (5, 4, 4)
    np.testing.assert_allclose(frames[0], start, atol=1e-12)
    np.testing.assert_allclose(frames[-1], end, atol=1e-12)
    # Straight line in position, constant angular speed in orientation
    np.testing.assert_allclose(frames[2][:3, 3], [0.5, 1, 1.5])
    np.testing.assert_allclose(frames[2][:3, :3], geometry.rz_matrix(np.pi / 4), atol=1e-12)
    np.testing.assert_allclose(frames[1][:3, :3], geometry.rz_matrix(np.pi / 8), atol=1e-12)
    np.testing.assert_array_equal(frames[:, 3], [[0, 0, 0, 1]] * 5)


def test_interpolate_frames_validates_inputs():
    with pytest.raises(ValueError):
        geometry.interpolate_frames(np.eye(4), np.eye(4), 1)
    with pytest.raises(ValueError):
        geometry.interpolate_frames(np.eye(3), np.eye(4), 3)


def _two_poses(ergo):
    n_links = len(ergo.links)
    start_joints = [0, 0.3, -0.5, 0.6, 0.2, -0.4, 0.1, 0][:n_links] + [0] * max(0, n_links - 8)
    end_joints = [0, -0.2, -0.9, 0.9, -0.3, 0.1, 0.4, 0][:n_links] + [0] * max(0, n_links - 8)
    return start_joints, end_joints, ergo.forward_kinematics(start_joints), ergo.forward_kinematics(end_joints)


def test_position_trajectory_reaches_every_point_and_stays_continuous(ergo):
    start_joints, _, start, end = _two_poses(ergo)
    frames = geometry.interpolate_frames(start, end, 15)
    trajectory = ergo.inverse_kinematics_trajectory_frame(frames, initial_position=start_joints)
    assert trajectory.shape == (15, len(ergo.links))
    for frame, joints in zip(frames, trajectory):
        np.testing.assert_allclose(ergo.forward_kinematics(joints)[:3, 3], frame[:3, 3], atol=1e-4)
    # Continuity: successive configurations stay close, no jump to another branch of solutions
    jumps = np.abs(np.diff(trajectory, axis=0)).max(axis=1)
    assert jumps.max() < 0.3
    # The first step starts where asked
    np.testing.assert_allclose(trajectory[0], start_joints, atol=1e-3)


def test_full_frame_trajectory_between_two_reachable_poses(ergo):
    """The interpolated frames in between are not all exactly reachable by a 6-DOF arm with joint limits: the
    solver returns the closest configuration, and the two ends, which are reachable, are reached"""
    start_joints, _, start, end = _two_poses(ergo)
    frames = geometry.interpolate_frames(start, end, 15)
    trajectory = ergo.inverse_kinematics_trajectory_frame(frames, initial_position=start_joints, orientation_mode="all")
    np.testing.assert_allclose(ergo.forward_kinematics(trajectory[0]), start, atol=1e-3)
    np.testing.assert_allclose(ergo.forward_kinematics(trajectory[-1]), end, atol=1e-3)
    for frame, joints in zip(frames, trajectory):
        np.testing.assert_allclose(ergo.forward_kinematics(joints)[:3, 3], frame[:3, 3], atol=2e-2)


def test_regularization_smooths_the_joint_motion(ergo):
    """Along a full-frame trajectory that is not exactly reachable, the local solver may jump between branches of
    solutions; pulling each step towards the previous one trades a little accuracy for a continuous motion"""
    start_joints, _, start, end = _two_poses(ergo)
    frames = geometry.interpolate_frames(start, end, 15)
    plain = ergo.inverse_kinematics_trajectory_frame(frames, initial_position=start_joints, orientation_mode="all")
    smooth = ergo.inverse_kinematics_trajectory_frame(frames, initial_position=start_joints, orientation_mode="all", regularization_parameter=0.1)
    jumps_plain = np.abs(np.diff(plain, axis=0)).max()
    jumps_smooth = np.abs(np.diff(smooth, axis=0)).max()
    assert jumps_smooth < 0.1 < jumps_plain
    for frame, joints in zip(frames, smooth):
        np.testing.assert_allclose(ergo.forward_kinematics(joints)[:3, 3], frame[:3, 3], atol=5e-2)


def test_trajectory_from_positions_and_orientations(ergo):
    n_links = len(ergo.links)
    joints = np.zeros((4, n_links))
    joints[:, 1] = np.linspace(0, 0.6, 4)
    joints[:, 2] = -0.5
    frames = np.array([ergo.forward_kinematics(q) for q in joints])

    positions_only = ergo.inverse_kinematics_trajectory(target_positions=frames[:, :3, 3], initial_position=joints[0])
    assert positions_only.shape == (4, n_links)
    for frame, q in zip(frames, positions_only):
        np.testing.assert_allclose(ergo.forward_kinematics(q)[:3, 3], frame[:3, 3], atol=1e-3)

    with_z = ergo.inverse_kinematics_trajectory(target_positions=frames[:, :3, 3], target_orientations=frames[:, :3, 2], orientation_mode="Z", initial_position=joints[0])
    for frame, q in zip(frames, with_z):
        np.testing.assert_allclose(ergo.forward_kinematics(q)[:3, 2], frame[:3, 2], atol=1e-2)

    orientations_only = ergo.inverse_kinematics_trajectory(target_orientations=frames[:, :3, :3], orientation_mode="all", initial_position=joints[0])
    for frame, q in zip(frames, orientations_only):
        np.testing.assert_allclose(ergo.forward_kinematics(q)[:3, :3], frame[:3, :3], atol=1e-2)


def test_trajectory_validates_inputs(ergo):
    with pytest.raises(ValueError):
        ergo.inverse_kinematics_trajectory()
    with pytest.raises(ValueError):
        ergo.inverse_kinematics_trajectory(target_positions=np.zeros((3, 3)), target_orientations=np.zeros((2, 3)), orientation_mode="X")
    with pytest.raises(ValueError):
        ergo.inverse_kinematics_trajectory_frame(np.zeros((3, 3)))
    assert ergo.inverse_kinematics_trajectory_frame(np.zeros((0, 4, 4))).shape == (0, len(ergo.links))


def test_trajectory_first_step_can_use_the_multistart(ergo):
    """An int or a dict as initial_position runs the multi-start on the first step only"""
    n_links = len(ergo.links)
    joints = [0, 0.5, -0.4, 0.7, 0.3, -0.2, 0.1, 0][:n_links] + [0] * max(0, n_links - 8)
    target = ergo.forward_kinematics(joints)
    frames = geometry.interpolate_frames(target, target, 3)
    trajectory = ergo.inverse_kinematics_trajectory_frame(frames, initial_position={"n": 5, "seed": 0}, orientation_mode="all")
    for q in trajectory:
        np.testing.assert_allclose(ergo.forward_kinematics(q)[:3, 3], target[:3, 3], atol=2e-3)
    np.testing.assert_allclose(trajectory[1], trajectory[0], atol=1e-3)
