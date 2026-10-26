import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.utils import geometry


ANGLES = [(0.0, 0.0, 0.0), (0.3, -0.7, 1.9), (-2.5, 1.2, 0.4), (np.pi / 2, 0.0, -np.pi / 2)]


@pytest.mark.parametrize("rpy", ANGLES)
def test_rpy_round_trip(rpy):
    """rotation_matrix_to_rpy is the inverse of rpy_matrix"""
    matrix = geometry.rpy_matrix(*rpy)
    recovered = geometry.rotation_matrix_to_rpy(matrix)
    np.testing.assert_allclose(geometry.rpy_matrix(*recovered), matrix, atol=1e-12)


@pytest.mark.parametrize("rpy", ANGLES)
def test_quaternion_round_trip(rpy):
    matrix = geometry.rpy_matrix(*rpy)
    for scalar_first in (False, True):
        quaternion = geometry.rotation_matrix_to_quaternion(matrix, scalar_first=scalar_first)
        assert quaternion.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0)
        np.testing.assert_allclose(
            geometry.quaternion_to_rotation_matrix(quaternion, scalar_first=scalar_first), matrix, atol=1e-12)


def test_quaternion_conventions():
    """A 90 degree rotation about z is (0, 0, sin(pi/4), cos(pi/4)) scalar-last"""
    matrix = geometry.rz_matrix(np.pi / 2)
    s = np.sin(np.pi / 4)
    np.testing.assert_allclose(geometry.rotation_matrix_to_quaternion(matrix), [0, 0, s, s], atol=1e-12)
    np.testing.assert_allclose(geometry.rotation_matrix_to_quaternion(matrix, scalar_first=True), [s, 0, 0, s], atol=1e-12)


def test_identity_quaternion():
    np.testing.assert_allclose(geometry.rotation_matrix_to_quaternion(np.eye(3)), [0, 0, 0, 1])
    np.testing.assert_allclose(geometry.quaternion_to_rotation_matrix([0, 0, 0, 1]), np.eye(3))


def test_unnormalized_quaternion_is_accepted():
    s = np.sin(np.pi / 4)
    np.testing.assert_allclose(
        geometry.quaternion_to_rotation_matrix([0, 0, 2 * s, 2 * s]), geometry.rz_matrix(np.pi / 2), atol=1e-12)


@pytest.mark.parametrize("seq", ["xyz", "ZYX", "zxz"])
def test_euler_round_trip(seq):
    matrix = geometry.rpy_matrix(0.3, -0.7, 1.9)
    angles = geometry.rotation_matrix_to_euler(matrix, seq)
    np.testing.assert_allclose(geometry.euler_to_rotation_matrix(angles, seq), matrix, atol=1e-12)
    degrees = geometry.rotation_matrix_to_euler(matrix, seq, degrees=True)
    np.testing.assert_allclose(np.deg2rad(degrees), angles)


def test_homogeneous_matrix_is_accepted():
    matrix = geometry.rpy_matrix(0.3, -0.7, 1.9)
    homogeneous = geometry.to_transformation_matrix([1, 2, 3], matrix)
    np.testing.assert_allclose(
        geometry.rotation_matrix_to_quaternion(homogeneous), geometry.rotation_matrix_to_quaternion(matrix))
    np.testing.assert_allclose(geometry.rotation_matrix_to_rpy(homogeneous), geometry.rotation_matrix_to_rpy(matrix))


def test_ik_from_quaternion_target(resources_path):
    """A quaternion target can be solved with orientation_mode='all'"""
    my_chain = chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")
    joints = [0, 0.3, -0.4, 0.5, 0.2, -0.3, 0]
    joints = joints + [0] * (len(my_chain.links) - len(joints))
    frame = my_chain.forward_kinematics(joints)
    quaternion = geometry.rotation_matrix_to_quaternion(frame)

    ik = my_chain.inverse_kinematics(
        target_position=frame[:3, 3],
        target_orientation=geometry.quaternion_to_rotation_matrix(quaternion),
        orientation_mode="all",
        initial_position=joints)
    np.testing.assert_allclose(my_chain.forward_kinematics(ik), frame, atol=1e-3)
