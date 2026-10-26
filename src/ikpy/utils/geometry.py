# coding= utf8
"""
.. module:: geometry_utils
This module contains helper functions used to compute 3D geometric transformations.
"""
import numpy as np
import sympy


def rx_matrix(theta):
    """Rotation matrix around the X axis"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def rz_matrix(theta):
    """Rotation matrix around the Z axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def symbolic_rz_matrix(symbolic_theta):
    """Matrice symbolique de rotation autour de l'axe Z"""
    return sympy.Matrix([
        [sympy.cos(symbolic_theta), -sympy.sin(symbolic_theta), 0],
        [sympy.sin(symbolic_theta), sympy.cos(symbolic_theta), 0],
        [0, 0, 1]
    ])


def ry_matrix(theta):
    """Rotation matrix around the Y axis"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotation_matrix(phi, theta, psi):
    """Return a rotation matrix using the given Euler angles"""
    return np.dot(rz_matrix(phi), np.dot(rx_matrix(theta), rz_matrix(psi)))


def symbolic_rotation_matrix(phi, theta, symbolic_psi):
    """Retourne une matrice de rotation où psi est symbolique"""
    return sympy.Matrix(rz_matrix(phi)) * sympy.Matrix(rx_matrix(theta)) * symbolic_rz_matrix(symbolic_psi)


def rpy_matrix(roll, pitch, yaw):
    """Return a rotation matrix described by the extrinsinc roll, pitch, yaw coordinates"""
    return np.dot(rz_matrix(yaw), np.dot(ry_matrix(pitch), rx_matrix(roll)))


def _axis_rotation_matrix_formula(x, y, z, c, s):
    return [
        [x ** 2 + (1 - x ** 2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y ** 2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z ** 2 + (1 - z ** 2) * c]
    ]


def axis_rotation_matrix(axis, theta):
    """Returns a rotation matrix around the given axis"""
    [x, y, z] = axis
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(_axis_rotation_matrix_formula(x, y, z, c, s))


def symbolic_axis_rotation_matrix(axis, symbolic_theta):
    """Returns a rotation matrix around the given axis"""
    [x, y, z] = axis
    c = sympy.cos(symbolic_theta)
    s = sympy.sin(symbolic_theta)
    return sympy.Matrix(_axis_rotation_matrix_formula(x, y, z, c, s))


def get_translation_matrix(mu):
    """Returns a translation matrix of the given mu"""
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = mu
    return translation_matrix


def get_symbolic_translation_matrix(mu):
    """Returns a translation matrix of the given mu"""
    translation_matrix = sympy.eye(4)
    translation_matrix[:3, 3] = mu
    return sympy.Matrix(translation_matrix)


def homogeneous_translation_matrix(trans_x, trans_y, trans_z):
    """Return a translation matrix the homogeneous space"""
    return np.array([[1, 0, 0, trans_x], [0, 1, 0, trans_y], [0, 0, 1, trans_z], [0, 0, 0, 1]])


def from_transformation_matrix(transformation_matrix):
    """Convert a transformation matrix to a tuple (translation_vector, rotation_matrix)"""
    return transformation_matrix[:, -1], transformation_matrix[:-1, :-1]


def to_transformation_matrix(translation, orientation_matrix=np.zeros((3, 3))):
    """Convert a tuple (translation_vector, orientation_matrix) to a transformation matrix

    Parameters
    ----------
    translation: numpy.array
        The translation of your frame presented as a 3D vector.
    orientation_matrix: numpy.array
        Optional : The orientation of your frame, presented as a 3x3 matrix.
    """
    matrix = np.eye(4)

    matrix[:-1, :-1] = orientation_matrix
    matrix[:-1, -1] = translation
    return matrix


def cartesian_to_homogeneous(cartesian_matrix, matrix_type="numpy"):
    """Converts a cartesian matrix to an homogeneous matrix"""
    dimension_x, dimension_y = cartesian_matrix.shape
    # Square matrix
    # Manage different types fo input matrixes
    if matrix_type == "numpy":
        homogeneous_matrix = np.eye(dimension_x + 1)
    elif matrix_type == "sympy":
        homogeneous_matrix = sympy.eye(dimension_x + 1)
    else:
        raise ValueError("Unknown matrix_type: {}".format(matrix_type))
    # Add a column filled with 0 and finishing with 1 to the cartesian matrix to transform it into an homogeneous one
    homogeneous_matrix[:-1, :-1] = cartesian_matrix

    return homogeneous_matrix


def cartesian_to_homogeneous_vectors(cartesian_vector, matrix_type="numpy"):
    """Converts a cartesian vector to an homogeneous vector"""
    dimension_x = cartesian_vector.shape[0]
    # Vector
    if matrix_type == "numpy":
        homogeneous_vector = np.zeros(dimension_x + 1)
        # Last item is a 1
        homogeneous_vector[-1] = 1
        homogeneous_vector[:-1] = cartesian_vector
    else:
        raise ValueError("Unknown matrix_type: {}".format(matrix_type))
    return homogeneous_vector


def homogeneous_to_cartesian_vectors(homogeneous_vector):
    """Convert an homogeneous vector to cartesian vector"""""
    return homogeneous_vector[:-1]


def homogeneous_to_cartesian(homogeneous_matrix):
    """Convert an homogeneous matrix to cartesian matrix"""
    return homogeneous_matrix[:-1, :-1]


# Conversions between orientation representations
# The homogeneous matrices used by IKPy are the reference representation; these helpers
# convert them to and from the representations used by 3D engines and ROS.

def _as_scipy_rotation(matrix):
    from scipy.spatial.transform import Rotation
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    return Rotation.from_matrix(matrix)


def rotation_matrix_to_rpy(matrix):
    """Return the (roll, pitch, yaw) angles of a rotation matrix.

    This is the inverse of :func:`rpy_matrix`: the angles are extrinsic, applied in the order x, y, z,
    as in URDF files.

    Parameters
    ----------
    matrix: numpy.array
        A 3x3 rotation matrix, or a 4x4 homogeneous matrix whose rotation part is used

    Returns
    -------
    numpy.array
        The (roll, pitch, yaw) angles, in radians
    """
    return _as_scipy_rotation(matrix).as_euler("xyz")


def rotation_matrix_to_euler(matrix, seq, degrees=False):
    """Return the Euler angles of a rotation matrix, for any axis sequence.

    Parameters
    ----------
    matrix: numpy.array
        A 3x3 rotation matrix, or a 4x4 homogeneous matrix whose rotation part is used
    seq: str
        The axis sequence, as in :meth:`scipy.spatial.transform.Rotation.as_euler`: lowercase letters
        ("xyz") for extrinsic rotations, uppercase ("ZYX") for intrinsic rotations
    degrees: bool
        Whether to return the angles in degrees instead of radians

    Returns
    -------
    numpy.array
        The three Euler angles
    """
    return _as_scipy_rotation(matrix).as_euler(seq, degrees=degrees)


def euler_to_rotation_matrix(angles, seq, degrees=False):
    """Return the 3x3 rotation matrix of three Euler angles, for any axis sequence.

    Parameters
    ----------
    angles: numpy.array
        The three Euler angles
    seq: str
        The axis sequence, see :func:`rotation_matrix_to_euler`
    degrees: bool
        Whether the angles are given in degrees instead of radians
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(seq, angles, degrees=degrees).as_matrix()


def rotation_matrix_to_quaternion(matrix, scalar_first=False):
    """Return the unit quaternion of a rotation matrix.

    Parameters
    ----------
    matrix: numpy.array
        A 3x3 rotation matrix, or a 4x4 homogeneous matrix whose rotation part is used
    scalar_first: bool
        If False (the default), the quaternion is returned as (x, y, z, w), the convention of Scipy, ROS and Unity.
        If True, it is returned as (w, x, y, z), the convention of Blender and Eigen.

    Returns
    -------
    numpy.array
        The quaternion, as a 4-vector
    """
    quaternion = _as_scipy_rotation(matrix).as_quat()
    if scalar_first:
        quaternion = np.roll(quaternion, 1)
    return quaternion


def quaternion_to_rotation_matrix(quaternion, scalar_first=False):
    """Return the 3x3 rotation matrix of a quaternion.

    The result can be passed as `target_orientation` to the inverse kinematics with `orientation_mode="all"`.

    Parameters
    ----------
    quaternion: numpy.array
        The quaternion, as a 4-vector. It does not need to be normalized.
    scalar_first: bool
        Whether the quaternion is given as (w, x, y, z) instead of (x, y, z, w), see
        :func:`rotation_matrix_to_quaternion`
    """
    from scipy.spatial.transform import Rotation
    quaternion = np.asarray(quaternion, dtype=float)
    if scalar_first:
        quaternion = np.roll(quaternion, -1)
    return Rotation.from_quat(quaternion).as_matrix()
