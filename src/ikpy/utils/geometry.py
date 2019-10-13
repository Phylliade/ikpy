# coding= utf8
"""
.. module:: geometry_utils
This module contains helper functions used to compute 3D geometric transformations.
"""
import numpy as np
import sympy


def Rx_matrix(theta):
    """Rotation matrix around the X axis"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def Rz_matrix(theta):
    """Rotation matrix around the Z axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def symbolic_Rz_matrix(symbolic_theta):
    """Matrice symbolique de rotation autour de l'axe Z"""
    return sympy.Matrix([
        [sympy.cos(symbolic_theta), -sympy.sin(symbolic_theta), 0],
        [sympy.sin(symbolic_theta), sympy.cos(symbolic_theta), 0],
        [0, 0, 1]
    ])


def Ry_matrix(theta):
    """Rotation matrix around the Y axis"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotation_matrix(phi, theta, psi):
    """Retourne la matrice de rotation décrite par les angles d'Euler donnés en paramètres"""
    return np.dot(Rz_matrix(phi), np.dot(Rx_matrix(theta), Rz_matrix(psi)))


def symbolic_rotation_matrix(phi, theta, symbolic_psi):
    """Retourne une matrice de rotation où psi est symbolique"""
    return sympy.Matrix(Rz_matrix(phi)) * sympy.Matrix(Rx_matrix(theta)) * symbolic_Rz_matrix(symbolic_psi)


def rpy_matrix(roll, pitch, yaw):
    """Returns a rotation matrix described by the extrinsinc roll, pitch, yaw coordinates"""
    return np.dot(Rz_matrix(yaw), np.dot(Ry_matrix(pitch), Rx_matrix(roll)))


def axis_rotation_matrix(axis, theta):
    """Returns a rotation matrix around the given axis"""
    [x, y, z] = axis
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [x**2 + (1 - x**2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y**2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z**2 + (1 - z**2) * c]
    ])


def symbolic_axis_rotation_matrix(axis, symbolic_theta):
    """Returns a rotation matrix around the given axis"""
    [x, y, z] = axis
    c = sympy.cos(symbolic_theta)
    s = sympy.sin(symbolic_theta)
    return sympy.Matrix([
        [x**2 + (1 - x**2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y**2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z**2 + (1 - z**2) * c]
    ])


def homogeneous_translation_matrix(trans_x, trans_y, trans_z):
    """Returns a translation matrix the homogeneous space"""
    return np.array([[1, 0, 0, trans_x], [0, 1, 0, trans_y], [0, 0, 1, trans_z], [0, 0, 0, 1]])


def from_transformation_matrix(transformation_matrix):
    """Converts a transformation matrix to a tuple (translation_vector, rotation_matrix)"""
    return (transformation_matrix[:, -1], transformation_matrix[:-1, :-1])


def to_transformation_matrix(translation, orientation_matrix=np.zeros((3, 3))):
    """Converts a tuple (translation_vector, orientation_matrix)  to a transformation matrix

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
    """Converts a cartesian matrix to an homogenous matrix"""
    dimension_x, dimension_y = cartesian_matrix.shape
    # Square matrix
    # Manage different types fo input matrixes
    if matrix_type == "numpy":
        homogeneous_matrix = np.eye(dimension_x + 1)
    elif matrix_type == "sympy":
        homogeneous_matrix = sympy.eye(dimension_x + 1)
    # Add a column filled with 0 and finishing with 1 to the cartesian matrix to transform it into an homogeneous one
    homogeneous_matrix[:-1, :-1] = cartesian_matrix

    return homogeneous_matrix


def cartesian_to_homogeneous_vectors(cartesian_vector, matrix_type="numpy"):
    """Converts a cartesian vector to an homogenous vector"""
    dimension_x = cartesian_vector.shape[0]
    # Vector
    if matrix_type == "numpy":
        homogeneous_vector = np.zeros(dimension_x + 1)
        # Last item is a 1
        homogeneous_vector[-1] = 1
        homogeneous_vector[:-1] = cartesian_vector
    return homogeneous_vector


def homogeneous_to_cartesian_vectors(homogeneous_vector):
    """Converts a cartesian vector to an homogenous vector"""
    return homogeneous_vector[:-1]


def homogeneous_to_cartesian(homogeneous_matrix):
    """Converts a cartesian vector to an homogenous matrix"""
    # Remove the last column
    return homogeneous_matrix[:-1, :-1]
