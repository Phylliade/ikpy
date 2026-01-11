# coding= utf8
"""
.. module:: jax_geometry
This module contains JAX-based helper functions for 3D geometric transformations.
These are JIT-compilable and differentiable versions of the numpy geometry functions.
"""
import jax.numpy as jnp
from jax import jit


@jit
def rx_matrix(theta):
    """Rotation matrix around the X axis"""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


@jit
def ry_matrix(theta):
    """Rotation matrix around the Y axis"""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


@jit
def rz_matrix(theta):
    """Rotation matrix around the Z axis"""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


@jit
def rotation_matrix(phi, theta, psi):
    """Return a rotation matrix using the given Euler angles"""
    return jnp.dot(rz_matrix(phi), jnp.dot(rx_matrix(theta), rz_matrix(psi)))


@jit
def rpy_matrix(roll, pitch, yaw):
    """Return a rotation matrix described by the extrinsic roll, pitch, yaw coordinates"""
    return jnp.dot(rz_matrix(yaw), jnp.dot(ry_matrix(pitch), rx_matrix(roll)))


@jit
def axis_rotation_matrix(axis, theta):
    """Returns a rotation matrix around the given axis using Rodrigues' rotation formula"""
    x, y, z = axis[0], axis[1], axis[2]
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    return jnp.array([
        [x ** 2 + (1 - x ** 2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y ** 2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z ** 2 + (1 - z ** 2) * c]
    ])


@jit
def homogeneous_translation_matrix(trans_x, trans_y, trans_z):
    """Return a translation matrix in homogeneous space"""
    return jnp.array([
        [1.0, 0.0, 0.0, trans_x],
        [0.0, 1.0, 0.0, trans_y],
        [0.0, 0.0, 1.0, trans_z],
        [0.0, 0.0, 0.0, 1.0]
    ])


@jit
def get_translation_matrix(mu):
    """Returns a translation matrix of the given mu"""
    translation_matrix = jnp.eye(4)
    translation_matrix = translation_matrix.at[:3, 3].set(mu)
    return translation_matrix


@jit
def cartesian_to_homogeneous(cartesian_matrix):
    """Converts a 3x3 cartesian matrix to a 4x4 homogeneous matrix"""
    homogeneous_matrix = jnp.eye(4)
    homogeneous_matrix = homogeneous_matrix.at[:3, :3].set(cartesian_matrix)
    return homogeneous_matrix


@jit
def cartesian_to_homogeneous_vectors(cartesian_vector):
    """Converts a 3D cartesian vector to a 4D homogeneous vector"""
    return jnp.array([cartesian_vector[0], cartesian_vector[1], cartesian_vector[2], 1.0])


@jit
def homogeneous_to_cartesian_vectors(homogeneous_vector):
    """Convert a homogeneous vector to cartesian vector"""
    return homogeneous_vector[:3]


@jit
def homogeneous_to_cartesian(homogeneous_matrix):
    """Convert a homogeneous matrix to cartesian matrix"""
    return homogeneous_matrix[:3, :3]


@jit
def from_transformation_matrix(transformation_matrix):
    """Convert a transformation matrix to a tuple (translation_vector, rotation_matrix)"""
    return transformation_matrix[:, -1], transformation_matrix[:3, :3]


@jit
def to_transformation_matrix(translation, orientation_matrix):
    """Convert a tuple (translation_vector, orientation_matrix) to a transformation matrix"""
    matrix = jnp.eye(4)
    matrix = matrix.at[:3, :3].set(orientation_matrix)
    matrix = matrix.at[:3, 3].set(translation)
    return matrix


def compute_link_frame_matrix_revolute(origin_translation, origin_orientation, rotation_axis, theta):
    """
    Compute the link frame matrix for a revolute joint.

    Parameters
    ----------
    origin_translation: jnp.ndarray
        The translation vector (3,)
    origin_orientation: jnp.ndarray
        The orientation as roll, pitch, yaw (3,)
    rotation_axis: jnp.ndarray
        The rotation axis (3,)
    theta: float
        The joint angle

    Returns
    -------
    jnp.ndarray
        4x4 transformation matrix
    """
    # Translation matrix
    frame_matrix = homogeneous_translation_matrix(
        origin_translation[0],
        origin_translation[1],
        origin_translation[2]
    )

    # Orientation matrix
    orientation = cartesian_to_homogeneous(
        rpy_matrix(origin_orientation[0], origin_orientation[1], origin_orientation[2])
    )
    frame_matrix = jnp.dot(frame_matrix, orientation)

    # Rotation matrix
    rotation = cartesian_to_homogeneous(axis_rotation_matrix(rotation_axis, theta))
    frame_matrix = jnp.dot(frame_matrix, rotation)

    return frame_matrix


def compute_link_frame_matrix_prismatic(origin_translation, origin_orientation, translation_axis, mu):
    """
    Compute the link frame matrix for a prismatic joint.

    Parameters
    ----------
    origin_translation: jnp.ndarray
        The translation vector (3,)
    origin_orientation: jnp.ndarray
        The orientation as roll, pitch, yaw (3,)
    translation_axis: jnp.ndarray
        The translation axis (3,)
    mu: float
        The joint translation

    Returns
    -------
    jnp.ndarray
        4x4 transformation matrix
    """
    # Translation matrix
    frame_matrix = homogeneous_translation_matrix(
        origin_translation[0],
        origin_translation[1],
        origin_translation[2]
    )

    # Orientation matrix
    orientation = cartesian_to_homogeneous(
        rpy_matrix(origin_orientation[0], origin_orientation[1], origin_orientation[2])
    )
    frame_matrix = jnp.dot(frame_matrix, orientation)

    # Translation
    translation_vector = translation_axis * mu
    frame_matrix = jnp.dot(frame_matrix, get_translation_matrix(translation_vector))

    return frame_matrix


def compute_link_frame_matrix_fixed(origin_translation, origin_orientation):
    """
    Compute the link frame matrix for a fixed joint.

    Parameters
    ----------
    origin_translation: jnp.ndarray
        The translation vector (3,)
    origin_orientation: jnp.ndarray
        The orientation as roll, pitch, yaw (3,)

    Returns
    -------
    jnp.ndarray
        4x4 transformation matrix
    """
    # Translation matrix
    frame_matrix = homogeneous_translation_matrix(
        origin_translation[0],
        origin_translation[1],
        origin_translation[2]
    )

    # Orientation matrix
    orientation = cartesian_to_homogeneous(
        rpy_matrix(origin_orientation[0], origin_orientation[1], origin_orientation[2])
    )
    frame_matrix = jnp.dot(frame_matrix, orientation)

    return frame_matrix
