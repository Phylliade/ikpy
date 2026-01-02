# coding= utf8
"""
Utility functions for MJCF parsing.
Contains conversion functions for various orientation representations used in MJCF.

MJCF supports multiple ways to specify frame orientations:
- quat: quaternion (w, x, y, z)
- axisangle: axis (x, y, z) and angle
- euler: Euler angles with configurable sequence
- xyaxes: X and Y axes of the frame
- zaxis: Z axis of the frame (minimal rotation from [0, 0, 1])
"""

import numpy as np


def quat_to_rotation_matrix(quat):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    quat: list or numpy.array
        Quaternion in (w, x, y, z) format (MuJoCo convention)

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Rotation matrix from quaternion
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def rotation_matrix_to_rpy(R):
    """
    Convert a rotation matrix to roll-pitch-yaw (RPY) angles.

    Uses the convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Parameters
    ----------
    R: numpy.array
        3x3 rotation matrix

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    # Handle gimbal lock cases
    if abs(R[2, 0]) >= 1 - 1e-10:
        # Gimbal lock
        yaw = 0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return [roll, pitch, yaw]


def quat_to_rpy(quat):
    """
    Convert a quaternion to RPY angles.

    Parameters
    ----------
    quat: list or numpy.array
        Quaternion in (w, x, y, z) format

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    R = quat_to_rotation_matrix(quat)
    return rotation_matrix_to_rpy(R)


def axisangle_to_rotation_matrix(axis, angle):
    """
    Convert axis-angle representation to rotation matrix.

    Parameters
    ----------
    axis: list or numpy.array
        Rotation axis (will be normalized)
    angle: float
        Rotation angle in radians

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    return np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])


def axisangle_to_rpy(axis, angle):
    """
    Convert axis-angle to RPY angles.

    Parameters
    ----------
    axis: list or numpy.array
        Rotation axis
    angle: float
        Rotation angle in radians

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    R = axisangle_to_rotation_matrix(axis, angle)
    return rotation_matrix_to_rpy(R)


def _euler_rotation_matrix(angles, sequence):
    """
    Build rotation matrix from Euler angles with given sequence.

    Parameters
    ----------
    angles: list
        Three Euler angles in radians
    sequence: str
        Three-letter sequence like 'xyz', 'zyx', etc.

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rotations = {'x': rx, 'y': ry, 'z': rz, 'X': rx, 'Y': ry, 'Z': rz}

    R = np.eye(3)
    for angle, axis in zip(angles, sequence.lower()):
        R = R @ rotations[axis](angle)

    return R


def euler_to_rpy(euler, sequence="xyz"):
    """
    Convert Euler angles to RPY.

    Parameters
    ----------
    euler: list
        Three Euler angles in radians
    sequence: str
        Euler sequence (e.g., 'xyz', 'zyx')

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    R = _euler_rotation_matrix(euler, sequence)
    return rotation_matrix_to_rpy(R)


def xyaxes_to_rotation_matrix(xyaxes):
    """
    Convert xyaxes representation to rotation matrix.
    The X and Y axes are given, Y is orthogonalized, Z is computed as cross product.

    Parameters
    ----------
    xyaxes: list
        6 values: [x_axis(3), y_axis(3)]

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])

    # Normalize X axis
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Make Y orthogonal to X
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Z is cross product
    z_axis = np.cross(x_axis, y_axis)

    # Build rotation matrix (columns are the axes)
    return np.column_stack([x_axis, y_axis, z_axis])


def xyaxes_to_rpy(xyaxes):
    """
    Convert xyaxes to RPY angles.

    Parameters
    ----------
    xyaxes: list
        6 values: [x_axis(3), y_axis(3)]

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    R = xyaxes_to_rotation_matrix(xyaxes)
    return rotation_matrix_to_rpy(R)


def zaxis_to_rotation_matrix(zaxis):
    """
    Convert zaxis representation to rotation matrix.
    Finds the minimal rotation that maps [0, 0, 1] to the given Z axis.

    Parameters
    ----------
    zaxis: list
        3 values defining the Z axis

    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    z_target = np.array(zaxis)
    z_target = z_target / np.linalg.norm(z_target)

    z_default = np.array([0, 0, 1])

    # If already aligned
    dot = np.dot(z_default, z_target)
    if dot > 1 - 1e-10:
        return np.eye(3)
    if dot < -1 + 1e-10:
        # Rotate 180 degrees around X
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Rotation axis is cross product
    axis = np.cross(z_default, z_target)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    angle = np.arccos(dot)

    return axisangle_to_rotation_matrix(axis, angle)


def zaxis_to_rpy(zaxis):
    """
    Convert zaxis to RPY angles.

    Parameters
    ----------
    zaxis: list
        3 values defining the Z axis

    Returns
    -------
    list
        [roll, pitch, yaw] in radians
    """
    R = zaxis_to_rotation_matrix(zaxis)
    return rotation_matrix_to_rpy(R)
