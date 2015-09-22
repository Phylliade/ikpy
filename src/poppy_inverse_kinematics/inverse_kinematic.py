# coding: utf8
import scipy.optimize
import numpy as np
from . import forward_kinematics


def get_squared_distance_to_target(robot_parameters, nodes_angles, target):
    """Calcule la distance au carré de l'extrêmité du robot à la target"""
    n = len(robot_parameters)

    # Extrêmité du robot
    # On prend bien la position n (au lieu de n-1)
    end_point = forward_kinematics.get_nodes(robot_parameters, nodes_angles)[0][n]

    return sum([(end_point_i - target_i) ** 2 for (end_point_i, target_i) in zip(end_point, target)])


def inverse_kinematic(robot_parameters, starting_nodes_angles, target, bounds=None):
    """Calcule les angles pour atteindre la target"""
    # Utilisation d'une optimisation L-BFGS-B
    res = scipy.optimize.minimize(lambda x: get_squared_distance_to_target(robot_parameters, x, target), starting_nodes_angles, method='L-BFGS-B', bounds=bounds)
    return(res.x)


def inverse_kinematic_jacobian_method(robot_parameters, starting_nodes_angles, target):
    pass


def inverse_kinematic_trajectory(robot_parameters, starting_nodes_angles, targets_x, targets_y, targets_z, bounds=None):
    """Renvoie la liste des angles pour suivre la trajectoire (liste "targets") donnée en argument"""
    IK_angles = []
    nodes_angles = starting_nodes_angles
    for target in zip(targets_x, targets_y, targets_z):
        IK_angles.append(inverse_kinematic(robot_parameters, nodes_angles, target, bounds))
        nodes_angles = IK_angles[-1]
    return IK_angles

