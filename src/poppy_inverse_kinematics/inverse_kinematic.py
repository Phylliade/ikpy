# coding= utf8
import scipy.optimize
import numpy as np
from . import forward_kinematics


def get_distance_to_target(robot_parameters, nodes_angles, target, end_point=None, model_type="custom", representation="euler"):
    """Calcule la distance au carré de l'extrêmité du robot à la target"""
    n = len(robot_parameters)

    # Extrêmité du robot
    # On prend bien la position n (au lieu de n-1)
    end_point = forward_kinematics.get_nodes(robot_parameters, nodes_angles, model_type=model_type, representation=representation)["positions"][n]
    return np.linalg.norm(end_point - target)
    # return sum([(end_point_i - target_i) ** 2 for (end_point_i, target_i) in zip(end_point, target)])


def inverse_kinematic(target, transformation_lambda, starting_nodes_angles, fk_method="default", bounds=None, **kwargs):
    """Calcule les angles pour atteindre la target"""
    print("Sarting optimisation with bounds : ", bounds)

    # Utilisation d'une optimisation L-BFGS-B
    def optimize_fun(x):
        y = np.append(starting_nodes_angles[:3], x)
        return np.linalg.norm(forward_kinematics.get_end_effector(y, method=fk_method, transformation_lambda=transformation_lambda, **kwargs) - target)

    res = scipy.optimize.minimize(optimize_fun, starting_nodes_angles[3:], method='L-BFGS-B', bounds=bounds[3:], options={"maxiter": 100})
    print(res.message, res.nit)
    return(np.append(starting_nodes_angles[:3], res.x))


def inverse_kinematic_trajectory(targets_x, targets_y, targets_z, transformation_lambda, starting_nodes_angles, fk_method="default", bounds=None, **kwargs):
    """Renvoie la liste des angles pour suivre la trajectoire (liste "targets") donnée en argument"""
    IK_angles = []
    nodes_angles = starting_nodes_angles
    for target in zip(targets_x, targets_y, targets_z):
        IK_angles.append(inverse_kinematic(target, transformation_lambda, nodes_angles, fk_method="default", bounds=None, **kwargs))
        nodes_angles = IK_angles[-1]
    return IK_angles
