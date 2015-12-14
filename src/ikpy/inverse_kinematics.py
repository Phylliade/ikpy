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


def inverse_kinematic(target, transformation_lambda, starting_nodes_angles, fk_method="default", bounds=None, first_active_joint=0, regularization_parameter=None, max_iter=None, **kwargs):
    """Calcule les angles pour atteindre la target"""
    # print("Sarting optimisation with bounds : ", bounds)

    # Compute squared distance to target
    def optimize_target(x):
        y = np.append(starting_nodes_angles[:first_active_joint], x)
        squared_distance = np.linalg.norm(forward_kinematics.get_end_effector(y, method=fk_method, transformation_lambda=transformation_lambda, **kwargs) - target)
        return squared_distance

    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - starting_nodes_angles[first_active_joint:])
            return optimize_target(x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)

    # Manage bounds
    if bounds is not None:
        real_bounds = bounds[first_active_joint:]
    else:
        real_bounds = None

    options = {}
    # Manage iterations maximum
    if max_iter is not None:
        options["maxiter"] = max_iter
        print(options)

    # Utilisation d'une optimisation L-BFGS-B
    res = scipy.optimize.minimize(optimize_total, starting_nodes_angles[first_active_joint:], method='L-BFGS-B', bounds=real_bounds, options=options)

    print("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))
    if first_active_joint == 0:
        return(res.x)
    else:
        return(np.append(starting_nodes_angles[:first_active_joint], res.x))


def inverse_kinematic_trajectory(targets_x, targets_y, targets_z, transformation_lambda, starting_nodes_angles, fk_method="default", bounds=None, **kwargs):
    """Renvoie la liste des angles pour suivre la trajectoire (liste "targets") donnée en argument"""
    IK_angles = []
    nodes_angles = starting_nodes_angles
    for target in zip(targets_x, targets_y, targets_z):
        IK_angles.append(inverse_kinematic(target, transformation_lambda, nodes_angles, fk_method="default", bounds=None, **kwargs))
        nodes_angles = IK_angles[-1]
    return IK_angles
