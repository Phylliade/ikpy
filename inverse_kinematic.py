# coding: utf8
import plot3D
import scipy.optimize
import numpy as np
import matplotlib.pyplot
import plot_utils
import forward_kinematics
import test_sets


def get_squared_distance_to_target(robot_parameters, nodes_angles, target):
    """Calcule la distance au carré de l'extrêmité du robot à la target"""
    n = len(robot_parameters)

    # Extrêmité du robot
    end_point = forward_kinematics.get_nodes(robot_parameters, nodes_angles)[0][n]

    return sum([(end_point_i - target_i) ** 2 for (end_point_i, target_i) in zip(end_point, target)])


def inverse_kinematic(robot_parameters, nodes_angles, target):
    """Calcule les angles pour atteindre la target"""
    # Utilisation d'une BFGS
    res = scipy.optimize.minimize(lambda x: get_squared_distance_to_target(robot_parameters, x, target), nodes_angles, method='BFGS')
    return(res.x)


def plot_target(target, ax):
    pass

if (__name__ == "__main__"):
    # Définition des paramètres
    robot_parameters = test_sets.classical_arm_parameters
    starting_nodes_angles = test_sets.classical_arm_default_angles
    target = [3, 1.7, 3]

    # Calcul de la réponse
    angles = inverse_kinematic(robot_parameters, starting_nodes_angles, target)

    # Affichage du résultat
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_utils.plot_robot(robot_parameters, angles, ax)
    plot_utils.plot_target(target, ax)
    print(get_squared_distance_to_target(robot_parameters, angles, target))
    matplotlib.pyplot.show()
