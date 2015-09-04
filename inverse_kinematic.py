import plot3D
import scipy.optimize
import numpy as np
import matplotlib.pyplot


def get_squared_distance_to_target(robot_parameters, nodes_angles, target):
    """Calcule la distance au carré de l'extrêmité du robot à la target"""
    n = len(robot_parameters)

    # Extrêmité du robot
    end_point = plot3D.get_nodes(robot_parameters, nodes_angles)[0][n]
    return sum([(end_point_i - target_i) ** 2 for (end_point_i, target_i) in zip(end_point, target)])


def inverse_kinematic(robot_parameters, nodes_angles, target):
    """Calcule les angles pour atteindre la target"""
    res = scipy.optimize.minimize(lambda x: get_squared_distance_to_target(robot_parameters, x, target), starting_nodes_angles, method='BFGS')
    return(res.x)

# Optimisation


if (__name__ == "__main__"):
    # Définition des paramètres
    robot_parameters = [(0, 0, 4), (0, np.pi / 2, 3), (0, np.pi / 2, 1)]
    starting_nodes_angles = [0, 0, np.pi / 2]
    target = [3, 1.7, 3]

    # Calcul de la réponse
    angles = inverse_kinematic(robot_parameters, starting_nodes_angles, target)

    # Affichage du résultat
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3D.plot_robot(robot_parameters, angles, ax)
    ax.scatter(target[0], target[1], target[2], c="red")
    print(get_squared_distance_to_target(robot_parameters, angles, target))
    matplotlib.pyplot.show()
