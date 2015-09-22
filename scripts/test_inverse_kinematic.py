# coding: utf8
from poppy_inverse_kinematics import inverse_kinematic as ik
from poppy_inverse_kinematics import test_sets
from poppy_inverse_kinematics import plot_utils
import matplotlib.pyplot


if (__name__ == "__main__"):
    # Définition des paramètres
    robot_parameters = test_sets.classical_arm_parameters
    starting_nodes_angles = test_sets.classical_arm_default_angles
    target = [3, 1.7, 3]

    # Calcul de la réponse
    angles = ik.inverse_kinematic(robot_parameters, starting_nodes_angles, target)

    # Affichage du résultat
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_utils.plot_robot(robot_parameters, angles, ax)
    plot_utils.plot_target(target, ax)
    print(get_squared_distance_to_target(robot_parameters, angles, target))
    matplotlib.pyplot.show()