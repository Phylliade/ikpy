# coding: utf8
from poppy_inverse_kinematics import forward_kinematics as fk
from poppy_inverse_kinematics import test_sets
from poppy_inverse_kinematics import plot_utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot

if (__name__ == "__main__"):
    # Paramètres du robot
    robot_parameters = test_sets.classical_arm_parameters
    nodes_angles = test_sets.classical_arm_default_angles
    x = 5
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_utils.plot_robot(robot_parameters, nodes_angles, ax)
    matplotlib.pyplot.show()
    for i in range(1, 100):
        print("test")
