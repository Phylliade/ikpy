# coding: utf8
# from poppy.creatures import PoppyRightArm
import plot_utils
import inverse_kinematic
import numpy as np
import matplotlib.pyplot
import test_sets
from mpl_toolkits.mplot3d import Axes3D


def init_third_arm(third_arm):
    """Initialise le bras"""

    third_arm.compliant = False
    third_arm.power_up()

    # Change PID of Dynamixel motors
    for m in filter(lambda m: hasattr(m, 'pid'), third_arm.motors):
        m.pid = (1.9, 5, 0)
    # Change PID of XL-320 motors
    for m in third_arm.r_gripper:
        m.pid = (4, 2, 0)
    # Change PID of Gripper (r_m5)
    third_arm.r_m5.pid = (8, 0, 0)
    # Reduce max torque to keep motor temperature low
    for m in third_arm.motors:
        m.torque_limit = 70

    third_arm.r_shoulder_x.goal_position = -20
    third_arm.r_m1.goal_position = 0
    third_arm.r_m4.goal_position = 90

    return third_arm

if __name__ == "__main__":
    real = False
    simulate = False
    animate = True

    # Paramètres du bras
    third_arm_parameters = test_sets.third_arm_parameters
    third_arm_starting_angles = test_sets.third_arm_default_angles
    third_arm_bounds = test_sets.third_arm_bounds

    # Cible
    target = [0.2, 0.2, -0.2]
    # target = [1, 0, 0]

    # Calcul de la position
    angles = inverse_kinematic.inverse_kinematic(third_arm_parameters, third_arm_starting_angles, target, bounds=third_arm_bounds)

    print(angles * 180 / np.pi)

    # Affichage du résultat simulé
    if (simulate):
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_utils.plot_robot(third_arm_parameters, angles, ax)
        ax.scatter(target[0], target[1], target[2], c="red", s=80)

    if (real):
        third_arm = PoppyRightArm()
        init_third_arm(third_arm)

        indirect_joints = [1, 3, 6, 8]

        joints = [third_arm.r_shoulder_y, third_arm.r_shoulder_x, third_arm.r_arm_z, third_arm.r_elbow_y, third_arm.r_m1, third_arm.r_m2, third_arm.r_m3, third_arm.r_m4, third_arm.r_m5]

        for index, joint in enumerate(joints):
            if index in indirect_joints:
                if index == 1:
                    joint.goal_position = -angles[index] / np.pi * 180 - 90
                else:
                    joint.goal_position = -angles[index] / np.pi * 180
            else:
                joint.goal_position = angles[index] / np.pi * 180

        third_arm.r_m5.goal_position = 0

    if (animate):
        x = np.arange(0, np.pi / 2, np.pi / 50)
        y = np.sin(x) * 0.3
        z = np.cos(x)**2 * 0.3
        targets = zip(z, y, z)
        fig = matplotlib.pyplot.figure()
        plot_utils.animate_IK(third_arm_parameters, third_arm_starting_angles, targets, fig, third_arm_bounds)
        matplotlib.pyplot.show()
