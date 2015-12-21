import unittest
import poppy_inverse_kinematics.creature
import numpy as np


class TestIK(unittest.TestCase):
    def test_ik_plot(self):
        robot = poppy_inverse_kinematics.creature.creature(creature_type="torso_left_arm", ik_regularization_parameter=None, max_ik_iterations=5)
        robot.target = [0.1, -0.2, 0.1]
        robot.goto_target()
        robot.plot_model()

    def test_local_ik(self):
        robot = poppy_inverse_kinematics.creature.creature(creature_type="torso_left_arm", ik_regularization_parameter=None, max_ik_iterations=5)
        robot.target = [0.1, -0.2, 0.1]
        robot.goto_target()
        robot.target = np.array([0.1, -0.2, 0.1]) + np.array([0.01, 0, 0])
        robot.goto_target()
        robot.plot_model()

if __name__ == '__main__':
    unittest.main(verbosity=2)
