import unittest
import poppy_inverse_kinematics.creature


class TestIK(unittest.TestCase):
    def test_ik_plot(self):
        robot = poppy_inverse_kinematics.creature.creature(creature_type="torso_left_arm", ik_regularization_parameter=0.01)
        robot.target = [0.1, -0.2, 0.1]
        robot.goto_target()
        robot.plot_model()

if __name__ == '__main__':
    unittest.main(verbosity=2)
