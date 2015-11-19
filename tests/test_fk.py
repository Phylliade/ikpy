import unittest
import poppy_inverse_kinematics.creature as creature
import numpy as np


class TestFK(unittest.TestCase):
    def setUp(self):
        self.torso_right_arm = creature.creature(creature_type="torso_right_arm")
        all_zeros = [0 for i in range(0, self.torso_right_arm.config.joints_number)]
        one_move = [0 for i in range(0, self.torso_right_arm.config.joints_number)]
        one_move[3] = 10
        self.test_pos = one_move

    def test_fk_creature(self):

        np.testing.assert_almost_equal(self.torso_right_arm.forward_kinematic(), [-0.3535, 0.0415, 0.004], decimal=3)

    def test_fk_creature_pypot(self):
        torso_right_arm = creature.creature(creature_type="torso_right_arm", interface_type="vrep")
        torso_right_arm.goal_joints = self.test_pos
        torso_right_arm.pypot_sync_goal_joints()
        torso_right_arm.plot_model()

if __name__ == '__main__':
    unittest.main(verbosity=2)
