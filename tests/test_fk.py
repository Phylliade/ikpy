import unittest
import poppy_inverse_kinematics.creature as creature
import numpy as np

creature_type = "torso_left_arm"


class TestFK(unittest.TestCase):
    def setUp(self):
        self.creature = creature.creature(creature_type=creature_type)
        all_zeros = [0 for i in range(0, self.creature.config.joints_number)]
        one_move = [0 for i in range(0, self.creature.config.joints_number)]
        one_move[5] = np.pi / 4
        one_move[6] = -np.pi / 2
        one_move[4] = -np.pi / 2
        self.test_pos = one_move

    def test_fk_creature(self):

        np.testing.assert_almost_equal(self.creature.forward_kinematic(), [-0.3535, 0.0415, 0.004], decimal=3)

    def test_fk_creature_pypot(self):
        creature_pypot = creature.creature(creature_type=creature_type, interface_type="vrep")
        creature_pypot.goal_joints = self.test_pos
        creature_pypot.pypot_sync_goal_joints(set_current=True)
        creature_pypot.plot_model()

if __name__ == '__main__':
    unittest.main(verbosity=2)
