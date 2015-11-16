import unittest
import poppy_inverse_kinematics.creature
import numpy as np


class TestModel(unittest.TestCase):
    def test_creature(self):
        creature = poppy_inverse_kinematics.creature.creature("torso_right_arm")
        np.testing.assert_almost_equal(creature.forward_kinematic(), [-0.3535, 0.0415, 0.004], decimal=3)

    def test_creature_vrep(self):
        creature = poppy_inverse_kinematics.creature.creature("ergo", pypot_type="vrep")

if __name__ == '__main__':
    unittest.main(verbosity=2)
