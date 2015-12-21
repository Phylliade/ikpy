import unittest
import poppy_inverse_kinematics.creature
import numpy as np


class TestCreature(unittest.TestCase):
    def test_creature(self):
        for creature_type in ["torso_right_arm", "torso_left_arm", "ergo"]:
            creature = poppy_inverse_kinematics.creature.creature(creature_type=creature_type)

    def test_creature_vrep(self):
        for creature_type in ["torso_right_arm", "torso_left_arm", "ergo"]:
            creature = poppy_inverse_kinematics.creature.creature(creature_type=creature_type, interface_type="vrep")


if __name__ == '__main__':
    unittest.main(verbosity=2)
