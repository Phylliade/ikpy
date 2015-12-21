import unittest
import numpy as np
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import poppy_inverse_kinematics.model_config
import poppy_inverse_kinematics.creature


class TestModel(unittest.TestCase):

    def test_vrep(self):
        creature = poppy_inverse_kinematics.creature.creature("ergo_jr", interface_type="vrep")
        creature.target = [1, 0.5, 0.5]
        creature.goto_target()


if __name__ == '__main__':
    unittest.main(verbosity=2)
