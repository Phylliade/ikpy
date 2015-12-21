import unittest
import numpy as np
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import poppy_inverse_kinematics.model_config
import config_test


class TestModel(unittest.TestCase):
    def test_urdf_import(self):
        params = poppy_inverse_kinematics.model_config.from_urdf_file(config_test.urdf_file, config_test.base_elements, config_test.last_link_vector, base_elements_type=config_test.base_elements_type)
        self.assertEqual(params.parameters, config_test.predicted_config)

if __name__ == '__main__':
    unittest.main(verbosity=2)
