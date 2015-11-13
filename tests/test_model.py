import unittest
import numpy as np
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import poppy_inverse_kinematics.model_config
import config_test


class TestModel(unittest.TestCase):
    def setUp(self):

        params = poppy_inverse_kinematics.model_config.from_urdf_file(config_test.urdf_file, config_test.base_link, config_test.last_link_vector)
        self.robots = []
        methods = ["default", "symbolic", "hybrid"]
        target = [-0.1, 0.1, 0.1]
        for index, method in enumerate(methods):
            self.robots.append(poppy_inverse_kinematics.model.Model(params, computation_method=method, simplify=False))
            self.robots[index].target = target

    def test_urdf_import(self):
        params = poppy_inverse_kinematics.model_config.from_urdf_file(config_test.urdf_file, config_test.base_link, config_test.last_link_vector)
        self.assertEqual(params.parameters, [([-0.000819116943949499, 0.0, 0.0395000000000191], [0.0, 0.0, 1.5707963267949], [0.0, 0.0, -1.0]), ([0.0, 0.0, 0.03], [1.5707963267949, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.07, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.03715, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), ([0.0, 0.0258500000000001, 0.0702000000000001], [1.5707963267949, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.07, 0.0], [0.0, 0.0, -9.36192041654558e-16], [-1.0, 0.0, 0.0])])

    def test_ik(self):
        for robot in self.robots:
            robot.goto_target()
            np.testing.assert_almost_equal(robot.forward_kinematic(), robot.target, decimal=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
