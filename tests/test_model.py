import unittest
import numpy as np
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import poppy_inverse_kinematics.model_config
import config_test


class TestModel(unittest.TestCase):
    def setUp(self):

        params = poppy_inverse_kinematics.model_config.from_urdf_file(config_test.urdf_file, config_test.base_elements, config_test.last_link_vector, base_elements_type=config_test.base_elements_type)
        self.robots = []
        methods = ["default", "symbolic", "hybrid"]
        target = [-0.1, 0.1, 0.1]
        for index, method in enumerate(methods):
            self.robots.append(poppy_inverse_kinematics.model.Model(params, computation_method=method, simplify=False))
            self.robots[index].target = target

    def test_ik(self):
        for robot in self.robots:
            robot.goto_target()
            robot.plot_model()
            np.testing.assert_almost_equal(robot.forward_kinematic(), robot.target, decimal=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
