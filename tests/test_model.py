import unittest
import numpy as np
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import scripts_config


class TestModel(unittest.TestCase):
    def setUp(self):

        urdf_params = poppy_inverse_kinematics.robot_utils.get_urdf_parameters(scripts_config.urdf_file, scripts_config.base_link, scripts_config.last_link_vector)
        self.robot = poppy_inverse_kinematics.model.Model(urdf_params, representation="rpy", model_type="URDF", computation_method="hybrid", simplify=False)

        target = [-0.1, 0.1, 0.1]

        self.robot.target = target

    def test_ik(self):
        self.robot.goto_target()
        self.robot.plot_model()
        np.testing.assert_almost_equal(self.robot.forward_kinematic(), self.robot.target, decimal=3)


if __name__ == '__main__':
    unittest.main()
