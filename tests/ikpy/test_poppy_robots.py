import unittest
import numpy as np
import resources_tests
from ikpy import chain
from ikpy import plot_utils


plot = True


class TestPoppyRobot(unittest.TestCase):
    def test_ergo(self):
        a = chain.Chain.from_urdf_file(resources_tests.resources_path + "/poppy_ergo.URDF")
        target = [0.1, -0.2, 0.1]
        frame_target = np.eye(4)
        frame_target[:3, 3] = target
        joints = [0] * len(a.links)
        ik = a.inverse_kinematics(frame_target, initial_position=joints)

        if plot:
            ax = plot_utils.init_3d_figure()

        if plot:
            a.plot(ik, ax, target=target)
            plot_utils.show_figure()


if __name__ == '__main__':
    unittest.main(verbosity=2)
