import unittest
import numpy as np
import sys
from ikpy import chain
from ikpy import plot_utils
import params

plot = params.interactive


class TestChain(unittest.TestCase):
    def setUp(self):
        if plot:
            self.ax = plot_utils.init_3d_figure()
        self.chain1 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, False, True, True, True, True, True])
        self.joints = [0] * len(self.chain1.links)
        self.joints[-4] = 0

    def test_chain(self):
        self.chain1 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, False, True, True, True, True, True])
        self.chain2 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "l_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, False, True, True, True, True, True])

        if plot:
            self.chain1.plot(self.joints, self.ax)
            self.chain2.plot(self.joints, self.ax)

    def test_ik(self):
        target = [0.1, -0.2, 0.1]
        frame_target = np.eye(4)
        frame_target[:3, 3] = target
        ik = self.chain1.inverse_kinematics(frame_target, initial_position=self.joints)

        if plot:
            self.chain1.plot(ik, self.ax, target=target)
            plot_utils.show_figure()

        np.testing.assert_almost_equal(self.chain1.forward_kinematics(ik)[:3, 3], target, decimal=3)


if __name__ == '__main__':
    unittest.main(verbosity=2, argv=[sys.argv[0]])
