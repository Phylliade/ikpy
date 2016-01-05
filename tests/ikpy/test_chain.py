import unittest
import numpy as np
import sys
from ikpy import chain
from ikpy import plot_utils
import params

plot = params.interactive


class TestChain(unittest.TestCase):
    def test_chain(self):
        a = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, False, True, True, True, True, True])
        b = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "l_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, False, True, True, True, True, True])

        joints = [0] * len(a.links)
        joints[-4] = 0

        if plot:
            ax = plot_utils.init_3d_figure()
            a.plot(joints, ax)
            b.plot(joints, ax)

        target = [0.1, -0.2, 0.1]
        frame_target = np.eye(4)
        frame_target[:3, 3] = target
        ik = a.inverse_kinematics(frame_target, initial_position=joints)

        if plot:
            a.plot(ik, ax, target=target)
            plot_utils.show_figure()

if __name__ == '__main__':
    unittest.main(verbosity=2, argv=[sys.argv[0]])
