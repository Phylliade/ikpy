import unittest
from ikpy import chain
import test_resources
from ikpy import plot_utils
import numpy as np


class TestChain(unittest.TestCase):
    def test_chain(self):
        a = chain.Chain.from_urdf_file(test_resources.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0])
        b = chain.Chain.from_urdf_file(test_resources.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "l_shoulder_y"], last_link_vector=[0, 0.18, 0])

        ax = plot_utils.init_3d_figure()
        joints = [0] * len(a.links)
        joints[-4] = 0
        a.plot(joints, ax)
        # b.plot(joints, ax)
        target = [0.1, -0.2, 0.1]

        a.plot(a.inverse_kinematic(target, initial_position=joints, first_active_joint=4), ax, target=target)
        plot_utils.show_figure()

if __name__ == '__main__':
    unittest.main(verbosity=2)
