import unittest
import numpy as np
import sys
from ikpy import chain
from ikpy import plot_utils
from ikpy import matrix_link
import sympy
import params

plot = params.interactive


class TestChain(unittest.TestCase):
    def setUp(self):
        if plot:
            self.ax = plot_utils.init_3d_figure()
        self.chain1 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, True, True, True, True, True])
        self.joints = [0] * self.chain1.get_num_params()
        self.joints[-4] = 0
        self.target = [0.1, -0.2, 0.1]
        self.frame_target = np.eye(4)
        self.frame_target[:3, 3] = self.target

    def test_chain(self):
        self.chain1 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, True, True, True, True, True])
        self.chain2 = chain.Chain.from_urdf_file(params.resources_path + "/poppy_torso.URDF", base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "l_shoulder_y"], last_link_vector=[0, 0.18, 0], active_links_mask=[False, False, False, True, True, True, True, True])

        for link in self.chain1.links:
            print(link.name + " : " + str(link.parent))
        
        if plot:
            self.chain1.plot(self.joints, self.ax)
            self.chain2.plot(self.joints, self.ax)

    def test_ik(self):

        ik = self.chain1.inverse_kinematics(self.frame_target, initial_position=self.joints)

        if plot:
            self.chain1.plot(ik, self.ax, target=self.target)
            plot_utils.show_figure()

        np.testing.assert_almost_equal(self.chain1.forward_kinematics(ik)[:3, 3], self.target, decimal=3)

    def test_ik_optimization(self):
        """Tests the IK optimization-based method"""
        args = {"max_iter": 3}
        ik = self.chain1.inverse_kinematics(self.frame_target, initial_position=self.joints, **args)
        # Check whether the results are almost equal
        np.testing.assert_almost_equal(self.chain1.forward_kinematics(ik)[:3, 3], self.target, decimal=1)
        print(self.chain1.forward_kinematics(ik))

    def test_matrix(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        l1 = matrix_link.VariableMatrixLink("test", ((y*sympy.cos(x),-y*sympy.sin(x),0,0),(y*sympy.sin(x),y*sympy.cos(x),0,0),(0,0,1,0),(0,0,0,1)), [x, y])
        l2 = matrix_link.ConstantMatrixLink("test", ((1,0,0,10),(0,1,0,0),(0,0,1,0),(0,0,0,1)))
        c = chain.Chain([l1, l2], [True, False])
        args = {"max_iter": 100}
        print(c.forward_kinematics([3.1415, 1]))
        print(c.inverse_kinematics(np.matrix(((1.0,0,0,-5),(0,1,0,0.01),(0,0,1,0),(0,0,0,1))), [3.1415, 1], **args))
        

if __name__ == '__main__':
    unittest.main(verbosity=2, argv=[sys.argv[0]])
