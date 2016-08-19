import unittest
import numpy as np
import sys
from ikpy import chain
from ikpy import plot_utils
from ikpy import matrix_link
from ikpy import blender_utils
import sympy
import params
import json
import subprocess

plot = params.interactive


class TestChain(unittest.TestCase):
    def setUp(self):
        if plot:
            self.ax = plot_utils.init_3d_figure()
            self.ax.set_xlim3d([-300.0, 300.0])
            self.ax.set_xlabel('X')
            self.ax.set_ylim3d([-300.0, 300.0])
            self.ax.set_ylabel('Y')
            self.ax.set_zlim3d([-300.0, 300.0])
            self.ax.set_zlabel('Z')

    def test_matrix(self):
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        l1 = matrix_link.VariableMatrixLink("r1", [[sympy.cos(x),-sympy.sin(x),0,0],[sympy.sin(x),sympy.cos(x),0,0],[0,0,1,0],[0,0,0,1]], [x, y])
        l2 = matrix_link.ConstantMatrixLink("test", [[10],[0],[0],[1]])
        c = chain.Chain([l1, l2], [True, False])
        args = {"max_iter": 100}
        target_matrix = np.eye(4)
        target = [0,10,0]
        target_matrix[:3,3] = target
        ik = c.inverse_kinematics(target_matrix, [3.1415, 0], **args)
        np.testing.assert_almost_equal(c.forward_kinematics(ik)[:3, 0], target, decimal=1)
        if plot:
            c.plot(ik, self.ax)
            plot_utils.show_figure()

    def test_blender(self):
        blend_path = "../../resources/buggybot.blend"
        #for side in ["left_back", "right_back", "left_front", "right_front"]:
        side = "left_back"
        endpoint = "armature/forearm_"+side+"/endpoint"
        c = chain.Chain.from_blend_file(blend_path, endpoint)
        target_matrix = np.eye(4)
        target = [0,0,0]
        initial = [0,0,0]
        if side == "left_back":
            target = [-100,100,-100]
            initial = [0,0,3.1415/2]
        if side == "right_back":
            target = [-100,-100,-100]
            initial = [0,0,3.1415/2]
        if side == "left_front":
            target = [100,100,-100]
            initial = [0,0,-3.1415/2]
        if side == "right_front":
            target = [100,-100,-100]
            initial = [0,0,-3.1415/2]
        target_matrix[:3,3] = target
        args = {"max_iter": 1000}
        ik = c.inverse_kinematics(target_matrix, initial, **args)
        if plot:
            c.plot(ik, self.ax)
        if plot:
            plot_utils.show_figure()

if __name__ == '__main__':
    unittest.main(verbosity=2, argv=[sys.argv[0]])
