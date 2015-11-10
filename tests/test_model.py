import unittest
from poppy_inverse_kinematics.model import Model
from poppy_inverse_kinematics.tools import transformations as tr
import math
import numpy as np


class TestModel(unittest.TestCase):

    def test_fk(self):
        links = [1, 1]
        m = Model(links)
        # test joint 1-1
        X = m.forward_kinematic([0, 0])
        np.testing.assert_almost_equal(X, [2, 0, 0])
        # test joint 1-2
        X = m.forward_kinematic([math.pi / 2., 0])
        np.testing.assert_almost_equal(X, [0, 2, 0])
        # test joint 1-3
        X = m.forward_kinematic([-math.pi / 2., 0])
        np.testing.assert_almost_equal(X, [0, -2, 0])
        # test joint 2-1
        X = m.forward_kinematic([0, math.pi / 2.])
        np.testing.assert_almost_equal(X, [1, 1, 0])
        # test joint 2-1
        X = m.forward_kinematic([0, -math.pi / 2.])
        np.testing.assert_almost_equal(X, [1, -1, 0])
        # test coupled 1
        X = m.forward_kinematic([math.pi / 2., math.pi / 2.])
        np.testing.assert_almost_equal(X, [-1, 1, 0])
        # test coupled 2
        X = m.forward_kinematic([-math.pi / 2., math.pi / 2.])
        np.testing.assert_almost_equal(X, [1, -1, 0])
        # test coupled 3
        X = m.forward_kinematic([math.pi / 2., -math.pi / 2.])
        np.testing.assert_almost_equal(X, [1, 1, 0])
        # test coupled 4
        X = m.forward_kinematic([-math.pi / 2., -math.pi / 2.])
        np.testing.assert_almost_equal(X, [-1, -1, 0])

    def test_ik(self):
        links = [1, 1, 1]
        m = Model(links)
        for i in range(100):
            # create end effector random pose
            X = np.random.uniform(-1, 1, 2)
            X = np.append(X, 0)
            # calculate inverse kinematic
            q = m.inverse_kinematic(X)
            # calculate forward kinematic from obtained q
            X2 = m.forward_kinematic(q)
            # assert equality between X and X2
            np.testing.assert_almost_equal(X, X2, decimal=3)

    def test_random_model(self):
        for i in range(100):
            nb_joints = np.random.randint(10) + 3
            links = np.random.uniform(1, 5, nb_joints)
            # create the model and check the ik
            m = Model(links)
            # create end effector random pose
            X = np.random.uniform(-3, 3, 2)
            X = np.append(X, 0)
            # calculate inverse kinematic
            q = m.inverse_kinematic(X)
            # calculate forward kinematic from obtained q
            X2 = m.forward_kinematic(q)
            # plot the model for testing
            # m.plot_model(q,X)
            # assert equality between X and X2
            np.testing.assert_almost_equal(X, X2, decimal=3)

    def test_base_transformation(self):
        links = [1, 1, 1]
        # shift the base by a random vector
        rot = np.eye(3)
        trans = np.random.uniform(-5, 5, 3)
        T = tr.transformation(rot, trans)
        # create the model
        m = Model(links, rot, trans)
        # check the forward kinematic 1
        Xd = tr.transform_point([2, 0, 0], T)
        X = m.forward_kinematic([0, 0])
        np.testing.assert_almost_equal(X, Xd)
        # check the forward kinematic 2
        Xd = tr.transform_point([1, -1, 0], T)
        X = m.forward_kinematic([0, -math.pi / 2.])
        np.testing.assert_almost_equal(X, Xd)
        # check the inverse kinematic
        X = np.random.uniform(-1, 1, 2)
        X = np.append(X, 0)
        WX = tr.transform_point(X, T)
        q = m.inverse_kinematic(WX)
        X2 = m.forward_kinematic(q)
        np.testing.assert_almost_equal(WX, X2, decimal=3)


if __name__ == '__main__':
    unittest.main()
