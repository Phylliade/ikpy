import numpy as np
import copy as cp
import math
from . import forward_kinematics as fk
from . import inverse_kinematic as ik
from . import plot_utils as pl

class Model(object):
    def __init__(self, links):
        self.links = links
        self.nb_joints = len(links)
        # initialize the parameters according to the kinematic library
        self.init_params(links)
        # initialize starting configuration
        self.current_joints = np.zeros(len(links))
        self.current_pose = self.forward_kinematic(self.current_joints)

    def init_params(self, links):
        # vectors = [([0,0,1],[1,0,0])]
        vectors = []
        for l in links:
            vectors.append(([0,0,1],[l,0,0]))
        self.parameters = fk.euler_from_URDF_parameters(vectors)

    def set_max_velocity(self, v):
        self.max_velocity = v

    def forward_kinematic(self, q=None):
        if q is None:
            q = self.current_joints
        X = fk.get_nodes(self.parameters,q)
        return X[0][-1]

    def inverse_kinematic(self, X, seed=None):
        if seed is None:
            seed = self.current_joints
        return ik.inverse_kinematic(self.parameters,seed,X)

    def set_current_joints(self, q):
        self.current_joints = q

    def plot_model(self, q=None, target=None):
        if q is None:
            q = self.current_joints
        ax = pl.init_3d_figure()
        pl.plot_robot(self.parameters, q, ax)
        if not target is None:
            pl.plot_target(target, ax)
        pl.show_figure()
