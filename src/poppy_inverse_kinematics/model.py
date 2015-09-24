import numpy as np
import copy as cp
import math
from . import forward_kinematics as fk
from . import inverse_kinematic as ik
from . import plot_utils as pl
from .tools import transformations as tr

class Model(object):
    def __init__(self, links, rot=None, trans=None):
        self.links = links
        self.nb_joints = len(links)
        # initialize the parameters according to the kinematic library
        self.init_params(links)
        # set the transformations from world to base
        self.set_base_transformations(rot,trans)
        # initialize starting configuration
        self.current_joints = np.zeros(len(links))
        self.current_pose = self.forward_kinematic(self.current_joints)

    def set_base_transformations(self, rot=None, trans=None):
        if rot is None:
            rot = np.eye(3)
        if trans is None:
            trans = [0,0,0]
        self.world_to_base = tr.transformation(rot,trans)
        self.base_to_world = tr.inverse_transform(self.world_to_base)

    def init_params(self, links):
        vectors = []
        for l in links:
            vectors.append(([0,0,1],[l,0,0]))
        self.parameters = fk.euler_from_URDF_parameters(vectors)

    def set_max_velocity(self, v):
        self.max_velocity = v

    def forward_kinematic(self, q=None):
        if q is None:
            q = self.current_joints
        # calculate the forward kinematic
        X = fk.get_nodes(self.parameters,q)
        # return the result in the world frame
        W_X = tr.transform_point(X[0][-1],self.world_to_base)
        return W_X

    def inverse_kinematic(self, W_X, seed=None):
        if seed is None:
            seed = self.current_joints
        # calculate the coordinate of the target in the robot frame
        X = tr.transform_point(W_X,self.base_to_world)
        # return the inverse kinematic
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
