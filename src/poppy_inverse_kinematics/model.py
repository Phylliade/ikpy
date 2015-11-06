"""
.. module:: model
"""
import numpy as np
from . import forward_kinematics as fk
from . import inverse_kinematic as ik
from . import plot_utils as pl
from .tools import transformations as tr


class Model(object):
    """Base model class
    :param numerateur: le numerateur de la division
    :type numerateur: int
    :param denominateur: le denominateur de la division
    :type denominateur: int
    :return: la valeur entiere de la division
    :rtype: int
    """

    def __init__(self, links, rot=None, trans=None, representation="euler", model_type="custom", pypot_object=None, computation_method="default", simplify=False):
        # Configuration 2D
        self.links = links
        self.nb_joints = len(links)
        self.parameters = links
        self.arm_length = self.get_robot_length()
        self.model_type = model_type
        self.representation = representation
        self.computation_method = computation_method
        self.pypot_object = pypot_object
        self.simplify = simplify
        self.transformation_lambda = fk.compute_transformation(self.parameters, method=self.computation_method, representation=self.representation, model_type=self.model_type, simplify=self.simplify)
        # set the transformations from world to base
        self.set_base_transformations(rot, trans)
        # initialize starting configuration
        self.current_joints = np.zeros(len(links))
        self.current_pose = self.forward_kinematic(self.current_joints)
        self.target = self.current_pose

    def sym_mat(self, *args, **kwargs):
        return self.symbolic_transformation_matrix(*args, **kwargs)

    def set_base_transformations(self, rot=None, trans=None):
        if rot is None:
            rot = np.eye(3)
        if trans is None:
            trans = [0, 0, 0]
        self.world_to_base = tr.transformation(rot, trans)
        self.base_to_world = tr.inverse_transform(self.world_to_base)

    def init_params_2d(self, links):
        vectors = []
        for l in links:
            vectors.append(([0, 0, 1], [l, 0, 0]))
        self.parameters = fk.euler_from_URDF_parameters(vectors)

    def set_max_velocity(self, v):
        """Set maximum velocity of the robot"""
        self.max_velocity = v

    def forward_kinematic(self, q=None):
        """Renvoie la position du end effector en fonction de la configuration des joints"""
        if q is None:
            q = self.current_joints
        # calculate the forward kinematic
        if self.computation_method == "default":
            # Special args for the default method
            X = fk.get_end_effector(nodes_angles=q, method=self.computation_method, transformation_lambda=self.transformation_lambda, representation=self.representation, model_type=self.model_type, robot_parameters=self.parameters)
        else:
            X = fk.get_end_effector(nodes_angles=q, method=self.computation_method, transformation_lambda=self.transformation_lambda)
        # return the result in the world frame
        W_X = tr.transform_point(X, self.world_to_base)
        return W_X

    def inverse_kinematic(self, absolute_target=None):
        """Computes the IK for given target"""
        # If absolute_target is not given, use self.target
        if absolute_target is None:
            absolute_target = self.target

        # Compute relative target
        target = tr.transform_point(absolute_target, self.base_to_world)
        # Choose computation method
        if self.computation_method == "default":
            return ik.inverse_kinematic(target, self.transformation_lambda, self.current_joints, fk_method=self.computation_method, model_type=self.model_type, representation=self.representation, robot_parameters=self.parameters)
        else:
            return ik.inverse_kinematic(target, self.transformation_lambda, self.current_joints, fk_method=self.computation_method)

    def set_current_joints(self, q):
        """Set the position of the current joints"""
        self.current_joints = q

    def goto_target(self):
        """Déplace le robot vers la target donnée"""
        self.goal_joints = self.inverse_kinematic(self.target)

        if self.pypot_object is not None:
            # Si un robot pypot est attaché au modèle, on demande au robot d'aller vers les angles voulus
            self.pypot_sync_goal_joints()

            # On actualise la position des joints
            self.pypot_sync_current_joints()

        else:
            # Sinon on place le modèle directement dans la position voulue
            self.current_joints = self.goal_joints

    def pypot_sync_goal_joints(self):
        """Synchronise les valeurs de goal_joints"""
        if self.pypot_object is not None:
            for i, m in enumerate(self.pypot_object.motors):
                m.goal_position = self.goal_joints[i] * 180 / (np.pi / 2)

    def pypot_sync_current_joints(self):
        """Synchronise les valeurs de current_joints"""
        pass

    def plot_model(self, q=None):
        """Affiche le modèle du robot"""
        if q is None:
            q = self.current_joints
        ax = pl.init_3d_figure()
        pl.plot_robot(self.parameters, q, ax, representation=self.representation, model_type=self.model_type)
        pl.plot_basis(self.parameters, ax, self.arm_length)

        # Plot the goal position
        if self.target is not None:
            pl.plot_target(self.target, ax)
        pl.show_figure()

    def get_robot_length(self):
        """Calcule la longueur du robot (tendu)"""
        translations_vectors = [x[0] for x in self.parameters]
        joints_lengths = [np.sqrt(sum([x**2 for x in vector]))
                          for vector in translations_vectors]
        return sum(joints_lengths)
