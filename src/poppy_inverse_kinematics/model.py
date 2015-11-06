"""
.. module:: model
"""
import numpy as np
from . import forward_kinematics as fk
from . import inverse_kinematic as ik
from . import plot_utils as pl


class Model():
    """Base model class

   :param configuration: The configuration of the robot
   :type configuration: model_config
   :param computation_method: Method for the computation of the Forward Kinematic
   :type computation_method: string
   :param simplify: Simplify symbolic expressions (hybrid and symbolic computation methods only)
   :type simplify: bool
    """

    def __init__(self, configuration, pypot_object=None, computation_method="default", simplify=False):
        # Configuration 2D
        self.config = configuration
        self.arm_length = self.get_robot_length()
        self.computation_method = computation_method
        self.pypot_object = pypot_object
        self.simplify = simplify
        self.transformation_lambda = fk.compute_transformation(self.config.parameters, method=self.computation_method, representation=self.config.representation, model_type=self.config.model_type, simplify=self.simplify)
        # initialize starting configuration
        self.current_joints = np.zeros(self.config.joints_number)
        self.current_pose = self.forward_kinematic(self.current_joints)
        self.target = self.current_pose

    def forward_kinematic(self, q=None):
        """Renvoie la position du end effector en fonction de la configuration des joints"""
        if q is None:
            q = self.current_joints
        # calculate the forward kinematic
        if self.computation_method == "default":
            # Special args for the default method
            X = fk.get_end_effector(nodes_angles=q, method=self.computation_method, transformation_lambda=self.transformation_lambda, representation=self.config.representation, model_type=self.config.model_type, robot_parameters=self.config.parameters)
        else:
            X = fk.get_end_effector(nodes_angles=q, method=self.computation_method, transformation_lambda=self.transformation_lambda)
        return X

    def inverse_kinematic(self, target=None):
        """Computes the IK for given target"""
        # If absolute_target is not given, use self.target
        if target is None:
            target = self.target

        # Choose computation method
        if self.computation_method == "default":
            return ik.inverse_kinematic(target, self.transformation_lambda, self.current_joints, fk_method=self.computation_method, model_type=self.config.model_type, representation=self.config.representation, robot_parameters=self.config.parameters)
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
            # self.pypot_sync_current_joints()

        # On place le modèle directement dans la position voulue
        self.current_joints = self.goal_joints

    def pypot_sync_goal_joints(self):
        """Synchronise les valeurs de goal_joints"""
        if self.pypot_object is not None:
            for i, m in enumerate(self.pypot_object.motors):
                m.goal_position = self.goal_joints[i] * 180 / (np.pi / 2)

    def pypot_sync_current_joints(self):
        """Synchronise les valeurs de current_joints"""
        if self.pypot_object is not None:
            for i, m in enumerate(self.pypot_object.motors):
                print(m.present_position)
                self.current_joints[i] = m.present_position * (np.pi / 2) / 180

    def plot_model(self, q=None):
        """Affiche le modèle du robot"""
        if q is None:
            q = self.current_joints
        ax = pl.init_3d_figure()
        pl.plot_robot(self.config.parameters, q, ax, representation=self.config.representation, model_type=self.config.model_type)
        pl.plot_basis(self.config.parameters, ax, self.arm_length)

        # Plot the goal position
        if self.target is not None:
            pl.plot_target(self.target, ax)
        pl.show_figure()

    def get_robot_length(self):
        """Calcule la longueur du robot (tendu)"""
        translations_vectors = [x[0] for x in self.config.parameters]
        joints_lengths = [np.sqrt(sum([x**2 for x in vector]))
                          for vector in translations_vectors]
        return sum(joints_lengths)
