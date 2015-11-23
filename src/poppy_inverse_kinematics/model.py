# coding= utf8
"""
.. module:: model
"""
import numpy as np
from . import forward_kinematics as fk
from . import inverse_kinematic as ik
from . import plot_utils as pl
from . import robot_utils
import matplotlib.pyplot


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

    def inverse_kinematic(self, target=None, initial_position=None):
        """Computes the IK for given target"""
        # If absolute_target is not given, use self.target
        if target is None:
            target = self.target

        if initial_position is None:
            initial_position = self.current_joints

        # Choose computation method
        if self.computation_method == "default":
            return ik.inverse_kinematic(target, self.transformation_lambda, initial_position, fk_method=self.computation_method, model_type=self.config.model_type, representation=self.config.representation, robot_parameters=self.config.parameters, bounds=self.config.bounds)
        else:
            return ik.inverse_kinematic(target, self.transformation_lambda, initial_position, fk_method=self.computation_method, bounds=self.config.bounds)

    def goto_target(self):
        """Déplace le robot vers la target donnée"""

        # Compute goal joints
        self.goal_joints = self.inverse_kinematic()

        # Go to goal joints
        self.goto_joints()

    def goto_joints(self):
        """Move the robot according to the goal joints"""
        self.sync_goal_joints()

        self.sync_current_joints()

    def sync_goal_joints(self):
        """Synchronize goal_joints value with goto_position value of Pypot object"""
        if self.pypot_object is not None:
            for index, joint in enumerate(self.config.parameters):
                if joint["name"] != "last_joint":
                    # If the joint is not the last (virtual) joint :
                    # Add offset
                    offset = self.config.motor_parameters[joint["name"]]["offset"]

                    if self.config.motor_parameters is not None:
                        # Manage orientation
                        if self.config.motor_parameters[joint["name"]]["orientation"] == "indirect":
                            orientation = "indirect"
                        else:
                            orientation = "direct"
                    else:
                        offset = 0
                        orientation = "direct"

                    angle = robot_utils.convert_angle_to_pypot(self.goal_joints[index], orientation, offset, joint_name=joint["name"])
                    print(joint["name"], self.goal_joints[index] * 180 / np.pi, angle)

                    # Use the name of the joint to map to the motor name
                    getattr(self.pypot_object, joint["name"]).goal_position = angle

    def sync_current_joints(self, pypot_sync=False):
        """Synchronise les valeurs de current_joints"""
        if self.pypot_object is not None and pypot_sync:
            for i, m in enumerate(self.pypot_object.motors):
                self.current_joints[i] = m.present_position * (np.pi / 2) / 180
        else:
            # On place le modèle directement dans la position voulue
            self.current_joints = self.goal_joints

    def plot_model(self, q=None, ax=None, show=True):
        """Affiche le modèle du robot"""
        if q is None:
            q = self.current_joints
        if ax is None:
            # If ax is not given, create one
            ax = pl.init_3d_figure()
        pl.plot_robot(self.config.parameters, q, ax, representation=self.config.representation, model_type=self.config.model_type)
        pl.plot_basis(self.config.parameters, ax, self.arm_length)

        # Plot the goal position
        if self.target is not None:
            pl.plot_target(self.target, ax)
        if(show):
            pl.show_figure()

    def animate_model(self, targets_x, targets_y, targets_z):
        """Animate the model moving along the trajectory"""
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Création d'un objet line
        line = ax.plot([0, 0], [0, 0], [0, 0])[0]

        # Plot de la trajectoire et du repère
        pl.plot_target_trajectory(targets_x, targets_y, targets_z, ax)
        pl.plot_basis(self.config.parameters, ax)

        IK_angles = []
        nodes_angles = self.current_joints
        for target in zip(targets_x, targets_y, targets_z):
            IK_angles.append(self.inverse_kinematic(target, initial_position=nodes_angles))
            nodes_angles = IK_angles[-1]

        animation = matplotlib.animation.FuncAnimation(fig, pl.update_line, len(IK_angles), fargs=(self.config.parameters, IK_angles, line, self.config.representation, self.config.model_type), interval=50)
        matplotlib.pyplot.show()

        return animation

    def get_robot_length(self):
        """Calcule la longueur du robot (tendu)"""
        translations_vectors = [x["translation"] for x in self.config.parameters]
        joints_lengths = [np.sqrt(sum([x**2 for x in vector]))
                          for vector in translations_vectors]
        return sum(joints_lengths)
