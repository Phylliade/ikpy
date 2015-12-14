# coding= utf8
from . import URDF_utils
from . import forward_kinematics as fk
from . import inverse_kinematics as ik
from . import plot_utils


class Chain(object):
    """The base Chain class

    :param list links: List of the links of the chain
    :param list active_links: The list of the positions of the active links
    """
    def __init__(self, links, active_links=0, profile=''"", ik_solver=None, **kwargs):
        self.links = links

    def forward_kinematics(self, joints):
        """Returns the transformation matrix of the forward kinematics

        :param list joints: The list of the positions of each joint
        :returns: The transformation matrix
        """
        if self.computation_method == "default":
            # Special args for the default method
            X = fk.get_end_effector(nodes_angles=joints, method=self.computation_method, transformation_lambda=self.transformation_lambda, representation=self.config.representation, model_type=self.config.model_type, robot_parameters=self.config.parameters)
        else:
            X = fk.get_end_effector(nodes_angles=joints, method=self.computation_method, transformation_lambda=self.transformation_lambda)
        return X

    def inverse_kinematic(self, target=None, initial_position=None, regularization_parameter=None, max_iterations=None):
        """Computes the inverse kinematic on the specified target

        :param numpy.array target: The target of the inverse kinematic
        :param numpy.array initial_position: the initial position of each joint of the chain
        """
        # Choose computation method
        if self.computation_method == "default":
            return ik.inverse_kinematic(target, self.transformation_lambda, initial_position, fk_method=self.computation_method, model_type=self.config.model_type, representation=self.config.representation, regularization_parameter=regularization_parameter, max_iter=max_iterations, robot_parameters=self.config.parameters, bounds=self.config.bounds, first_active_joint=self.config.first_active_joint)
        else:
            return ik.inverse_kinematic(target, self.transformation_lambda, initial_position, fk_method=self.computation_method, bounds=self.config.bounds, first_active_joint=self.config.first_active_joint, regularization_parameter=regularization_parameter, max_iter=max_iterations)

    def plot(self, joints, ax, target=None, show=False):
        """Plots the Chain using Matplotlib

        :param list joints: The list of the positions of each joint
        :param matplotlib.axes.Axes ax: A matplotlib axes
        :param numpy.array target: An optional target
        :param bool show: Display the axe. Defaults to False
        """
        if ax is None:
            # If ax is not given, create one
            ax = plot_utils.init_3d_figure()
        plot_utils.plot_robot(self.config.parameters, joints, ax, representation=self.config.representation, model_type=self.config.model_type)
        plot_utils.plot_basis(self.config.parameters, ax, self.arm_length)

        # Plot the goal position
        if target is not None:
            plot_utils.plot_target(target, ax)
        if(show):
            plot_utils.show_figure()

    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements=["base_link"], last_link_vector=None, base_elements_type="joint"):
        """Creates a chain from an URDF file

       :param urdf_file: The path of the URDF file
       :type urdf_file: string
       :param base_elements: List of the links beginning the chain
       :type base_elements: list of strings
       :param last_link_vector: Optional : The translation vector of the tip.
       :type last_link_vector: numpy.array
        """
        links = URDF_utils.get_urdf_parameters(urdf_file)
        return cls(links)


def pinv():
    pass
