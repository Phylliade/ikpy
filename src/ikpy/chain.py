# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""

from . import URDF_utils
from . import inverse_kinematics as ik
from . import plot_utils
import numpy as np
from . import link as link_lib


class Chain(object):
    """The base Chain class

    :param list links: List of the links of the chain
    :param list active_links: The list of the positions of the active links
    """
    def __init__(self, links, active_links=0, profile=''"", **kwargs):
        self.links = [link_lib.OriginLink()] + links
        self._length = sum([link._length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link._length == 0:
                link._axis_length = self.links[index - 1]._axis_length
        # Temporary
        self.first_active_joint = active_links

    def forward_kinematics(self, joints, full_kinematics=False):
        """Returns the transformation matrix of the forward kinematics

        :param list joints: The list of the positions of each joint. Note : Inactive joints must be in the list.
        :param bool full_kinematics: Return the transorfmation matrixes of each joint
        :returns: The transformation matrix
        """
        frame_matrix = np.eye(4)

        if full_kinematics:
            frame_matrixes = []

        for index, (link, joint_angle) in enumerate(zip(self.links, joints)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_transformation_matrix(joint_angle)))
            if full_kinematics:
                # rotation_axe = np.dot(frame_matrix, link.rotation)
                frame_matrixes.append(frame_matrix)

        # Return the matrix, or matrixes
        if full_kinematics:
            return frame_matrixes
        else:
            return frame_matrix

    def inverse_kinematics(self, target, initial_position, **kwargs):
        """Computes the inverse kinematic on the specified target

        :param numpy.array target: The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        :param numpy.array initial_position: the initial position of each joint of the chain
        :returns: The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")

        return ik.inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)

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
        plot_utils.plot_chain(self, joints, ax)
        plot_utils.plot_basis(ax, self._length)

        # Plot the goal position
        if target is not None:
            plot_utils.plot_target(target, ax)
        if(show):
            plot_utils.show_figure()

    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements=["base_link"], last_link_vector=None, base_elements_type="joint", active_links=0):
        """Creates a chain from an URDF file

       :param urdf_file: The path of the URDF file
       :type urdf_file: string
       :param base_elements: List of the links beginning the chain
       :type base_elements: list of strings
       :param last_link_vector: Optional : The translation vector of the tip.
       :type last_link_vector: numpy.array
       :param list active_links: The active links
        """
        links = URDF_utils.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_elements_type=base_elements_type)
        return cls(links, active_links=active_links)


def pinv():
    pass
