# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""
import numpy as np

# IKPY imports
from ikpy.utils import URDF
from . import inverse_kinematics as ik
from . import link as link_lib


class Chain(object):
    """The base Chain class

    Parameters
    ----------
    links: list[ikpy.link.Link]
        List of the links of the chain
    active_links_mask: list
        A list of boolean indicating that whether or not the corresponding link is active
    name: str
        The name of the Chain
    """
    def __init__(self, links, active_links_mask=None, name="chain", profile=''"", **kwargs):
        self.name = name
        self.links = links
        self._length = sum([link.length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link.length == 0:
                link.axis_length = self.links[index - 1].axis_length

        # If the active_links_mask is not given, set it to True for every link
        if active_links_mask is not None:
            if len(active_links_mask) != len(self.links):
                raise ValueError("Your active links mask length of {} is different from the number of your links, which is {}".format(len(active_links_mask), len(self.links)))
            self.active_links_mask = np.array(active_links_mask)
            # Always set the last link to True
            self.active_links_mask[-1] = False
        else:
            self.active_links_mask = np.array([True] * len(links))

    def __repr__(self):
        return "Kinematic chain name={} links={} active_links={}".format(self.name, self.links, self.active_links_mask)

    def forward_kinematics(self, joints, full_kinematics=False):
        """Returns the transformation matrix of the forward kinematics

        Parameters
        ----------
        joints: list
            The list of the positions of each joint. Note : Inactive joints must be in the list.
        full_kinematics: bool
            Return the transformation matrices of each joint

        Returns
        -------
        frame_matrix:
            The transformation matrix
        """
        frame_matrix = np.eye(4)

        if full_kinematics:
            frame_matrixes = []

        if len(self.links) != len(joints):
            raise ValueError("Your joints vector length is {} but you have {} links".format(len(joints), len(self.links)))

        for index, (link, joint_angle) in enumerate(zip(self.links, joints)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix({"theta": joint_angle})))
            if full_kinematics:
                # rotation_axe = np.dot(frame_matrix, link.rotation)
                frame_matrixes.append(frame_matrix)

        # Return the matrix, or matrixes
        if full_kinematics:
            return frame_matrixes
        else:
            return frame_matrix

    def inverse_kinematics(self, target_vector=None, target_orientation=None, orientation_mode=None, **kwargs):
        """

        Parameters
        ----------
        target_vector: np.ndarray
            Vector of shape (3,): the target point
        target_orientation: np.ndarray
            Vector of shape (3,): the target orientation
        orientation_mode: str
            Orientation to target. Choices:
            * None: No orientation
            * "X": Target the X axis
            * "Y": Target the Y axis
            * "Z": Target the Z axis
            * "all": Target the entire frame (e.g. the three axes) (not currently supported)
        kwargs

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        frame_target = np.eye(4)
        frame_target[:3, 0] = target_orientation
        frame_target[:3, 3] = target_vector

        return self.inverse_kinematics_frame(target=frame_target, orientation_mode=orientation_mode, **kwargs)

    def inverse_kinematics_frame(self, target, initial_position=None, **kwargs):
        """Computes the inverse kinematic on the specified target

        Parameters
        ----------
        target: numpy.array
            The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        initial_position: numpy.array
            Optional : the initial position of each joint of the chain. Defaults to 0 for each joint

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")

        if initial_position is None:
            initial_position = [0] * len(self.links)

        return ik.inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)

    def plot(self, joints, ax, target=None, show=False):
        """Plots the Chain using Matplotlib

        Parameters
        ----------
        joints: list
            The list of the positions of each joint
        ax: matplotlib.axes.Axes
            A matplotlib axes
        target: numpy.array
            An optional target
        show: bool
            Display the axe. Defaults to False
        """
        from ikpy.utils import plot

        if ax is None:
            # If ax is not given, create one
            ax = plot.init_3d_figure()
        plot.plot_chain(self, joints, ax, length=self._length)
        plot.plot_basis(ax, self._length)

        # Plot the goal position
        if target is not None:
            plot.plot_target(target, ax)
        if show:
            plot.show_figure()

    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements=None, last_link_vector=None, base_element_type="link", active_links_mask=None, name="chain", symbolic=True):
        """Creates a chain from an URDF file

        Parameters
        ----------
        urdf_file: str
            The path of the URDF file
        base_elements: list of strings
            List of the links beginning the chain
        last_link_vector: numpy.array
            Optional : The translation vector of the tip.
        name: str
            The name of the Chain
        base_element_type: str
        active_links_mask: list[bool]
        symbolic: bool
            Use symoblic computations


        Note
        ----
        IKPY works with links, whereras URDF works with joints and links. The mapping is currently misleading:

        * URDF joints = IKPY links
        * URDF links are not used by IKPY. They are thrown away when parsing
        """
        # FIXME: Rename links to joints, to be coherent with URDF?
        if base_elements is None:
            base_elements = ["base_link"]

        links = URDF.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type, symbolic=symbolic)
        # Add an origin link at the beginning
        return cls([link_lib.OriginLink()] + links, active_links_mask=active_links_mask, name=name)

    def active_to_full(self, active_joints, initial_position):
        full_joints = np.array(initial_position, copy=True, dtype=np.float)
        np.place(full_joints, self.active_links_mask, active_joints)
        return full_joints

    def active_from_full(self, joints):
        return np.compress(self.active_links_mask, joints, axis=0)

    @classmethod
    def concat(cls, chain1, chain2):
        return cls(links=chain1.links + chain2.links, active_links_mask=chain1.active_links_mask + chain2.active_links_mask)
