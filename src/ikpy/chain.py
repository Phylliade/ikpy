# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""
import numpy as np
import json
import os
from typing import List
import warnings

# IKPY imports
from .urdf import URDF
from . import inverse_kinematics as ik
from . import link as link_lib


class Chain:
    """The base Chain class

    Parameters
    ----------
    links: list[ikpy.link.Link]
        List of the links of the chain
    active_links_mask: list
        A list of boolean indicating that whether or not the corresponding link is active
    name: str
        The name of the Chain
    urdf_metadata
        Technical attribute
    """
    def __init__(self, links, active_links_mask=None, name="chain", urdf_metadata=None, **kwargs):
        self.name = name
        self.links = links
        self._length = sum([link.length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link.length == 0:
                link.axis_length = self.links[index - 1].axis_length
        # Optional argument
        self._urdf_metadata = urdf_metadata

        # If the active_links_mask is not given, set it to True for every link
        if active_links_mask is not None:
            if len(active_links_mask) != len(self.links):
                raise ValueError("Your active links mask length of {} is different from the number of your links, which is {}".format(len(active_links_mask), len(self.links)))
            self.active_links_mask = np.array(active_links_mask)

        else:
            self.active_links_mask = np.array([True] * len(links))

        # Always set the last link to True
        if self.active_links_mask[-1] is True:
            warnings.warn("active_link_mask[-1] is True, but it should be set to False. Overriding and setting to False")
            self.active_links_mask[-1] = False

        # Check that none of the active links are fixed
        for link_index, (link_active, link) in enumerate(zip(self.active_links_mask, self.links)):
            if link.joint_type == "fixed" and link_active:
                warnings.warn("Link {} (index: {}) is of type 'fixed' but set as active in the active_links_mask. In practice, this fixed link doesn't provide any transformation so is as it were inactive".format(link.name, link_index))

    def __repr__(self):
        return "Kinematic chain name={} links={} active_links={}".format(self.name, [link.name for link in self.links], self.active_links_mask)

    def __len__(self):
        return len(self.links)

    def forward_kinematics(self, joints: List, full_kinematics=False):
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

        for index, (link, joint_parameters) in enumerate(zip(self.links, joints)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            # FIXME: The casting to array is a loss of time
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix(joint_parameters)))
            if full_kinematics:
                # rotation_axe = np.dot(frame_matrix, link.rotation)
                frame_matrixes.append(frame_matrix)

        # Return the matrix, or matrixes
        if full_kinematics:
            return frame_matrixes
        else:
            return frame_matrix

    def inverse_kinematics(self, target_position=None, target_orientation=None, orientation_mode=None, **kwargs):
        """

        Parameters
        ----------
        target_position: np.ndarray
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
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        frame_target = np.eye(4)

        # Compute orientation
        if orientation_mode is not None:
            if orientation_mode == "X":
                frame_target[:3, 0] = target_orientation
            elif orientation_mode == "Y":
                frame_target[:3, 1] = target_orientation
            elif orientation_mode == "Z":
                frame_target[:3, 2] = target_orientation
            elif orientation_mode == "all":
                frame_target[:3, :3] = target_orientation
            else:
                raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

        # Compute target
        if target_position is None:
            no_position = True
        else:
            no_position = False
            frame_target[:3, 3] = target_position

        return self.inverse_kinematics_frame(target=frame_target, orientation_mode=orientation_mode, no_position=no_position, **kwargs)

    def inverse_kinematics_frame(self, target, initial_position=None, **kwargs):
        """Computes the inverse kinematic on the specified target

        Parameters
        ----------
        target: numpy.array
            The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        initial_position: numpy.array
            Optional : the initial position of each joint of the chain. Defaults to 0 for each joint
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

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
        plot.plot_chain(self, joints, ax, name=self.name)

        # Plot the goal position
        if target is not None:
            plot.plot_target(target, ax)
        if show:
            plot.show_figure()

    @property
    def _json_path(self):
        """Path where the JSON is expected to be"""
        return os.path.dirname(self._urdf_metadata["urdf_file"]) + "/" + self.name + ".json"

    @classmethod
    def from_json_file(cls, json_file):
        """
        Load a chain serialized in the JSON format.
        This is basically a URDF file with some metadata

        Parameters
        ----------
        json_file: str
            Path to the json file serializing the robot

        Returns
        -------

        """
        with open(json_file, "r") as fd:
            chain_config = json.load(fd)

        # The path where is stored the URDF file
        chain_basedir = os.path.dirname(json_file)

        # Get the different attributes
        urdf_file = chain_config["urdf_file"]
        elements = chain_config["elements"]
        if elements == "":
            elements = None
        active_links_mask = chain_config["active_links_mask"]
        if active_links_mask == "":
            active_links_mask = None
        last_link_vector = chain_config["last_link_vector"]
        if last_link_vector == "":
            last_link_vector = None

        return cls.from_urdf_file(
            urdf_file=chain_basedir + "/" + urdf_file,
            base_elements=elements,
            active_links_mask=active_links_mask,
            last_link_vector=last_link_vector,
            name=chain_config["name"]
        )

    def to_json_file(self, force=False):
        """
        Serialize the chain into a json that will be saved next to the original URDF, with the name of the chain

        Parameters
        ----------
        force: bool
            Overwrite if existing

        Returns
        -------
        str:
            Path of the exported JSON

        """
        chain_dict = {
            "elements": self._urdf_metadata["base_elements"],
            "urdf_file": os.path.basename(self._urdf_metadata["urdf_file"]),
            "active_links_mask": [bool(x) for x in self.active_links_mask],
            "last_link_vector": self._urdf_metadata["last_link_vector"],
            "name": self.name,
            "version": "v1"
        }

        if os.path.exists(self._json_path) and not force:
            raise OSError("File {} exists".format(self._json_path))

        # And create the json file
        with open(self._json_path, "w") as fd:
            json.dump(chain_dict, fd, indent=2)

        return self._json_path

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
        urdf_metadata = {
            "base_elements": base_elements,
            "urdf_file": urdf_file,
            "last_link_vector": last_link_vector
        }

        # TODO: Remove this default and force the user to provide something
        if base_elements is None:
            base_elements = ["base_link"]

        links = URDF.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type, symbolic=symbolic)
        # Add an origin link at the beginning
        chain = cls([link_lib.OriginLink()] + links, active_links_mask=active_links_mask, name=name, urdf_metadata=urdf_metadata)

        # Save some useful metadata
        # FIXME: We have attributes specific to objects created in this style, not great...
        chain.urdf_file = urdf_file
        chain.base_elements = base_elements

        return chain

    def active_to_full(self, active_joints, initial_position):
        full_joints = np.array(initial_position, copy=True, dtype=np.float)
        np.place(full_joints, self.active_links_mask, active_joints)
        return full_joints

    def active_from_full(self, joints):
        return np.compress(self.active_links_mask, joints, axis=0)

    @classmethod
    def concat(cls, chain1, chain2):
        """Concatenate two chains"""
        return cls(links=chain1.links + chain2.links, active_links_mask=chain1.active_links_mask + chain2.active_links_mask)
