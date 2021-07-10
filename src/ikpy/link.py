# coding= utf8
"""
.. module:: link
This module implements the Link class.
"""
import numpy as np
import sympy

# Ikpy imports
from ikpy.utils import geometry
from typing import Optional


class Link:
    """
    Base Link class.

    Parameters
    ----------
    name: string
        The name of the link
    bounds: tuple
        Optional : The bounds of the link. Defaults to None

    Attributes
    ----------
    has_rotation: bool
        Whether the link provides a rotation
    length: float
        Length of the link
    """

    def __init__(self, name, length, bounds=(None, None), is_final=False):
        self.bounds = bounds
        self.name = name
        self.length = length
        self.axis_length = length
        self.is_final = is_final
        self.has_rotation = False
        self.joint_type = None

    def __repr__(self):
        return "Link name={} bounds={}".format(self.name, self.bounds)

    def get_rotation_axis(self):
        """

        Returns
        -------
        coords:
            coordinates of the rotation axis in the frame of the joint

        """
        # Defaults to None
        raise ValueError("This Link doesn't have a rotation axis")

    def get_link_frame_matrix(self, actuator_parameters: dict):
        """
        Return the frame matrix corresponding to the link, parameterized with theta

        Parameters
        ----------
        actuator_parameters: dict
            Values for the actuator movements

        Note
        ----
        Theta works for rotations, and for other one-dimensional actuators (ex: prismatic joints), even if the name can be misleading
        """
        raise NotImplementedError


class URDFLink(Link):
    """Link in URDF representation.

    Parameters
    ----------
    name: str
        The name of the link
    bounds: tuple
        Optional : The bounds of the link. Defaults to None
    origin_translation: numpy.array
        The translation vector. (In URDF, attribute "xyz" of the "origin" element)
    origin_orientation: numpy.array
        The orientation of the link. (In URDF, attribute "rpy" of the "origin" element)
    rotation: numpy.array
        The rotation axis of the link. (In URDF, attribute "xyz" of the "axis" element)
    angle_representation: str
        Optional : The representation used by the angle. Currently supported representations : rpy. Defaults to rpy, the URDF standard.
    use_symbolic_matrix: bool
        whether the transformation matrix is stored as a Numpy array or as a Sympy symbolic matrix.
    joint_type: str
        The URDF "type" attribute of the joint. Only support for revolute and prismatic joint for the moment

    Returns
    -------
    URDFLink
        The link object

    Example
    -------

    URDFlink()
    """

    def __init__(self,
                 name: str,
                 origin_translation: np.ndarray,
                 origin_orientation: np.ndarray,
                 rotation: Optional[np.ndarray] = None,
                 translation: Optional[np.ndarray] = None,
                 bounds=(None, None),
                 angle_representation="rpy",
                 use_symbolic_matrix=True,
                 joint_type: str = "revolute"
                 ):
        Link.__init__(self, name=name, bounds=bounds, length=np.linalg.norm(origin_translation))
        self.use_symbolic_matrix = use_symbolic_matrix
        self.origin_translation = np.array(origin_translation)
        self.origin_orientation = np.array(origin_orientation)
        if rotation is not None:
            self.rotation = np.array(rotation)
            self.has_rotation = True
        else:
            self.rotation = None
            self.has_rotation = False
        # FIXME: We cast to np array, but the type already asks for a np array
        if translation is not None:
            self.has_translation = True
            self.translation = np.array(translation)
        else:
            self.has_translation = False
            self.translation = None

        # Check that the given joint type matches the given parameters
        if joint_type == "revolute":
            if not(self.has_rotation and not self.has_translation):
                raise ValueError("Joint type is 'revolute' but rotation axis = {} and translation axis = {}".format(self.has_rotation, self.has_translation))
        elif joint_type == "prismatic":
            if not(not self.has_rotation and self.has_translation):
                raise ValueError("Joint type is 'prismatic' but rotation axis = {} and translation axis = {}".format(self.has_rotation, self.has_translation))
        elif joint_type == "fixed":
            if not(not self.has_rotation and not self.has_translation):
                raise ValueError("Joint type is 'fixed' but rotation axis = {} and translation axis = {}".format(self.has_rotation, self.has_translation))

        else:
            raise ValueError("Unknown joint type: {}".format(joint_type))
        self.joint_type = joint_type

        if use_symbolic_matrix:
            # Angle symbolique qui paramètre la rotation du joint en cours
            theta = sympy.symbols("theta")
            mu = sympy.symbols("mu")
            self.symbolic_transformation_matrix = self._apply_geometric_transformations(theta=theta, mu=mu, symbolic=True)

    def __repr__(self):
        return("""URDF Link {} :
    Type : {}
    Bounds : {}
    Origin Translation : {}
    Origin Orientation : {}
    Rotation : {}
    Translation: {}""".format(self.name, self.joint_type, self.bounds, self.origin_translation, self.origin_orientation, self.rotation, self.translation))

    def get_rotation_axis(self):
        if self.rotation is not None:
            return np.dot(
                geometry.homogeneous_translation_matrix(*self.origin_translation),
                np.dot(
                    geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.origin_orientation)),
                    geometry.cartesian_to_homogeneous_vectors(self.rotation * self.axis_length)
                )
            )
        else:
            raise ValueError("This link doesn't provide a rotation")

    def get_translation_axis(self):
        if self.has_translation:
            return np.dot(
                geometry.homogeneous_translation_matrix(*self.origin_translation),
                np.dot(
                    geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.origin_orientation)),
                    geometry.cartesian_to_homogeneous_vectors(self.translation * self.axis_length)
                )
            )
        else:
            raise ValueError("This link doesn't provide a translation")

    def get_link_frame_matrix(self, parameters):
        if self.joint_type == "revolute":
            theta = parameters
            mu = None
        elif self.joint_type == "prismatic":
            theta = None
            mu = parameters
        elif self.joint_type == "fixed":
            theta = None
            mu = None
        else:
            raise ValueError
        if self.use_symbolic_matrix:
            frame_matrix = self.symbolic_transformation_matrix(theta, mu)
        else:
            frame_matrix = self._apply_geometric_transformations(theta, mu, symbolic=False)

        return frame_matrix

    def _apply_geometric_transformations(self, theta, mu, symbolic):

        if symbolic:
            # Angle symbolique qui paramètre la rotation du joint en cours
            symbolic_frame_matrix = np.eye(4)

            # Apply translation matrix
            symbolic_frame_matrix = symbolic_frame_matrix * sympy.Matrix(geometry.homogeneous_translation_matrix(*self.origin_translation))

            # Apply orientation matrix
            symbolic_frame_matrix = symbolic_frame_matrix * geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.origin_orientation))

            # Apply rotation matrix
            if self.has_rotation:
                symbolic_frame_matrix = symbolic_frame_matrix * geometry.cartesian_to_homogeneous(geometry.symbolic_axis_rotation_matrix(self.rotation, theta), matrix_type="sympy")

            # Apply translation
            if self.has_translation:
                translation_vector = self.translation * mu
                symbolic_frame_matrix = symbolic_frame_matrix * geometry.get_symbolic_translation_matrix(translation_vector)

            symbolic_frame_matrix = sympy.lambdify([theta, mu], symbolic_frame_matrix, "numpy")

            return symbolic_frame_matrix

        else:
            # Init the transformation matrix
            frame_matrix = np.eye(4)

            # First, apply translation matrix
            frame_matrix = np.dot(frame_matrix, geometry.homogeneous_translation_matrix(*self.origin_translation))

            # Apply orientation
            frame_matrix = np.dot(frame_matrix, geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.origin_orientation)))

            # Apply rotation matrix
            if self.has_rotation:
                frame_matrix = np.dot(frame_matrix, geometry.cartesian_to_homogeneous(geometry.axis_rotation_matrix(self.rotation, theta)))

            # Apply translation
            if self.has_translation:
                translation_vector = self.translation * mu
                frame_matrix = np.dot(frame_matrix, geometry.get_translation_matrix(translation_vector))

            return frame_matrix


class DHLink(Link):
    """Link in Denavit-Hartenberg representation.

    Parameters
    ----------
    name: str
        The name of the link
    bounds: tuple
        Optional : The bounds of the link. Defaults to None
    d: float
        offset along previous z to the common normal
    a: float
        offset along previous   to the common normal
    use_symbolic_matrix: bool
        whether the transformation matrix is stored as Numpy array or as a Sympy symbolic matrix.

    Returns
    -------
    DHLink:
        The link object
    """

    def __init__(self, name, d=0, a=0, bounds=None, use_symbolic_matrix=True):
        Link.__init__(self, use_symbolic_matrix)
        self.d = d
        self.a = a

    def get_link_frame_matrix(self, parameters):
        """ Computes the homogeneous transformation matrix for this link. """
        theta = parameters
        ct = np.cos(theta + self.theta)
        st = np.sin(theta + self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        return np.matrix(((ct, -st * ca, st * sa, self.a * ct),
                          (st, ct * ca, -ct * sa, self.a * st),
                          (0, sa, ca, self.d),
                          (0, 0, 0, 1)))


class OriginLink(Link):
    """The link at the origin of the robot"""
    def __init__(self):
        Link.__init__(self, name="Base link", length=1)
        self.has_rotation = False
        self.has_translation = False
        self.joint_type = "fixed"

    def get_rotation_axis(self):
        return [0, 0, 0, 1]

    def get_link_frame_matrix(self, theta):
        return np.eye(4)
