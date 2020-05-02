# coding= utf8
"""
.. module:: link
This module implements the Link class.
"""
import numpy as np
import sympy

# Ikpy imports
from ikpy.utils import geometry


class Link(object):
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

    def get_link_frame_matrix(self, actuator_parameters):
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
    translation_vector: numpy.array
        The translation vector. (In URDF, attribute "xyz" of the "origin" element)
    orientation: numpy.array
        The orientation of the link. (In URDF, attribute "rpy" of the "origin" element)
    rotation: numpy.array
        The rotation axis of the link. (In URDF, attribute "xyz" of the "axis" element)
    angle_representation: str
        Optional : The representation used by the angle. Currently supported representations : rpy. Defaults to rpy, the URDF standard.
    use_symbolic_matrix: bool
        whether the transformation matrix is stored as a Numpy array or as a Sympy symbolic matrix.

    Returns
    -------
    URDFLink
        The link object

    Example
    -------

    URDFlink()
    """

    def __init__(self, name, translation_vector, orientation, rotation=None, bounds=(None, None), angle_representation="rpy", use_symbolic_matrix=True):
        Link.__init__(self, name=name, bounds=bounds, length=np.linalg.norm(translation_vector))
        self.use_symbolic_matrix = use_symbolic_matrix
        self.translation_vector = np.array(translation_vector)
        self.orientation = np.array(orientation)
        if rotation is not None:
            self.rotation = np.array(rotation)
            self.has_rotation = True
        else:
            self.rotation = None
            self.has_rotation = False

        if use_symbolic_matrix:
            # Angle symbolique qui paramètre la rotation du joint en cours
            theta = sympy.symbols("theta")
            self.symbolic_transformation_matrix = self._apply_geometric_transformations(theta=theta, symbolic=self.use_symbolic_matrix)

    def __str__(self):
        return("""URDF Link {} :
    Bounds : {}
    Translation : {}
    Orientation : {}
    Rotation : {}""".format(self.name, self.bounds, self.translation_vector, self.orientation, self.rotation))

    def get_rotation_axis(self):
        if self.rotation is not None:
            return np.dot(
                geometry.homogeneous_translation_matrix(*self.translation_vector),
                np.dot(
                    geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.orientation)),
                    geometry.cartesian_to_homogeneous_vectors(self.rotation * self.axis_length)
                )
            )
        else:
            raise ValueError("This link doesn't provide a rotation")

    def get_link_frame_matrix(self, parameters):
        theta = parameters["theta"]
        if self.use_symbolic_matrix:
            frame_matrix = self.symbolic_transformation_matrix(theta)
        else:
            frame_matrix = self._apply_geometric_transformations(theta, symbolic=False)

        return frame_matrix

    def _apply_geometric_transformations(self, theta, symbolic):

        if symbolic:
            # Angle symbolique qui paramètre la rotation du joint en cours
            symbolic_frame_matrix = np.eye(4)

            # Apply translation matrix
            symbolic_frame_matrix = symbolic_frame_matrix * sympy.Matrix(geometry.homogeneous_translation_matrix(*self.translation_vector))

            # Apply orientation matrix
            symbolic_frame_matrix = symbolic_frame_matrix * geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.orientation))

            # Apply rotation matrix
            if self.rotation is not None:
                symbolic_frame_matrix = symbolic_frame_matrix * geometry.cartesian_to_homogeneous(geometry.symbolic_axis_rotation_matrix(self.rotation, theta), matrix_type="sympy")

            symbolic_frame_matrix = sympy.lambdify(theta, symbolic_frame_matrix, "numpy")

            return symbolic_frame_matrix

        else:
            # Init the transformation matrix
            frame_matrix = np.eye(4)

            # First, apply translation matrix
            frame_matrix = np.dot(frame_matrix, geometry.homogeneous_translation_matrix(*self.translation_vector))

            # Apply orientation
            frame_matrix = np.dot(frame_matrix, geometry.cartesian_to_homogeneous(geometry.rpy_matrix(*self.orientation)))

            # Apply rotation matrix
            if self.rotation is not None:
                frame_matrix = np.dot(frame_matrix, geometry.cartesian_to_homogeneous(geometry.axis_rotation_matrix(self.rotation, theta)))

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

    def get_link_frame_matrix(self, theta, a):
        """ Computes the homogeneous transformation matrix for this link. """
        ct = np.cos(theta + self.theta)
        st = np.sin(theta + self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        return np.matrix(((ct, -st * ca, st * sa, a * ct),
                          (st, ct * ca, -ct * sa, a * st),
                          (0, sa, ca, self.d),
                          (0, 0, 0, 1)))


class OriginLink(Link):
    """The link at the origin of the robot"""
    def __init__(self):
        Link.__init__(self, name="Base link", length=1)

    def get_rotation_axis(self):
        return [0, 0, 0, 1]

    def get_link_frame_matrix(self, theta):
        return np.eye(4)
