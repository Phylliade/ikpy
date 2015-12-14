# coding= utf8
import numpy as np


class Link(object):
    """Base Link class.

    :param name: The name of the link
    :type name: string
    :param bounds: Optional : The bounds of the link. Defaults to None
    :type bounds: tuple
    :param use_symbolic_matrix: wether the transformation matrix is stored as Numpy array or as a Sympy symbolic matrix.
    :type use_symbolic_matrix: bool
    """

    def __init__(self, name, bounds=None, use_symbolic_matrix=True):
        self.use_symbolic_matrix = use_symbolic_matrix

    def get_transformartion_matrix(theta):
        raise NotImplementedError


class URDFLink(Link):
    """Link in URDF representation.

    :param name: The name of the link
    :type name: string
    :param bounds: Optional : The bounds of the link. Defaults to None
    :type bounds: tuple
   :param translation_vector: The translation vector. (In URDF, attribute "xyz" of the "origin" element)
   :type translation_vector: numpy.array
   :param orientation: The orientation of the link. (In URDF, attribute "rpy" of the "origin" element)
   :type orientation: numpy.array
   :param rotation: The rotation axis of the link. (In URDF, attribute "xyz" of the "axis" element)
   :type rotation: numpy.array
   :param angle_representation: Optionnal : The representation used by the angle. Currently supported representations : rpy. Defaults to rpy, the URDF standard.
   :type angle_representation: string
   :param use_symbolic_matrix: wether the transformation matrix is stored as a Numpy array or as a Sympy symbolic matrix.
   :type use_symbolic_matrix: bool
   :returns: The link object
   :rtype: URDFLink
   :Example:

   URDFlink()
    """

    def __init__(self, name, translation_vector, orientation, rotation, bounds=None, angle_representation="rpy", use_symbolic_matrix=True):
        Link.__init__(self, name, use_symbolic_matrix)


class DHLink(Link):
    """Link in Denavit-Hartenberg representation.

   :param name: The name of the link
   :type name: string
   :param bounds: Optional : The bounds of the link. Defaults to None
   :type bounds: tuple
   :param float d: offset along previous z to the common normal
   :param float a: offset along previous   to the common normal
   :param use_symbolic_matrix: wether the transformation matrix is stored as Numpy array or as a Sympy symbolic matrix.
   :type use_symbolic_matrix: bool
   :returns: The link object
   :rtype: DHLink
    """

    def __init__(self, name, d=0, a=0, bounds=None, use_symbolic_matrix=True):
        Link.__init__(self, use_symbolic_matrix)

    def get_transformation_matrix(self, theta, a):
        """ Computes the homogeneous transformation matrix for this link. """
        ct = np.cos(theta + self.theta)
        st = np.sin(theta + self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)

        return np.matrix(((ct, -st * ca, st * sa, a * ct),
                          (st, ct * ca, -ct * sa, a * st),
                          (0, sa, ca, self.d),
                          (0, 0, 0, 1)))
