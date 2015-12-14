# coding= utf8
from . import URDF_utils


class Chain(object):
    def __init__(self, links, profile=''"", ik_solver=None, **kwargs):
        self.links = links

    def forward_kinematics(self, joints):
        pass

    def inverse_kinematics(self, end_effector, ik_solver=None, **kwargs):
        return ik_solver(end_effector, **kwargs)

    def plot(self, ax, joints):
        pass

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
