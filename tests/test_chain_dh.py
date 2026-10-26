from ikpy.chain import Chain
from ikpy.link import DHLink, OriginLink
import pytest
from ikpy.utils import plot

import matplotlib.pyplot as plt

import numpy as np
from math import pi


class UR10():
    def __init__(self):
        self.robot_name = 'UR10'
        self.home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
        self.dh_params = np.array([
            [0.1273, 0., pi / 2, 0.],
            [0., -0.612, 0, 0.],
            [0., -0.5723, 0, 0.],
            [0.163941, 0., pi / 2, 0.],
            [0.1147, 0., -pi / 2, 0.],
            [0.0922, 0., 0, 0.]])

        self.joint_limits = [
            (-360, 360),
            (-360, 360),
            (-360, 360),
            (-360, 360),
            (-360, 360),
            (-360, 360)]


def create_dh_robot(robot):
    # Create a list of links for the robot
    links = []
    for i, dh in enumerate(robot.dh_params):
        link = DHLink(d=dh[0], a=dh[1], alpha=dh[2], theta=dh[3], length=abs(dh[1]))
        link.bounds = robot.joint_limits[i]
        links.append(link)

    # Create a chain using the robot links
    chain = Chain(links, name=robot.robot_name)
    return chain


def test_dh_chain():
    robot = UR10()
    chain = create_dh_robot(robot)
    frame = [[1.112918581, -0.209413742, 0.19382176], [0.0, 1.0, 0.0]]
    target_position, target_orientation = frame
    joint_angles = chain.inverse_kinematics(target_position=target_position, target_orientation=target_orientation, orientation_mode="Z")

    print(joint_angles)

    fig, ax = plot.init_3d_figure()
    chain.plot(robot.home_config, ax)
    chain.plot(joint_angles, ax)
    plt.savefig("out/UR10.png")


def test_dh_link_name_and_repr():
    link = DHLink(name="shoulder", d=0.1, a=0.2, alpha=pi / 2)
    assert link.name == "shoulder"
    assert "shoulder" in repr(link)
    assert link.convention == "classic"


def test_dh_unknown_convention_raises():
    with pytest.raises(ValueError):
        DHLink(name="j", convention="craig")


def test_modified_dh_matches_explicit_product():
    """Modified DH frame is Rx(alpha) Tx(a) Rz(theta) Tz(d)"""
    from ikpy.utils import geometry
    d, a, alpha, theta = 0.3, 0.5, pi / 3, 0.7
    link = DHLink(name="j", d=d, a=a, alpha=alpha, convention="modified")
    expected = (
        geometry.cartesian_to_homogeneous(geometry.rx_matrix(alpha))
        @ geometry.homogeneous_translation_matrix(a, 0, 0)
        @ geometry.cartesian_to_homogeneous(geometry.rz_matrix(theta))
        @ geometry.homogeneous_translation_matrix(0, 0, d)
    )
    np.testing.assert_allclose(link.get_link_frame_matrix(theta), expected, atol=1e-12)


def test_classic_dh_matches_explicit_product():
    """Classic DH frame is Rz(theta) Tz(d) Tx(a) Rx(alpha)"""
    from ikpy.utils import geometry
    d, a, alpha, theta = 0.3, 0.5, pi / 3, 0.7
    link = DHLink(name="j", d=d, a=a, alpha=alpha, convention="classic")
    expected = (
        geometry.cartesian_to_homogeneous(geometry.rz_matrix(theta))
        @ geometry.homogeneous_translation_matrix(0, 0, d)
        @ geometry.homogeneous_translation_matrix(a, 0, 0)
        @ geometry.cartesian_to_homogeneous(geometry.rx_matrix(alpha))
    )
    np.testing.assert_allclose(link.get_link_frame_matrix(theta), expected, atol=1e-12)


def test_modified_dh_chain_forward_kinematics():
    """A planar 2R arm in modified DH: a_0 = 0, a_1 = L1, tool at L2 along x"""
    L1, L2 = 1.0, 0.5
    links = [
        DHLink(name="j1", a=0, alpha=0, convention="modified"),
        DHLink(name="j2", a=L1, alpha=0, convention="modified"),
        DHLink(name="tool", a=L2, alpha=0, convention="modified", bounds=(0, 0)),
    ]
    my_chain = Chain([OriginLink()] + links, active_links_mask=[False, True, True, False])
    q1, q2 = 0.4, -0.9
    frame = my_chain.forward_kinematics([0, q1, q2, 0])
    expected = [L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L1 * np.sin(q1) + L2 * np.sin(q1 + q2), 0]
    np.testing.assert_allclose(frame[:3, 3], expected, atol=1e-12)
