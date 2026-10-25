import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.link import URDFLink
from ikpy.urdf import URDF


CONTINUOUS_URDF = """<?xml version="1.0" ?>
<robot name="continuous_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <joint name="joint1" type="continuous">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
</robot>
"""

BASE_ELEMENTS = ["base_link", "joint1", "link1", "joint2", "link2"]


def _write_urdf(tmp_path, content=CONTINUOUS_URDF):
    urdf_file = tmp_path / "robot.URDF"
    urdf_file.write_text(content)
    return str(urdf_file)


def test_continuous_joint_is_parsed(tmp_path):
    """A continuous joint must be parsed instead of raising 'Unknown joint type'"""
    links = URDF.get_urdf_parameters(
        _write_urdf(tmp_path), base_elements=BASE_ELEMENTS)

    continuous = [link for link in links if link.name == "joint1"]
    assert len(continuous) == 1
    assert continuous[0].joint_type == "continuous"
    np.testing.assert_almost_equal(continuous[0].rotation, [0, 0, 1])
    # A continuous joint has no limits
    assert continuous[0].bounds == (-np.inf, np.inf)


def test_continuous_joint_ignores_limit_element(tmp_path):
    """Even when a limit element is present, a continuous joint stays unbounded,
    while its revolute sibling keeps the limits from the file"""
    content = CONTINUOUS_URDF.replace(
        '<axis xyz="0 0 1"/>',
        '<axis xyz="0 0 1"/><limit lower="-0.1" upper="0.1"/>')
    links = URDF.get_urdf_parameters(
        _write_urdf(tmp_path, content), base_elements=BASE_ELEMENTS)

    by_name = {link.name: link for link in links}
    assert by_name["joint1"].bounds == (-np.inf, np.inf)
    assert by_name["joint2"].bounds == (-1.5, 1.5)


def test_continuous_link_rotates_like_revolute():
    """The frame matrix of a continuous link is the revolute one, for any angle,
    including angles beyond +/- pi"""
    kwargs = dict(
        name="joint",
        origin_translation=[0, 0, 0.5],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        use_symbolic_matrix=False,
    )
    continuous = URDFLink(joint_type="continuous", **kwargs)
    revolute = URDFLink(joint_type="revolute", **kwargs)

    for theta in [0.0, 0.3, -1.2, np.pi, 4.0, -7.5]:
        np.testing.assert_almost_equal(
            continuous.get_link_frame_matrix(theta),
            revolute.get_link_frame_matrix(theta))


def test_continuous_link_rejects_translation_axis():
    """A continuous joint is a rotation: passing a translation axis must fail,
    exactly like for a revolute joint"""
    with pytest.raises(ValueError):
        URDFLink(
            name="joint",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            translation=[0, 0, 1],
            use_symbolic_matrix=False,
            joint_type="continuous",
        )


def test_continuous_chain_forward_kinematics(tmp_path):
    """FK of a chain mixing continuous and revolute joints matches the same
    chain written with only revolute joints"""
    continuous_chain = chain.Chain.from_urdf_file(
        _write_urdf(tmp_path),
        base_elements=BASE_ELEMENTS,
        last_link_vector=[0.5, 0, 0])
    revolute_chain = chain.Chain.from_urdf_file(
        _write_urdf(tmp_path, CONTINUOUS_URDF.replace("continuous", "revolute")),
        base_elements=BASE_ELEMENTS,
        last_link_vector=[0.5, 0, 0])

    joints = [0, 2.5, 0.4, 0]
    joints = joints + [0] * (len(continuous_chain.links) - len(joints))
    np.testing.assert_almost_equal(
        continuous_chain.forward_kinematics(joints),
        revolute_chain.forward_kinematics(joints))


def test_continuous_chain_inverse_kinematics(tmp_path):
    """IK round-trip on a chain with a continuous joint"""
    continuous_chain = chain.Chain.from_urdf_file(
        _write_urdf(tmp_path),
        base_elements=BASE_ELEMENTS,
        last_link_vector=[0.5, 0, 0])

    initial = [0] * len(continuous_chain.links)
    initial[1] = 0.7
    initial[2] = 0.3
    fk = continuous_chain.forward_kinematics(initial)
    ik = continuous_chain.inverse_kinematics_frame(fk)
    np.testing.assert_almost_equal(
        fk, continuous_chain.forward_kinematics(ik), decimal=3)


def test_unknown_joint_type_still_raises(tmp_path):
    """Joint types outside the URDF spec support list keep raising"""
    content = CONTINUOUS_URDF.replace("continuous", "planar")
    with pytest.raises(ValueError, match="Unknown joint type"):
        URDF.get_urdf_parameters(
            _write_urdf(tmp_path, content), base_elements=BASE_ELEMENTS)
