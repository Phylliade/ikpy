import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.urdf import URDF


def _urdf(joint_type, axis_element):
    return """<?xml version="1.0" ?>
<robot name="default_axis_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="{joint_type}">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    {axis_element}
  </joint>
</robot>
""".format(joint_type=joint_type, axis_element=axis_element)


BASE_ELEMENTS = ["base_link", "joint1", "link1"]


def _write_urdf(tmp_path, content):
    urdf_file = tmp_path / "robot.URDF"
    urdf_file.write_text(content)
    return str(urdf_file)


def _single_link(tmp_path, content):
    links = URDF.get_urdf_parameters(
        _write_urdf(tmp_path, content), base_elements=BASE_ELEMENTS)
    assert len(links) == 1
    return links[0]


def test_revolute_without_axis_element_defaults_to_x(tmp_path):
    """A revolute joint with no <axis> element uses the URDF default (1, 0, 0)
    instead of failing in URDFLink validation"""
    link = _single_link(tmp_path, _urdf("revolute", ""))
    assert link.joint_type == "revolute"
    np.testing.assert_almost_equal(link.rotation, [1, 0, 0])


def test_prismatic_without_axis_element_defaults_to_x(tmp_path):
    """Same default for a prismatic joint"""
    link = _single_link(tmp_path, _urdf("prismatic", ""))
    assert link.joint_type == "prismatic"
    np.testing.assert_almost_equal(link.translation, [1, 0, 0])


def test_axis_element_without_xyz_defaults_to_x(tmp_path):
    """An <axis> element without its xyz attribute also defaults to (1, 0, 0)"""
    link = _single_link(tmp_path, _urdf("revolute", "<axis/>"))
    np.testing.assert_almost_equal(link.rotation, [1, 0, 0])


def test_fixed_without_axis_element_still_parses(tmp_path):
    """Fixed joints never needed an axis: no regression"""
    link = _single_link(tmp_path, _urdf("fixed", ""))
    assert link.joint_type == "fixed"
    assert link.rotation is None
    assert link.translation is None


def test_explicit_axis_is_unchanged(tmp_path):
    """An explicit axis keeps being used as before"""
    link = _single_link(
        tmp_path, _urdf("revolute", '<axis xyz="0 0 1"/>'))
    np.testing.assert_almost_equal(link.rotation, [0, 0, 1])


def test_fk_matches_explicit_x_axis(tmp_path):
    """A chain relying on the default axis produces the same FK as the same
    chain with an explicit <axis xyz="1 0 0"/>"""
    for joint_type in ("revolute", "prismatic"):
        implicit = chain.Chain.from_urdf_file(
            _write_urdf(tmp_path, _urdf(joint_type, "")),
            base_elements=BASE_ELEMENTS,
            last_link_vector=[0.5, 0, 0])
        explicit = chain.Chain.from_urdf_file(
            _write_urdf(tmp_path, _urdf(joint_type, '<axis xyz="1 0 0"/>')),
            base_elements=BASE_ELEMENTS,
            last_link_vector=[0.5, 0, 0])

        joints = [0] * len(implicit.links)
        joints[1] = 0.6
        np.testing.assert_almost_equal(
            implicit.forward_kinematics(joints),
            explicit.forward_kinematics(joints))


def test_continuous_without_axis_element_defaults_to_x(tmp_path):
    """A continuous joint follows the same default as a revolute joint"""
    link = _single_link(tmp_path, _urdf("continuous", ""))
    assert link.joint_type == "continuous"
    np.testing.assert_almost_equal(link.rotation, [1, 0, 0])


def test_axis_element_without_xyz_warns(tmp_path):
    """The xyz attribute is required by the URDF spec: defaulting it is reported"""
    with pytest.warns(UserWarning, match="without an 'xyz' attribute"):
        _single_link(tmp_path, _urdf("revolute", "<axis/>"))
