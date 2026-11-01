"""URDF conventions aligned on ROS (urdfdom and KDL)"""
import numpy as np
import pytest

# ikpy imports
from ikpy import chain


PANDA = "../resources/franka_panda/panda.urdf"


def _urdf(joint_type, axis, extra=""):
    return """<?xml version="1.0" ?>
<robot name="kdl_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="{joint_type}">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    {axis}
    {extra}
  </joint>
</robot>
""".format(joint_type=joint_type, axis=axis, extra=extra)


def _write(tmp_path, content, name="robot.urdf"):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_non_unit_axis_is_normalized(tmp_path):
    with pytest.warns(UserWarning, match="non-unit axis"):
        scaled = chain.Chain.from_urdf_file(_write(tmp_path, _urdf("revolute", '<axis xyz="0 0 2"/>')), symbolic=False)
    unit = chain.Chain.from_urdf_file(_write(tmp_path, _urdf("revolute", '<axis xyz="0 0 1"/>'), "unit.urdf"), symbolic=False)
    np.testing.assert_allclose(scaled.links[1].rotation, [0, 0, 1])
    np.testing.assert_allclose(scaled.forward_kinematics([0, 0.7]), unit.forward_kinematics([0, 0.7]))
    with pytest.warns(UserWarning, match="non-unit axis"):
        prismatic = chain.Chain.from_urdf_file(_write(tmp_path, _urdf("prismatic", '<axis xyz="3 0 4"/>')), symbolic=False)
    np.testing.assert_allclose(prismatic.links[1].translation, [0.6, 0, 0.8])
    with pytest.raises(ValueError, match="zero axis"):
        chain.Chain.from_urdf_file(_write(tmp_path, _urdf("revolute", '<axis xyz="0 0 0"/>')), symbolic=False)


@pytest.mark.parametrize("joint_type", ["floating", "planar"])
def test_floating_and_planar_joints_are_fixed_like_kdl(tmp_path, joint_type):
    with pytest.warns(UserWarning, match="treated as a fixed joint, as KDL does"):
        my_chain = chain.Chain.from_urdf_file(_write(tmp_path, _urdf(joint_type, '<axis xyz="0 0 1"/>')), symbolic=False)
    assert my_chain.links[1].joint_type == "fixed"
    np.testing.assert_allclose(my_chain.forward_kinematics([0, 0])[:3, 3], [0, 0, 0.5])


def test_xacro_files_are_refused_with_a_hint(tmp_path):
    with pytest.raises(ValueError, match="xacro file.*expanded first"):
        chain.Chain.from_urdf_file(_write(tmp_path, _urdf("revolute", '<axis xyz="0 0 1"/>', '<origin xyz="$(optenv ROBOT_XYZ 0 0 0)"/>')))
    with pytest.raises(ValueError, match="xacro file"):
        chain.Chain.from_urdf_file(_write(tmp_path, _urdf("revolute", '<axis xyz="0 0 1"/>', '<origin xyz="${length} 0 0"/>')))
    with pytest.raises(ValueError, match="xacro file"):
        chain.Chain.from_urdf_file(_write(tmp_path, '<robot name="x" xmlns:xacro="http://ros.org/wiki/xacro"><xacro:property name="l" value="1"/><link name="base_link"/></robot>'))
    with pytest.raises(ValueError, match="root element 'robot'"):
        chain.Chain.from_urdf_file(_write(tmp_path, '<mujoco><worldbody/></mujoco>'))


def test_panda_loads_with_mimic_and_safety_elements():
    """Every chain of the Panda loads; the mimic finger is an ordinary prismatic link for now"""
    arm = chain.Chain.from_urdf_file(PANDA, base="panda_link0", tip="panda_hand", symbolic=False)
    assert [link.name for link in arm.links][1:] == ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_joint8", "panda_hand_joint"]
    assert arm.links[1].bounds == (-2.9671, 2.9671)
    finger = chain.Chain.from_urdf_file(PANDA, base="panda_hand", tip="panda_rightfinger", symbolic=False)
    assert finger.links[-1].joint_type == "prismatic"
    joints = [0, 0.3, -0.5, 0.2, -1.8, 0.1, 1.5, 0.4, 0, 0]
    np.testing.assert_allclose(arm.forward_kinematics(joints)[:3, 3], arm.forward_kinematics(joints)[:3, 3])
    with pytest.raises(ValueError, match="Several links"):
        chain.Chain.from_urdf_file(PANDA, symbolic=False)


MIMIC_URDF = """<?xml version="1.0" ?>
<robot name="mimic_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="link3"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <mimic joint="joint1" multiplier="2" offset="0.1"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <mimic joint="joint2" multiplier="-1" offset="0.05"/>
  </joint>
</robot>
"""


def _mimic_chain(tmp_path):
    with pytest.warns(UserWarning, match="mimics link"):
        return chain.Chain.from_urdf_file(_write(tmp_path, MIMIC_URDF, "mimic.urdf"), symbolic=False)


def test_mimic_attributes_are_parsed(tmp_path):
    my_chain = _mimic_chain(tmp_path)
    assert my_chain.links[1].mimic_joint is None
    assert my_chain.links[2].mimic_joint == "joint1"
    assert my_chain.links[2].mimic_multiplier == 2.0
    assert my_chain.links[2].mimic_offset == 0.1
    # A joint can mimic another mimicking joint
    assert my_chain.links[3].mimic_joint == "joint2"
    assert my_chain.links[3].mimic_multiplier == -1.0
    assert my_chain.links[3].mimic_offset == 0.05


def test_mimic_joints_are_not_active(tmp_path):
    my_chain = _mimic_chain(tmp_path)
    # Only the origin and joint1 are active: joint2 and joint3 are derived
    assert list(my_chain.active_links_mask) == [True, True, False, False]
    assert [dep[0] for dep in my_chain.mimic_dependencies] == [2, 3]


def test_mimic_values_are_derived_in_active_to_full(tmp_path):
    my_chain = _mimic_chain(tmp_path)
    full = my_chain.active_to_full([0, 0.3], [0, 0, 0, 0])
    # joint2 = 2 * 0.3 + 0.1, joint3 = -joint2 + 0.05
    np.testing.assert_allclose(full, [0, 0.3, 0.7, -0.65])
    # The end-effector pose is the one of the derived configuration
    plain = chain.Chain.from_urdf_file(_write(tmp_path, MIMIC_URDF.replace("<mimic joint=\"joint1\" multiplier=\"2\" offset=\"0.1\"/>", "").replace("<mimic joint=\"joint2\" multiplier=\"-1\" offset=\"0.05\"/>", ""), "plain.urdf"), symbolic=False)
    np.testing.assert_allclose(my_chain.forward_kinematics(full), plain.forward_kinematics([0, 0.3, 0.7, -0.65]))


def test_mimic_inverse_kinematics(tmp_path):
    my_chain = _mimic_chain(tmp_path)
    target = my_chain.forward_kinematics(my_chain.active_to_full([0, 0.3], [0, 0, 0, 0]))
    solution = my_chain.inverse_kinematics(target_position=target[:3, 3], initial_position=[0, 0, 0, 0])
    np.testing.assert_allclose(my_chain.forward_kinematics(solution)[:3, 3], target[:3, 3], atol=1e-4)
    # The solution derives the mimicking joints instead of moving them freely
    np.testing.assert_allclose(solution[2], 2 * solution[1] + 0.1, atol=1e-6)
    np.testing.assert_allclose(solution[3], -solution[2] + 0.05, atol=1e-6)


def test_panda_mimic_joint_is_parsed():
    parameters = chain.URDF.get_urdf_parameters(PANDA, base="panda_hand", tip="panda_rightfinger", symbolic=False)
    finger2 = parameters[-1]
    assert finger2.name == "panda_finger_joint2"
    assert finger2.mimic_joint == "panda_finger_joint1"
    # No multiplier or offset in the file: the URDF defaults apply
    assert finger2.mimic_multiplier == 1.0
    assert finger2.mimic_offset == 0.0


def test_mimic_master_outside_the_chain_warns_and_keeps_the_joint():
    # The Panda fingers are siblings: a chain to one finger does not
    # contain the master of its mimic joint, so nothing can be derived
    with pytest.warns(UserWarning, match="not part of this chain"):
        finger = chain.Chain.from_urdf_file(PANDA, base="panda_hand", tip="panda_rightfinger", symbolic=False)
    assert finger.mimic_dependencies == []
    assert finger.links[-1].joint_type == "prismatic"
