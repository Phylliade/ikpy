"""Selecting a chain with base and tip, in the three formats, and the deprecated base_elements"""
import json
import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy import selection
from ikpy.bvh import BVH
from ikpy.urdf import URDF


BAXTER = "../resources/baxter/baxter.urdf"
POPPY_TORSO = "../resources/poppy_torso/poppy_torso.URDF"
ERGO = "../resources/poppy_ergo.URDF"
PRISMATIC = "../resources/prismatic/prismatic_robot.URDF"
UR5E = "../resources/mjcf/ur5e_simplified.xml"
SIMPLE_BVH = "../resources/bvh/simple_skeleton.bvh"

BAXTER_LEFT_ARM_ELEMENTS = ["base", "torso_t0", "torso", "left_torso_arm_mount", "left_arm_mount", "left_s0", "left_upper_shoulder", "left_s1", "left_lower_shoulder", "left_e0", "left_upper_elbow", "left_e1", "left_lower_elbow", "left_w0", "left_upper_forearm", "left_w1", "left_lower_forearm", "left_w2", "left_wrist", "left_hand", "left_hand", "left_gripper_base", "left_gripper_base", "left_endpoint", "left_gripper"]


def names(my_chain):
    return [link.name for link in my_chain.links]


# Tree helpers

def test_path_between_and_leaves():
    parent_of = {"b": "a", "c": "b", "d": "b", "e": "d"}
    children_of = {"a": ["b"], "b": ["c", "d"], "c": [], "d": ["e"], "e": []}
    assert selection.path_between(parent_of, "a", "e") == ["a", "b", "d", "e"]
    assert selection.path_between(parent_of, "e", "e") == ["e"]
    with pytest.raises(ValueError, match="not below"):
        selection.path_between(parent_of, "c", "e")
    assert selection.leaves_below(children_of, "a") == ["c", "e"]
    assert selection.unique_leaf(children_of, "d", "nodes") == "e"
    with pytest.raises(ValueError, match="Several nodes"):
        selection.unique_leaf(children_of, "a", "nodes")
    assert selection.unique_root(["a"], "nodes") == "a"
    with pytest.raises(ValueError, match="2 root nodes"):
        selection.unique_root(["a", "z"], "nodes")


def test_resolve_selection_translates_and_warns():
    assert selection.resolve_selection(base="a", tip="b", tip_offset=[1, 0, 0]) == ("a", "b", [1, 0, 0], None)
    with pytest.warns(DeprecationWarning, match="tip_offset"):
        assert selection.resolve_selection(last_link_vector=[1, 0, 0])[2] == [1, 0, 0]
    with pytest.warns(DeprecationWarning, match="base='a'"):
        assert selection.resolve_selection(base_elements=["a", "b"])[3] == (["a", "b"], "link")
    with pytest.warns(DeprecationWarning):
        assert selection.resolve_selection(base_elements=["j"], base_element_type="joint")[3] == (["j"], "joint")
    with pytest.raises(ValueError, match="not both"):
        selection.resolve_selection(base="a", base_elements=["a"])


# URDF

def test_urdf_base_and_tip_give_the_legacy_chain():
    new = chain.Chain.from_urdf_file(BAXTER, base="base", tip="left_gripper", symbolic=False)
    with pytest.warns(DeprecationWarning):
        legacy = chain.Chain.from_urdf_file(BAXTER, base_elements=BAXTER_LEFT_ARM_ELEMENTS, symbolic=False)
    assert names(new) == names(legacy)
    assert names(new)[-1] == "left_endpoint"
    assert new.base == "base" and new.tip == "left_gripper"


def test_urdf_tip_and_base_can_be_joints():
    by_link = chain.Chain.from_urdf_file(BAXTER, base="base", tip="left_gripper", symbolic=False)
    by_joint = chain.Chain.from_urdf_file(BAXTER, base="base", tip="left_endpoint", symbolic=False)
    assert names(by_link) == names(by_joint)
    from_joint = chain.Chain.from_urdf_file(BAXTER, base="left_s0", tip="left_gripper", symbolic=False)
    assert names(from_joint)[1] == "left_s0"
    from_link = chain.Chain.from_urdf_file(BAXTER, base="left_upper_shoulder", tip="left_gripper", symbolic=False)
    assert names(from_link)[1] == "left_s1"
    # A name shared by a joint and its child link means the link: both readings give the same chain
    shared = chain.Chain.from_urdf_file(BAXTER, base="base", tip="left_hand", symbolic=False)
    assert names(shared)[-1] == "left_hand"


def test_urdf_dict_says_link_or_joint():
    # Baxter has a joint and a link both named left_hand: as a base, the two readings differ
    as_link = chain.Chain.from_urdf_file(BAXTER, base="left_hand", tip="left_gripper", symbolic=False)
    explicit_link = chain.Chain.from_urdf_file(BAXTER, base={"name": "left_hand", "type": "link"}, tip="left_gripper", symbolic=False)
    as_joint = chain.Chain.from_urdf_file(BAXTER, base={"name": "left_hand", "type": "joint"}, tip={"name": "left_gripper", "type": "link"}, symbolic=False)
    assert names(as_link) == names(explicit_link) == ["Base link", "left_gripper_base", "left_endpoint"]
    assert names(as_joint) == ["Base link", "left_hand", "left_gripper_base", "left_endpoint"]
    by_joint_tip = chain.Chain.from_urdf_file(BAXTER, base="base", tip={"name": "left_endpoint", "type": "joint"}, symbolic=False)
    by_link_tip = chain.Chain.from_urdf_file(BAXTER, base="base", tip={"name": "left_gripper", "type": "link"}, symbolic=False)
    assert names(by_joint_tip) == names(by_link_tip)
    with pytest.raises(ValueError, match="is not a joint of the URDF"):
        chain.Chain.from_urdf_file(BAXTER, base={"name": "left_gripper", "type": "joint"}, symbolic=False)
    with pytest.raises(ValueError, match="is not a link of the URDF"):
        chain.Chain.from_urdf_file(BAXTER, base="base", tip={"name": "left_endpoint", "type": "link"}, symbolic=False)
    with pytest.raises(ValueError, match='must be {"name": ..., "type": "link"}'):
        chain.Chain.from_urdf_file(BAXTER, base={"name": "base", "type": "body"}, symbolic=False)
    with pytest.raises(ValueError, match='must be {"name": ..., "type": "link"}'):
        chain.Chain.from_urdf_file(BAXTER, base={"link": "base"}, symbolic=False)


def test_urdf_defaults_root_and_unique_leaf():
    # poppy_ergo: one root, one leaf, nothing to say
    assert names(chain.Chain.from_urdf_file(ERGO, symbolic=False)) == ["Base link", "m1", "m2", "m3", "m4", "m5", "m6"]
    # prismatic_robot has no link named base_link: the root is found anyway
    assert names(chain.Chain.from_urdf_file(PRISMATIC, symbolic=False)) == ["Base link", "joint_baseLink_childA"]
    # A branching robot needs a tip
    with pytest.raises(ValueError, match="Several links end the tree below 'base'.*tip="):
        chain.Chain.from_urdf_file(POPPY_TORSO, symbolic=False)
    with pytest.raises(ValueError, match="r_forearm"):
        chain.Chain.from_urdf_file(POPPY_TORSO, base="chest", symbolic=False)
    # Below a branch, the leaf is unique
    arm = chain.Chain.from_urdf_file(POPPY_TORSO, base="r_shoulder_y", symbolic=False)
    assert names(arm)[1:] == ["r_shoulder_y", "r_shoulder_x", "r_arm_z", "r_elbow_y"]


def test_urdf_bad_names_raise():
    with pytest.raises(ValueError, match="neither a link nor a joint"):
        chain.Chain.from_urdf_file(BAXTER, base="nope", symbolic=False)
    with pytest.raises(ValueError, match="neither a link nor a joint"):
        chain.Chain.from_urdf_file(BAXTER, base="base", tip="nope", symbolic=False)
    with pytest.raises(ValueError, match="not below"):
        chain.Chain.from_urdf_file(BAXTER, base="left_s0", tip="right_gripper", symbolic=False)


def test_urdf_tip_offset_and_deprecated_arguments():
    new = chain.Chain.from_urdf_file(POPPY_TORSO, base="base", tip="r_elbow_y", tip_offset=[0, 0.18, 0], symbolic=False)
    assert names(new)[-1] == "last_joint"
    np.testing.assert_array_equal(new.links[-1].origin_translation, [0, 0.18, 0])
    with pytest.warns(DeprecationWarning, match="tip_offset"):
        legacy = chain.Chain.from_urdf_file(POPPY_TORSO, base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"], last_link_vector=[0, 0.18, 0], symbolic=False)
    assert names(new) == names(legacy)
    np.testing.assert_allclose(new.forward_kinematics([0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0]), legacy.forward_kinematics([0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0]))
    with pytest.raises(ValueError, match="not both"):
        chain.Chain.from_urdf_file(POPPY_TORSO, base="base", base_elements=["base"], symbolic=False)
    with pytest.raises(TypeError):
        chain.Chain.from_urdf_file(POPPY_TORSO, ["base"])  # the selection is keyword-only


def test_urdf_legacy_default_is_base_link():
    with pytest.warns(DeprecationWarning):
        legacy = chain.Chain.from_urdf_file(ERGO, base_element_type="link", symbolic=False)
    assert names(legacy) == ["Base link", "m1", "m2", "m3", "m4", "m5", "m6"]
    with pytest.warns(DeprecationWarning):
        links = URDF.get_urdf_parameters(ERGO, base_elements=["base_link"], symbolic=False)
    assert [link.name for link in links] == ["m1", "m2", "m3", "m4", "m5", "m6"]


def test_json_round_trip_both_formats(tmp_path):
    new = chain.Chain.from_urdf_file(POPPY_TORSO, base={"name": "base", "type": "link"}, tip="r_elbow_y", tip_offset=[0, 0.18, 0], name="arm_v2", symbolic=False)
    new._urdf_metadata["urdf_file"] = str(tmp_path / "poppy_torso.URDF")
    import shutil
    shutil.copy(POPPY_TORSO, tmp_path / "poppy_torso.URDF")
    path = new.to_json_file()
    saved = json.load(open(path))
    assert saved["version"] == "v2" and saved["base"] == {"name": "base", "type": "link"} and saved["tip"] == "r_elbow_y" and saved["tip_offset"] == [0, 0.18, 0]
    loaded = chain.Chain.from_json_file(path)
    assert names(loaded) == names(new)

    # The v1 resources still load, without a deprecation warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        baxter = chain.Chain.from_json_file("../resources/baxter/baxter_left_arm.json")
    assert names(baxter)[-2:] == ["left_endpoint", "last_joint"]


# MJCF

def test_mjcf_base_and_tip():
    new = chain.Chain.from_mjcf_file(UR5E, base="base", tip="wrist_3_link", symbolic=False)
    with pytest.warns(DeprecationWarning):
        legacy = chain.Chain.from_mjcf_file(UR5E, base_elements=["base"], symbolic=False)
    assert names(new) == names(legacy)
    assert names(chain.Chain.from_mjcf_file(UR5E, symbolic=False)) == names(new)
    with pytest.raises(ValueError, match="Body 'nope' not found"):
        chain.Chain.from_mjcf_file(UR5E, base="nope")


def test_mjcf_branching_needs_a_tip(tmp_path):
    mjcf_file = tmp_path / "branched_arm.xml"
    mjcf_file.write_text("""
<mujoco model="branched_arm">
  <worldbody>
    <body name="base" pos="1 0 0">
      <body name="left_arm" pos="0 -2 0"/>
      <body name="right_arm" pos="0 2 0">
        <joint name="right_shoulder" type="hinge" axis="0 0 1"/>
        <body name="tool" pos="1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
    with pytest.raises(ValueError, match="Several bodies.*left_arm.*tool"):
        chain.Chain.from_mjcf_file(str(mjcf_file), symbolic=False)
    right = chain.Chain.from_mjcf_file(str(mjcf_file), tip="tool", symbolic=False)
    assert names(right) == ["Base link", "base", "right_shoulder", "tool"]
    # The chain stops at the tip
    short = chain.Chain.from_mjcf_file(str(mjcf_file), base="base", tip="right_arm", symbolic=False)
    assert names(short) == ["Base link", "base", "right_shoulder"]
    with pytest.raises(ValueError, match="not below"):
        chain.Chain.from_mjcf_file(str(mjcf_file), base="left_arm", tip="tool", symbolic=False)


# BVH

def test_bvh_base_and_tip():
    arm = chain.Chain.from_bvh_file(SIMPLE_BVH, base="Hips", tip="LeftForeArm", symbolic=False)
    assert names(arm)[-1] == "LeftForeArm_end_site"
    with pytest.warns(DeprecationWarning):
        legacy = chain.Chain.from_bvh_file(SIMPLE_BVH, base_elements=["Hips", "LeftForeArm"], symbolic=False)
    assert names(arm) == names(legacy)
    # A branching skeleton needs a tip, a branch with one leaf does not
    with pytest.raises(ValueError, match="Several joints.*LeftForeArm.*Head.*LeftLeg"):
        chain.Chain.from_bvh_file(SIMPLE_BVH, symbolic=False)
    assert names(chain.Chain.from_bvh_file(SIMPLE_BVH, base="LeftArm", symbolic=False))[-1] == "LeftForeArm_end_site"
    # The motion follows the same selection
    trajectory, _ = BVH.get_bvh_motion(SIMPLE_BVH, base="Hips", tip="LeftForeArm")
    assert trajectory.shape[1] == len(arm.links)
    with pytest.raises(ValueError, match="not a descendant"):
        chain.Chain.from_bvh_file(SIMPLE_BVH, base="Spine", tip="LeftLeg", symbolic=False)
