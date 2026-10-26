import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.bvh import BVH
from ikpy.utils import geometry


SIMPLE = "../resources/bvh/simple_skeleton.bvh"
CMU = "../resources/bvh/cmu_01_01_short.bvh"


def reference_joint_positions(skeleton, frame_index):
    """An independent forward kinematics of a BVH skeleton, following the file semantics joint by joint:
    T(position) . R_channel1 . R_channel2 ..., with position = the position channels when present, else the offset.
    Returns the world position of every joint and of every end site, by name"""
    positions = {}
    frame = skeleton.motion[frame_index]

    def visit(joint, parent_transform):
        values = dict(zip(joint.channels, frame[joint.channel_index:joint.channel_index + len(joint.channels)]))
        position = joint.offset.copy()
        for axis in "XYZ":
            if axis + "position" in values:
                position["XYZ".index(axis)] = values[axis + "position"]
        transform = parent_transform @ geometry.homogeneous_translation_matrix(*position)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                angle = np.deg2rad(values[channel])
                rotation = {"X": geometry.rx_matrix, "Y": geometry.ry_matrix, "Z": geometry.rz_matrix}[channel[0]](angle)
                transform = transform @ geometry.cartesian_to_homogeneous(rotation)
        positions[joint.name] = transform[:3, 3].copy()
        if joint.end_site is not None:
            positions[joint.name + "_end_site"] = (transform @ geometry.homogeneous_translation_matrix(*joint.end_site))[:3, 3]
        for child in joint.children:
            visit(child, transform)

    for root in skeleton.roots:
        visit(root, np.eye(4))
    return positions


def chain_joint_positions(my_chain, joints):
    """The world position of each link of a chain, by name"""
    frames = my_chain.forward_kinematics(joints, full_kinematics=True)
    return {link.name: frame[:3, 3] for link, frame in zip(my_chain.links, frames)}


# Parsing

def test_parse_simple_skeleton():
    skeleton = BVH.parse_bvh(SIMPLE)
    assert [root.name for root in skeleton.roots] == ["Hips"]
    assert [joint.name for joint in skeleton.iter_joints()] == ["Hips", "Spine", "LeftArm", "LeftForeArm", "Head", "LeftUpLeg", "LeftLeg"]
    assert skeleton.n_channels == 24
    assert skeleton.motion.shape == (3, 24)
    assert skeleton.frame_time == 0.04

    hips = skeleton.roots[0]
    assert hips.channels == ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]
    np.testing.assert_array_equal(hips.offset, [0, 1, 0])
    assert [child.name for child in hips.children] == ["Spine", "LeftUpLeg"]
    assert hips.end_site is None

    forearm = skeleton.find("LeftForeArm")
    assert forearm.parent.name == "LeftArm"
    assert forearm.is_leaf
    assert forearm.channel_index == 12
    np.testing.assert_array_equal(forearm.end_site, [0.4, 0, 0])
    assert skeleton.find("Nope") is None


def test_get_joint_names():
    assert BVH.get_joint_names(SIMPLE)[:3] == ["Hips", "Spine", "LeftArm"]
    assert len(BVH.get_joint_names(CMU)) == 31


def test_parse_real_capture():
    """CMU 01_01: tabs and zero offsets"""
    skeleton = BVH.parse_bvh(CMU)
    assert skeleton.roots[0].name == "Hips"
    assert skeleton.n_channels == 96
    assert skeleton.motion.shape == (8, 96)
    assert skeleton.find("LHipJoint").channels == ["Zrotation", "Yrotation", "Xrotation"]
    np.testing.assert_array_equal(skeleton.find("LHipJoint").offset, [0, 0, 0])
    assert sum(1 for joint in skeleton.iter_joints() if joint.end_site is not None) == 7


def test_parse_windows_line_endings_trailing_spaces_and_lowercase_channels(tmp_path):
    content = open(SIMPLE).read().replace("\n", " \r\n").replace("Zrotation", "ZROTATION").replace("Xposition", "xposition")
    bvh_file = tmp_path / "crlf.bvh"
    bvh_file.write_bytes(content.encode())
    skeleton = BVH.parse_bvh(str(bvh_file))
    assert skeleton.roots[0].channels[0] == "Xposition"
    assert skeleton.roots[0].channels[3] == "Zrotation"
    assert skeleton.motion.shape == (3, 24)


def test_parse_hierarchy_only(tmp_path):
    content = open(SIMPLE).read().split("MOTION")[0]
    bvh_file = tmp_path / "hierarchy.bvh"
    bvh_file.write_text(content)
    skeleton = BVH.parse_bvh(str(bvh_file))
    assert skeleton.n_frames == 0
    assert skeleton.frame_time is None
    my_chain = chain.Chain.from_bvh_file(str(bvh_file), symbolic=False)
    assert len(my_chain.links) > 1


def test_parse_truncated_motion_warns(tmp_path):
    lines = open(SIMPLE).read().splitlines()
    bvh_file = tmp_path / "truncated.bvh"
    bvh_file.write_text("\n".join(lines[:-1]) + "\n")  # drop the last frame, but keep "Frames: 3"
    with pytest.warns(UserWarning, match="announces 3 frames"):
        skeleton = BVH.parse_bvh(str(bvh_file))
    assert skeleton.n_frames == 2


@pytest.mark.parametrize("corruption, message", [
    ("CHANNELS 3 Zrotation Xrotation Yrotation\n\t\tJOINT LeftArm", "CHANNELS 3 Zrotation Wrotation Yrotation\n\t\tJOINT LeftArm"),
    ("ROOT Hips", "ROOTS Hips"),
    ("OFFSET 0.5 0.0 0.0", "OFFSET 0.5 abc 0.0"),
])
def test_malformed_files_raise(tmp_path, corruption, message):
    content = open(SIMPLE).read()
    assert corruption in content
    bvh_file = tmp_path / "bad.bvh"
    bvh_file.write_text(content.replace(corruption, message, 1))
    with pytest.raises(ValueError):
        BVH.parse_bvh(str(bvh_file))


# Chain construction

def test_default_chain_follows_first_children():
    my_chain = chain.Chain.from_bvh_file(SIMPLE, symbolic=False)
    assert [link.name for link in my_chain.links] == [
        "Base link",
        "Hips_Xposition", "Hips_Yposition", "Hips_Zposition", "Hips_Zrotation", "Hips_Xrotation", "Hips_Yrotation",
        "Spine_Zrotation", "Spine_Xrotation", "Spine_Yrotation",
        "LeftArm_Xrotation", "LeftArm_Yrotation", "LeftArm_Zrotation",
        "LeftForeArm_Xrotation", "LeftForeArm_Yrotation", "LeftForeArm_Zrotation",
        "LeftForeArm_end_site",
    ]
    types = [link.joint_type for link in my_chain.links]
    assert types[1:4] == ["prismatic"] * 3
    assert types[4:16] == ["revolute"] * 12
    assert types[16] == "fixed"
    # The first link of a joint carries its offset, the others have a zero origin
    np.testing.assert_array_equal(my_chain.links[1].origin_translation, [0, 1, 0])
    np.testing.assert_array_equal(my_chain.links[2].origin_translation, [0, 0, 0])
    np.testing.assert_array_equal(my_chain.links[7].origin_translation, [0, 0.5, 0])
    np.testing.assert_array_equal(my_chain.links[16].origin_translation, [0.4, 0, 0])
    # Axes follow the channel letters
    np.testing.assert_array_equal(my_chain.links[1].translation, [1, 0, 0])
    np.testing.assert_array_equal(my_chain.links[4].rotation, [0, 0, 1])
    np.testing.assert_array_equal(my_chain.links[5].rotation, [1, 0, 0])
    # BVH joints have no limits
    assert my_chain.links[4].bounds == (-np.inf, np.inf)


def test_base_elements_fill_the_gaps_and_stop_at_the_last_name():
    my_chain = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "LeftLeg"], symbolic=False)
    names = [link.name for link in my_chain.links]
    assert names[7:] == ["LeftUpLeg_Zrotation", "LeftUpLeg_Xrotation", "LeftUpLeg_Yrotation",
                         "LeftLeg_Zrotation", "LeftLeg_Xrotation", "LeftLeg_Yrotation", "LeftLeg_end_site"]

    # Stopping before a leaf: no end site link
    my_chain = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "Spine"], symbolic=False)
    assert [link.name for link in my_chain.links][-1] == "Spine_Yrotation"

    # A single name starts there and follows the first children
    my_chain = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Spine"], symbolic=False)
    names = [link.name for link in my_chain.links]
    assert names[1] == "Spine_Zrotation" and names[-1] == "LeftForeArm_end_site"

    # The full path is also accepted
    full = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "Spine", "LeftArm", "LeftForeArm"], symbolic=False)
    short = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "LeftForeArm"], symbolic=False)
    assert [link.name for link in full.links] == [link.name for link in short.links]


def test_bad_base_elements_raise():
    with pytest.raises(ValueError, match="not found"):
        chain.Chain.from_bvh_file(SIMPLE, base_elements=["Pelvis"])
    with pytest.raises(ValueError, match="not a descendant"):
        chain.Chain.from_bvh_file(SIMPLE, base_elements=["Spine", "LeftLeg"])
    with pytest.raises(ValueError, match="not a descendant"):
        chain.Chain.from_bvh_file(SIMPLE, base_elements=["Spine", "Spine"])


def test_last_link_vector_overrides_the_end_site():
    my_chain = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "LeftForeArm"], last_link_vector=[0.1, 0.2, 0.3], symbolic=False)
    assert my_chain.links[-1].name == "last_joint"
    np.testing.assert_array_equal(my_chain.links[-1].origin_translation, [0.1, 0.2, 0.3])


def test_scale():
    my_chain = chain.Chain.from_bvh_file(SIMPLE, scale=0.01, symbolic=False)
    np.testing.assert_allclose(my_chain.links[1].origin_translation, [0, 0.01, 0])
    np.testing.assert_allclose(my_chain.links[-1].origin_translation, [0.004, 0, 0])
    unscaled = chain.Chain.from_bvh_file(SIMPLE, symbolic=False)
    trajectory, _ = BVH.get_bvh_motion(SIMPLE, scale=0.01)
    reference, _ = BVH.get_bvh_motion(SIMPLE)
    np.testing.assert_allclose(my_chain.forward_kinematics(trajectory[1])[:3, 3], unscaled.forward_kinematics(reference[1])[:3, 3] * 0.01)
    # Angles are not scaled, displacements are
    np.testing.assert_allclose(trajectory[1, 4:], reference[1, 4:])
    np.testing.assert_allclose(trajectory[1, 1:4], reference[1, 1:4] * 0.01)


def test_rest_pose_is_the_zero_vector():
    """With all joint values at zero, the chain is in the rest pose given by the offsets"""
    skeleton = BVH.parse_bvh(SIMPLE)
    my_chain = chain.Chain.from_bvh_file(SIMPLE, base_elements=["Hips", "LeftForeArm"], symbolic=False)
    tip = my_chain.forward_kinematics([0] * len(my_chain.links))[:3, 3]
    expected = sum((skeleton.find(name).offset for name in ["Hips", "Spine", "LeftArm", "LeftForeArm"]), np.zeros(3)) + skeleton.find("LeftForeArm").end_site
    np.testing.assert_allclose(tip, expected)


# Motion replay

@pytest.mark.parametrize("bvh_file", [SIMPLE, CMU])
@pytest.mark.parametrize("symbolic", [False, True])
def test_motion_matches_reference_forward_kinematics(bvh_file, symbolic):
    """Replaying the motion on every chain of the skeleton gives the joint positions of an independent FK"""
    skeleton = BVH.parse_bvh(bvh_file)
    root = skeleton.roots[0]
    leaves = [joint for joint in root.iter_joints() if joint.is_leaf]
    frames = range(skeleton.n_frames) if bvh_file == SIMPLE else [0, 3, 7]
    for leaf in leaves:
        base_elements = [root.name, leaf.name]
        my_chain = chain.Chain.from_bvh_file(bvh_file, base_elements=base_elements, symbolic=symbolic)
        trajectory, frame_time = BVH.get_bvh_motion(bvh_file, base_elements=base_elements)
        assert trajectory.shape == (skeleton.n_frames, len(my_chain.links))
        assert frame_time == skeleton.frame_time
        for frame_index in frames:
            reference = reference_joint_positions(skeleton, frame_index)
            positions = chain_joint_positions(my_chain, trajectory[frame_index])
            for joint in root.path_to(leaf):
                # The last link of a joint is at the position of the joint
                last_link = "{}_{}".format(joint.name, joint.channels[-1])
                np.testing.assert_allclose(positions[last_link], reference[joint.name], atol=1e-10, err_msg="{} frame {}".format(joint.name, frame_index))
            np.testing.assert_allclose(positions[leaf.name + "_end_site"], reference[leaf.name + "_end_site"], atol=1e-10)


def test_position_channels_replace_the_offset(tmp_path):
    """Position channels are absolute: a joint whose position channels repeat its offset does not move"""
    content = open(SIMPLE).read()
    content = content.replace("CHANNELS 3 Zrotation Xrotation Yrotation\n\t\tJOINT LeftArm",
                              "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n\t\tJOINT LeftArm", 1)
    lines = content.splitlines()
    for i in range(-3, 0):
        values = lines[i].split()
        lines[i] = " ".join(values[:6] + ["0.0", "0.5", "0.0"] + values[6:])
    bvh_file = tmp_path / "spine_position.bvh"
    bvh_file.write_text("\n".join(lines) + "\n")

    base_elements = ["Hips", "LeftForeArm"]
    original = chain.Chain.from_bvh_file(SIMPLE, base_elements=base_elements, symbolic=False)
    modified = chain.Chain.from_bvh_file(str(bvh_file), base_elements=base_elements, symbolic=False)
    assert "Spine_Yposition" in [link.name for link in modified.links]
    original_motion, _ = BVH.get_bvh_motion(SIMPLE, base_elements=base_elements)
    modified_motion, _ = BVH.get_bvh_motion(str(bvh_file), base_elements=base_elements)
    np.testing.assert_allclose(modified_motion[:, 7:10], 0)
    for frame_index in range(3):
        np.testing.assert_allclose(modified.forward_kinematics(modified_motion[frame_index]), original.forward_kinematics(original_motion[frame_index]), atol=1e-12)


def test_inverse_kinematics_on_a_capture():
    """IK on the left arm of the CMU skeleton, with the root locked: reaches the pose of a frame"""
    base_elements = ["Hips", "LeftHand"]
    my_chain = chain.Chain.from_bvh_file(CMU, base_elements=base_elements, symbolic=False, scale=0.0254)
    trajectory, _ = BVH.get_bvh_motion(CMU, base_elements=base_elements, scale=0.0254)
    # Lock the root (position and orientation) and the tip
    mask = [not (link.name.startswith("Hips_") or link.joint_type == "fixed") for link in my_chain.links]
    my_chain = chain.Chain.from_bvh_file(CMU, base_elements=base_elements, active_links_mask=mask, symbolic=False, scale=0.0254)

    target = my_chain.forward_kinematics(trajectory[5])
    initial = trajectory[0]
    ik = my_chain.inverse_kinematics_frame(target, initial_position=initial)
    np.testing.assert_allclose(my_chain.forward_kinematics(ik)[:3, 3], target[:3, 3], atol=1e-3)
    # Inactive links kept their initial values
    np.testing.assert_array_equal(ik[:7], initial[:7])
