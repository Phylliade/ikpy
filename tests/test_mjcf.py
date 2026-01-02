# coding= utf8
"""Tests for the MJCF parser module."""
import os
import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy.mjcf import MJCF
from ikpy.mjcf import utils as mjcf_utils


# Test fixtures
@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), "../resources")


@pytest.fixture
def simple_arm_mjcf(resources_path):
    return os.path.join(resources_path, "mjcf/simple_arm.xml")


@pytest.fixture
def ur5e_mjcf(resources_path):
    return os.path.join(resources_path, "mjcf/ur5e_simplified.xml")


@pytest.fixture
def prismatic_mjcf(resources_path):
    return os.path.join(resources_path, "mjcf/prismatic_test.xml")


# ==================== Utils Tests ====================

class TestMJCFUtils:
    """Test MJCF utility functions for rotation conversions."""

    def test_quat_to_rpy_identity(self):
        """Test quaternion to RPY for identity rotation."""
        quat = [1, 0, 0, 0]  # w, x, y, z - identity
        rpy = mjcf_utils.quat_to_rpy(quat)
        np.testing.assert_array_almost_equal(rpy, [0, 0, 0])

    def test_quat_to_rpy_rotation_z(self):
        """Test quaternion to RPY for 90 degree rotation around Z."""
        angle = np.pi / 2
        quat = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # 90 deg around Z
        rpy = mjcf_utils.quat_to_rpy(quat)
        # For rotation around Z, yaw should be pi/2
        np.testing.assert_almost_equal(rpy[2], np.pi/2, decimal=5)

    def test_axisangle_to_rpy(self):
        """Test axis-angle to RPY conversion."""
        # 90 degree rotation around Z axis
        axis = [0, 0, 1]
        angle = np.pi / 2
        rpy = mjcf_utils.axisangle_to_rpy(axis, angle)
        np.testing.assert_almost_equal(rpy[2], np.pi/2, decimal=5)

    def test_euler_to_rpy(self):
        """Test Euler angles to RPY conversion."""
        # Identity
        euler = [0, 0, 0]
        rpy = mjcf_utils.euler_to_rpy(euler, "xyz")
        np.testing.assert_array_almost_equal(rpy, [0, 0, 0])

    def test_xyaxes_to_rpy(self):
        """Test xyaxes to RPY conversion."""
        # Standard frame (no rotation)
        xyaxes = [1, 0, 0, 0, 1, 0]
        rpy = mjcf_utils.xyaxes_to_rpy(xyaxes)
        np.testing.assert_array_almost_equal(rpy, [0, 0, 0], decimal=5)

    def test_zaxis_to_rpy(self):
        """Test zaxis to RPY conversion."""
        # Z pointing up (no rotation)
        zaxis = [0, 0, 1]
        rpy = mjcf_utils.zaxis_to_rpy(zaxis)
        np.testing.assert_array_almost_equal(rpy, [0, 0, 0], decimal=5)

    def test_rotation_roundtrip(self):
        """Test that rotation matrix conversions are consistent."""
        # Create a rotation matrix from axis-angle
        axis = [1, 1, 1]
        axis = np.array(axis) / np.linalg.norm(axis)
        angle = np.pi / 4

        R = mjcf_utils.axisangle_to_rotation_matrix(axis, angle)
        rpy = mjcf_utils.rotation_matrix_to_rpy(R)

        # Convert RPY back to rotation matrix using numpy implementation
        from ikpy.utils.geometry import rpy_matrix
        R_back = rpy_matrix(*rpy)

        np.testing.assert_array_almost_equal(R, R_back, decimal=5)


# ==================== Parser Tests ====================

class TestMJCFParser:
    """Test MJCF parsing functionality."""

    def test_get_body_names(self, simple_arm_mjcf):
        """Test extracting body names from MJCF."""
        names = MJCF.get_body_names(simple_arm_mjcf)
        assert "base" in names
        assert "link1" in names
        assert "link2" in names
        assert "link3" in names

    def test_get_joint_names(self, simple_arm_mjcf):
        """Test extracting joint names from MJCF."""
        names = MJCF.get_joint_names(simple_arm_mjcf)
        assert "joint1" in names
        assert "joint2" in names
        assert "joint3" in names

    def test_get_mjcf_parameters_simple(self, simple_arm_mjcf):
        """Test parsing simple arm MJCF."""
        links = MJCF.get_mjcf_parameters(simple_arm_mjcf, base_elements=["base"])
        # Should have: base (fixed), link1 with joint1, link2 with joint2, link3 with joint3
        assert len(links) >= 3  # At least the 3 revolute joints

        # Check that joints are parsed correctly
        joint_names = [link.name for link in links]
        assert "joint1" in joint_names
        assert "joint2" in joint_names
        assert "joint3" in joint_names

    def test_get_mjcf_parameters_ur5e(self, ur5e_mjcf):
        """Test parsing UR5e MJCF."""
        links = MJCF.get_mjcf_parameters(ur5e_mjcf, base_elements=["base"])
        # UR5e has 6 joints
        joint_names = [link.name for link in links if link.joint_type == "revolute"]
        assert len(joint_names) == 6

        expected_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        for joint in expected_joints:
            assert joint in joint_names, f"Joint {joint} not found"

    def test_prismatic_joint(self, prismatic_mjcf):
        """Test parsing prismatic joints."""
        links = MJCF.get_mjcf_parameters(prismatic_mjcf, base_elements=["base"])

        # Find the prismatic joint
        prismatic_links = [link for link in links if link.joint_type == "prismatic"]
        assert len(prismatic_links) == 1
        assert prismatic_links[0].name == "slide_joint"

    def test_last_link_vector(self, simple_arm_mjcf):
        """Test adding last link vector."""
        last_vec = [0.1, 0, 0]
        links = MJCF.get_mjcf_parameters(
            simple_arm_mjcf,
            base_elements=["base"],
            last_link_vector=last_vec
        )
        # Last link should be the tip
        assert links[-1].name == "last_joint"
        np.testing.assert_array_equal(links[-1].origin_translation, last_vec)


# ==================== Chain Tests ====================

class TestMJCFChain:
    """Test Chain creation from MJCF files."""

    def test_chain_from_mjcf_simple(self, simple_arm_mjcf):
        """Test creating a chain from simple MJCF."""
        my_chain = chain.Chain.from_mjcf_file(
            simple_arm_mjcf,
            base_elements=["base"]
        )

        # Should have origin link + parsed links
        assert len(my_chain.links) >= 4

        # Test forward kinematics with zero configuration
        joints = [0] * len(my_chain.links)
        fk = my_chain.forward_kinematics(joints)
        assert fk.shape == (4, 4)

    def test_chain_from_mjcf_ur5e(self, ur5e_mjcf):
        """Test creating a chain from UR5e MJCF."""
        my_chain = chain.Chain.from_mjcf_file(
            ur5e_mjcf,
            base_elements=["base"],
            name="ur5e"
        )

        assert my_chain.name == "ur5e"

        # Test forward kinematics
        joints = [0] * len(my_chain.links)
        fk = my_chain.forward_kinematics(joints)
        assert fk.shape == (4, 4)

    def test_chain_forward_kinematics(self, simple_arm_mjcf):
        """Test forward kinematics calculation."""
        my_chain = chain.Chain.from_mjcf_file(
            simple_arm_mjcf,
            base_elements=["base"]
        )

        # Zero configuration
        joints_zero = [0] * len(my_chain.links)
        fk_zero = my_chain.forward_kinematics(joints_zero)

        # Non-zero configuration
        joints_moved = joints_zero.copy()
        # Find the first active joint and move it
        for i, link in enumerate(my_chain.links):
            if link.joint_type == "revolute":
                joints_moved[i] = np.pi / 4
                break

        fk_moved = my_chain.forward_kinematics(joints_moved)

        # The end effector position should be different
        assert not np.allclose(fk_zero[:3, 3], fk_moved[:3, 3])

    def test_chain_inverse_kinematics(self, simple_arm_mjcf):
        """Test inverse kinematics calculation."""
        my_chain = chain.Chain.from_mjcf_file(
            simple_arm_mjcf,
            base_elements=["base"],
            last_link_vector=[0.1, 0, 0]
        )

        # Create active links mask (first and last are fixed)
        active_mask = [False] + [link.joint_type != "fixed" for link in my_chain.links[1:-1]] + [False]
        my_chain.active_links_mask = np.array(active_mask)

        # Get a target from forward kinematics
        joints_target = [0] * len(my_chain.links)
        for i, link in enumerate(my_chain.links):
            if link.joint_type == "revolute":
                joints_target[i] = 0.3
        fk_target = my_chain.forward_kinematics(joints_target)
        target_position = fk_target[:3, 3]

        # Compute inverse kinematics
        ik_result = my_chain.inverse_kinematics(target_position=target_position)

        # Forward kinematics of the result should be close to target
        fk_result = my_chain.forward_kinematics(ik_result)
        np.testing.assert_array_almost_equal(
            fk_result[:3, 3],
            target_position,
            decimal=2  # Allow some tolerance
        )

    def test_chain_with_prismatic(self, prismatic_mjcf):
        """Test chain with prismatic joints."""
        my_chain = chain.Chain.from_mjcf_file(
            prismatic_mjcf,
            base_elements=["base"]
        )

        # Check that we have a prismatic joint
        has_prismatic = any(link.joint_type == "prismatic" for link in my_chain.links)
        assert has_prismatic

        # Test forward kinematics
        joints = [0] * len(my_chain.links)
        fk = my_chain.forward_kinematics(joints)
        assert fk.shape == (4, 4)

    def test_chain_full_kinematics(self, simple_arm_mjcf):
        """Test full kinematics (all intermediate frames)."""
        my_chain = chain.Chain.from_mjcf_file(
            simple_arm_mjcf,
            base_elements=["base"]
        )

        joints = [0] * len(my_chain.links)
        frames = my_chain.forward_kinematics(joints, full_kinematics=True)

        assert len(frames) == len(my_chain.links)
        for frame in frames:
            assert frame.shape == (4, 4)


# ==================== Integration Tests ====================

class TestMJCFIntegration:
    """Integration tests comparing MJCF results with expected behavior."""

    def test_simple_arm_geometry(self, simple_arm_mjcf):
        """Test that simple arm geometry is parsed correctly."""
        my_chain = chain.Chain.from_mjcf_file(
            simple_arm_mjcf,
            base_elements=["base"]
        )

        # At zero configuration, the arm should extend along X axis
        joints = [0] * len(my_chain.links)
        fk = my_chain.forward_kinematics(joints)

        # The end effector should be at positive X (since all links extend in X)
        # This is a sanity check based on the model structure
        assert fk[0, 3] > 0, "End effector should be at positive X"
