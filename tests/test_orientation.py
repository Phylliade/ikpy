import numpy as np


def test_orientation(baxter_left_arm):
    target_position = [0.1, 0.5, -0.1]
    # Begin to place the arm in the right position
    ik = baxter_left_arm.inverse_kinematics(target_position)

    # Check orientation in all of the three axes
    for axis_index, axis in enumerate(["X", "Y", "Z"]):
        target_orientation = [0, 0, 1]

        ik = baxter_left_arm.inverse_kinematics(target_position, target_orientation, initial_position=ik, orientation_mode=axis)

        position = baxter_left_arm.forward_kinematics(ik)[:3, 3]
        orientation = baxter_left_arm.forward_kinematics(ik)[:3, axis_index]

        # Check
        np.testing.assert_almost_equal(position, target_position, decimal=5)
        np.testing.assert_almost_equal(orientation, target_orientation, decimal=5)


def test_orientation_full_frame(baxter_left_arm):
    target_position = [0.1, 0.4, -0.1]
    target_orientation = np.eye(3)

    # Begin to place the arm in the right position
    ik = baxter_left_arm.inverse_kinematics(target_position)
    ik = baxter_left_arm.inverse_kinematics(target_position, target_orientation, initial_position=ik, orientation_mode='all')

    position = baxter_left_arm.forward_kinematics(ik)[:3, 3]
    orientation = baxter_left_arm.forward_kinematics(ik)[:3, :3]

    # Check
    np.testing.assert_almost_equal(position, target_position, decimal=5)
    np.testing.assert_almost_equal(orientation, target_orientation, decimal=5)
