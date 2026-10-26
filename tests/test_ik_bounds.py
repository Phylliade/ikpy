import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy import inverse_kinematics as ik


XY_TABLE_ARM = "../resources/prismatic/xy_table_arm.urdf"


def test_strictly_inside_bounds():
    bounds = np.array([[-np.inf, np.inf], [0, 3], [0, 3], [-1.25, 1.25], [0, 0.001]])
    x0 = ik.strictly_inside_bounds([0, 0, 3, 0, 0], bounds)
    np.testing.assert_allclose(x0, [0, 1e-3, 3 - 1e-3, 0, 0.00025])
    # Inside the bounds, by more than the margin: untouched
    np.testing.assert_array_equal(ik.strictly_inside_bounds([5, 1, 2, -1, 0.0005], bounds), [5, 1, 2, -1, 0.0005])


def test_outside_initial_position_is_clipped_with_a_warning():
    bounds = np.array([[0, 3], [-1.25, 1.25]])
    with pytest.warns(UserWarning, match="outside the bounds of the links \\[0, 1\\]"):
        x0 = ik.strictly_inside_bounds([-2, 7], bounds)
    np.testing.assert_allclose(x0, [1e-3, 1.25 - 1e-3])


@pytest.mark.parametrize("optimizer", ["least_squares", "scalar"])
def test_ik_from_the_default_position_with_a_joint_on_its_lower_bound(optimizer):
    """Issue #142: an XY table (prismatic joints with lower="0") carrying a two-link arm. Starting from the default
    initial position, every joint at zero, the optimizer used to stop immediately on the starting point"""
    my_chain = chain.Chain.from_urdf_file(XY_TABLE_ARM, last_link_vector=[0, 0, 0.5])
    for target in ([1, 1.5, 1.5], [0.5, 0.5, 1.1], [2.5, 0.2, 0.9]):
        joints = my_chain.inverse_kinematics(target, optimizer=optimizer)
        np.testing.assert_allclose(my_chain.forward_kinematics(joints)[:3, 3], target, atol=1e-4)
        lower, upper = np.array([link.bounds for link in my_chain.links]).T
        assert np.all(joints >= lower) and np.all(joints <= upper)


def test_ik_with_an_initial_position_outside_the_bounds():
    my_chain = chain.Chain.from_urdf_file(XY_TABLE_ARM, last_link_vector=[0, 0, 0.5])
    target = [1, 1.5, 1.5]
    with pytest.warns(UserWarning, match="outside the bounds"):
        joints = my_chain.inverse_kinematics(target, initial_position=[0, -1, 5, 0, 0, 0, 0])
    np.testing.assert_allclose(my_chain.forward_kinematics(joints)[:3, 3], target, atol=1e-4)
