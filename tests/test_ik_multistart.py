import numpy as np
import pytest

# ikpy imports
from ikpy import chain
from ikpy import inverse_kinematics as ik
from ikpy.link import OriginLink, URDFLink


def planar_2r():
    """A planar two-link arm: every reachable point has an elbow-up and an elbow-down solution"""
    links = [
        OriginLink(),
        URDFLink(name="shoulder", origin_translation=[0, 0, 0], origin_orientation=[0, 0, 0], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), use_symbolic_matrix=False),
        URDFLink(name="elbow", origin_translation=[1, 0, 0], origin_orientation=[0, 0, 0], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), use_symbolic_matrix=False),
        URDFLink(name="tip", origin_translation=[0.8, 0, 0], origin_orientation=[0, 0, 0], use_symbolic_matrix=False, joint_type="fixed"),
    ]
    return chain.Chain(links, active_links_mask=[False, True, True, False])


def test_sample_initial_positions():
    my_chain = planar_2r()
    starts = my_chain.sample_initial_positions(5, seed=1)
    assert starts.shape == (5, 4)
    # Inactive links stay at zero, active ones are inside the bounds
    np.testing.assert_array_equal(starts[:, 0], 0)
    np.testing.assert_array_equal(starts[:, 3], 0)
    assert np.all(np.abs(starts[:, 1:3]) <= np.pi)
    assert not np.allclose(starts[0], starts[1])
    # Reproducible
    np.testing.assert_array_equal(starts, my_chain.sample_initial_positions(5, seed=1))
    with pytest.raises(ValueError):
        my_chain.sample_initial_positions(0)
    with pytest.raises(ValueError):
        my_chain.sample_initial_positions(3, radius=0.1)


def test_sample_initial_positions_around_a_configuration():
    my_chain = planar_2r()
    around = [0.5, 3.0, -2.0, 0.7]
    starts = my_chain.sample_initial_positions(50, seed=2, around=around, radius=0.3)
    # Inactive links take the values of the configuration
    np.testing.assert_array_equal(starts[:, 0], 0.5)
    np.testing.assert_array_equal(starts[:, 3], 0.7)
    # Active links stay within the radius, clipped to the bounds
    assert np.all(np.abs(starts[:, 2] + 2.0) <= 0.3)
    assert np.all(starts[:, 1] >= 2.7) and np.all(starts[:, 1] <= np.pi)


def test_sample_initial_positions_without_bounds(resources_path):
    my_chain = chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")
    assert any(not np.isfinite(link.bounds[0]) for link in my_chain.links)
    starts = my_chain.sample_initial_positions(20, seed=3)
    assert np.all(np.abs(starts) <= np.pi + 1e-12)
    around = np.full(len(my_chain.links), 2.0)
    starts = my_chain.sample_initial_positions(20, seed=3, around=around)
    assert np.all(np.abs(starts - around) <= np.pi + 1e-12)


def test_target_error_matches_forward_kinematics():
    my_chain = planar_2r()
    joints = [0, 0.4, -0.9, 0]
    fk = my_chain.forward_kinematics(joints)
    target = np.eye(4)
    target[:3, 3] = [1.2, 0.3, 0]
    np.testing.assert_allclose(ik.target_error(my_chain, joints, target), fk[:3, 3] - target[:3, 3])
    error = ik.target_error(my_chain, joints, target, orientation_mode="all")
    assert error.shape == (12,)
    np.testing.assert_allclose(error[3:], (fk[:3, :3] - np.eye(3)).ravel())
    np.testing.assert_allclose(ik.target_error(my_chain, joints, target, orientation_mode="X", no_position=True), fk[:3, 0] - target[:3, 0])
    with pytest.raises(ValueError):
        ik.target_error(my_chain, joints, target, no_position=True)


def test_solutions_finds_both_elbow_configurations():
    my_chain = planar_2r()
    target = [1.2, 0.5, 0]
    starts = my_chain.sample_initial_positions(12, seed=0)
    frame = my_chain._target_frame(target, None, None)[0]
    every_minimum = my_chain._inverse_kinematics_solutions_frame(frame, starts)
    solutions = my_chain._inverse_kinematics_solutions_frame(frame, starts, max_error=1e-3)
    assert solutions.shape[1] == 4
    assert len(solutions) == 2
    for solution in solutions:
        np.testing.assert_allclose(my_chain.forward_kinematics(solution)[:3, 3], target, atol=1e-5)
    # Elbow up and elbow down: opposite elbow angles
    np.testing.assert_allclose(solutions[0][2], -solutions[1][2], atol=1e-4)
    # Without max_error, the starting points stuck in a local minimum are reported too, after the good ones
    assert len(every_minimum) >= 2
    np.testing.assert_array_equal(every_minimum[:2], solutions)
    errors = [np.linalg.norm(ik.target_error(my_chain, s, frame)) for s in every_minimum]
    assert errors == sorted(errors)
    # The public call returns the best one
    np.testing.assert_array_equal(my_chain.inverse_kinematics(target, initial_position=starts), every_minimum[0])


def test_initial_position_forms():
    my_chain = planar_2r()
    target = [1.2, 0.5, 0]
    frame = my_chain._target_frame(target, None, None)[0]
    reached = lambda q: np.allclose(my_chain.forward_kinematics(q)[:3, 3], target, atol=1e-5)  # noqa: E731
    # A configuration: the single start, as before
    assert reached(my_chain.inverse_kinematics(target, initial_position=[0, 0.3, 0.3, 0]))
    # An int: that many points drawn anywhere in the limits
    assert reached(my_chain.inverse_kinematics(target, initial_position=6))
    # A dict: given points plus drawn ones
    spec = {"points": [[0, 0.3, 0.3, 0], [0, -0.3, -0.3, 0]], "n": 4, "seed": 0, "around": [0, 0.1, 0.2, 0], "radius": 3.0}
    from_dict = my_chain.inverse_kinematics(target, initial_position=spec)
    expected_starts = np.concatenate([spec["points"], my_chain.sample_initial_positions(4, seed=0, around=[0, 0.1, 0.2, 0], radius=3.0)])
    np.testing.assert_array_equal(from_dict, my_chain._inverse_kinematics_solutions_frame(frame, expected_starts)[0])
    # Only given points: both elbow configurations, one per point
    both = my_chain._inverse_kinematics_solutions_frame(frame, {"points": spec["points"]}, max_error=1e-3)
    assert len(both) == 2
    # An array of starting points
    assert reached(my_chain.inverse_kinematics(target, initial_position=my_chain.sample_initial_positions(5, seed=1)))
    # The expert function accepts a single configuration too, and defaults to 10 drawn points
    assert my_chain._inverse_kinematics_solutions_frame(frame, [0, 0.3, 0.3, 0]).shape == (1, 4)
    assert reached(my_chain._inverse_kinematics_solutions_frame(frame)[0])
    # Validation
    with pytest.raises(ValueError, match="Unknown keys"):
        my_chain.inverse_kinematics(target, initial_position={"n": 3, "sigma": 1})
    with pytest.raises(ValueError, match="no starting point"):
        my_chain.inverse_kinematics(target, initial_position={"seed": 3})
    with pytest.raises(ValueError, match="points"):
        my_chain.inverse_kinematics(target, initial_position={"points": [0, 0.3, 0.3, 0]})
    with pytest.raises(ValueError, match="initial_position must be"):
        my_chain.inverse_kinematics(target, initial_position=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="initial_position must be"):
        my_chain.inverse_kinematics(target, initial_position=np.zeros(3))
    with pytest.raises(ValueError):
        my_chain.inverse_kinematics(target, initial_position=0)


def test_unreachable_target_gives_the_best_approximation():
    my_chain = planar_2r()
    frame = my_chain._target_frame([5, 0, 0], None, None)[0]
    starts = my_chain.sample_initial_positions(4, seed=0)
    assert my_chain._inverse_kinematics_solutions_frame(frame, starts, max_error=1e-3).shape == (0, 4)
    best = my_chain.inverse_kinematics([5, 0, 0], initial_position=starts)
    np.testing.assert_allclose(my_chain.forward_kinematics(best)[:3, 3], [1.8, 0, 0], atol=1e-3)


def test_multistart_near_a_bad_configuration(resources_path):
    """Full-frame IK on a 6-DOF arm, from a deliberately bad initial position: starts drawn around it reach the
    frame when the single start does not"""
    my_chain = chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")
    joints = [0, 0.8, -0.6, 0.9, 0.5, -0.7, 0.2, 0][:len(my_chain.links)]
    joints = joints + [0] * (len(my_chain.links) - len(joints))
    target = my_chain.forward_kinematics(joints)
    bad_initial = [0, -0.8, 0.6, -0.9, -0.5, 0.7, -0.2, 0][:len(my_chain.links)]
    bad_initial = bad_initial + [0] * (len(my_chain.links) - len(bad_initial))

    single = my_chain.inverse_kinematics_frame(target, initial_position=bad_initial, orientation_mode="all")
    multi = my_chain.inverse_kinematics_frame(target, initial_position={"points": [bad_initial], "n": 7, "seed": 0, "around": bad_initial, "radius": 2.0}, orientation_mode="all")
    error = lambda q: np.linalg.norm(ik.target_error(my_chain, q, target, "all"))  # noqa: E731
    assert error(multi) <= error(single) + 1e-9
    assert error(multi) < 1e-3
