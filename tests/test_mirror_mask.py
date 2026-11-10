import warnings

import numpy as np
import pytest

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def link(name, joint_type="revolute", bounds=None, axis=(0, 0, 1)):
    kwargs = {"translation": axis} if joint_type == "prismatic" else {"rotation": axis}
    if joint_type == "fixed":
        kwargs = {}
    return URDFLink(name=name, origin_translation=[0, 0, 0.1], origin_orientation=[0, 0, 0],
                    joint_type=joint_type, bounds=bounds, use_symbolic_matrix=False, **kwargs)


def arm(mask, bounds=(-2, 2)):
    """A planar arm: a held base rotation, then three active revolute joints, then the tip"""
    links = [OriginLink(),
             link("base", bounds=(-1, 1)),
             URDFLink(name="shoulder", origin_translation=[0, 0, 0.1], origin_orientation=[0, 0, 0],
                      rotation=[0, 1, 0], bounds=bounds, use_symbolic_matrix=False),
             URDFLink(name="elbow", origin_translation=[0, 0, 0.5], origin_orientation=[0, 0, 0],
                      rotation=[0, 1, 0], bounds=bounds, use_symbolic_matrix=False),
             URDFLink(name="wrist", origin_translation=[0, 0, 0.5], origin_orientation=[0, 0, 0],
                      rotation=[0, 1, 0], bounds=bounds, use_symbolic_matrix=False),
             link("tip", joint_type="fixed")]
    return Chain(links, active_links_mask=mask)


def single_joint(joint_type, bounds):
    return Chain([OriginLink(), link("joint", joint_type=joint_type, bounds=bounds,
                                     axis=(1, 0, 0) if joint_type == "prismatic" else (0, 0, 1)),
                  link("tip", joint_type="fixed")], active_links_mask=[0, True, 0])


MASK = [0, 0.3, True, True, True, 0]


def test_mask_holds_links_at_their_value():
    chain = arm(MASK)
    np.testing.assert_array_equal(chain.active_links_mask, [False, False, True, True, True, False])
    np.testing.assert_array_equal(chain.fixed_values, [0, 0.3, np.nan, np.nan, np.nan, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        arm(MASK)


def test_inverse_kinematics_uses_the_held_values():
    chain = arm(MASK)
    target = chain.forward_kinematics([0, 0.3, 0.4, 0.6, -0.2, 0])[:3, 3]
    for start in (None, [False, False, 0.1, 0.1, 0.1, False], 5, {"points": [[False, False, 0, 0.5, 0, False]], "n": 3, "seed": 0}):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            joints = chain.inverse_kinematics(target, initial_position=start)
        assert joints[1] == 0.3
        np.testing.assert_allclose(chain.forward_kinematics(joints)[:3, 3], target, atol=1e-4)


def test_multistart_and_trajectory_chain_their_starts_without_warning():
    chain = arm(MASK)
    target = chain.forward_kinematics([0, 0.3, 0.4, 0.6, -0.2, 0])[:3, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trajectory = chain.inverse_kinematics_trajectory([target, target], initial_position=[False, False, 0, 0, 0, False])
        chain.inverse_kinematics_trajectory([target, target], initial_position=4)
    np.testing.assert_array_equal(trajectory[:, 1], 0.3)


@pytest.mark.parametrize("value", [0.9, 0.3])
def test_a_number_for_a_held_link_is_overridden_with_a_warning(value):
    chain = arm(MASK)
    with pytest.warns(UserWarning, match="base"):
        start = chain._resolve_initial_position([False, value, 0.1, 0.1, 0.1, False])
    assert start[1] == 0.3
    with pytest.warns(UserWarning, match="base"):
        chain._resolve_initial_position({"points": [[False, value, 0.1, 0.1, 0.1, False]]})


def test_mirror_errors():
    chain = arm(MASK)
    with pytest.raises(ValueError, match="shoulder"):
        chain._resolve_initial_position([False, False, False, 0.1, 0.1, False])
    with pytest.raises(ValueError, match="not True"):
        chain._resolve_initial_position([True, True, 0.1, 0.1, 0.1, True])
    with pytest.raises(ValueError, match="shape"):
        chain._resolve_initial_position([False, 0.1])


def test_mask_errors():
    with pytest.raises(ValueError, match="no active link"):
        arm([0, 1, 1, 1, 1, 0])
    with pytest.raises(ValueError, match="True for an active link"):
        arm([0, "a", True, True, True, 0])
    with pytest.warns(UserWarning, match="outside of its bounds"):
        arm([0, 5, True, True, True, 0])


def test_false_in_the_mask_is_deprecated_and_reads_the_start():
    with pytest.warns(FutureWarning, match="base"):
        chain = arm([False, False, True, True, True, False])
    target = chain.forward_kinematics([0, 0.3, 0.4, 0.6, -0.2, 0])[:3, 3]
    joints = chain.inverse_kinematics(target, initial_position=[0, 0.3, 0, 0, 0, 0])
    assert joints[1] == 0.3


def test_concat_and_json_keep_the_held_values():
    chain = arm(MASK)
    assert chain.links_mask == [0.0, 0.3, True, True, True, 0.0]
    concatenated = Chain.concat(chain, arm([0, -0.2, True, True, True, 0]))
    np.testing.assert_array_equal(concatenated.fixed_values[[1, 7]], [0.3, -0.2])


@pytest.mark.parametrize("bounds,low,high", [((-1, 2), -1, 2), (None, -np.pi, np.pi),
                                             ((10, np.inf), 10, 10 + 2 * np.pi), ((-np.inf, -10), -10 - 2 * np.pi, -10)])
def test_revolute_links_are_drawn_within_their_bounds_or_a_full_turn(bounds, low, high):
    samples = single_joint("revolute", bounds).sample_initial_positions(200, seed=0)[:, 1]
    assert np.all((low <= samples) & (samples <= high))
    assert np.ptp(samples) > 0.8 * (high - low)


@pytest.mark.parametrize("bounds", [(10, np.inf), (-np.inf, -10), None])
def test_an_active_prismatic_link_needs_bounds(bounds):
    with pytest.raises(ValueError, match="joint"):
        single_joint("prismatic", bounds)
    single_joint("prismatic", (-1, 1)).sample_initial_positions(5, seed=0)


def test_the_default_mask_holds_the_fixed_and_the_unbounded_prismatic_links():
    links = [OriginLink(), link("slide", joint_type="prismatic", bounds=(10, np.inf), axis=(1, 0, 0)),
             link("bounded", joint_type="prismatic", bounds=(-1, 1), axis=(0, 1, 0)), link("turn", bounds=None),
             link("tip", joint_type="fixed")]
    with pytest.warns(UserWarning, match="slide"):
        chain = Chain(links)
    assert chain.links_mask == [0.0, 10.0, True, True, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Chain(links, active_links_mask=chain.links_mask)
        assert Chain(links[2:4]).links_mask == [True, True]


def test_drawing_checks_the_bounds_changed_after_the_creation():
    chain = single_joint("prismatic", (-1, 1))
    chain.links[1].bounds = (-1, np.inf)
    with pytest.raises(ValueError, match="joint"):
        chain.sample_initial_positions(5, seed=0)


def test_sampling_is_reproducible_and_keeps_the_held_values():
    chain = arm(MASK)
    samples = chain.sample_initial_positions(30, seed=3)
    np.testing.assert_array_equal(samples, chain.sample_initial_positions(30, seed=3))
    np.testing.assert_array_equal(samples[:, [0, 1, 5]], np.tile([0, 0.3, 0], (30, 1)))
    assert np.all(np.abs(samples[:, 2:5]) <= 2)


def test_a_held_prismatic_link_without_a_bound_does_not_prevent_drawing():
    chain = Chain([OriginLink(), link("slide", joint_type="prismatic", bounds=(10, np.inf), axis=(1, 0, 0)),
                   link("turn", bounds=None), link("tip", joint_type="fixed")], active_links_mask=[0, 11.0, True, 0])
    samples = chain.sample_initial_positions(5, seed=0)
    np.testing.assert_array_equal(samples[:, 1], 11.0)


def test_around_and_radius_are_deprecated_and_stay_in_the_bounds():
    chain = single_joint("revolute", (10, np.inf))
    with pytest.warns(FutureWarning, match="deprecated"):
        samples = chain.sample_initial_positions(20, seed=0, around=[0, 0, 0], radius=1)
    assert np.all((10 <= samples[:, 1]) & (samples[:, 1] <= 11))
    with pytest.warns(FutureWarning, match="deprecated"):
        chain._resolve_initial_position({"n": 2, "around": [0, 0, 0]})
