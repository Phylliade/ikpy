import numpy as np
import pytest

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def prismatic_chain(bounds):
    return Chain([
        OriginLink(),
        URDFLink(name="slide", origin_translation=[0, 0, 0],
                 origin_orientation=[0, 0, 0], translation=[1, 0, 0],
                 joint_type="prismatic", bounds=bounds, use_symbolic_matrix=False),
        URDFLink(name="tip", origin_translation=[0, 0, 0],
                 origin_orientation=[0, 0, 0], joint_type="fixed",
                 use_symbolic_matrix=False),
    ], active_links_mask=[False, True, False])


@pytest.mark.parametrize("bounds", [(10, np.inf), (-np.inf, -10)])
@pytest.mark.parametrize("center", [None, 0, -20, 20])
def test_sampling_with_one_sided_bounds(bounds, center):
    robot = prismatic_chain(bounds)
    around = None if center is None else [0.5, center, 0.7]
    samples = robot.sample_initial_positions(20, seed=2, around=around)
    assert samples.shape == (20, 3)
    assert np.all(np.isfinite(samples))
    assert np.all(samples[:, 1] >= bounds[0])
    assert np.all(samples[:, 1] <= bounds[1])
    assert np.ptp(samples[:, 1]) > 0
    np.testing.assert_array_equal(samples, robot.sample_initial_positions(20, seed=2, around=around))
    np.testing.assert_array_equal(samples[:, 0], 0 if around is None else 0.5)
    np.testing.assert_array_equal(samples[:, 2], 0 if around is None else 0.7)
    if around is not None:
        assert around == [0.5, center, 0.7]


@pytest.mark.parametrize("bounds,target", [((10, np.inf), 12), ((-np.inf, -10), -12)])
def test_multistart_with_one_sided_bounds(bounds, target):
    robot = prismatic_chain(bounds)
    joints = robot.inverse_kinematics([target, 0, 0], initial_position=3)
    np.testing.assert_allclose(robot.forward_kinematics(joints)[:3, 3], [target, 0, 0], atol=1e-6)
    assert bounds[0] <= joints[1] <= bounds[1]
