import numpy as np
import pytest
from ikpy.chain import Chain
from ikpy.link import URDFLink


def make_chain(prefix, mask):
    links = [
        URDFLink(
            name=prefix + str(i),
            origin_translation=[0, 0, 1],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            use_symbolic_matrix=False,
        )
        for i in range(len(mask))
    ]
    return Chain(links, active_links_mask=mask)


@pytest.mark.parametrize(
    "left,right",
    [
        ([True, False], [False, True]),
        ([True, False], [True, False, False]),
        ([False], [True, False]),
    ],
)
def test_concat_preserves_masks_and_forward_kinematics(left, right):
    first, second = make_chain("a", left), make_chain("b", right)
    first_mask, second_mask = (
        first.active_links_mask.copy(),
        second.active_links_mask.copy(),
    )
    combined = Chain.concat(first, second)
    expected = np.array(left + right, dtype=bool)
    np.testing.assert_array_equal(combined.active_links_mask, expected)
    assert combined.active_links_mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(
        combined.active_from_full(np.arange(len(expected))), np.flatnonzero(expected)
    )
    angles = np.linspace(0.1, 0.5, len(expected))
    np.testing.assert_allclose(
        combined.forward_kinematics(angles),
        first.forward_kinematics(angles[: len(left)])
        @ second.forward_kinematics(angles[len(left) :]),
    )
    combined.active_links_mask[0] = not combined.active_links_mask[0]
    np.testing.assert_array_equal(first.active_links_mask, first_mask)
    np.testing.assert_array_equal(second.active_links_mask, second_mask)
