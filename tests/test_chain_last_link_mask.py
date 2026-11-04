import warnings

import numpy as np
import pytest

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def make_links(last_joint_type="revolute"):
    links = [OriginLink()]
    for index, joint_type in enumerate(["revolute", last_joint_type]):
        links.append(URDFLink(
            name="joint_{}".format(index),
            origin_translation=[0, 0, 1],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1] if joint_type != "fixed" else None,
            joint_type=joint_type,
            use_symbolic_matrix=False,
        ))
    return links


def last_link_warnings(caught):
    return [w for w in caught if "The last link" in str(w.message)]


def test_explicit_active_last_link_warns_without_override():
    with pytest.warns(UserWarning, match="The last link joint_1"):
        chain = Chain(make_links(), active_links_mask=[False, True, True])
    np.testing.assert_array_equal(chain.active_links_mask, [False, True, True])


def test_default_mask_is_silent_and_keeps_last_link_active():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chain = Chain(make_links())
    assert last_link_warnings(caught) == []
    np.testing.assert_array_equal(chain.active_links_mask, [True, True, True])


def test_inactive_last_link_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Chain(make_links(), active_links_mask=[False, True, False])
    assert last_link_warnings(caught) == []


def test_fixed_last_link_only_gets_the_fixed_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Chain(make_links("fixed"), active_links_mask=[False, True, True])
    assert last_link_warnings(caught) == []
    assert any("is of type 'fixed'" in str(w.message) for w in caught)
