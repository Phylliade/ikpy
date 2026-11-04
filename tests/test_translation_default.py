import numpy as np
import pytest
from ikpy.utils.geometry import to_transformation_matrix


@pytest.mark.parametrize(
    "translation", [[0, 0, 0], [1, -2, 3], np.array([0.25, 4.0, -8.0])]
)
def test_translation_only_preserves_points_and_axes(translation):
    frame = to_transformation_matrix(translation)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-3.0, 1.0, 2.0]])
    mapped = (frame @ np.column_stack((points, np.ones(3))).T).T
    np.testing.assert_allclose(mapped[:, :3], points + translation)
    np.testing.assert_allclose(mapped[:, 3], 1)
    assert np.linalg.det(frame) == pytest.approx(1)


def test_explicit_orientation_is_preserved():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame = to_transformation_matrix([1, 2, 3], rotation)
    np.testing.assert_array_equal(frame[:3, :3], rotation)
    np.testing.assert_array_equal(frame[:3, 3], [1, 2, 3])
