import numpy as np
import pytest

from ikpy.utils import geometry


@pytest.mark.parametrize("translation", [[0, 0, 0], [1.5, -2, 3]])
@pytest.mark.parametrize("angles", [[0, 0, 0], [0.3, -0.7, 1.2]])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_transformation_roundtrip(translation, angles, backend):
    if backend == "jax":
        pytest.importorskip("jax")
        from ikpy.utils import jax_geometry as implementation
    else:
        implementation = geometry
    rotation = geometry.rpy_matrix(*angles)
    frame = implementation.to_transformation_matrix(np.array(translation), rotation)
    position, orientation = implementation.from_transformation_matrix(frame)
    assert position.shape == (3,)
    assert orientation.shape == (3, 3)
    np.testing.assert_allclose(position, translation)
    np.testing.assert_allclose(orientation, rotation, atol=1e-6)
    np.testing.assert_allclose(
        implementation.to_transformation_matrix(position, orientation), frame
    )
