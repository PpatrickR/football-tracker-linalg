import numpy as np
import pytest

from football_tracker.registration import FieldRegistration


def test_identity_homography_roundtrip():
    pixel = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    field = pixel.copy()
    reg = FieldRegistration.from_correspondences(pixel, field)
    out = reg.pixel_to_field(pixel)
    assert np.allclose(out, field, atol=1e-6)
    assert np.allclose(reg.field_to_pixel(out), pixel, atol=1e-6)


def test_known_scale_and_offset():
    """Pixels at (0,0)..(1000,500) map to field (10,0)..(110,53.333)."""
    pixel = np.array(
        [[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64
    )
    field = np.array(
        [[10, 0], [110, 0], [110, 53.333], [10, 53.333]], dtype=np.float64
    )
    reg = FieldRegistration.from_correspondences(pixel, field)
    midpoint = reg.pixel_to_field(np.array([[500, 250]]))[0]
    assert abs(midpoint[0] - 60.0) < 1e-6
    assert abs(midpoint[1] - 53.333 / 2) < 1e-6


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        FieldRegistration.from_correspondences(
            np.array([[0, 0], [1, 0], [1, 1]]),
            np.array([[0, 0], [1, 0], [1, 1]]),
        )
