import numpy as np
import pytest

from volcal.calibration.bounds import (
    ParameterBounds,
    heston_default_bounds,
    validate_parameter_vector,
)


def test_parameter_bounds_convert_to_arrays() -> None:
    bounds = ParameterBounds(lower=(0.0, 1.0), upper=(2.0, 3.0))

    lower, upper = bounds.as_arrays()

    assert np.allclose(lower, np.array([0.0, 1.0]))
    assert np.allclose(upper, np.array([2.0, 3.0]))


def test_parameter_bounds_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ParameterBounds(lower=(0.0,), upper=(1.0, 2.0))


def test_parameter_bounds_reject_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="strictly less"):
        ParameterBounds(lower=(1.0,), upper=(1.0,))


def test_heston_default_bounds_have_five_parameters() -> None:
    bounds = heston_default_bounds()

    lower, upper = bounds.as_arrays()

    assert lower.shape == (5,)
    assert upper.shape == (5,)


def test_valid_heston_parameter_vector_passes() -> None:
    bounds = heston_default_bounds()
    params = np.array([2.0, 0.04, 0.5, -0.7, 0.04])

    validate_parameter_vector(params, bounds)


def test_wrong_shape_parameter_vector_raises_error() -> None:
    bounds = heston_default_bounds()
    params = np.array([2.0, 0.04])

    with pytest.raises(ValueError, match="wrong shape"):
        validate_parameter_vector(params, bounds)


def test_out_of_bounds_parameter_vector_raises_error() -> None:
    bounds = heston_default_bounds()
    params = np.array([2.0, 0.04, 0.5, -1.5, 0.04])

    with pytest.raises(ValueError, match="outside the allowed bounds"):
        validate_parameter_vector(params, bounds)