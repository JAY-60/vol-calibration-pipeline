import numpy as np
import pytest

from volcal.calibration.bounds import ParameterBounds
from volcal.calibration.optimizers import run_least_squares


def test_run_least_squares_recovers_scalar_parameter() -> None:
    bounds = ParameterBounds(lower=(0.0,), upper=(5.0,))

    def objective(params: np.ndarray) -> np.ndarray:
        return np.array([params[0] - 2.0])

    result = run_least_squares(
        objective=objective,
        initial_guess=np.array([0.5]),
        bounds=bounds,
    )

    assert result.success
    assert result.params[0] == pytest.approx(2.0, abs=1e-6)
    assert result.sum_squared_error == pytest.approx(0.0, abs=1e-10)


def test_run_least_squares_respects_upper_bound() -> None:
    bounds = ParameterBounds(lower=(0.0,), upper=(5.0,))

    def objective(params: np.ndarray) -> np.ndarray:
        return np.array([params[0] - 10.0])

    result = run_least_squares(
        objective=objective,
        initial_guess=np.array([1.0]),
        bounds=bounds,
    )

    assert result.success
    assert result.params[0] <= 5.0


def test_run_least_squares_rejects_wrong_initial_guess_shape() -> None:
    bounds = ParameterBounds(lower=(0.0, 0.0), upper=(5.0, 5.0))

    def objective(params: np.ndarray) -> np.ndarray:
        return params

    with pytest.raises(ValueError, match="wrong shape"):
        run_least_squares(
            objective=objective,
            initial_guess=np.array([1.0]),
            bounds=bounds,
        )


def test_run_least_squares_rejects_initial_guess_outside_bounds() -> None:
    bounds = ParameterBounds(lower=(0.0,), upper=(5.0,))

    def objective(params: np.ndarray) -> np.ndarray:
        return params

    with pytest.raises(ValueError, match="inside the supplied bounds"):
        run_least_squares(
            objective=objective,
            initial_guess=np.array([10.0]),
            bounds=bounds,
        )


def test_run_least_squares_rejects_non_positive_max_evaluations() -> None:
    bounds = ParameterBounds(lower=(0.0,), upper=(5.0,))

    def objective(params: np.ndarray) -> np.ndarray:
        return params

    with pytest.raises(ValueError, match="max_nfev must be positive"):
        run_least_squares(
            objective=objective,
            initial_guess=np.array([1.0]),
            bounds=bounds,
            max_nfev=0,
        )