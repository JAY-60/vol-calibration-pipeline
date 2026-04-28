import numpy as np
import pytest

from volcal.calibration.objective import (
    calibration_error_summary,
    iv_residuals,
    make_iv_objective,
)


def test_iv_residuals_returns_model_minus_target() -> None:
    model_ivs = [0.20, 0.25, 0.30]
    target_ivs = [0.19, 0.26, 0.31]

    residuals = iv_residuals(model_ivs, target_ivs)

    expected = np.array([0.01, -0.01, -0.01])
    assert np.allclose(residuals, expected)


def test_calibration_error_summary_basic_case() -> None:
    target_ivs = [0.20, 0.25, 0.30]
    model_ivs = [0.21, 0.24, 0.33]

    summary = calibration_error_summary(model_ivs=model_ivs, target_ivs=target_ivs)

    expected_residuals = np.array([0.01, -0.01, 0.03])
    expected_mae = np.mean(np.abs(expected_residuals))
    expected_rmse = np.sqrt(np.mean(expected_residuals**2))

    assert summary.mae == pytest.approx(expected_mae)
    assert summary.rmse == pytest.approx(expected_rmse)
    assert summary.max_abs_error == pytest.approx(0.03)
    assert summary.n_points == 3


def test_iv_residuals_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        iv_residuals(model_ivs=[0.20, 0.25], target_ivs=[0.20])


def test_iv_residuals_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        iv_residuals(model_ivs=[], target_ivs=[])


def test_iv_residuals_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite values"):
        iv_residuals(model_ivs=[0.20, np.nan], target_ivs=[0.20, 0.25])


def test_make_iv_objective_returns_expected_residuals() -> None:
    target_ivs = np.array([0.20, 0.25, 0.30])

    def flat_vol_model(params: np.ndarray) -> np.ndarray:
        sigma = params[0]
        return np.full_like(target_ivs, fill_value=sigma, dtype=float)

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=flat_vol_model,
    )

    residuals = objective(np.array([0.25]))

    expected = np.array([0.05, 0.00, -0.05])
    assert np.allclose(residuals, expected)


def test_make_iv_objective_rejects_wrong_model_output_shape() -> None:
    target_ivs = np.array([0.20, 0.25, 0.30])

    def bad_model(params: np.ndarray) -> np.ndarray:
        return np.array([params[0], params[0]])

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=bad_model,
    )

    with pytest.raises(ValueError, match="same shape"):
        objective(np.array([0.25]))