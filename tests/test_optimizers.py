import numpy as np
import pytest

from volcal.calibration.objective import make_iv_objective
from volcal.calibration.optimizers import (
    run_differential_evolution_then_least_squares,
    run_least_squares,
)


def test_run_least_squares_recovers_flat_vol_parameter() -> None:
    target_ivs = np.array([0.25, 0.25, 0.25])

    def flat_vol_model(params: np.ndarray) -> np.ndarray:
        sigma = params[0]
        return np.full_like(target_ivs, fill_value=sigma, dtype=float)

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=flat_vol_model,
    )

    result = run_least_squares(
        objective=objective,
        initial_guess=np.array([0.10]),
        lower_bounds=np.array([0.01]),
        upper_bounds=np.array([2.00]),
    )

    assert result.success
    assert result.params[0] == pytest.approx(0.25, abs=1e-6)
    assert np.linalg.norm(result.residuals) < 1e-6


def test_global_then_local_recovers_flat_vol_parameter() -> None:
    target_ivs = np.array([0.35, 0.35, 0.35])

    def flat_vol_model(params: np.ndarray) -> np.ndarray:
        sigma = params[0]
        return np.full_like(target_ivs, fill_value=sigma, dtype=float)

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=flat_vol_model,
    )

    result = run_differential_evolution_then_least_squares(
        objective=objective,
        lower_bounds=np.array([0.01]),
        upper_bounds=np.array([2.00]),
        maxiter=20,
        seed=123,
    )

    assert result.success
    assert result.params[0] == pytest.approx(0.35, abs=1e-5)
    assert np.linalg.norm(result.residuals) < 1e-5


def test_least_squares_rejects_initial_guess_outside_bounds() -> None:
    target_ivs = np.array([0.25, 0.25, 0.25])

    def flat_vol_model(params: np.ndarray) -> np.ndarray:
        return np.full_like(target_ivs, fill_value=params[0], dtype=float)

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=flat_vol_model,
    )

    with pytest.raises(ValueError, match="Initial guess must lie within the bounds"):
        run_least_squares(
            objective=objective,
            initial_guess=np.array([3.00]),
            lower_bounds=np.array([0.01]),
            upper_bounds=np.array([2.00]),
        )


def test_optimizers_reject_invalid_bounds() -> None:
    target_ivs = np.array([0.25, 0.25, 0.25])

    def flat_vol_model(params: np.ndarray) -> np.ndarray:
        return np.full_like(target_ivs, fill_value=params[0], dtype=float)

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=flat_vol_model,
    )

    with pytest.raises(ValueError, match="strictly less"):
        run_least_squares(
            objective=objective,
            initial_guess=np.array([0.25]),
            lower_bounds=np.array([1.00]),
            upper_bounds=np.array([1.00]),
        )