from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volcal.calibration.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


NumericVector = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationErrorSummary:
    """
    Summary statistics for calibration error.
    """

    rmse: float
    mae: float
    max_abs_error: float
    n_points: int


def _as_1d_float_array(values: NumericVector, name: str) -> NDArray[np.float64]:
    """
    Convert input values to a one-dimensional NumPy float array.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def iv_residuals(
    model_ivs: NumericVector,
    target_ivs: NumericVector,
) -> NDArray[np.float64]:
    """
    Compute implied-volatility residuals.

    Residual convention:
        residual = model implied volatility - target implied volatility
    """
    model = _as_1d_float_array(model_ivs, "model_ivs")
    target = _as_1d_float_array(target_ivs, "target_ivs")

    if model.shape != target.shape:
        raise ValueError("model_ivs and target_ivs must have the same shape.")

    return model - target


def calibration_error_summary(
    model_ivs: NumericVector,
    target_ivs: NumericVector,
) -> CalibrationErrorSummary:
    """
    Compute calibration error summary statistics.
    """
    model = _as_1d_float_array(model_ivs, "model_ivs")
    target = _as_1d_float_array(target_ivs, "target_ivs")

    residuals = iv_residuals(model, target)

    return CalibrationErrorSummary(
        rmse=root_mean_squared_error(actual=target, predicted=model),
        mae=mean_absolute_error(actual=target, predicted=model),
        max_abs_error=float(np.max(np.abs(residuals))),
        n_points=int(residuals.size),
    )


def make_iv_objective(
    target_ivs: NumericVector,
    model_iv_function: Callable[[NDArray[np.float64]], NumericVector],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Build an implied-volatility calibration objective.

    The returned function maps model parameters to residuals:

        params -> model_ivs(params) - target_ivs
    """
    target = _as_1d_float_array(target_ivs, "target_ivs")

    def objective(params: NDArray[np.float64]) -> NDArray[np.float64]:
        params_array = _as_1d_float_array(params, "params")
        model_ivs = _as_1d_float_array(
            model_iv_function(params_array),
            "model_ivs",
        )

        if model_ivs.shape != target.shape:
            raise ValueError("model_ivs and target_ivs must have the same shape.")

        return model_ivs - target

    return objective