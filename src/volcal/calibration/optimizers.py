from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from volcal.calibration.bounds import ParameterBounds


@dataclass(frozen=True)
class LeastSquaresResult:
    """
    Result returned by the least-squares calibration wrapper.
    """

    params: np.ndarray
    residuals: np.ndarray
    sum_squared_error: float
    success: bool
    message: str
    nfev: int


def _as_1d_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """
    Convert input values into a one-dimensional NumPy float array.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _validate_initial_guess(initial_guess: np.ndarray, bounds: ParameterBounds) -> None:
    """
    Check that the initial guess is compatible with the supplied bounds.
    """
    lower, upper = bounds.as_arrays()

    if initial_guess.shape != lower.shape:
        raise ValueError("initial_guess has the wrong shape.")

    if np.any(initial_guess < lower) or np.any(initial_guess > upper):
        raise ValueError("initial_guess must lie inside the supplied bounds.")


def run_least_squares(
    objective: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    initial_guess: Sequence[float] | np.ndarray,
    bounds: ParameterBounds,
    max_nfev: int = 500,
) -> LeastSquaresResult:
    """
    Run bounded nonlinear least-squares optimisation.

    The objective function should map a parameter vector to a residual vector.
    """
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive.")

    x0 = _as_1d_float_array(initial_guess, "initial_guess")
    _validate_initial_guess(x0, bounds)

    lower, upper = bounds.as_arrays()

    def residual_wrapper(params: np.ndarray) -> np.ndarray:
        residuals = objective(params)
        return _as_1d_float_array(residuals, "residuals")

    result = least_squares(
        fun=residual_wrapper,
        x0=x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
    )

    residuals = _as_1d_float_array(result.fun, "residuals")

    return LeastSquaresResult(
        params=np.asarray(result.x, dtype=float),
        residuals=residuals,
        sum_squared_error=float(np.sum(residuals**2)),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )