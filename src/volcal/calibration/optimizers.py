from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares


@dataclass(frozen=True)
class OptimizerResult:
    """
    Lightweight container for calibration optimiser results.
    """

    params: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    success: bool
    message: str
    nfev: int


def _validate_bounds(
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> None:
    """
    Validate lower and upper parameter bounds.
    """
    if lower_bounds.ndim != 1 or upper_bounds.ndim != 1:
        raise ValueError("Bounds must be one-dimensional arrays.")

    if lower_bounds.shape != upper_bounds.shape:
        raise ValueError("Lower and upper bounds must have the same shape.")

    if not np.all(np.isfinite(lower_bounds)):
        raise ValueError("Lower bounds must be finite.")

    if not np.all(np.isfinite(upper_bounds)):
        raise ValueError("Upper bounds must be finite.")

    if not np.all(lower_bounds < upper_bounds):
        raise ValueError("Each lower bound must be strictly less than each upper bound.")


def _validate_initial_guess(
    initial_guess: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> None:
    """
    Validate that the initial guess is compatible with the bounds.
    """
    if initial_guess.ndim != 1:
        raise ValueError("Initial guess must be one-dimensional.")

    if initial_guess.shape != lower_bounds.shape:
        raise ValueError("Initial guess and bounds must have the same shape.")

    if not np.all(np.isfinite(initial_guess)):
        raise ValueError("Initial guess must contain only finite values.")

    if np.any(initial_guess < lower_bounds) or np.any(initial_guess > upper_bounds):
        raise ValueError("Initial guess must lie within the bounds.")


def run_least_squares(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial_guess: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    max_nfev: int = 500,
) -> OptimizerResult:
    """
    Run bounded nonlinear least-squares calibration.
    """
    initial_guess = np.asarray(initial_guess, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    _validate_bounds(lower_bounds, upper_bounds)
    _validate_initial_guess(initial_guess, lower_bounds, upper_bounds)

    result = least_squares(
        fun=objective,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=max_nfev,
    )

    residuals = np.asarray(result.fun, dtype=float)

    return OptimizerResult(
        params=np.asarray(result.x, dtype=float),
        residuals=residuals,
        cost=float(2.0 * result.cost),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def run_differential_evolution_then_least_squares(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    maxiter: int = 100,
    seed: int = 42,
    polish_with_least_squares: bool = True,
) -> OptimizerResult:
    """
    Run global differential evolution, optionally followed by local least-squares polishing.
    """
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    _validate_bounds(lower_bounds, upper_bounds)

    bounds = list(zip(lower_bounds, upper_bounds))

    def scalar_loss(params: NDArray[np.float64]) -> float:
        residuals = np.asarray(objective(np.asarray(params, dtype=float)), dtype=float)
        if residuals.ndim != 1:
            raise ValueError("Objective must return a one-dimensional residual vector.")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("Objective residuals must be finite.")
        return float(np.sum(residuals**2))

    global_result = differential_evolution(
        func=scalar_loss,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        polish=False,
    )

    if polish_with_least_squares:
        return run_least_squares(
            objective=objective,
            initial_guess=np.asarray(global_result.x, dtype=float),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    residuals = np.asarray(objective(np.asarray(global_result.x, dtype=float)), dtype=float)

    return OptimizerResult(
        params=np.asarray(global_result.x, dtype=float),
        residuals=residuals,
        cost=float(np.sum(residuals**2)),
        success=bool(global_result.success),
        message=str(global_result.message),
        nfev=int(global_result.nfev),
    )