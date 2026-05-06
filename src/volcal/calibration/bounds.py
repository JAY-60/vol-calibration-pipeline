from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterBounds:
    """
    Lower and upper bounds for a parameter vector.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper bounds must have the same length.")

        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("each lower bound must be strictly less than its upper bound.")

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return bounds as NumPy arrays.
        """
        return np.array(self.lower, dtype=float), np.array(self.upper, dtype=float)


def heston_default_bounds() -> ParameterBounds:
    """
    Default parameter bounds for Heston calibration.

    Parameter order:
    kappa, theta, vol_of_vol, rho, v0
    """
    return ParameterBounds(
        lower=(1e-4, 1e-4, 1e-4, -0.999, 1e-4),
        upper=(10.0, 2.0, 5.0, 0.999, 2.0),
    )


def validate_parameter_vector(params: np.ndarray, bounds: ParameterBounds) -> None:
    """
    Validate that a parameter vector is inside the supplied bounds.
    """
    lower, upper = bounds.as_arrays()
    params = np.asarray(params, dtype=float)

    if params.shape != lower.shape:
        raise ValueError("parameter vector has the wrong shape.")

    if np.any(params < lower) or np.any(params > upper):
        raise ValueError("parameter vector is outside the allowed bounds.")