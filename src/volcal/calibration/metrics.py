from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_non_empty(values: Sequence[float], name: str) -> None:
    """
    Ensure a sequence is not empty before computing a metric.
    """
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_same_length(actual: Sequence[float], predicted: Sequence[float]) -> None:
    """
    Ensure actual and predicted sequences have the same length.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length.")


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Mean absolute error between actual and predicted values.
    """
    _validate_non_empty(actual, "actual")
    _validate_non_empty(predicted, "predicted")
    _validate_same_length(actual, predicted)

    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return sum(absolute_errors) / len(absolute_errors)


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """
    Root mean squared error between actual and predicted values.
    """
    _validate_non_empty(actual, "actual")
    _validate_non_empty(predicted, "predicted")
    _validate_same_length(actual, predicted)

    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))