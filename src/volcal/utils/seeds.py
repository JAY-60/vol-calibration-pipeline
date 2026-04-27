from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python's random module and NumPy.
    """
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)