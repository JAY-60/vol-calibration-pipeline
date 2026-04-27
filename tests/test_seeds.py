import numpy as np
import pytest

from volcal.utils.seeds import set_global_seed


def test_numpy_random_numbers_are_reproducible() -> None:
    set_global_seed(42)
    first = np.random.normal(size=5)

    set_global_seed(42)
    second = np.random.normal(size=5)

    assert np.allclose(first, second)


def test_negative_seed_raises_error() -> None:
    with pytest.raises(ValueError, match="Seed must be non-negative"):
        set_global_seed(-1)