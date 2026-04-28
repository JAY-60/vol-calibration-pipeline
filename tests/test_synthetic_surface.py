import numpy as np
import pytest

from volcal.data.synthetic_surface import (
    SyntheticSurfaceConfig,
    generate_synthetic_surface,
    surface_to_arrays,
    synthetic_iv,
)


def test_generate_synthetic_surface_has_expected_shape() -> None:
    config = SyntheticSurfaceConfig(
        maturities=(0.5, 1.0),
        moneyness=(0.9, 1.0, 1.1),
    )

    surface = generate_synthetic_surface(config)

    assert len(surface) == 6


def test_generate_synthetic_surface_has_required_columns() -> None:
    surface = generate_synthetic_surface()

    expected_columns = {
        "maturity",
        "moneyness",
        "strike",
        "target_iv",
        "target_call_price",
    }

    assert expected_columns.issubset(surface.columns)


def test_synthetic_surface_values_are_positive() -> None:
    surface = generate_synthetic_surface()

    assert (surface["maturity"] > 0.0).all()
    assert (surface["strike"] > 0.0).all()
    assert (surface["target_iv"] > 0.0).all()
    assert (surface["target_call_price"] > 0.0).all()


def test_synthetic_iv_is_positive() -> None:
    config = SyntheticSurfaceConfig()

    iv = synthetic_iv(moneyness=1.0, maturity=1.0, config=config)

    assert iv > 0.0


def test_noise_is_reproducible_with_same_seed() -> None:
    config = SyntheticSurfaceConfig(noise_std=0.01, seed=123)

    surface_1 = generate_synthetic_surface(config)
    surface_2 = generate_synthetic_surface(config)

    assert np.allclose(surface_1["target_iv"], surface_2["target_iv"])


def test_surface_to_arrays_returns_matching_lengths() -> None:
    surface = generate_synthetic_surface()

    maturities, strikes, target_ivs = surface_to_arrays(surface)

    assert len(maturities) == len(surface)
    assert len(strikes) == len(surface)
    assert len(target_ivs) == len(surface)


def test_invalid_config_negative_spot_raises_error() -> None:
    config = SyntheticSurfaceConfig(spot=-100.0)

    with pytest.raises(ValueError, match="spot must be positive"):
        generate_synthetic_surface(config)


def test_invalid_config_negative_noise_raises_error() -> None:
    config = SyntheticSurfaceConfig(noise_std=-0.01)

    with pytest.raises(ValueError, match="noise_std cannot be negative"):
        generate_synthetic_surface(config)
