import numpy as np
import pytest

from volcal.pricing.heston_pricer import (
    HestonParams,
    feller_condition_value,
    heston_call_price,
    heston_call_prices,
    validate_heston_params,
)


def test_valid_heston_params_pass_validation() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    validate_heston_params(params)


def test_invalid_negative_kappa_raises_error() -> None:
    params = HestonParams(
        kappa=-1.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    with pytest.raises(ValueError, match="kappa must be positive"):
        validate_heston_params(params)


def test_invalid_rho_raises_error() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-1.0,
        v0=0.04,
    )

    with pytest.raises(ValueError, match="rho must be strictly between -1 and 1"):
        validate_heston_params(params)


def test_feller_condition_value_is_float() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    value = feller_condition_value(params)

    assert isinstance(value, float)


def test_heston_call_price_is_positive_and_below_spot() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    price = heston_call_price(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        params=params,
    )

    assert price > 0.0
    assert price < 100.0


def test_heston_call_price_increases_with_spot() -> None:
    params = HestonParams(
        kappa=1.5,
        theta=0.04,
        vol_of_vol=0.4,
        rho=-0.5,
        v0=0.04,
    )

    low_spot_price = heston_call_price(
        S=90.0,
        K=100.0,
        T=1.0,
        r=0.02,
        params=params,
    )

    high_spot_price = heston_call_price(
        S=110.0,
        K=100.0,
        T=1.0,
        r=0.02,
        params=params,
    )

    assert high_spot_price > low_spot_price


def test_heston_call_price_at_expiry_equals_intrinsic_value() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    price = heston_call_price(
        S=120.0,
        K=100.0,
        T=0.0,
        r=0.02,
        params=params,
    )

    assert price == pytest.approx(20.0)


def test_heston_call_prices_returns_matching_shape() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    strikes = np.array([90.0, 100.0, 110.0])
    maturities = np.array([0.5, 1.0, 1.5])

    prices = heston_call_prices(
        S=100.0,
        strikes=strikes,
        maturities=maturities,
        r=0.02,
        params=params,
    )

    assert prices.shape == strikes.shape
    assert np.all(prices > 0.0)


def test_heston_call_prices_rejects_shape_mismatch() -> None:
    params = HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )

    strikes = np.array([90.0, 100.0])
    maturities = np.array([0.5, 1.0, 1.5])

    with pytest.raises(ValueError, match="same shape"):
        heston_call_prices(
            S=100.0,
            strikes=strikes,
            maturities=maturities,
            r=0.02,
            params=params,
        )