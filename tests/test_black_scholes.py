import math

import pytest

from volcal.pricing.black_scholes import (
    bs_call_price,
    bs_put_price,
    bs_vega,
)


def test_call_price_is_positive() -> None:
    price = bs_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert price > 0.0


def test_put_price_is_positive() -> None:
    price = bs_put_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert price > 0.0


def test_call_price_increases_with_spot() -> None:
    low_spot = bs_call_price(S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    high_spot = bs_call_price(S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert high_spot > low_spot


def test_call_price_decreases_with_strike() -> None:
    low_strike = bs_call_price(S=100.0, K=90.0, T=1.0, r=0.05, sigma=0.2)
    high_strike = bs_call_price(S=100.0, K=110.0, T=1.0, r=0.05, sigma=0.2)
    assert low_strike > high_strike


def test_vega_is_positive_for_standard_case() -> None:
    vega = bs_vega(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    assert vega > 0.0


def test_intrinsic_value_at_expiry_for_call() -> None:
    price = bs_call_price(S=120.0, K=100.0, T=0.0, r=0.05, sigma=0.2)
    assert price == pytest.approx(20.0)


def test_intrinsic_value_at_expiry_for_put() -> None:
    price = bs_put_price(S=80.0, K=100.0, T=0.0, r=0.05, sigma=0.2)
    assert price == pytest.approx(20.0)


def test_invalid_spot_raises_error() -> None:
    with pytest.raises(ValueError, match="Spot price S must be positive."):
        bs_call_price(S=0.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_invalid_sigma_raises_error() -> None:
    with pytest.raises(ValueError, match="Volatility sigma must be positive."):
        bs_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.0)