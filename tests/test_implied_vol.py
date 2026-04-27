import pytest

from volcal.pricing.black_scholes import bs_call_price
from volcal.pricing.implied_vol import implied_vol_call


def test_implied_vol_recovers_true_sigma_at_the_money() -> None:
    true_sigma = 0.2
    price = bs_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=true_sigma)

    recovered_sigma = implied_vol_call(
        price=price,
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
    )

    assert recovered_sigma == pytest.approx(true_sigma, abs=1e-6)


def test_implied_vol_recovers_true_sigma_out_of_the_money() -> None:
    true_sigma = 0.35
    price = bs_call_price(S=100.0, K=120.0, T=1.5, r=0.03, sigma=true_sigma)

    recovered_sigma = implied_vol_call(
        price=price,
        S=100.0,
        K=120.0,
        T=1.5,
        r=0.03,
    )

    assert recovered_sigma == pytest.approx(true_sigma, abs=1e-6)


def test_implied_vol_recovers_true_sigma_high_vol_case() -> None:
    true_sigma = 0.8
    price = bs_call_price(S=100.0, K=100.0, T=2.0, r=0.01, sigma=true_sigma)

    recovered_sigma = implied_vol_call(
        price=price,
        S=100.0,
        K=100.0,
        T=2.0,
        r=0.01,
    )

    assert recovered_sigma == pytest.approx(true_sigma, abs=1e-6)


def test_price_below_lower_bound_raises_error() -> None:
    with pytest.raises(ValueError, match="below the no-arbitrage lower bound"):
        implied_vol_call(
            price=0.0001,
            S=100.0,
            K=50.0,
            T=1.0,
            r=0.05,
        )


def test_price_above_upper_bound_raises_error() -> None:
    with pytest.raises(ValueError, match="above the no-arbitrage upper bound"):
        implied_vol_call(
            price=150.0,
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
        )


def test_zero_maturity_raises_error() -> None:
    with pytest.raises(ValueError, match="undefined for T <= 0"):
        implied_vol_call(
            price=10.0,
            S=100.0,
            K=100.0,
            T=0.0,
            r=0.05,
        )


def test_non_positive_price_raises_error() -> None:
    with pytest.raises(ValueError, match="Option price must be positive"):
        implied_vol_call(
            price=0.0,
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
        )