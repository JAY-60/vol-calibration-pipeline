from __future__ import annotations

import math

from volcal.pricing.black_scholes import bs_call_price, bs_vega


def _call_price_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """
    No-arbitrage lower and upper bounds for a European call option.
    """
    discounted_strike = K * math.exp(-r * T)
    lower = max(S - discounted_strike, 0.0)
    upper = S
    return lower, upper


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma0: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Compute the Black-Scholes implied volatility for a European call option.

    Uses Newton-Raphson first, with a bisection fallback if Newton becomes unstable.
    """
    if price <= 0.0:
        raise ValueError("Option price must be positive.")
    if S <= 0.0:
        raise ValueError("Spot price S must be positive.")
    if K <= 0.0:
        raise ValueError("Strike K must be positive.")
    if T <= 0.0:
        raise ValueError("Implied volatility is undefined for T <= 0.")
    if tol <= 0.0:
        raise ValueError("Tolerance must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    lower_bound, upper_bound = _call_price_bounds(S, K, T, r)

    if price < lower_bound - 1e-12:
        raise ValueError("Option price is below the no-arbitrage lower bound.")
    if price > upper_bound + 1e-12:
        raise ValueError("Option price is above the no-arbitrage upper bound.")

    # First attempt: Newton-Raphson.
    sigma = max(sigma0, 1e-6)

    for _ in range(max_iter):
        model_price = bs_call_price(S, K, T, r, sigma)
        diff = model_price - price

        if abs(diff) < tol:
            return sigma

        vega = bs_vega(S, K, T, r, sigma)

        if vega < 1e-12:
            break

        sigma_next = sigma - diff / vega

        if sigma_next <= 0.0 or sigma_next > 5.0:
            break

        sigma = sigma_next

    # Fallback: bisection.
    low_sigma = 1e-6
    high_sigma = 5.0

    low_price = bs_call_price(S, K, T, r, low_sigma)
    high_price = bs_call_price(S, K, T, r, high_sigma)

    if price < low_price or price > high_price:
        raise ValueError("Could not bracket the implied volatility in [1e-6, 5.0].")

    for _ in range(max_iter):
        mid_sigma = 0.5 * (low_sigma + high_sigma)
        mid_price = bs_call_price(S, K, T, r, mid_sigma)

        if abs(mid_price - price) < tol:
            return mid_sigma

        if mid_price < price:
            low_sigma = mid_sigma
        else:
            high_sigma = mid_sigma

    return 0.5 * (low_sigma + high_sigma)