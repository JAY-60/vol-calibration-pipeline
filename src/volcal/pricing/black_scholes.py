from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    """
    Basic domain checks for Black-Scholes pricing inputs.
    """
    if S <= 0.0:
        raise ValueError("Spot price S must be positive.")
    if K <= 0.0:
        raise ValueError("Strike K must be positive.")
    if T < 0.0:
        raise ValueError("Time to maturity T cannot be negative.")
    if sigma <= 0.0:
        raise ValueError("Volatility sigma must be positive.")


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes d1 term.
    """
    _validate_inputs(S, K, T, sigma)

    if T == 0.0:
        raise ValueError("d1 is undefined at T = 0. Use intrinsic value directly.")

    numerator = math.log(S / K) + (r + 0.5 * sigma * sigma) * T
    denominator = sigma * math.sqrt(T)
    return numerator / denominator


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes d2 term.
    """
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European call option.
    """
    _validate_inputs(S, K, T, sigma)

    if T == 0.0:
        return max(S - K, 0.0)

    d1_value = d1(S, K, T, r, sigma)
    d2_value = d2(S, K, T, r, sigma)

    return S * norm_cdf(d1_value) - K * math.exp(-r * T) * norm_cdf(d2_value)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European put option.
    """
    _validate_inputs(S, K, T, sigma)

    if T == 0.0:
        return max(K - S, 0.0)

    d1_value = d1(S, K, T, r, sigma)
    d2_value = d2(S, K, T, r, sigma)

    return K * math.exp(-r * T) * norm_cdf(-d2_value) - S * norm_cdf(-d1_value)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega: derivative of option price with respect to volatility.
    """
    _validate_inputs(S, K, T, sigma)

    if T == 0.0:
        return 0.0

    d1_value = d1(S, K, T, r, sigma)
    return S * norm_pdf(d1_value) * math.sqrt(T)