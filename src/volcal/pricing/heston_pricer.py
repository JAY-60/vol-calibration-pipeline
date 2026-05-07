from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass(frozen=True)
class HestonParams:
    """
    Parameters for the Heston stochastic volatility model.

    kappa: speed of variance mean reversion
    theta: long-run variance level
    vol_of_vol: volatility of the variance process
    rho: correlation between asset and variance shocks
    v0: initial variance
    """

    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    v0: float

    def as_array(self) -> np.ndarray:
        """
        Return parameters in the order used by the calibration pipeline.
        """
        return np.array(
            [self.kappa, self.theta, self.vol_of_vol, self.rho, self.v0],
            dtype=float,
        )


def validate_heston_params(params: HestonParams) -> None:
    """
    Validate Heston model parameters.
    """
    values = params.as_array()

    if not np.all(np.isfinite(values)):
        raise ValueError("Heston parameters must be finite.")

    if params.kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if params.theta <= 0.0:
        raise ValueError("theta must be positive.")
    if params.vol_of_vol <= 0.0:
        raise ValueError("vol_of_vol must be positive.")
    if not (-1.0 < params.rho < 1.0):
        raise ValueError("rho must be strictly between -1 and 1.")
    if params.v0 <= 0.0:
        raise ValueError("v0 must be positive.")


def feller_condition_value(params: HestonParams) -> float:
    """
    Return the Heston Feller condition margin.

    The classical Feller condition is:

        2 * kappa * theta >= vol_of_vol^2

    Positive values indicate the condition is satisfied.
    """
    validate_heston_params(params)
    return 2.0 * params.kappa * params.theta - params.vol_of_vol**2


def _validate_option_inputs(S: float, K: float, T: float, r: float) -> None:
    """
    Validate standard European option inputs.
    """
    if S <= 0.0:
        raise ValueError("spot price S must be positive.")
    if K <= 0.0:
        raise ValueError("strike K must be positive.")
    if T < 0.0:
        raise ValueError("maturity T cannot be negative.")
    if not math.isfinite(r):
        raise ValueError("risk-free rate r must be finite.")


def _heston_characteristic_function(
    u: complex,
    S: float,
    T: float,
    r: float,
    params: HestonParams,
) -> complex:
    """
    Heston characteristic function for log-price.

    This is the numerical engine used by the semi-closed-form Heston call price.
    """
    i = 1j

    x = math.log(S)
    kappa = params.kappa
    theta = params.theta
    sigma = params.vol_of_vol
    rho = params.rho
    v0 = params.v0

    d = np.sqrt((rho * sigma * i * u - kappa) ** 2 + sigma**2 * (i * u + u**2))

    g = (kappa - rho * sigma * i * u - d) / (
        kappa - rho * sigma * i * u + d
    )

    exp_neg_dT = np.exp(-d * T)

    C = (
        i * u * (x + r * T)
        + (kappa * theta / sigma**2)
        * (
            (kappa - rho * sigma * i * u - d) * T
            - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
        )
    )

    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    )

    return complex(np.exp(C + D * v0))


def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    integration_limit: float = 100.0,
    quad_limit: int = 200,
) -> float:
    """
    Price a European call option under the Heston model.

    Uses the Heston characteristic-function representation and numerical
    integration for the risk-neutral probabilities.
    """
    _validate_option_inputs(S, K, T, r)
    validate_heston_params(params)

    if integration_limit <= 0.0:
        raise ValueError("integration_limit must be positive.")
    if quad_limit <= 0:
        raise ValueError("quad_limit must be positive.")

    if T == 0.0:
        return max(S - K, 0.0)

    i = 1j
    log_strike = math.log(K)

    phi_minus_i = _heston_characteristic_function(
        -i,
        S=S,
        T=T,
        r=r,
        params=params,
    )

    def integrand_p1(u: float) -> float:
        numerator = np.exp(-i * u * log_strike) * _heston_characteristic_function(
            u - i,
            S=S,
            T=T,
            r=r,
            params=params,
        )
        denominator = i * u * phi_minus_i
        return float(np.real(numerator / denominator))

    def integrand_p2(u: float) -> float:
        numerator = np.exp(-i * u * log_strike) * _heston_characteristic_function(
            u,
            S=S,
            T=T,
            r=r,
            params=params,
        )
        denominator = i * u
        return float(np.real(numerator / denominator))

    lower = 1e-8

    p1_integral, _ = quad(
        integrand_p1,
        lower,
        integration_limit,
        limit=quad_limit,
        epsabs=1e-7,
        epsrel=1e-7,
    )

    p2_integral, _ = quad(
        integrand_p2,
        lower,
        integration_limit,
        limit=quad_limit,
        epsabs=1e-7,
        epsrel=1e-7,
    )

    p1 = 0.5 + p1_integral / math.pi
    p2 = 0.5 + p2_integral / math.pi

    price = S * p1 - K * math.exp(-r * T) * p2

    if price < 0.0 and price > -1e-8:
        price = 0.0

    return float(price)


def heston_call_prices(
    S: float,
    strikes: Sequence[float] | np.ndarray,
    maturities: Sequence[float] | np.ndarray,
    r: float,
    params: HestonParams,
) -> np.ndarray:
    """
    Price a grid/vector of European call options under Heston.
    """
    strikes_array = np.asarray(strikes, dtype=float)
    maturities_array = np.asarray(maturities, dtype=float)

    if strikes_array.shape != maturities_array.shape:
        raise ValueError("strikes and maturities must have the same shape.")

    prices = [
        heston_call_price(
            S=S,
            K=float(K),
            T=float(T),
            r=r,
            params=params,
        )
        for K, T in zip(strikes_array.ravel(), maturities_array.ravel())
    ]

    return np.asarray(prices, dtype=float).reshape(strikes_array.shape)