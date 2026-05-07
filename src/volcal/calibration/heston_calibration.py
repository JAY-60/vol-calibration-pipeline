from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from volcal.calibration.objective import iv_residuals
from volcal.pricing.heston_pricer import (
    HestonParams,
    heston_call_prices,
    validate_heston_params,
)
from volcal.pricing.implied_vol import implied_vol_call


@dataclass(frozen=True)
class HestonCalibrationMarket:
    """
    Market/synthetic option grid used for Heston calibration.
    """

    spot: float
    rate: float
    strikes: np.ndarray
    maturities: np.ndarray
    target_ivs: np.ndarray


def heston_params_from_vector(params: Sequence[float] | np.ndarray) -> HestonParams:
    """
    Convert a calibration parameter vector into HestonParams.

    Expected order:
    kappa, theta, vol_of_vol, rho, v0
    """
    array = np.asarray(params, dtype=float)

    if array.shape != (5,):
        raise ValueError("Heston parameter vector must have shape (5,).")

    heston_params = HestonParams(
        kappa=float(array[0]),
        theta=float(array[1]),
        vol_of_vol=float(array[2]),
        rho=float(array[3]),
        v0=float(array[4]),
    )

    validate_heston_params(heston_params)
    return heston_params


def _validate_market(market: HestonCalibrationMarket) -> None:
    """
    Validate the calibration market grid.
    """
    if market.spot <= 0.0:
        raise ValueError("spot must be positive.")

    strikes = np.asarray(market.strikes, dtype=float)
    maturities = np.asarray(market.maturities, dtype=float)
    target_ivs = np.asarray(market.target_ivs, dtype=float)

    if strikes.shape != maturities.shape:
        raise ValueError("strikes and maturities must have the same shape.")

    if strikes.shape != target_ivs.shape:
        raise ValueError("target_ivs must have the same shape as strikes.")

    if strikes.size == 0:
        raise ValueError("market grid must not be empty.")

    if np.any(strikes <= 0.0):
        raise ValueError("all strikes must be positive.")

    if np.any(maturities <= 0.0):
        raise ValueError("all maturities must be positive.")

    if np.any(target_ivs <= 0.0):
        raise ValueError("all target implied volatilities must be positive.")


def heston_model_implied_vols(
    market: HestonCalibrationMarket,
    params: HestonParams,
) -> np.ndarray:
    """
    Compute Heston model implied volatilities on a calibration grid.

    Steps:
    1. Price options under Heston.
    2. Convert each Heston price into Black-Scholes implied volatility.
    """
    _validate_market(market)
    validate_heston_params(params)

    strikes = np.asarray(market.strikes, dtype=float)
    maturities = np.asarray(market.maturities, dtype=float)

    prices = heston_call_prices(
        S=market.spot,
        strikes=strikes,
        maturities=maturities,
        r=market.rate,
        params=params,
    )

    implied_vols = [
        implied_vol_call(
            price=float(price),
            S=market.spot,
            K=float(strike),
            T=float(maturity),
            r=market.rate,
        )
        for price, strike, maturity in zip(
            prices.ravel(),
            strikes.ravel(),
            maturities.ravel(),
        )
    ]

    return np.asarray(implied_vols, dtype=float).reshape(strikes.shape)


def heston_iv_residuals(
    params_vector: Sequence[float] | np.ndarray,
    market: HestonCalibrationMarket,
) -> np.ndarray:
    """
    Compute Heston implied-volatility residuals.

    residual = model implied volatility - target implied volatility
    """
    params = heston_params_from_vector(params_vector)
    model_ivs = heston_model_implied_vols(market=market, params=params)

    target_ivs = np.asarray(market.target_ivs, dtype=float)

    return iv_residuals(model_ivs=model_ivs, target_ivs=target_ivs)