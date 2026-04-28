from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volcal.pricing.black_scholes import bs_call_price


@dataclass(frozen=True)
class SyntheticSurfaceConfig:
    """
    Configuration for generating a synthetic implied-volatility surface.
    """

    spot: float = 100.0
    rate: float = 0.02
    maturities: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    moneyness: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    base_vol: float = 0.20
    skew: float = -0.15
    curvature: float = 0.25
    term_slope: float = 0.03
    noise_std: float = 0.0
    seed: int = 42


def _validate_config(config: SyntheticSurfaceConfig) -> None:
    """
    Validate the synthetic surface configuration.
    """
    if config.spot <= 0.0:
        raise ValueError("spot must be positive.")
    if len(config.maturities) == 0:
        raise ValueError("maturities must not be empty.")
    if len(config.moneyness) == 0:
        raise ValueError("moneyness must not be empty.")
    if any(T <= 0.0 for T in config.maturities):
        raise ValueError("all maturities must be positive.")
    if any(m <= 0.0 for m in config.moneyness):
        raise ValueError("all moneyness values must be positive.")
    if config.base_vol <= 0.0:
        raise ValueError("base_vol must be positive.")
    if config.noise_std < 0.0:
        raise ValueError("noise_std cannot be negative.")


def synthetic_iv(
    moneyness: float,
    maturity: float,
    config: SyntheticSurfaceConfig,
) -> float:
    """
    Parametric synthetic implied-volatility function.

    The surface is built from:
    - base volatility,
    - skew in log-moneyness,
    - curvature in log-moneyness,
    - term-structure adjustment.
    """
    if moneyness <= 0.0:
        raise ValueError("moneyness must be positive.")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive.")

    log_moneyness = np.log(moneyness)

    iv = (
        config.base_vol
        + config.skew * log_moneyness
        + config.curvature * log_moneyness**2
        + config.term_slope * np.sqrt(maturity)
    )

    return max(float(iv), 1e-4)


def generate_synthetic_surface(config: SyntheticSurfaceConfig | None = None) -> pd.DataFrame:
    """
    Generate a synthetic implied-volatility surface and corresponding call prices.

    Returns a DataFrame with columns:
    - maturity
    - moneyness
    - strike
    - target_iv
    - target_call_price
    """
    if config is None:
        config = SyntheticSurfaceConfig()

    _validate_config(config)

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float]] = []

    for maturity in config.maturities:
        for moneyness in config.moneyness:
            strike = config.spot * moneyness
            iv = synthetic_iv(moneyness=moneyness, maturity=maturity, config=config)

            if config.noise_std > 0.0:
                iv += rng.normal(loc=0.0, scale=config.noise_std)
                iv = max(float(iv), 1e-4)

            call_price = bs_call_price(
                S=config.spot,
                K=strike,
                T=maturity,
                r=config.rate,
                sigma=iv,
            )

            rows.append(
                {
                    "maturity": float(maturity),
                    "moneyness": float(moneyness),
                    "strike": float(strike),
                    "target_iv": float(iv),
                    "target_call_price": float(call_price),
                }
            )

    return pd.DataFrame(rows)


def surface_to_arrays(surface: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a synthetic surface DataFrame into arrays useful for calibration.

    Returns:
    - maturities
    - strikes
    - target implied volatilities
    """
    required_columns = {"maturity", "strike", "target_iv"}
    missing_columns = required_columns.difference(surface.columns)

    if missing_columns:
        raise ValueError(f"surface is missing columns: {sorted(missing_columns)}")

    maturities = surface["maturity"].to_numpy(dtype=float)
    strikes = surface["strike"].to_numpy(dtype=float)
    target_ivs = surface["target_iv"].to_numpy(dtype=float)

    return maturities, strikes, target_ivs