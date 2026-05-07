from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volcal.calibration.bounds import heston_default_bounds
from volcal.calibration.heston_calibration import (
    HestonCalibrationMarket,
    heston_iv_residuals,
    heston_model_implied_vols,
)
from volcal.calibration.optimizers import run_least_squares
from volcal.pricing.heston_pricer import HestonParams


RESULTS_DIR = Path("results") / "tables"


def true_heston_params() -> HestonParams:
    """
    Parameters used to generate the synthetic Heston target surface.
    """
    return HestonParams(
        kappa=1.5,
        theta=0.04,
        vol_of_vol=0.35,
        rho=-0.55,
        v0=0.04,
    )


def build_synthetic_heston_market() -> HestonCalibrationMarket:
    """
    Build a small synthetic calibration market from known Heston parameters.
    """
    spot = 100.0
    rate = 0.02

    strikes = np.array([95.0, 100.0, 105.0])
    maturities = np.array([0.5, 1.0, 1.5])

    placeholder_market = HestonCalibrationMarket(
        spot=spot,
        rate=rate,
        strikes=strikes,
        maturities=maturities,
        target_ivs=np.full_like(strikes, 0.20, dtype=float),
    )

    target_ivs = heston_model_implied_vols(
        market=placeholder_market,
        params=true_heston_params(),
    )

    return HestonCalibrationMarket(
        spot=spot,
        rate=rate,
        strikes=strikes,
        maturities=maturities,
        target_ivs=target_ivs,
    )


def safe_heston_residuals(
    params_vector: np.ndarray,
    market: HestonCalibrationMarket,
) -> np.ndarray:
    """
    Compute Heston residuals, returning a large penalty if pricing fails.
    """
    try:
        residuals = heston_iv_residuals(
            params_vector=params_vector,
            market=market,
        )
        return np.asarray(residuals, dtype=float).ravel()
    except Exception:
        return np.full(market.target_ivs.size, 1e3, dtype=float)


def summarise_stage(
    stage: str,
    params: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float | str]:
    """
    Build a summary row for the recovery experiment.
    """
    residuals = np.asarray(residuals, dtype=float)

    return {
        "stage": stage,
        "kappa": float(params[0]),
        "theta": float(params[1]),
        "vol_of_vol": float(params[2]),
        "rho": float(params[3]),
        "v0": float(params[4]),
        "sse": float(np.sum(residuals**2)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def main() -> None:
    """
    Run a synthetic Heston recovery experiment.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    market = build_synthetic_heston_market()

    initial_guess = np.array([0.8, 0.08, 0.80, -0.10, 0.09], dtype=float)

    initial_residuals = safe_heston_residuals(
        params_vector=initial_guess,
        market=market,
    )

    result = run_least_squares(
        objective=lambda params: safe_heston_residuals(params, market),
        initial_guess=initial_guess,
        bounds=heston_default_bounds(),
        max_nfev=100,
    )

    rows = [
        summarise_stage(
            stage="initial_guess",
            params=initial_guess,
            residuals=initial_residuals,
        ),
        summarise_stage(
            stage="calibrated",
            params=result.params,
            residuals=result.residuals,
        ),
    ]

    summary = pd.DataFrame(rows)

    output_path = RESULTS_DIR / "heston_recovery_summary.csv"
    summary.to_csv(output_path, index=False)

    print(summary)
    print(f"Saved: {output_path}")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")


if __name__ == "__main__":
    main()