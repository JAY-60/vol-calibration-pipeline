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
    Build a small synthetic Heston calibration market.

    The target implied volatilities are generated from known Heston parameters.
    This gives us a controlled experiment where the data-generating model is known.
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


def initial_guess_grid() -> list[np.ndarray]:
    """
    Candidate starting guesses for robustness testing.

    Each vector has order:
    [kappa, theta, vol_of_vol, rho, v0]
    """
    return [
        np.array([0.8, 0.08, 0.80, -0.10, 0.09], dtype=float),
        np.array([0.5, 0.02, 0.25, -0.30, 0.02], dtype=float),
        np.array([1.0, 0.05, 0.50, -0.70, 0.05], dtype=float),
        np.array([2.5, 0.10, 1.00, -0.90, 0.08], dtype=float),
        np.array([3.0, 0.12, 1.50, -0.20, 0.12], dtype=float),
    ]


def safe_heston_residuals(
    params_vector: np.ndarray,
    market: HestonCalibrationMarket,
) -> np.ndarray:
    """
    Compute Heston implied-volatility residuals.

    If the numerical pricer or implied-vol inversion fails, return a large
    penalty vector instead of crashing the optimiser.
    """
    try:
        residuals = heston_iv_residuals(
            params_vector=params_vector,
            market=market,
        )
        return np.asarray(residuals, dtype=float).ravel()
    except Exception:
        return np.full(market.target_ivs.size, 1e3, dtype=float)


def error_summary(residuals: np.ndarray) -> dict[str, float]:
    """
    Summarise residual error using SSE, RMSE, and MAE.
    """
    residuals = np.asarray(residuals, dtype=float)

    return {
        "sse": float(np.sum(residuals**2)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def run_single_calibration(
    initial_guess: np.ndarray,
    market: HestonCalibrationMarket,
    max_nfev: int = 100,
):
    """
    Run one bounded least-squares Heston calibration.
    """
    return run_least_squares(
        objective=lambda params: safe_heston_residuals(params, market),
        initial_guess=initial_guess,
        bounds=heston_default_bounds(),
        max_nfev=max_nfev,
    )


def summarise_run(
    run_id: int,
    initial_guess: np.ndarray,
    result,
    market: HestonCalibrationMarket,
) -> dict[str, float | int | bool | str]:
    """
    Build one summary row for the robustness experiment.
    """
    initial_residuals = safe_heston_residuals(initial_guess, market)
    final_residuals = np.asarray(result.residuals, dtype=float)

    initial_error = error_summary(initial_residuals)
    final_error = error_summary(final_residuals)

    return {
        "run_id": run_id,
        "initial_kappa": float(initial_guess[0]),
        "initial_theta": float(initial_guess[1]),
        "initial_vol_of_vol": float(initial_guess[2]),
        "initial_rho": float(initial_guess[3]),
        "initial_v0": float(initial_guess[4]),
        "final_kappa": float(result.params[0]),
        "final_theta": float(result.params[1]),
        "final_vol_of_vol": float(result.params[2]),
        "final_rho": float(result.params[3]),
        "final_v0": float(result.params[4]),
        "initial_sse": initial_error["sse"],
        "final_sse": final_error["sse"],
        "initial_rmse": initial_error["rmse"],
        "final_rmse": final_error["rmse"],
        "initial_mae": initial_error["mae"],
        "final_mae": final_error["mae"],
        "improvement_factor": float(
            initial_error["sse"] / max(final_error["sse"], 1e-16)
        ),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }


def main() -> None:
    """
    Run a Heston robustness experiment across multiple initial guesses.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    market = build_synthetic_heston_market()

    rows = []

    for run_id, initial_guess in enumerate(initial_guess_grid(), start=1):
        result = run_single_calibration(
            initial_guess=initial_guess,
            market=market,
            max_nfev=100,
        )

        rows.append(
            summarise_run(
                run_id=run_id,
                initial_guess=initial_guess,
                result=result,
                market=market,
            )
        )

    summary = pd.DataFrame(rows).sort_values("final_rmse")

    output_path = RESULTS_DIR / "heston_robustness_summary.csv"
    summary.to_csv(output_path, index=False)

    display_columns = [
        "run_id",
        "initial_sse",
        "final_sse",
        "initial_rmse",
        "final_rmse",
        "improvement_factor",
        "success",
        "nfev",
    ]

    print(summary[display_columns])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()