from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volcal.calibration.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from volcal.calibration.objective import make_iv_objective
from volcal.calibration.optimizers import (
    OptimizerResult,
    run_differential_evolution_then_least_squares,
    run_least_squares,
)
from volcal.data.synthetic_surface import (
    SyntheticSurfaceConfig,
    generate_synthetic_surface,
)


RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRUE_PARAMS = {
    "base_vol": 0.20,
    "skew": -0.15,
    "curvature": 0.25,
    "term_slope": 0.03,
}


def parametric_iv_model(
    params: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Parametric implied-volatility model used for recovery testing.

    params = [base_vol, skew, curvature, term_slope]
    """
    base_vol, skew, curvature, term_slope = params
    log_moneyness = np.log(moneyness)

    model_ivs = (
        base_vol
        + skew * log_moneyness
        + curvature * log_moneyness**2
        + term_slope * np.sqrt(maturities)
    )

    return np.maximum(model_ivs, 1e-4)


def build_results_row(
    optimizer_name: str,
    result: OptimizerResult,
    target_ivs: np.ndarray,
    model_ivs: np.ndarray,
) -> dict[str, float | str | bool | int]:
    """
    Build one row of calibration results.
    """
    residuals = model_ivs - target_ivs

    return {
        "optimizer": optimizer_name,
        "success": result.success,
        "nfev": result.nfev,
        "cost": result.cost,
        "rmse": root_mean_squared_error(actual=target_ivs, predicted=model_ivs),
        "mae": mean_absolute_error(actual=target_ivs, predicted=model_ivs),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "true_base_vol": TRUE_PARAMS["base_vol"],
        "fit_base_vol": float(result.params[0]),
        "true_skew": TRUE_PARAMS["skew"],
        "fit_skew": float(result.params[1]),
        "true_curvature": TRUE_PARAMS["curvature"],
        "fit_curvature": float(result.params[2]),
        "true_term_slope": TRUE_PARAMS["term_slope"],
        "fit_term_slope": float(result.params[3]),
    }


def main() -> None:
    """
    Run a synthetic parameter recovery experiment.
    """
    config = SyntheticSurfaceConfig(
        base_vol=TRUE_PARAMS["base_vol"],
        skew=TRUE_PARAMS["skew"],
        curvature=TRUE_PARAMS["curvature"],
        term_slope=TRUE_PARAMS["term_slope"],
        noise_std=0.0,
        seed=42,
    )

    surface = generate_synthetic_surface(config)

    maturities = surface["maturity"].to_numpy(dtype=float)
    moneyness = surface["moneyness"].to_numpy(dtype=float)
    target_ivs = surface["target_iv"].to_numpy(dtype=float)

    def model_from_params(params: np.ndarray) -> np.ndarray:
        return parametric_iv_model(
            params=params,
            moneyness=moneyness,
            maturities=maturities,
        )

    objective = make_iv_objective(
        target_ivs=target_ivs,
        model_iv_function=model_from_params,
    )

    initial_guess = np.array([0.10, 0.00, 0.10, 0.00])
    lower_bounds = np.array([0.01, -2.00, 0.00, -1.00])
    upper_bounds = np.array([2.00, 2.00, 2.00, 1.00])

    local_result = run_least_squares(
        objective=objective,
        initial_guess=initial_guess,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    global_result = run_differential_evolution_then_least_squares(
        objective=objective,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        maxiter=50,
        seed=42,
    )

    local_model_ivs = model_from_params(local_result.params)
    global_model_ivs = model_from_params(global_result.params)

    results = pd.DataFrame(
        [
            build_results_row(
                optimizer_name="least_squares",
                result=local_result,
                target_ivs=target_ivs,
                model_ivs=local_model_ivs,
            ),
            build_results_row(
                optimizer_name="differential_evolution_then_least_squares",
                result=global_result,
                target_ivs=target_ivs,
                model_ivs=global_model_ivs,
            ),
        ]
    )

    output_path = RESULTS_DIR / "synthetic_recovery_results.csv"
    results.to_csv(output_path, index=False)

    print("Synthetic recovery experiment complete.")
    print(f"Saved results to: {output_path}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()