from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("results") / "tables" / "heston_robustness_summary.csv"
OUTPUT_DIR = Path("results") / "figures"


REQUIRED_COLUMNS = {
    "run_id",
    "initial_rmse",
    "final_rmse",
    "initial_sse",
    "final_sse",
    "success",
    "nfev",
}


def load_robustness_results(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """
    Load and validate the Heston robustness summary table.
    """
    summary = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(summary.columns)

    if missing_columns:
        raise ValueError(
            f"Robustness summary is missing required columns: {sorted(missing_columns)}"
        )

    return summary.sort_values("run_id").reset_index(drop=True)


def plot_rmse_comparison(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Plot initial RMSE versus final RMSE for each robustness run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ids = summary["run_id"].astype(str).to_numpy()
    x_positions = range(len(run_ids))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        x_positions,
        summary["initial_rmse"],
        marker="o",
        label="Initial RMSE",
    )

    ax.plot(
        x_positions,
        summary["final_rmse"],
        marker="o",
        label="Final RMSE",
    )

    ax.set_title("Heston Calibration Robustness: Initial vs Final RMSE")
    ax.set_xlabel("Robustness run")
    ax.set_ylabel("RMSE")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(run_ids)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_sse_comparison(
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Plot initial SSE versus final SSE for each robustness run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ids = summary["run_id"].astype(str).to_numpy()
    x_positions = range(len(run_ids))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        x_positions,
        summary["initial_sse"],
        marker="o",
        label="Initial SSE",
    )

    ax.plot(
        x_positions,
        summary["final_sse"],
        marker="o",
        label="Final SSE",
    )

    ax.set_title("Heston Calibration Robustness: Initial vs Final SSE")
    ax.set_xlabel("Robustness run")
    ax.set_ylabel("SSE")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(run_ids)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def main() -> None:
    """
    Generate robustness diagnostic plots from the saved robustness CSV.
    """
    summary = load_robustness_results()

    rmse_path = plot_rmse_comparison(
        summary=summary,
        output_path=OUTPUT_DIR / "heston_robustness_rmse.png",
    )

    sse_path = plot_sse_comparison(
        summary=summary,
        output_path=OUTPUT_DIR / "heston_robustness_sse.png",
    )

    print(f"Saved: {rmse_path}")
    print(f"Saved: {sse_path}")


if __name__ == "__main__":
    main()