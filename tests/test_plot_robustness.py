from pathlib import Path

import pandas as pd
import pytest

from scripts.plot_robustness import (
    load_robustness_results,
    plot_rmse_comparison,
    plot_sse_comparison,
)


def test_load_robustness_results_validates_required_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_robustness.csv"

    pd.DataFrame(
        {
            "run_id": [1],
            "initial_rmse": [0.1],
        }
    ).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError):
        load_robustness_results(bad_csv)


def test_plot_robustness_figures_are_created(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "run_id": [1, 2, 3],
            "initial_rmse": [0.10, 0.20, 0.15],
            "final_rmse": [0.001, 0.002, 0.0015],
            "initial_sse": [0.03, 0.04, 0.05],
            "final_sse": [1e-6, 2e-6, 1.5e-6],
            "success": [True, True, True],
            "nfev": [20, 25, 22],
        }
    )

    rmse_path = plot_rmse_comparison(
        summary=summary,
        output_path=tmp_path / "rmse.png",
    )

    sse_path = plot_sse_comparison(
        summary=summary,
        output_path=tmp_path / "sse.png",
    )

    assert rmse_path.exists()
    assert sse_path.exists()
    assert rmse_path.stat().st_size > 0
    assert sse_path.stat().st_size > 0