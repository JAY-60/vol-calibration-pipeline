from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from volcal.data.synthetic_surface import (
    SyntheticSurfaceConfig,
    generate_synthetic_surface,
)


FIGURE_DIR = Path("results") / "figures"


def _ensure_figure_dir() -> None:
    """
    Ensure the figure output directory exists.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_synthetic_iv_smiles(surface: pd.DataFrame) -> Path:
    """
    Plot implied-volatility smiles for each maturity.
    """
    _ensure_figure_dir()

    fig, ax = plt.subplots(figsize=(8, 5))

    for maturity, group in surface.groupby("maturity"):
        group = group.sort_values("moneyness")
        ax.plot(
            group["moneyness"],
            group["target_iv"],
            marker="o",
            label=f"T={maturity:g}",
        )

    ax.set_title("Synthetic Implied-Volatility Smiles")
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Implied volatility")
    ax.legend(title="Maturity")
    ax.grid(True, alpha=0.3)

    output_path = FIGURE_DIR / "synthetic_iv_smiles.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_synthetic_iv_surface(surface: pd.DataFrame) -> Path:
    """
    Plot a 3D synthetic implied-volatility surface.
    """
    _ensure_figure_dir()

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        surface["moneyness"],
        surface["maturity"],
        surface["target_iv"],
        linewidth=0.2,
        antialiased=True,
    )

    ax.set_title("Synthetic Implied-Volatility Surface")
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied volatility")

    output_path = FIGURE_DIR / "synthetic_iv_surface.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main() -> None:
    """
    Generate all project figures.
    """
    config = SyntheticSurfaceConfig(noise_std=0.0)
    surface = generate_synthetic_surface(config)

    smiles_path = plot_synthetic_iv_smiles(surface)
    surface_path = plot_synthetic_iv_surface(surface)

    print(f"Saved: {smiles_path}")
    print(f"Saved: {surface_path}")


if __name__ == "__main__":
    main()