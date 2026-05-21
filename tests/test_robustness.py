import numpy as np

from scripts.run_robustness import (
    build_synthetic_heston_market,
    initial_guess_grid,
    run_single_calibration,
    safe_heston_residuals,
)


def test_initial_guess_grid_contains_valid_heston_vectors() -> None:
    guesses = initial_guess_grid()

    assert len(guesses) >= 3

    for guess in guesses:
        assert guess.shape == (5,)
        assert np.all(np.isfinite(guess))
        assert guess[0] > 0.0
        assert guess[1] > 0.0
        assert guess[2] > 0.0
        assert -1.0 < guess[3] < 1.0
        assert guess[4] > 0.0


def test_single_robustness_calibration_reduces_error() -> None:
    market = build_synthetic_heston_market()
    initial_guess = initial_guess_grid()[0]

    initial_residuals = safe_heston_residuals(
        params_vector=initial_guess,
        market=market,
    )
    initial_sse = float(np.sum(initial_residuals**2))

    result = run_single_calibration(
        initial_guess=initial_guess,
        market=market,
        max_nfev=300,
    )

    final_sse = float(np.sum(np.asarray(result.residuals) ** 2))

    assert result.params.shape == (5,)
    assert final_sse < initial_sse