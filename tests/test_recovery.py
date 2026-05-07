import numpy as np

from volcal.calibration.heston_calibration import (
    HestonCalibrationMarket,
    heston_iv_residuals,
    heston_model_implied_vols,
)
from volcal.pricing.heston_pricer import HestonParams


def _true_params() -> HestonParams:
    return HestonParams(
        kappa=1.5,
        theta=0.04,
        vol_of_vol=0.35,
        rho=-0.55,
        v0=0.04,
    )


def _synthetic_market() -> HestonCalibrationMarket:
    spot = 100.0
    rate = 0.02
    strikes = np.array([95.0, 105.0])
    maturities = np.array([0.5, 1.0])

    placeholder_market = HestonCalibrationMarket(
        spot=spot,
        rate=rate,
        strikes=strikes,
        maturities=maturities,
        target_ivs=np.full_like(strikes, 0.20, dtype=float),
    )

    target_ivs = heston_model_implied_vols(
        market=placeholder_market,
        params=_true_params(),
    )

    return HestonCalibrationMarket(
        spot=spot,
        rate=rate,
        strikes=strikes,
        maturities=maturities,
        target_ivs=target_ivs,
    )


def test_true_heston_params_have_near_zero_self_generated_residuals() -> None:
    market = _synthetic_market()
    params_vector = _true_params().as_array()

    residuals = heston_iv_residuals(
        params_vector=params_vector,
        market=market,
    )

    assert np.allclose(residuals, np.zeros_like(residuals), atol=1e-8)


def test_perturbed_heston_params_produce_nonzero_residuals() -> None:
    market = _synthetic_market()
    perturbed_params = np.array([0.8, 0.08, 0.80, -0.10, 0.09])

    residuals = heston_iv_residuals(
        params_vector=perturbed_params,
        market=market,
    )

    assert np.sum(residuals**2) > 1e-8