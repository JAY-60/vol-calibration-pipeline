import numpy as np
import pytest

from volcal.calibration.heston_calibration import (
    HestonCalibrationMarket,
    heston_iv_residuals,
    heston_model_implied_vols,
    heston_params_from_vector,
)
from volcal.pricing.heston_pricer import HestonParams


def _small_market() -> HestonCalibrationMarket:
    return HestonCalibrationMarket(
        spot=100.0,
        rate=0.02,
        strikes=np.array([95.0, 100.0]),
        maturities=np.array([0.5, 1.0]),
        target_ivs=np.array([0.20, 0.22]),
    )


def _params() -> HestonParams:
    return HestonParams(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        v0=0.04,
    )


def test_heston_params_from_vector_returns_heston_params() -> None:
    params = heston_params_from_vector(np.array([2.0, 0.04, 0.5, -0.7, 0.04]))

    assert isinstance(params, HestonParams)
    assert params.kappa == pytest.approx(2.0)
    assert params.theta == pytest.approx(0.04)


def test_heston_params_from_vector_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        heston_params_from_vector(np.array([2.0, 0.04]))


def test_heston_model_implied_vols_returns_matching_shape() -> None:
    market = _small_market()
    params = _params()

    model_ivs = heston_model_implied_vols(market=market, params=params)

    assert model_ivs.shape == market.target_ivs.shape
    assert np.all(model_ivs > 0.0)


def test_heston_iv_residuals_returns_matching_shape() -> None:
    market = _small_market()
    params_vector = np.array([2.0, 0.04, 0.5, -0.7, 0.04])

    residuals = heston_iv_residuals(params_vector=params_vector, market=market)

    assert residuals.shape == market.target_ivs.shape
    assert np.all(np.isfinite(residuals))


def test_heston_iv_residuals_rejects_bad_market_shape() -> None:
    market = HestonCalibrationMarket(
        spot=100.0,
        rate=0.02,
        strikes=np.array([95.0, 100.0]),
        maturities=np.array([0.5]),
        target_ivs=np.array([0.20, 0.22]),
    )

    params_vector = np.array([2.0, 0.04, 0.5, -0.7, 0.04])

    with pytest.raises(ValueError, match="same shape"):
        heston_iv_residuals(params_vector=params_vector, market=market)