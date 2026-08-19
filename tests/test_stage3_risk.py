"""Tests for stage3_forecast.DriftForecaster and risk.compute_risk_index.

Fixtures are built inline from marsad.spectra (gaussian_feature / BAND_GRID)
and seeded numpy — marsad.synth is deliberately NOT imported here.
"""
from __future__ import annotations

import numpy as np
import pytest

from marsad.risk import (AMBER_THRESHOLD, PROXIMITY_RADIUS_KM, RED_THRESHOLD,
                         RiskAssessment, RiskLevel, compute_risk_index)
from marsad.spectra import BAND_GRID, gaussian_feature
from marsad.stage3_forecast import DriftForecaster, Forecast

_LEVEL_ORDER = {RiskLevel.GREEN: 0, RiskLevel.AMBER: 1, RiskLevel.RED: 2}


def _noisy_history(n_days: int = 30, base: float = 0.4, seed: int = 42) -> np.ndarray:
    """Seeded bounded random-walk risk history ending near `base`."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.02, size=n_days)
    return np.clip(base + np.cumsum(steps) - np.sum(steps), 0.05, 0.95)


def _rising_history(n_days: int = 30) -> np.ndarray:
    """Smooth monotonically rising history: the left flank of a spectral
    Gaussian re-used as a time series (all 30 samples lie left of the
    feature centre at BAND_GRID[45], so values strictly increase)."""
    flank = gaussian_feature(center_nm=float(BAND_GRID[45]),
                             fwhm_nm=400.0, amplitude=0.8)[:n_days]
    assert np.all(np.diff(flank) > 0)  # fixture sanity
    return flank


# ---------------------------------------------------------------- forecast

def test_forecast_shapes_ordering_clipping():
    fc = DriftForecaster(horizon_days=7).forecast(_noisy_history(), 2.0)
    assert isinstance(fc, Forecast)
    for arr in (fc.mean, fc.lo, fc.hi):
        assert arr.shape == (7,)
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
    assert np.all(fc.lo <= fc.mean) and np.all(fc.mean <= fc.hi)
    assert isinstance(fc.method, str) and fc.method


def test_forecast_custom_horizon_and_short_history():
    fc = DriftForecaster(horizon_days=3).forecast(np.array([0.5]))
    assert fc.mean.shape == (3,)
    np.testing.assert_allclose(fc.mean, 0.5, atol=1e-12)
    with pytest.raises(ValueError):
        DriftForecaster().forecast(np.array([]))


def test_flat_history_near_flat_mean_and_widening_band():
    fc = DriftForecaster(horizon_days=7).forecast(np.full(30, 0.4))
    np.testing.assert_allclose(fc.mean, 0.4, atol=0.02)
    width = fc.hi - fc.lo
    # band widens ~sqrt(lead): strictly wider at every further lead
    assert np.all(np.diff(width) > 0)
    assert width[-1] > width[0]


def test_rising_history_gives_rising_forecast():
    hist = _rising_history()
    fc = DriftForecaster(horizon_days=7).forecast(hist)
    assert np.all(np.diff(fc.mean) > 0)          # forecast keeps rising
    assert fc.mean[-1] > hist[-1]                # above the last observation


def test_positive_drift_raises_forecast_negative_does_not():
    hist = _noisy_history(base=0.35, seed=7)
    base = DriftForecaster(horizon_days=7).forecast(hist, 0.0)
    toward = DriftForecaster(horizon_days=7).forecast(hist, 3.0)
    away = DriftForecaster(horizon_days=7).forecast(hist, -5.0)
    assert np.all(toward.mean >= base.mean)
    assert toward.mean[-1] > base.mean[-1]       # strictly higher at range
    np.testing.assert_allclose(away.mean, base.mean)  # never lowers risk


# -------------------------------------------------------------------- risk

def test_risk_score_monotone_in_bloom_prob():
    probs = np.linspace(0.0, 1.0, 21)
    scores = [compute_risk_index(p, 5.0, 0.01, 3.0, 0.1).score for p in probs]
    assert np.all(np.diff(scores) > 0)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_level_thresholds_green_amber_red():
    # isolate the probability term: no chl, no trend, far from intake
    green = compute_risk_index(0.45, 0.0, 0.0, 20.0, 0.0)
    amber = compute_risk_index(0.55, 0.0, 0.0, 20.0, 0.0)
    amber_hi = compute_risk_index(0.90, 0.0, 0.0, 20.0, 0.0)
    red = compute_risk_index(0.99, 0.0, 0.0, 20.0, 0.0)
    assert green.level is RiskLevel.GREEN and green.score < AMBER_THRESHOLD
    assert amber.level is RiskLevel.AMBER
    assert AMBER_THRESHOLD <= amber.score < RED_THRESHOLD
    assert amber_hi.level is RiskLevel.AMBER and amber_hi.score < RED_THRESHOLD
    assert red.level is RiskLevel.RED and red.score >= RED_THRESHOLD


def test_levels_monotone_across_prob_sweep():
    levels = [_LEVEL_ORDER[compute_risk_index(p, 0.0, 0.0, 20.0, 0.0).level]
              for p in np.linspace(0.0, 1.0, 101)]
    assert all(b >= a for a, b in zip(levels, levels[1:]))
    assert set(levels) == {0, 1, 2}  # all three levels reachable


def test_trend_and_proximity_boost_risk():
    base = compute_risk_index(0.7, 5.0, 0.0, 20.0, 0.0).score
    trending = compute_risk_index(0.7, 5.0, 0.04, 20.0, 0.0).score
    near = compute_risk_index(0.7, 5.0, 0.0, 2.0, 0.0).score
    assert trending > base
    assert near > base


def test_uncertainty_promotes_borderline_amber_to_red_near_intake():
    # score ~0.57: AMBER within PROMOTION_MARGIN of RED_THRESHOLD (0.65)
    args = dict(bloom_prob=0.7, chl_mg_m3=10.0, trend_per_day=0.01,
                distance_km=3.0)
    calm = compute_risk_index(uncertainty=0.05, **args)
    uncertain = compute_risk_index(uncertainty=0.5, **args)
    assert calm.level is RiskLevel.AMBER
    assert uncertain.level is RiskLevel.RED
    assert uncertain.score == calm.score  # promotion changes level, not score
    assert any("precautionary" in r for r in uncertain.rationale)
    # far from the intake, uncertainty alone must not promote
    far = compute_risk_index(0.7, 10.0, 0.01, PROXIMITY_RADIUS_KM + 5.0, 0.9)
    assert far.level is RiskLevel.AMBER
    # a MID-band AMBER (score well below the margin) must NOT promote:
    # otherwise every amber near an intake turns RED and RED loses meaning
    mid = compute_risk_index(0.6, 5.0, 0.01, 3.0, 0.9)
    assert mid.level is RiskLevel.AMBER


def test_uncertainty_never_downgrades():
    for p, chl, dist in [(0.05, 0.0, 3.0),   # GREEN stays GREEN
                         (0.6, 5.0, 3.0),    # AMBER may only go up
                         (0.99, 40.0, 1.0)]: # RED stays RED
        low = compute_risk_index(p, chl, 0.0, dist, 0.0)
        high = compute_risk_index(p, chl, 0.0, dist, 1.0)
        assert _LEVEL_ORDER[high.level] >= _LEVEL_ORDER[low.level]
    still_red = compute_risk_index(0.99, 40.0, 0.02, 1.0, 1.0)
    assert still_red.level is RiskLevel.RED


def test_rationale_is_concrete_and_extremes_are_clipped():
    hot = compute_risk_index(0.82, 32.0, 0.03, 3.2, 0.1)
    assert isinstance(hot, RiskAssessment)
    assert hot.rationale and all(isinstance(r, str) for r in hot.rationale)
    assert any("82%" in r for r in hot.rationale)
    assert any("3.2 km" in r and "approaching" in r for r in hot.rationale)
    # out-of-range inputs are clipped, score stays in [0, 1]
    worst = compute_risk_index(1.5, 1e4, 9.0, -2.0, 2.0)
    assert worst.score <= 1.0 and worst.level is RiskLevel.RED
    calm = compute_risk_index(-0.3, -5.0, -1.0, -1.0, -0.2)
    assert calm.score >= 0.0 and calm.level is RiskLevel.GREEN
    assert calm.rationale  # GREEN still explains itself
