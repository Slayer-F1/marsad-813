"""Tests for Stage 1 - ShallowWaterCorrector.

Fixtures are built inline from ``marsad.spectra`` primitives (NOT from
``marsad.synth``, which is developed in parallel): clean spectra are smooth
water-like Gaussian combinations, and observed spectra add the three
contamination terms the corrector is meant to remove - a flat sunglint
offset, a broad bottom-reflectance bump, and multiplicative sensor noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from marsad.spectra import BAND_GRID, N_BANDS, gaussian_feature
from marsad.stage1_correction import ShallowWaterCorrector

SEED = 42
N_TRAIN = 400
N_TEST = 80


def _make_pairs(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (rrs_observed, rrs_true), each (n, N_BANDS) float64.

    Clean: blue shoulder + green peak (+ occasional red-edge bump), all
    Gaussians that decay to ~0 in the SWIR like real water.
    Observed: clean + flat glint offset + broad sandy-bottom bump
    + ~1.5 % multiplicative noise.
    """
    clean = np.empty((n, N_BANDS))
    observed = np.empty((n, N_BANDS))
    for i in range(n):
        blue = gaussian_feature(450.0, 200.0, rng.uniform(0.004, 0.020))
        green = gaussian_feature(555.0, 120.0, rng.uniform(0.005, 0.025))
        red_edge = gaussian_feature(708.0, 30.0, rng.uniform(0.0, 0.005))
        rrs_true = blue + green + red_edge

        glint = rng.uniform(0.001, 0.008)  # spectrally flat, survives into SWIR
        bottom = gaussian_feature(600.0, 400.0, rng.uniform(0.0, 0.012))
        noise = 1.0 + 0.015 * rng.standard_normal(N_BANDS)

        clean[i] = rrs_true
        observed[i] = (rrs_true + glint + bottom) * noise
    return observed, clean


@pytest.fixture(scope="module")
def data() -> dict:
    rng = np.random.default_rng(SEED)
    obs_train, true_train = _make_pairs(N_TRAIN, rng)
    obs_test, true_test = _make_pairs(N_TEST, rng)
    return {
        "obs_train": obs_train,
        "true_train": true_train,
        "obs_test": obs_test,
        "true_test": true_test,
    }


@pytest.fixture(scope="module")
def fitted(data) -> ShallowWaterCorrector:
    corrector = ShallowWaterCorrector(hidden=(64, 32, 64), max_iter=200, seed=SEED)
    result = corrector.fit(data["obs_train"], data["true_train"])
    assert result is corrector  # fit returns self for chaining
    return corrector


def test_transform_shape_and_finite(fitted, data):
    corrected = fitted.transform(data["obs_test"])
    assert corrected.shape == (N_TEST, N_BANDS)
    assert corrected.dtype == np.float64
    assert np.all(np.isfinite(corrected))


def test_rmse_after_beats_rmse_before_on_holdout(fitted, data):
    metrics = fitted.score(data["obs_test"], data["true_test"])
    assert set(metrics) == {"rmse_before", "rmse_after"}
    assert isinstance(metrics["rmse_before"], float)
    assert isinstance(metrics["rmse_after"], float)
    assert metrics["rmse_before"] > 0.0
    # Headline Stage 1 behaviour: the learned correction removes most of
    # the glint + bottom contamination on held-out data.
    assert metrics["rmse_after"] < metrics["rmse_before"]
    assert metrics["rmse_after"] < 0.8 * metrics["rmse_before"]


def test_swir_glint_removed(fitted, data):
    """Clean water is ~black in SWIR; corrected spectra should be too."""
    swir = BAND_GRID >= 1300.0
    corrected = fitted.transform(data["obs_test"])
    obs_swir = float(np.mean(np.abs(data["obs_test"][:, swir])))
    cor_swir = float(np.mean(np.abs(corrected[:, swir])))
    assert cor_swir < obs_swir  # the flat glint offset shrank


def test_save_load_roundtrip(fitted, data, tmp_path):
    path = tmp_path / "stage1.joblib"
    fitted.save(path)
    reloaded = ShallowWaterCorrector.load(path)
    a = fitted.transform(data["obs_test"])
    b = reloaded.transform(data["obs_test"])
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    m1 = fitted.score(data["obs_test"], data["true_test"])
    m2 = reloaded.score(data["obs_test"], data["true_test"])
    assert m1 == m2


def test_unfitted_transform_raises():
    with pytest.raises(RuntimeError):
        ShallowWaterCorrector().transform(np.zeros((3, N_BANDS)))


def test_bad_shape_raises(fitted):
    with pytest.raises(ValueError):
        fitted.transform(np.zeros((5, N_BANDS - 1)))
    with pytest.raises(ValueError):
        ShallowWaterCorrector().fit(
            np.zeros((4, N_BANDS)), np.zeros((5, N_BANDS))
        )
