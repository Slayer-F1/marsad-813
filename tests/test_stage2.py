"""Tests for marsad.stage2_classifier.BloomClassifier.

Fixtures are built inline from marsad.spectra (gaussian_feature / BAND_GRID)
— deliberately NOT from marsad.synth, which is developed in parallel.
Three easy synthetic classes:

- 0 no_bloom: flat-ish blue-green water hump, low chl.
- 1 dinoflagellate: strong 443/675 nm chl absorption dips, 555 nm green
  peak, 708 nm red-edge peak; chl high.
- 2 cyanobacteria: same as dino plus the 620 nm phycocyanin dip.

All randomness is seeded; every test runs in well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from marsad.spectra import BAND_GRID, N_BANDS, gaussian_feature
from marsad.stage2_classifier import BloomClassifier


# --------------------------------------------------------------------------
# Inline fixture generation
# --------------------------------------------------------------------------

def _base_water(rng: np.random.Generator, n: int) -> np.ndarray:
    """Blue-green Rrs hump peaking ~470 nm, decaying to ~0 in the SWIR."""
    shape = np.exp(-0.5 * ((BAND_GRID - 470.0) / 150.0) ** 2)
    amps = rng.uniform(0.8, 1.25, size=(n, 1))
    return 0.006 * amps * shape[None, :]


def _make_class(rng: np.random.Generator, n: int, label: int):
    """Return (rrs, chl) for one synthetic class."""
    rrs = _base_water(rng, n)
    if label == 0:
        chl = rng.uniform(0.1, 1.0, n)
        # mild productive-water variability, no bloom signature
        rrs += rng.uniform(0.0, 3e-4, (n, 1)) * gaussian_feature(555.0, 60.0)[None, :]
    else:
        chl = np.exp(rng.uniform(np.log(8.0), np.log(80.0), n))
        s = (np.log1p(chl) / np.log1p(80.0))[:, None]  # strength ~0.5..1
        rrs -= 0.004 * s * gaussian_feature(443.0, 35.0)[None, :]
        rrs -= 0.003 * s * gaussian_feature(675.0, 30.0)[None, :]
        rrs += 0.003 * s * gaussian_feature(555.0, 70.0)[None, :]
        rrs += 0.004 * s * gaussian_feature(708.0, 25.0)[None, :]
        if label == 2:
            rrs -= 0.0035 * s * gaussian_feature(620.0, 25.0)[None, :]
    # ~1.5 % multiplicative sensor noise, keep Rrs positive
    rrs *= 1.0 + 0.015 * rng.standard_normal(rrs.shape)
    return np.clip(rrs, 1e-6, None), chl


def make_dataset(n_per_class: int = 60, seed: int = 0, classes=(0, 1, 2)):
    """Shuffled (rrs, labels, chl) fixture with the requested classes."""
    rng = np.random.default_rng(seed)
    xs, ys, cs = [], [], []
    for label in classes:
        r, c = _make_class(rng, n_per_class, label)
        xs.append(r)
        ys.append(np.full(n_per_class, label, dtype=int))
        cs.append(c)
    rrs = np.vstack(xs)
    labels = np.concatenate(ys)
    chl = np.concatenate(cs)
    perm = rng.permutation(len(labels))
    return rrs[perm], labels[perm], chl[perm]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_proba_shape_sum_and_predict_consistency():
    rrs, labels, chl = make_dataset(50, seed=1)
    clf = BloomClassifier(seed=0).fit(rrs, labels, chl)
    proba = clf.predict_proba(rrs[:20])
    assert proba.shape == (20, 3)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(clf.predict(rrs[:20]), proba.argmax(axis=1))


def test_proba_columns_follow_label_order_with_missing_class():
    # Train WITHOUT dinoflagellates: column 1 must still exist and be ~0.
    rrs, labels, chl = make_dataset(40, seed=2, classes=(0, 2))
    clf = BloomClassifier(seed=0).fit(rrs, labels, chl)
    proba = clf.predict_proba(rrs)
    assert proba.shape == (len(labels), 3)
    np.testing.assert_allclose(proba[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # Column order 0,1,2 respected: each true class peaks in its own column.
    assert np.mean(proba[labels == 2].argmax(axis=1) == 2) > 0.9
    assert np.mean(proba[labels == 0].argmax(axis=1) == 0) > 0.9


def test_single_class_degenerate_fit():
    rrs, labels, chl = make_dataset(30, seed=3, classes=(1,))
    clf = BloomClassifier(seed=0).fit(rrs, labels, chl)
    proba = clf.predict_proba(rrs[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba[:, 1], 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_accuracy_and_evaluate_on_easy_fixture():
    rrs_tr, y_tr, chl_tr = make_dataset(80, seed=4)
    rrs_te, y_te, _ = make_dataset(40, seed=5)
    clf = BloomClassifier(seed=0).fit(rrs_tr, y_tr, chl_tr)
    report = clf.evaluate(rrs_te, y_te)

    assert report["accuracy"] > 0.85
    cm = report["confusion"]
    assert isinstance(cm, list) and len(cm) == 3
    assert all(isinstance(row, list) and len(row) == 3 for row in cm)
    assert all(isinstance(v, int) for row in cm for v in row)
    assert sum(sum(row) for row in cm) == len(y_te)
    diag = sum(cm[i][i] for i in range(3))
    assert diag / len(y_te) == pytest.approx(report["accuracy"])


def test_chl_estimates_positively_correlated():
    rrs_tr, y_tr, chl_tr = make_dataset(80, seed=6)
    rrs_te, _, chl_te = make_dataset(40, seed=7)
    clf = BloomClassifier(seed=0).fit(rrs_tr, y_tr, chl_tr)
    est = clf.estimate_chl(rrs_te)
    assert est.shape == (len(chl_te),)
    assert np.all(np.isfinite(est)) and np.all(est >= 0.0)
    r = np.corrcoef(np.log1p(chl_te), np.log1p(est))[0, 1]
    assert r > 0.6


def test_save_load_round_trip(tmp_path):
    rrs, labels, chl = make_dataset(40, seed=8)
    clf = BloomClassifier(seed=0).fit(rrs, labels, chl)
    path = tmp_path / "stage2.joblib"
    clf.save(path)
    loaded = BloomClassifier.load(path)
    np.testing.assert_allclose(loaded.predict_proba(rrs), clf.predict_proba(rrs))
    np.testing.assert_allclose(loaded.estimate_chl(rrs), clf.estimate_chl(rrs))
    assert np.array_equal(loaded.predict(rrs), clf.predict(rrs))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        BloomClassifier().predict_proba(np.zeros((2, N_BANDS)))
