"""Tests for marsad.uncertainty (ensemble uncertainty + calibration).

Fixtures are built inline from marsad.spectra (gaussian_feature /
BAND_GRID) - deliberately NOT from marsad.synth, which is developed in
parallel. Two contrasting populations drive the whole file:

- **separable**: the three MARSAD classes with their textbook
  signatures (no_bloom water; dinoflagellate bloom with 443/675 nm
  absorption, 555 nm green peak and a 708 nm red edge; cyanobacteria =
  the same plus a full-depth 620 nm phycocyanin dip). The model should
  be confident here.
- **ambiguous**: bloom spectra whose 620 nm dip sits exactly halfway
  between the two bloom classes, with the label drawn by a coin flip.
  No classifier can do better than 50/50 on these, so the uncertainty
  has to come out high - that is the whole point of the module.

The ensemble is fitted ONCE per module (module-scoped fixture) because
each boosted member costs ~2 s; three members keep the file in
single-digit seconds while still producing member disagreement.

Honesty note: these fixtures are a hand-built forward model, like
marsad.synth. Numbers here are self-consistency checks against a
simulation, never validation against real Gulf water.
"""
from __future__ import annotations

import numpy as np
import pytest

from marsad.spectra import BAND_GRID, N_BANDS, gaussian_feature
from marsad.stage2_classifier import BloomClassifier
from marsad.uncertainty import (
    EnsembleClassifier,
    expected_calibration_error,
    reliability_curve,
    review_queue,
)

# Full-depth phycocyanin (620 nm) dip amplitude for a dense cyano bloom.
_PC_FULL = 0.0035


# --------------------------------------------------------------------------
# Inline fixture generation
# --------------------------------------------------------------------------

def _base_water(rng: np.random.Generator, n: int) -> np.ndarray:
    """Blue-green Rrs hump peaking ~470 nm, decaying to ~0 in the SWIR."""
    shape = np.exp(-0.5 * ((BAND_GRID - 470.0) / 150.0) ** 2)
    return 0.006 * rng.uniform(0.8, 1.25, size=(n, 1)) * shape[None, :]


def _clear(rng: np.random.Generator, n: int, noise: float = 0.015):
    """Class 0: productive-but-clear water, low chl, no bloom signature."""
    rrs = _base_water(rng, n)
    chl = rng.uniform(0.1, 1.0, n)
    rrs += rng.uniform(0.0, 3e-4, (n, 1)) * gaussian_feature(555.0, 60.0)[None, :]
    rrs *= 1.0 + noise * rng.standard_normal(rrs.shape)
    return np.clip(rrs, 1e-6, None), chl


def _bloom(rng: np.random.Generator, n: int, pc_fraction, noise: float = 0.015):
    """Dense bloom with a tunable 620 nm phycocyanin dip.

    ``pc_fraction`` 0 gives a dinoflagellate spectrum, 1 a cyanobacteria
    spectrum, and 0.5 a spectrum that is genuinely between the two.
    """
    rrs = _base_water(rng, n)
    chl = np.exp(rng.uniform(np.log(8.0), np.log(80.0), n))
    strength = (np.log1p(chl) / np.log1p(80.0))[:, None]
    rrs -= 0.004 * strength * gaussian_feature(443.0, 35.0)[None, :]
    rrs -= 0.003 * strength * gaussian_feature(675.0, 30.0)[None, :]
    rrs += 0.003 * strength * gaussian_feature(555.0, 70.0)[None, :]
    rrs += 0.004 * strength * gaussian_feature(708.0, 25.0)[None, :]
    pc = np.asarray(pc_fraction, dtype=float).reshape(-1, 1)
    rrs -= _PC_FULL * pc * strength * gaussian_feature(620.0, 25.0)[None, :]
    rrs *= 1.0 + noise * rng.standard_normal(rrs.shape)
    return np.clip(rrs, 1e-6, None), chl


def make_separable(n_per_class: int = 50, seed: int = 0):
    """Confidently separable (rrs, labels, chl) covering all three classes."""
    rng = np.random.default_rng(seed)
    r0, c0 = _clear(rng, n_per_class)
    r1, c1 = _bloom(rng, n_per_class, np.zeros(n_per_class))
    r2, c2 = _bloom(rng, n_per_class, np.ones(n_per_class))
    rrs = np.vstack([r0, r1, r2])
    chl = np.concatenate([c0, c1, c2])
    labels = np.repeat([0, 1, 2], n_per_class)
    perm = rng.permutation(len(labels))
    return rrs[perm], labels[perm], chl[perm]


def make_ambiguous(n: int = 60, seed: int = 0):
    """Half-depth 620 nm dip with coin-flip dino/cyano labels.

    Irreducible (aleatoric) ambiguity by construction: the spectrum
    carries no information about which of the two bloom labels applies.
    """
    rng = np.random.default_rng(seed)
    rrs, chl = _bloom(rng, n, rng.uniform(0.47, 0.53, n), noise=0.05)
    labels = rng.integers(1, 3, n)
    return rrs, labels, chl


@pytest.fixture(scope="module")
def fitted():
    """Ensemble trained on separable + ambiguous water, fitted once."""
    rrs_s, y_s, chl_s = make_separable(60, seed=1)
    rrs_a, y_a, chl_a = make_ambiguous(120, seed=2)
    rrs = np.vstack([rrs_s, rrs_a])
    labels = np.concatenate([y_s, y_a])
    chl = np.concatenate([chl_s, chl_a])
    return EnsembleClassifier(n_members=3, seed=0).fit(rrs, labels, chl)


@pytest.fixture(scope="module")
def holdout():
    """Fresh separable and ambiguous test sets (different seeds)."""
    return make_separable(30, seed=11), make_ambiguous(40, seed=12)


# --------------------------------------------------------------------------
# Ensemble API
# --------------------------------------------------------------------------

def test_defaults_and_unfitted_raises():
    ens = EnsembleClassifier()
    assert ens.n_members == 5  # contract default
    assert ens.members == []
    with pytest.raises(RuntimeError):
        ens.predict_proba(np.zeros((2, N_BANDS)))
    with pytest.raises(RuntimeError):
        ens.uncertainty(np.zeros((2, N_BANDS)))
    with pytest.raises(ValueError):
        EnsembleClassifier(n_members=0)


def test_bootstrap_resample_keeps_every_class():
    # One single cyanobacteria row in 60: a plain bootstrap misses it
    # about a third of the time, and the resample must repair that.
    labels = np.zeros(60, dtype=int)
    labels[1:20] = 1
    labels[0] = 2
    rng = np.random.default_rng(0)
    for _ in range(200):
        idx = EnsembleClassifier._bootstrap_index(rng, labels.size, labels)
        assert idx.shape == (labels.size,)
        assert set(np.unique(labels[idx]).tolist()) == {0, 1, 2}


def test_predict_proba_shape_sums_and_members(fitted, holdout):
    (rrs, labels, _), _ = holdout
    proba = fitted.predict_proba(rrs)
    assert proba.shape == (len(labels), 3)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.array_equal(fitted.predict(rrs), proba.argmax(axis=1))

    members = fitted.member_probas(rrs)
    assert members.shape == (fitted.n_members, len(labels), 3)
    np.testing.assert_allclose(members.mean(axis=0), proba)
    # The separable holdout is easy: the ensemble should nail it.
    assert np.mean(fitted.predict(rrs) == labels) > 0.9


def test_estimate_chl_tracks_truth(fitted, holdout):
    (rrs, _, chl), _ = holdout
    est = fitted.estimate_chl(rrs)
    assert est.shape == (len(chl),)
    assert np.all(np.isfinite(est)) and np.all(est >= 0.0)
    assert np.corrcoef(np.log1p(chl), np.log1p(est))[0, 1] > 0.6


def test_uncertainty_keys_shapes_and_ranges(fitted, holdout):
    (rrs, labels, _), _ = holdout
    unc = fitted.uncertainty(rrs)
    assert set(unc) == {"total", "epistemic", "aleatoric", "confidence"}
    for key, value in unc.items():
        assert value.shape == (len(labels),), key
        assert np.all(np.isfinite(value)), key
        assert np.all(value >= 0.0) and np.all(value <= 1.0), key

    # Mutual information can never exceed the predictive entropy.
    assert np.all(unc["epistemic"] <= unc["total"] + 1e-9)
    # The contract's decomposition, exactly.
    np.testing.assert_allclose(
        unc["aleatoric"], unc["total"] - unc["epistemic"], atol=1e-12
    )
    # Confidence is the top mean probability, so it is bounded below by 1/3.
    assert np.all(unc["confidence"] >= 1.0 / 3.0 - 1e-9)
    np.testing.assert_allclose(
        unc["confidence"], fitted.predict_proba(rrs).max(axis=1)
    )


def test_confident_input_low_uncertainty_ambiguous_input_high(fitted, holdout):
    (rrs_sep, _, _), (rrs_amb, _, _) = holdout
    sep = fitted.uncertainty(rrs_sep)
    amb = fitted.uncertainty(rrs_amb)

    assert sep["total"].mean() < 0.15
    assert amb["total"].mean() > 0.25
    assert amb["total"].mean() > 3.0 * sep["total"].mean()
    # A coin-flip label is aleatoric, so that term must rise too, and the
    # ensemble must be visibly less confident overall.
    assert amb["aleatoric"].mean() > sep["aleatoric"].mean()
    assert amb["confidence"].mean() < sep["confidence"].mean()
    # Normalisation sanity: a 50/50 posterior over 2 of 3 classes has
    # entropy log(2)/log(3) = 0.631, which is the ceiling these fixtures
    # can reach.
    assert amb["total"].max() <= 1.0


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------

def test_review_queue_flags_ambiguous_not_confident(fitted, holdout):
    (rrs_sep, _, _), (rrs_amb, _, _) = holdout
    flag_sep = review_queue(fitted.uncertainty(rrs_sep))
    flag_amb = review_queue(fitted.uncertainty(rrs_amb))

    assert flag_sep.dtype == bool and flag_amb.dtype == bool
    assert flag_sep.shape == (rrs_sep.shape[0],)
    assert flag_amb.shape == (rrs_amb.shape[0],)
    assert flag_sep.mean() < 0.15   # confident pixels auto-alert
    assert flag_amb.mean() > 0.5    # ambiguous pixels go to an analyst


def test_review_queue_threshold_semantics():
    unc = {"total": np.array([0.0, 0.34, 0.35, 0.36, 1.0])}
    np.testing.assert_array_equal(
        review_queue(unc), [False, False, False, True, True]
    )
    assert review_queue(unc, threshold=0.0).sum() == 4      # strict ">"
    assert review_queue(unc, threshold=1.0).sum() == 0
    with pytest.raises(KeyError):
        review_queue({"epistemic": np.zeros(3)})


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_ece_bounds_and_extremes(fitted, holdout):
    (rrs_sep, y_sep, _), (rrs_amb, y_amb, _) = holdout
    probs = fitted.predict_proba(np.vstack([rrs_sep, rrs_amb]))
    labels = np.concatenate([y_sep, y_amb])
    ece = expected_calibration_error(probs, labels)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0

    # Perfectly confident and perfectly right: ECE 0.
    one_hot = np.eye(3)[np.array([0, 1, 2, 1])]
    assert expected_calibration_error(one_hot, [0, 1, 2, 1]) == pytest.approx(0.0)
    # Perfectly confident and always wrong: ECE 1.
    assert expected_calibration_error(one_hot, [1, 2, 0, 0]) == pytest.approx(1.0)
    # A uniform posterior is 1/3 confident; accuracy here is 1/3 too.
    uniform = np.full((3, 3), 1.0 / 3.0)
    assert expected_calibration_error(uniform, [0, 1, 1]) == pytest.approx(0.0)


def test_reliability_curve_shape_and_empty_bins(fitted, holdout):
    (rrs_sep, y_sep, _), (rrs_amb, y_amb, _) = holdout
    probs = fitted.predict_proba(np.vstack([rrs_sep, rrs_amb]))
    labels = np.concatenate([y_sep, y_amb])
    curve = reliability_curve(probs, labels, n_bins=10)

    assert set(curve) == {"bin_confidence", "bin_accuracy", "bin_count"}
    n_kept = len(curve["bin_count"])
    assert n_kept == len(curve["bin_confidence"]) == len(curve["bin_accuracy"])
    assert 0 < n_kept <= 10                       # empty bins dropped
    assert all(c > 0 for c in curve["bin_count"])  # ... and none survives empty
    assert sum(curve["bin_count"]) == len(labels)
    assert all(0.0 <= v <= 1.0 for v in curve["bin_confidence"])
    assert all(0.0 <= v <= 1.0 for v in curve["bin_accuracy"])
    # JSON-serialisable plain Python types for dashboard/data.js.
    assert all(isinstance(v, float) for v in curve["bin_confidence"])
    assert all(isinstance(v, int) for v in curve["bin_count"])


def test_reliability_curve_perfect_model_sits_on_the_diagonal():
    one_hot = np.eye(3)[np.array([0, 1, 2, 2, 1])]
    curve = reliability_curve(one_hot, [0, 1, 2, 2, 1], n_bins=5)
    assert curve["bin_count"] == [5]              # one bin, four empty ones gone
    assert curve["bin_confidence"] == [pytest.approx(1.0)]
    assert curve["bin_accuracy"] == [pytest.approx(1.0)]


def test_calibration_input_validation():
    with pytest.raises(ValueError):
        expected_calibration_error(np.ones((4, 2)) / 2.0, [0, 1, 0, 1])
    with pytest.raises(ValueError):
        expected_calibration_error(np.eye(3)[[0, 1, 2]], [0, 1])
    with pytest.raises(ValueError):
        reliability_curve(np.eye(3)[[0, 1, 2]], [0, 1, 2], n_bins=0)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_save_load_round_trip(fitted, holdout, tmp_path):
    (rrs, _, _), _ = holdout
    path = tmp_path / "ensemble.joblib"
    fitted.save(path)
    loaded = EnsembleClassifier.load(path)

    assert loaded.n_members == fitted.n_members
    assert len(loaded.members) == len(fitted.members)
    np.testing.assert_allclose(loaded.predict_proba(rrs), fitted.predict_proba(rrs))
    np.testing.assert_allclose(loaded.estimate_chl(rrs), fitted.estimate_chl(rrs))
    before = fitted.uncertainty(rrs)
    after = loaded.uncertainty(rrs)
    for key in before:
        np.testing.assert_allclose(after[key], before[key])


def test_bloom_classifier_accepts_sample_weight():
    """The v0.2 pass-through the ensemble contract depends on.

    Weights must actually reach the boosting heads, not be swallowed:
    re-weighting the training set has to change the fitted posterior.
    """
    rrs, labels, chl = make_separable(8, seed=21)
    plain = BloomClassifier(seed=0).fit(rrs, labels, chl)
    weights = np.where(labels == 0, 5.0, 1.0)
    weighted = BloomClassifier(seed=0).fit(rrs, labels, chl, sample_weight=weights)

    assert isinstance(weighted, BloomClassifier)
    probs = weighted.predict_proba(rrs)
    assert probs.shape == (len(labels), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert not np.allclose(probs, plain.predict_proba(rrs))
    assert weighted.estimate_chl(rrs).shape == (len(labels),)

    with pytest.raises(ValueError):
        BloomClassifier(seed=0).fit(rrs, labels, chl, sample_weight=np.ones(len(labels) + 1))


def test_fit_input_validation():
    rrs, labels, chl = make_separable(4, seed=3)
    ens = EnsembleClassifier(n_members=2, seed=0)
    with pytest.raises(ValueError):
        ens.fit(rrs[:, :10], labels, chl)
    with pytest.raises(ValueError):
        ens.fit(rrs, labels[:-1], chl)
