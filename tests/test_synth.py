"""Tests for marsad.synth — synthetic Gulf-water scene generator.

Fast, seeded checks of the contract: shapes/dtypes, physical value ranges,
label mixture, phycocyanin confined to cyanobacteria, the shallow-water
contamination asymmetry (observed vs true differ on shallow pixels, converge
on deep ones, and naive band ratios degrade under contamination), class
separability on clean spectra, and the history endpoint/bound conditions.
"""
import numpy as np
import pytest

from marsad.spectra import BAND_GRID, N_BANDS, band_index
from marsad.synth import LABELS, SynthDataset, generate_dataset, generate_history


def test_labels_mapping():
    assert LABELS == {0: "no_bloom", 1: "dinoflagellate", 2: "cyanobacteria"}


def test_shapes_and_dtypes():
    n = 400
    ds = generate_dataset(n, seed=1)
    assert isinstance(ds, SynthDataset)
    assert ds.rrs_observed.shape == (n, N_BANDS)
    assert ds.rrs_true.shape == (n, N_BANDS)
    assert ds.rrs_observed.dtype == np.float64
    assert ds.rrs_true.dtype == np.float64
    for field in (ds.labels, ds.chl, ds.tss, ds.depth_m, ds.phycocyanin):
        assert field.shape == (n,)
    assert np.issubdtype(ds.labels.dtype, np.integer)


def test_seed_determinism():
    a = generate_dataset(120, seed=42)
    b = generate_dataset(120, seed=42)
    c = generate_dataset(120, seed=43)
    np.testing.assert_array_equal(a.rrs_observed, b.rrs_observed)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.array_equal(a.rrs_observed, c.rrs_observed)


def test_value_ranges():
    ds = generate_dataset(1500, seed=2)
    assert np.isfinite(ds.rrs_true).all() and np.isfinite(ds.rrs_observed).all()
    # Clean Rrs: non-negative, typical ocean-colour magnitudes (< 0.05 sr^-1).
    assert ds.rrs_true.min() >= 0.0
    assert ds.rrs_true.max() < 0.05
    # Contaminated spectra may exceed 0.05 but stay physically bounded.
    assert ds.rrs_observed.min() >= 0.0
    assert ds.rrs_observed.max() < 0.25
    # Clean water is ~black in the SWIR; observed SWIR carries glint.
    swir = BAND_GRID > 1500.0
    blue = (BAND_GRID > 420.0) & (BAND_GRID < 500.0)
    assert ds.rrs_true[:, swir].mean() < 1e-3
    assert ds.rrs_true[:, swir].mean() < 0.05 * ds.rrs_true[:, blue].mean()
    assert ds.rrs_observed[:, swir].mean() > ds.rrs_true[:, swir].mean()
    # Biogeochemistry positive; blooms are high-chl.
    assert (ds.chl > 0).all() and (ds.tss > 0).all() and (ds.depth_m > 0).all()
    assert (ds.phycocyanin >= 0).all()
    assert np.median(ds.chl[ds.labels == 1]) > 5 * np.median(ds.chl[ds.labels == 0])


def test_label_distribution():
    ds = generate_dataset(3000, seed=0)
    counts = np.bincount(ds.labels, minlength=3)
    assert counts.sum() == 3000
    fracs = counts / 3000.0
    assert (fracs > 0.15).all() and (fracs < 0.55).all()
    assert set(np.unique(ds.labels)) == {0, 1, 2}


def test_phycocyanin_concentrated_in_label2():
    ds = generate_dataset(2000, seed=4)
    cy = ds.labels == 2
    assert (ds.phycocyanin[cy] > 0).all()
    assert (ds.phycocyanin[~cy] == 0).all()
    # The 620 nm phycocyanin absorption line is deeper for cyanobacteria than
    # for a comparably dense dinoflagellate bloom (their key spectral contrast).
    i600, i620, i640 = band_index(600), band_index(620), band_index(640)
    line = 0.5 * (ds.rrs_true[:, i600] + ds.rrs_true[:, i640]) - ds.rrs_true[:, i620]
    assert line[cy].mean() > 1.5 * line[ds.labels == 1].mean()


def test_cyanobacteria_pixels_are_shallow():
    ds = generate_dataset(1000, seed=5, shallow_fraction=0.0)
    cy = ds.labels == 2
    # Inland reservoirs are always shallow, even with shallow_fraction=0 ...
    assert (ds.depth_m[cy] < 10.0).all()
    # ... while sea pixels are all deep at shallow_fraction=0.
    assert (ds.depth_m[~cy] > 10.0).all()
    # And at shallow_fraction=1 everything is shallow.
    ds_all = generate_dataset(500, seed=5, shallow_fraction=1.0)
    assert (ds_all.depth_m < 10.0).all()


def test_observed_differs_on_shallow_converges_on_deep():
    ds = generate_dataset(2000, seed=3)
    shallow = ds.depth_m < 10.0
    diff = np.abs(ds.rrs_observed - ds.rrs_true).mean(axis=1)
    # Bottom reflectance makes shallow pixels genuinely contaminated ...
    assert diff[shallow].mean() > 0.0015
    # ... while deep pixels carry only glint + 1-2 % noise ...
    assert diff[~shallow].mean() < 0.0015
    # ... so the contamination is strongly asymmetric (the Stage 1 target).
    assert diff[shallow].mean() > 2.5 * diff[~shallow].mean()


def test_naive_band_ratio_degrades_under_contamination():
    """The project's core claim: contamination corrupts naive band ratios.

    The blue/green ratio (443/555) tracks log-chl almost perfectly on clean
    spectra, but flat bottom + glint offsets push the ratio toward 1 on
    shallow pixels, weakening the relationship.
    """
    ds = generate_dataset(2000, seed=7)
    shallow = ds.depth_m < 10.0
    i443, i555 = band_index(443), band_index(555)
    logchl = np.log10(ds.chl[shallow])
    r_true = np.corrcoef(
        ds.rrs_true[shallow, i443] / ds.rrs_true[shallow, i555], logchl
    )[0, 1]
    obs_ratio = ds.rrs_observed[shallow, i443] / np.clip(
        ds.rrs_observed[shallow, i555], 1e-9, None
    )
    r_obs = np.corrcoef(obs_ratio, logchl)[0, 1]
    assert abs(r_true) > 0.9  # clean ratio is an excellent chl proxy
    assert abs(r_true) > abs(r_obs) + 0.05  # contamination measurably degrades it


def test_classes_separable_on_clean_spectra():
    """A plain linear classifier must exceed 0.85 holdout accuracy on rrs_true."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = generate_dataset(1500, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(
        ds.rrs_true, ds.labels, test_size=0.33, random_state=0, stratify=ds.labels
    )
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
    assert clf.score(scaler.transform(Xte), yte) > 0.85


def test_history_endpoint_and_bounds():
    for score in (0.0, 0.37, 1.0):
        h = generate_history(score, n_days=30, seed=5)
        assert h.shape == (30,)
        assert h[-1] == score  # EXACT endpoint condition
        assert (h >= 0.0).all() and (h <= 1.0).all()
    # Single-day history is just today's score.
    np.testing.assert_array_equal(generate_history(0.42, n_days=1, seed=0), [0.42])
    # Out-of-range inputs are clipped into [0, 1].
    assert generate_history(1.7, n_days=5, seed=0)[-1] == 1.0
    with pytest.raises(ValueError):
        generate_history(0.5, n_days=0, seed=0)


def test_history_autocorrelated_walk():
    h = generate_history(0.5, n_days=200, seed=2)
    # Not constant, but smooth day to day (bounded random walk, no jumps).
    assert h.std() > 0.01
    assert np.abs(np.diff(h)).max() < 0.3
    # Strong positive lag-1 autocorrelation (AR(1) with phi ~ 0.88).
    dev = h - h.mean()
    lag1 = (dev[:-1] * dev[1:]).mean() / (dev**2).mean()
    assert lag1 > 0.5
    # Deterministic in the seed.
    np.testing.assert_array_equal(h, generate_history(0.5, n_days=200, seed=2))
