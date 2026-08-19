"""Tests for marsad.baselines - the standard operational algorithms.

The fixtures here are built inline from :func:`marsad.spectra.gaussian_feature`
(deliberately not from :mod:`marsad.synth`, so this file tests the algorithms
rather than a second forward model). Two families:

* ``_clean_spectra`` - stylized optically deep, Case-1-like water: a blue-bright
  continuum decaying into the red, chlorophyll-a red absorption at 675 nm, an
  optional phycocyanin absorption at 620 nm, a red-edge peak at 708 nm for
  dense blooms, and a green peak whose amplitude is solved so the blue-green
  ratio follows the empirical Case-1 relation OC4 was calibrated on, plus 1.5 %
  multiplicative sensor noise. On this water OC4 is *supposed* to work, so it
  is the right place to check the implementation (band selection, coefficient
  order, polynomial evaluation) rather than the physics.
* ``_contaminate`` - the same spectra plus the two contaminations that define
  the Gulf-coast problem: a broad sandy-bottom reflectance lift attenuated by
  the two-way water path, and a spectrally flat sunglint offset.

The asymmetry between those two families is the project's core claim, so it is
asserted numerically (test_oc4_degrades_under_bottom_and_glint), not narrated.

Honesty: these are constructed fixtures from our own forward model, so nothing
here validates any algorithm against real Gulf water. It shows that a published
band-ratio algorithm behaves as published on water that satisfies its
assumptions and breaks in the documented direction when those assumptions are
violated, which is consistent with the Case-2 water literature.
"""
import warnings

import numpy as np
import pytest

from marsad import baselines
from marsad.spectra import BAND_GRID, N_BANDS, band_index, gaussian_feature

# Nominal band indices on the 813 grid, used for the hand-computed checks.
I443, I490, I510 = band_index(443.0), band_index(490.0), band_index(510.0)
I555 = band_index(555.0)
I600, I620, I650 = band_index(600.0), band_index(620.0), band_index(650.0)
I665, I708 = band_index(665.0), band_index(708.0)


def _oc4_target_ratio(chl: np.ndarray) -> np.ndarray:
    """Blue-green ratio that the OC4 polynomial maps back to ``chl``.

    Numeric inverse of the published polynomial, which is strictly decreasing
    over the band-ratio range where OC4 is defined (R in -0.35 to 1.0, i.e.
    chl from ~0.02 to ~57 mg m^-3). Building a fixture through this inverse is
    what makes it "Case-1-like": its blue-green ratio obeys the empirical
    relation the algorithm was fitted to.
    """
    r_grid = np.linspace(-0.35, 1.0, 30001)
    log_chl = np.polynomial.polynomial.polyval(r_grid, baselines.OC4_COEFFS)
    return 10.0 ** np.interp(np.log10(chl), log_chl[::-1], r_grid[::-1])


def _clean_spectra(chl, phycocyanin=0.0, noise=0.015, seed=0) -> np.ndarray:
    """Stylized clean, optically deep Rrs spectra (n, N_BANDS) in sr^-1.

    ``phycocyanin`` is the fractional depth of the 620 nm absorption feature
    (0 for a dinoflagellate bloom, ~0.5 for cyanobacteria).
    """
    rng = np.random.default_rng(seed)
    chl = np.atleast_1d(np.asarray(chl, dtype=float))
    n = chl.size

    # Blue-bright continuum: Rrs falls roughly exponentially into the red, and
    # is effectively black beyond ~900 nm (pure-water absorption).
    amplitude = rng.uniform(0.009, 0.013, n)
    decay_nm = rng.uniform(100.0, 118.0, n)
    base = amplitude[:, None] * np.exp(-(BAND_GRID - 400.0)[None, :] / decay_nm[:, None])

    # Pigment absorption dips act multiplicatively on the upwelling light;
    # forcing is logarithmic in chl because absorption saturates (packaging).
    chl_forcing = np.clip((np.log10(chl) + 1.0) / 3.0, 0.0, 1.0)
    pc = np.broadcast_to(np.asarray(phycocyanin, dtype=float), (n,))
    dips = (0.40 * chl_forcing)[:, None] * gaussian_feature(675.0, 30.0) + pc[
        :, None
    ] * gaussian_feature(620.0, 30.0)
    spec = base * np.clip(1.0 - dips, 0.05, None)

    # Red-edge scattering peak, dense surface blooms only (onset ~20 mg/m3).
    rededge = 0.0020 / (1.0 + np.exp(-(chl - 20.0) / 4.0))
    spec = spec + rededge[:, None] * gaussian_feature(708.0, 22.0)

    # Solve the green-peak amplitude so max(443, 490, 510) / 555 equals the
    # Case-1 ratio for this chl. Bisection: the ratio decreases monotonically
    # in the green amplitude (the 555 nm band gains far more than 510 nm).
    green = gaussian_feature(555.0, 40.0)
    target = _oc4_target_ratio(chl)
    lo, hi = np.zeros(n), np.full(n, 0.08)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        trial = spec + mid[:, None] * green[None, :]
        ratio = trial[:, [I443, I490, I510]].max(axis=1) / trial[:, I555]
        need_more_green = ratio > target
        lo = np.where(need_more_green, mid, lo)
        hi = np.where(need_more_green, hi, mid)
    spec = spec + (0.5 * (lo + hi))[:, None] * green[None, :]

    # 1.5 % multiplicative sensor noise, so recovery is close but not exact.
    spec = spec * (1.0 + noise * rng.standard_normal((n, N_BANDS)))
    return np.clip(spec, 0.0, None)


def _contaminate(rrs: np.ndarray, chl: np.ndarray, seed: int = 0) -> np.ndarray:
    """Add a broad sandy-bottom lift and a flat sunglint offset.

    Bottom reflectance reaches the sensor as ``R_bottom * exp(-2 * Kd * depth)``:
    bright in the green-red where the sand is bright and the water is clear,
    extinguished beyond ~730 nm where water absorption takes over. Sunglint is
    a spectrally flat specular offset. Neither carries any information about
    chlorophyll, and both move the blue-green ratio.
    """
    rng = np.random.default_rng(seed)
    n = rrs.shape[0]
    chl = np.atleast_1d(np.asarray(chl, dtype=float))

    bottom_shape = 0.6 + 0.4 * (BAND_GRID - 400.0) / 1300.0
    kd = (0.045 + 2.5 / (1.0 + np.exp(-(BAND_GRID - 730.0) / 55.0)))[None, :] + (
        0.035 * np.log1p(chl) + 0.09
    )[:, None]
    depth_m = rng.uniform(3.0, 7.0, n)
    albedo = rng.uniform(0.15, 0.35, n)
    bottom = (
        (albedo / np.pi)[:, None]
        * bottom_shape[None, :]
        * np.exp(-2.0 * kd * depth_m[:, None])
    )
    glint = rng.uniform(0.0004, 0.0014, n)
    return rrs + bottom + glint[:, None]


def _log10_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Absolute log10 error, the standard ocean-colour retrieval metric."""
    return np.abs(np.log10(estimate / truth))


@pytest.fixture(scope="module")
def deep_water():
    """300 clean deep-water pixels spanning 0.3-45 mg/m3 chlorophyll-a."""
    rng = np.random.default_rng(0)
    chl = 10.0 ** rng.uniform(np.log10(0.3), np.log10(45.0), 300)
    return chl, _clean_spectra(chl, seed=1)


# --------------------------------------------------------------- API shape


def test_shapes_dtypes_and_validation(deep_water):
    chl, rrs = deep_water
    n = rrs.shape[0]
    for fn in (
        baselines.oc4_chl,
        baselines.oc3m_chl,
        baselines.ndci,
        baselines.ndci_chl,
        baselines.red_nir_ratio,
        baselines.phycocyanin_line_height,
        baselines.turbidity_proxy,
    ):
        out = fn(rrs)
        assert out.shape == (n,), fn.__name__
        assert out.dtype == np.float64, fn.__name__
        assert np.isfinite(out).all(), fn.__name__
        # A single spectrum is accepted and promoted to one row.
        assert fn(rrs[0]).shape == (1,), fn.__name__

    labels = baselines.classify_baseline(rrs)
    assert labels.shape == (n,)
    assert np.issubdtype(labels.dtype, np.integer)

    with pytest.raises(ValueError):
        baselines.oc4_chl(np.zeros((4, N_BANDS - 3)))


def test_published_coefficients_unchanged():
    """The contract pins the literature coefficients; guard against drift."""
    assert baselines.OC4_COEFFS == (0.3272, -2.9940, 2.7218, -1.2259, -0.5683)
    assert baselines.OC3M_COEFFS == (0.2424, -2.7423, 1.8017, 0.0015, -1.2280)
    assert baselines.NDCI_CHL_COEFFS == (14.039, 86.115, 194.325)


# ------------------------------------------------- OC4 / OC3M chlorophyll


def test_oc4_recovers_chl_on_clean_deep_water(deep_water):
    """OC4 works where its Case-1 assumption holds - to within sensor noise."""
    chl, rrs = deep_water
    est = baselines.oc4_chl(rrs)
    assert (est > 0).all() and np.isfinite(est).all()

    err = _log10_error(est, chl)
    # Median within a few percent, worst case well inside a factor of two.
    assert np.median(err) < 0.10
    assert np.percentile(err, 90) < 0.20
    assert err.max() < 0.30
    # And it is a retrieval, not a constant: it tracks chl over two decades.
    assert np.corrcoef(np.log10(est), np.log10(chl))[0, 1] > 0.98


def test_oc4_degrades_under_bottom_and_glint(deep_water):
    """The core MARSAD claim, stated numerically.

    Identical water, identical chlorophyll: adding a bright shallow bottom and
    sunglint - contaminations that carry no chlorophyll information at all -
    costs OC4 more than an order of magnitude in retrieval error. This is the
    asymmetry Stage 1 exists to remove. It is measured on our own forward
    model, so it is a self-consistency check that reproduces the documented
    Case-2 failure mode, not independent validation on real Gulf water.
    """
    chl, clean = deep_water
    contaminated = _contaminate(clean, chl, seed=3)

    err_clean = _log10_error(baselines.oc4_chl(clean), chl)
    err_dirty = _log10_error(baselines.oc4_chl(contaminated), chl)

    # The contamination is a realistic perturbation, not a wrecking ball: the
    # added signal is comparable to the water-leaving signal itself.
    lift = np.median((contaminated - clean)[:, I443] / clean[:, I443])
    assert 0.2 < lift < 3.0

    # Median pixel lands close to a factor of two out, the tail far worse,
    # against a clean median error of a few percent.
    assert np.median(err_dirty) > 0.25
    assert np.percentile(err_dirty, 90) > 0.45
    assert np.median(err_dirty) > 4.0 * np.median(err_clean)
    assert (err_dirty > err_clean).mean() > 0.75
    # Every retrieval stays finite and non-negative even when badly wrong.
    assert np.isfinite(err_dirty).all()


def test_oc3m_tracks_chl_and_degrades_too(deep_water):
    """OC3M is the same empirical family, so it inherits the same failure."""
    chl, clean = deep_water
    contaminated = _contaminate(clean, chl, seed=4)
    est_clean = baselines.oc3m_chl(clean)
    est_dirty = baselines.oc3m_chl(contaminated)

    assert (est_clean > 0).all() and np.isfinite(est_dirty).all()
    assert np.corrcoef(np.log10(est_clean), np.log10(chl))[0, 1] > 0.98
    assert np.median(_log10_error(est_dirty, chl)) > np.median(
        _log10_error(est_clean, chl)
    )


def test_oc4_uses_the_maximum_blue_band(deep_water):
    """The 'maximum band ratio' really is a maximum over 443/490/510."""
    _, rrs = deep_water
    boosted = rrs.copy()
    # Lift only the weakest blue band far above the others: if the maximum is
    # honoured, the ratio rises and the retrieved chl must fall.
    weakest = np.argmin(rrs[:, [I443, I490, I510]], axis=1)
    cols = np.array([I443, I490, I510])[weakest]
    boosted[np.arange(rrs.shape[0]), cols] = rrs[:, [I443, I490, I510]].max(axis=1) * 3.0
    assert (baselines.oc4_chl(boosted) < baselines.oc4_chl(rrs)).all()


# ------------------------------------------------------------ red / red-edge


def test_ndci_definition_and_range(deep_water):
    _, rrs = deep_water
    got = baselines.ndci(rrs)
    expected = (rrs[:, I708] - rrs[:, I665]) / (rrs[:, I708] + rrs[:, I665])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert (np.abs(got) <= 1.0).all()


def test_ndci_and_red_nir_rise_with_bloom_density():
    clear = _clean_spectra(10.0 ** np.random.default_rng(7).uniform(-0.5, 0.3, 60), seed=8)
    bloom = _clean_spectra(
        10.0 ** np.random.default_rng(9).uniform(np.log10(25.0), np.log10(45.0), 60), seed=10
    )
    assert np.median(baselines.ndci(bloom)) > np.median(baselines.ndci(clear))
    # Red edge below the red band in ordinary water, above it in a bloom.
    assert np.median(baselines.red_nir_ratio(clear)) < 1.0
    assert np.median(baselines.red_nir_ratio(bloom)) > 1.0


def test_ndci_chl_matches_published_polynomial(deep_water):
    _, rrs = deep_water
    index = baselines.ndci(rrs)
    expected = 14.039 + 86.115 * index + 194.325 * index**2
    np.testing.assert_allclose(baselines.ndci_chl(rrs), expected, rtol=1e-12)
    # Mishra & Mishra's quadratic floors near 14 mg/m3 at NDCI = 0, so it can
    # never report oligotrophic water; that limitation is part of the baseline.
    assert baselines.ndci_chl(np.zeros((1, N_BANDS)))[0] == pytest.approx(14.039)
    assert (baselines.ndci_chl(rrs) >= 0).all()


# ------------------------------------------------------------- phycocyanin


def test_phycocyanin_line_height_matches_manual_continuum(deep_water):
    _, rrs = deep_water
    w = (BAND_GRID[I620] - BAND_GRID[I600]) / (BAND_GRID[I650] - BAND_GRID[I600])
    expected = (1.0 - w) * rrs[:, I600] + w * rrs[:, I650] - rrs[:, I620]
    np.testing.assert_allclose(
        baselines.phycocyanin_line_height(rrs), expected, rtol=1e-12, atol=1e-15
    )


def test_phycocyanin_line_height_separates_cyanobacteria():
    """The only classical speciation signal: the 620 nm absorption depth.

    Same chlorophyll, same bloom density; the only difference is the marker
    pigment. The dinoflagellate spectra still return a small positive value
    (the continuum between 600 and 650 nm is convex), so the separation has to
    be a margin, not a sign test - which is exactly why the operational
    threshold sits at 5e-4 sr^-1.
    """
    rng = np.random.default_rng(11)
    chl = 10.0 ** rng.uniform(np.log10(22.0), np.log10(45.0), 80)
    dino = _clean_spectra(chl, phycocyanin=0.0, seed=5)
    cyano = _clean_spectra(chl, phycocyanin=0.55, seed=5)

    lh_dino = baselines.phycocyanin_line_height(dino)
    lh_cyano = baselines.phycocyanin_line_height(cyano)

    assert np.median(lh_cyano) > 3.0 * np.median(lh_dino)
    assert lh_cyano.min() > lh_dino.max()
    # Cleanly on opposite sides of the contract's default decision threshold.
    assert lh_cyano.min() > 0.0005 > lh_dino.max()


# ---------------------------------------------------------------- turbidity


def test_turbidity_proxy_is_monotone_and_bounded():
    rrs = np.zeros((6, N_BANDS))
    rrs[:, I665] = np.array([0.0, 0.001, 0.005, 0.01, 0.03, 0.2])
    turbidity = baselines.turbidity_proxy(rrs)
    assert (turbidity >= 0).all() and np.isfinite(turbidity).all()
    assert (np.diff(turbidity) > 0).all()
    # Nechad's hyperbolic form has a pole at rho_w = C; we hold short of it.
    assert turbidity[-1] < 1e4
    # Only the red band drives it.
    other = np.zeros((1, N_BANDS))
    other[:, I443] = 0.02
    assert baselines.turbidity_proxy(other)[0] == 0.0


# ------------------------------------------------------ the operator's tree


def test_classify_baseline_on_constructed_classes():
    """The classical decision tree on textbook-clean examples of each class."""
    rng = np.random.default_rng(12)
    clear_chl = 10.0 ** rng.uniform(np.log10(0.3), np.log10(2.0), 60)
    bloom_chl = 10.0 ** rng.uniform(np.log10(22.0), np.log10(45.0), 60)
    cases = {
        0: _clean_spectra(clear_chl, seed=13),
        1: _clean_spectra(bloom_chl, phycocyanin=0.0, seed=14),
        2: _clean_spectra(bloom_chl, phycocyanin=0.55, seed=15),
    }
    for expected, rrs in cases.items():
        labels = baselines.classify_baseline(rrs)
        assert set(np.unique(labels)).issubset({0, 1, 2})
        assert (labels == expected).mean() > 0.9, f"class {expected}"


def test_classify_baseline_thresholds_are_policy_knobs():
    rng = np.random.default_rng(16)
    bloom_chl = 10.0 ** rng.uniform(np.log10(22.0), np.log10(45.0), 40)
    cyano = _clean_spectra(bloom_chl, phycocyanin=0.55, seed=17)

    assert (baselines.classify_baseline(cyano) == 2).all()
    # Raise the pigment threshold out of reach: cyanobacteria are then read as
    # dinoflagellates (the speciation call collapses, the bloom call survives).
    assert (baselines.classify_baseline(cyano, pc_lh_threshold=1.0) == 1).all()
    # Raise the bloom threshold out of reach: everything reads as clear water.
    assert (baselines.classify_baseline(cyano, chl_threshold=1e6) == 0).all()


def test_classify_baseline_labels_are_in_range_for_any_input():
    rng = np.random.default_rng(18)
    weird = np.concatenate(
        [
            np.zeros((3, N_BANDS)),
            np.full((3, N_BANDS), -0.01),
            rng.normal(0.0, 0.05, (20, N_BANDS)),
            np.full((2, N_BANDS), 1e6),
        ]
    )
    labels = baselines.classify_baseline(weird)
    assert labels.shape == (weird.shape[0],)
    assert np.isin(labels, (0, 1, 2)).all()


# ------------------------------------------------------- numerical hygiene


@pytest.mark.parametrize(
    "name, rrs",
    [
        ("zeros", np.zeros((4, N_BANDS))),
        ("negative", np.full((4, N_BANDS), -0.002)),
        ("mixed_sign", np.tile(np.linspace(-0.01, 0.01, N_BANDS), (4, 1))),
        ("huge", np.full((4, N_BANDS), 1e6)),
        ("non_finite", np.full((4, N_BANDS), np.nan)),
    ],
)
def test_no_nan_inf_or_warnings_on_degenerate_input(name, rrs):
    """Zero, negative and non-finite Rrs are routine in real L2 products.

    Over-corrected atmospheres give negative reflectance and masked pixels give
    NaN, so an operational baseline must not answer with NaN, inf, or a numpy
    warning - a silent NaN would poison every median downstream.
    """
    functions = (
        baselines.oc4_chl,
        baselines.oc3m_chl,
        baselines.ndci,
        baselines.ndci_chl,
        baselines.red_nir_ratio,
        baselines.phycocyanin_line_height,
        baselines.turbidity_proxy,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            for fn in functions:
                out = fn(rrs)
                assert np.isfinite(out).all(), f"{fn.__name__} on {name}"
            labels = baselines.classify_baseline(rrs)
    assert np.isin(labels, (0, 1, 2)).all()

    # Physical clipping holds even on garbage: chl and turbidity never negative.
    for fn in (baselines.oc4_chl, baselines.oc3m_chl, baselines.ndci_chl,
               baselines.red_nir_ratio, baselines.turbidity_proxy):
        assert (fn(rrs) >= 0).all(), f"{fn.__name__} on {name}"
