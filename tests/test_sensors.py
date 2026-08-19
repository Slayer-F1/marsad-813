"""Tests for marsad.sensors - band tables, SRF resampling, grid lift.

Fixtures are built inline from marsad.spectra (BAND_GRID / gaussian_feature)
and seeded numpy; marsad.synth is deliberately not imported, so these tests
exercise the resampling maths rather than the forward model.

The headline behaviour under test is the one the hyperspectral claim rests on:
the 620 nm phycocyanin line survives a sensor that has a 620 nm band and is
interpolated away by one that does not. The spatial tests add the other half of
"what a sensor can see": a 100 m intake-scale patch fills about a ninth of a
300 m OLCI pixel and one hundredth of a 1 km MODIS pixel, so even a sensor with
the right band arrives with a diluted signal. Both halves are statements about
band tables, pixel geometry and linear algebra applied to our own simulated
spectra, not claims about real Gulf water.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from marsad.sensors import (
    ASSUMED_813_GSD_M,
    SENSORS,
    Band,
    Sensor,
    mix_subpixel,
    resample,
    resample_to_grid,
    subpixel_fill_fraction,
)
from marsad.spectra import BAND_GRID, N_BANDS, band_index, gaussian_feature

MULTISPECTRAL = tuple(k for k in SENSORS if k != "marsad_813")


def _smooth_water_spectrum() -> np.ndarray:
    """A smooth, band-limited Case-2-ish Rrs shape on BAND_GRID.

    Green peak, weak blue shoulder, a broad red-edge bump, and near-zero SWIR.
    Deliberately free of narrow features so that any round-trip error is the
    resampling itself, not aliasing of a line the grid cannot carry.
    """
    return (
        0.020 * gaussian_feature(560.0, 224.0)
        + 0.006 * gaussian_feature(700.0, 106.0)
        + 0.002 * gaussian_feature(450.0, 141.0)
    )


def _line_height_620(spectrum: np.ndarray) -> float:
    """Baseline-subtracted 620 nm absorption depth (the classical PC index).

    Linear baseline 600 -> 650 nm evaluated at 620 nm, minus the observed
    value there. Positive means an absorption dip is present.
    """
    i600, i620, i650 = (band_index(w) for w in (600.0, 620.0, 650.0))
    w600, w620, w650 = BAND_GRID[i600], BAND_GRID[i620], BAND_GRID[i650]
    frac = (w620 - w600) / (w650 - w600)
    baseline = spectrum[i600] + frac * (spectrum[i650] - spectrum[i600])
    return float(baseline - spectrum[i620])


# --------------------------------------------------------------------------
# registry and band tables
# --------------------------------------------------------------------------

def test_registry_has_the_contract_keys():
    assert set(SENSORS) == {
        "marsad_813",
        "sentinel2_msi",
        "sentinel3_olci",
        "modis_aqua",
        "landsat8_oli",
    }
    for key, sensor in SENSORS.items():
        assert isinstance(sensor, Sensor)
        assert sensor.key == key
        assert sensor.label.strip()
        assert sensor.note.strip()
        assert sensor.n_bands == len(sensor.bands)
        assert all(isinstance(b, Band) for b in sensor.bands)


def test_band_centres_inside_813_range_and_sorted():
    for key, sensor in SENSORS.items():
        centers = sensor.centers_nm
        assert centers.shape == (sensor.n_bands,)
        assert centers.dtype == np.float64
        assert centers.min() >= BAND_GRID[0] - 1e-9, key
        assert centers.max() <= BAND_GRID[-1] + 1e-9, key
        assert np.all(np.diff(centers) > 0), f"{key} band table must be ascending"
        assert all(b.fwhm_nm > 0 for b in sensor.bands), key


def test_expected_band_counts():
    assert SENSORS["marsad_813"].n_bands == N_BANDS
    assert SENSORS["sentinel2_msi"].n_bands == 12
    assert SENSORS["sentinel3_olci"].n_bands == 17
    assert SENSORS["modis_aqua"].n_bands == 15
    assert SENSORS["landsat8_oli"].n_bands == 6


def test_dataclasses_are_frozen():
    band = SENSORS["landsat8_oli"].bands[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        band.center_nm = 500.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        SENSORS["landsat8_oli"].note = "rewritten"


# --------------------------------------------------------------------------
# the 620 nm phycocyanin question
# --------------------------------------------------------------------------

def test_sentinel2_and_modis_have_no_620_band_olci_has_one():
    for key in ("sentinel2_msi", "modis_aqua", "landsat8_oli"):
        near = np.abs(SENSORS[key].centers_nm - 620.0) < 10.0
        assert near.sum() == 0, f"{key} unexpectedly has a band near 620 nm"
    olci = np.abs(SENSORS["sentinel3_olci"].centers_nm - 620.0) < 10.0
    assert olci.sum() == 1
    assert SENSORS["sentinel3_olci"].bands[int(np.argmax(olci))].name == "Oa7"


def test_notes_state_the_620_nm_situation_honestly():
    s2 = SENSORS["sentinel2_msi"].note.lower()
    assert "620" in s2 and "no band at 620" in s2 and "phycocyanin" in s2

    olci = SENSORS["sentinel3_olci"].note.lower()
    assert "620" in olci and "does carry" in olci      # credit where due
    assert "cannot" in olci and "300 m" in olci        # and the honest limit
    assert "sediment" in olci and "chlorophyll" in olci

    modis = SENSORS["modis_aqua"].note.lower()
    assert "620" in modis and "1 km" in modis

    assert "205 contiguous bands, 400-1700 nm" in SENSORS["marsad_813"].note
    assert "6 usable water bands" in SENSORS["landsat8_oli"].note.lower()


def test_620_line_survives_olci_and_is_erased_by_sentinel2():
    """The enabling-sensor claim, reduced to its spectral core.

    A phycocyanin dip on a smooth baseline keeps most of its depth through the
    813 grid, keeps a measurable part of it through OLCI (which has Oa7 at
    620 nm), and is interpolated to nothing by Sentinel-2 and MODIS, whose
    nearest bands sit 45 nm and 25 nm away. This is linear algebra on our own
    simulated spectra, not a field comparison of the instruments.
    """
    spectrum = _smooth_water_spectrum() - 0.0025 * gaussian_feature(620.0, 30.0)
    truth = _line_height_620(spectrum)
    assert truth > 0.001

    def round_trip(key: str) -> float:
        lifted = resample_to_grid(resample(spectrum[None, :], key), key)[0]
        return _line_height_620(lifted)

    assert round_trip("marsad_813") == pytest.approx(truth, rel=1e-12)
    assert round_trip("sentinel3_olci") > 0.3 * truth
    for blind in ("sentinel2_msi", "modis_aqua", "landsat8_oli"):
        assert abs(round_trip(blind)) < 0.1 * truth, blind


# --------------------------------------------------------------------------
# resample
# --------------------------------------------------------------------------

def test_marsad_813_resample_is_identity():
    rng = np.random.default_rng(3)
    rrs = 0.03 * rng.random((16, N_BANDS))
    out = resample(rrs, "marsad_813")
    assert out.shape == rrs.shape
    assert np.abs(out - rrs).max() < 1e-12


def test_marsad_813_full_round_trip_is_identity():
    rng = np.random.default_rng(4)
    rrs = 0.03 * rng.random((8, N_BANDS))
    back = resample_to_grid(resample(rrs, "marsad_813"), "marsad_813")
    assert back.shape == (8, N_BANDS)
    assert np.abs(back - rrs).max() < 1e-12


def test_resample_reduces_band_count_for_every_multispectral_sensor():
    rng = np.random.default_rng(5)
    rrs = 0.03 * rng.random((7, N_BANDS))
    for key in MULTISPECTRAL:
        sensor = SENSORS[key]
        out = resample(rrs, key)
        assert sensor.n_bands < N_BANDS, key
        assert out.shape == (7, sensor.n_bands), key
        assert np.isfinite(out).all(), key
        assert (out >= 0.0).all(), key  # non-negative weights, non-negative input


def test_srf_rows_are_normalised():
    """A flat spectrum must come back flat: band weights sum to one."""
    flat = np.full((1, N_BANDS), 0.017)
    for key in SENSORS:
        out = resample(flat, key)
        assert np.abs(out - 0.017).max() < 1e-12, key


def test_resample_is_a_local_average_of_the_grid():
    """Each band value must lie between the min and max of its neighbourhood."""
    spectrum = _smooth_water_spectrum()[None, :]
    for key in MULTISPECTRAL:
        sensor = SENSORS[key]
        values = resample(spectrum, key)[0]
        for value, band in zip(values, sensor.bands):
            lo_nm = band.center_nm - 3.0 * band.fwhm_nm
            hi_nm = band.center_nm + 3.0 * band.fwhm_nm
            window = spectrum[0][(BAND_GRID >= lo_nm) & (BAND_GRID <= hi_nm)]
            assert window.min() - 1e-12 <= value <= window.max() + 1e-12


def test_sensor_object_and_key_agree():
    rng = np.random.default_rng(6)
    rrs = 0.02 * rng.random((3, N_BANDS))
    sensor = SENSORS["sentinel3_olci"]
    assert np.array_equal(resample(rrs, sensor), resample(rrs, "sentinel3_olci"))
    assert np.array_equal(
        resample_to_grid(resample(rrs, sensor), sensor),
        resample_to_grid(resample(rrs, "sentinel3_olci"), "sentinel3_olci"),
    )


def test_single_spectrum_is_accepted_and_returns_1d():
    spectrum = _smooth_water_spectrum()
    out = resample(spectrum, "modis_aqua")
    assert out.shape == (SENSORS["modis_aqua"].n_bands,)
    back = resample_to_grid(out, "modis_aqua")
    assert back.shape == (N_BANDS,)
    assert np.allclose(back, resample_to_grid(out[None, :], "modis_aqua")[0])


# --------------------------------------------------------------------------
# resample_to_grid
# --------------------------------------------------------------------------

def test_resample_to_grid_shape_and_finiteness():
    rng = np.random.default_rng(7)
    for key in SENSORS:
        sensor = SENSORS[key]
        values = 0.02 * rng.random((5, sensor.n_bands))
        out = resample_to_grid(values, key)
        assert out.shape == (5, N_BANDS), key
        assert out.dtype == np.float64, key
        assert np.isfinite(out).all(), key
        assert (out >= 0.0).all(), key


def test_resample_to_grid_matches_numpy_interp_with_edge_hold():
    rng = np.random.default_rng(8)
    for key in MULTISPECTRAL:
        sensor = SENSORS[key]
        values = 0.02 * rng.random((4, sensor.n_bands))
        out = resample_to_grid(values, key)
        for row, expected_row in zip(out, values):
            expected = np.interp(BAND_GRID, sensor.centers_nm, expected_row)
            assert np.abs(row - expected).max() < 1e-12, key


def test_resample_to_grid_holds_the_edges():
    sensor = SENSORS["sentinel3_olci"]  # bands stop at 1020 nm
    values = np.linspace(0.01, 0.02, sensor.n_bands)[None, :]
    out = resample_to_grid(values, sensor)[0]
    below = BAND_GRID < sensor.centers_nm[0]
    above = BAND_GRID > sensor.centers_nm[-1]
    assert above.any()
    assert np.allclose(out[above], values[0, -1])
    if below.any():
        assert np.allclose(out[below], values[0, 0])
    # Each band value reappears at its own wavelength, to well within one
    # band-to-band increment: the piecewise-linear lift is itself written onto
    # the 6.4 nm grid, so a kink at a band centre is sampled, not landed on.
    step = float(np.max(np.abs(np.diff(values[0]))))
    for center, value in zip(sensor.centers_nm, values[0]):
        assert np.interp(center, BAND_GRID, out) == pytest.approx(value, abs=0.5 * step)


def test_smooth_spectrum_survives_the_round_trip():
    spectrum = _smooth_water_spectrum()
    scale = float(spectrum.max())

    for key in MULTISPECTRAL:
        back = resample_to_grid(resample(spectrum[None, :], key), key)[0]
        assert back.shape == (N_BANDS,)
        assert np.isfinite(back).all()
        rmse = float(np.sqrt(np.mean((back - spectrum) ** 2)))
        assert rmse < 0.10 * scale, f"{key} rmse {rmse:.5f}"

    # OLCI samples the visible densely, so it should be close everywhere.
    olci = resample_to_grid(resample(spectrum[None, :], "sentinel3_olci"), "sentinel3_olci")[0]
    assert np.abs(olci - spectrum).max() < 0.08 * scale
    assert float(np.sqrt(np.mean((olci - spectrum) ** 2))) < 0.03 * scale


def test_round_trip_keeps_array_width_for_the_ablation():
    """Every sensor must hand the models the same N_BANDS-wide input."""
    rng = np.random.default_rng(9)
    rrs = 0.02 * rng.random((6, N_BANDS))
    for key in SENSORS:
        back = resample_to_grid(resample(rrs, key), key)
        assert back.shape == rrs.shape, key


# --------------------------------------------------------------------------
# ground sampling distance and sub-pixel dilution
# --------------------------------------------------------------------------

def test_every_sensor_declares_a_positive_ground_sampling_distance():
    for key, sensor in SENSORS.items():
        assert isinstance(sensor.gsd_m, float), key
        assert np.isfinite(sensor.gsd_m) and sensor.gsd_m > 0.0, key


def test_documented_ground_sampling_distances():
    """Published figures, for the water-relevant bands of each instrument."""
    assert SENSORS["sentinel2_msi"].gsd_m == 20.0     # red edge B5-B7, not 10 m
    assert SENSORS["sentinel3_olci"].gsd_m == 300.0   # full resolution
    assert SENSORS["modis_aqua"].gsd_m == 1000.0      # ocean-colour bands
    assert SENSORS["landsat8_oli"].gsd_m == 30.0


def test_sentinel2_note_explains_the_20_m_red_edge():
    """The bloom signal rides on B5-B7, which are 20 m, so 10 m is not the
    number that matters for this application."""
    note = SENSORS["sentinel2_msi"].note.lower()
    assert "20 m" in note
    assert "b5-b7" in note
    assert "10 m" in note


def test_813_gsd_is_labelled_an_assumption_not_a_published_fact():
    """813 has published no GSD, so ours must never read as a specification."""
    assert SENSORS["marsad_813"].gsd_m == ASSUMED_813_GSD_M == 30.0
    note = SENSORS["marsad_813"].note.lower()
    assert "assum" in note
    assert "enmap" in note and "prisma" in note


def test_assumed_813_gsd_is_a_greppable_module_constant():
    import marsad.sensors as sensors_module

    source = Path(sensors_module.__file__).read_text(encoding="utf-8")
    assert "ASSUMED_813_GSD_M = 30.0" in source
    assert "assum" in (sensors_module.__doc__ or "").lower()


def test_new_sensors_default_to_the_assumed_813_gsd():
    """The field is defaulted, so existing construction sites keep working."""
    ad_hoc = Sensor(
        key="ad_hoc",
        label="band table built without a gsd",
        bands=(Band("B1", 560.0, 30.0),),
        note="no ground sampling distance given",
    )
    assert ad_hoc.gsd_m == ASSUMED_813_GSD_M


def test_non_positive_gsd_is_rejected():
    for bad in (0.0, -30.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gsd_m"):
            Sensor(
                key="bad_gsd",
                label="impossible pixel",
                bands=(Band("B1", 560.0, 30.0),),
                note="a pixel must have a positive size",
                gsd_m=bad,
            )


def test_fill_fraction_is_one_when_the_patch_covers_the_pixel():
    for key, sensor in SENSORS.items():
        assert subpixel_fill_fraction(sensor.gsd_m, key) == 1.0, key
        assert subpixel_fill_fraction(4.0 * sensor.gsd_m, key) == 1.0, key


def test_fill_fraction_falls_as_the_square_of_the_size_ratio():
    """A 100 m intake-scale patch, seen by each pixel size."""
    assert subpixel_fill_fraction(100.0, "sentinel3_olci") == pytest.approx(0.111, abs=1e-3)
    assert subpixel_fill_fraction(100.0, "modis_aqua") == pytest.approx(0.01, abs=1e-12)
    # the 20-30 m class resolves the same patch outright
    assert subpixel_fill_fraction(100.0, "marsad_813") == 1.0
    assert subpixel_fill_fraction(100.0, "sentinel2_msi") == 1.0
    assert subpixel_fill_fraction(100.0, "landsat8_oli") == 1.0

    # quadratic below the pixel: halving the patch quarters the fill
    for key in ("sentinel3_olci", "modis_aqua"):
        gsd = SENSORS[key].gsd_m
        assert subpixel_fill_fraction(50.0, key) == pytest.approx(
            0.25 * subpixel_fill_fraction(100.0, key)
        ), key
        assert subpixel_fill_fraction(30.0, key) == pytest.approx((30.0 / gsd) ** 2), key


def test_fill_fraction_ranks_the_sensors_by_pixel_size():
    fills = {key: subpixel_fill_fraction(100.0, key) for key in SENSORS}
    assert fills["marsad_813"] == 1.0
    assert fills["marsad_813"] > fills["sentinel3_olci"] > fills["modis_aqua"] > 0.0


def test_fill_fraction_accepts_a_sensor_object_or_a_key():
    sensor = SENSORS["sentinel3_olci"]
    assert subpixel_fill_fraction(100.0, sensor) == subpixel_fill_fraction(
        100.0, "sentinel3_olci"
    )
    with pytest.raises(KeyError):
        subpixel_fill_fraction(100.0, "hyperion")


def test_fill_fraction_rejects_non_positive_or_non_finite_patch_size():
    for bad in (0.0, -1.0, -1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="patch_size_m"):
            subpixel_fill_fraction(bad, "modis_aqua")


def test_mix_with_a_full_pixel_returns_the_target_exactly():
    rng = np.random.default_rng(21)
    target = 0.03 * rng.random((5, N_BANDS))
    background = 0.01 * rng.random((5, N_BANDS))
    out = mix_subpixel(target, background, 300.0, "sentinel3_olci")
    assert np.array_equal(out, target)


def test_mix_with_a_tiny_patch_returns_almost_the_background():
    rng = np.random.default_rng(22)
    target = 0.03 * rng.random((5, N_BANDS))
    background = 0.01 * rng.random((5, N_BANDS))
    fill = subpixel_fill_fraction(5.0, "modis_aqua")   # (5/1000)**2 = 2.5e-5
    assert fill < 1e-4
    out = mix_subpixel(target, background, 5.0, "modis_aqua")
    assert not np.array_equal(out, background)         # not silently discarded
    assert np.abs(out - background).max() < 1e-5


def test_mixed_output_stays_between_its_two_inputs_elementwise():
    rng = np.random.default_rng(23)
    target = 0.03 * rng.random((7, N_BANDS))
    background = 0.02 * rng.random((7, N_BANDS))
    lo = np.minimum(target, background)
    hi = np.maximum(target, background)
    for key in SENSORS:
        for patch_m in (5.0, 100.0, 500.0, 5000.0):
            out = mix_subpixel(target, background, patch_m, key)
            assert out.shape == target.shape, key
            assert out.dtype == np.float64, key
            assert np.isfinite(out).all(), key
            assert (out >= lo - 1e-12).all(), key
            assert (out <= hi + 1e-12).all(), key


def test_mix_broadcasts_a_single_background_across_many_targets():
    rng = np.random.default_rng(24)
    target = 0.03 * rng.random((4, N_BANDS))
    background = 0.01 * rng.random(N_BANDS)
    fill = subpixel_fill_fraction(100.0, "sentinel3_olci")

    out = mix_subpixel(target, background, 100.0, "sentinel3_olci")
    assert out.shape == (4, N_BANDS)
    assert out.dtype == np.float64
    expected = fill * target + (1.0 - fill) * background[None, :]
    assert np.abs(out - expected).max() < 1e-15

    # one spectrum at a time gives the same rows, and stays 1-D
    for i in range(4):
        single = mix_subpixel(target[i], background, 100.0, "sentinel3_olci")
        assert single.shape == (N_BANDS,)
        assert np.allclose(single, out[i])


def test_mix_rejects_bad_shapes_row_counts_and_patch_sizes():
    target = np.zeros((3, N_BANDS))
    with pytest.raises(ValueError):
        mix_subpixel(target, np.zeros((2, N_BANDS)), 100.0, "modis_aqua")
    with pytest.raises(ValueError):
        mix_subpixel(np.zeros((3, N_BANDS - 1)), target, 100.0, "modis_aqua")
    with pytest.raises(ValueError):
        mix_subpixel(target, np.zeros((3, 4)), 100.0, "modis_aqua")
    with pytest.raises(ValueError, match="patch_size_m"):
        mix_subpixel(target, np.zeros((3, N_BANDS)), 0.0, "modis_aqua")
    with pytest.raises(KeyError):
        mix_subpixel(target, np.zeros((3, N_BANDS)), 100.0, "hyperion")


def test_subpixel_dilution_shrinks_the_620_line_olci_would_otherwise_see():
    """The spatial term on top of the spectral one.

    OLCI carries Oa7 at 620 nm, so on a patch that fills the pixel it keeps a
    real part of the phycocyanin line. Put the same patch at intake scale,
    100 m, inside a 300 m OLCI pixel and the line arrives at about a ninth of
    its depth, because the pixel is mostly the clear water around the bloom.
    Both spectra are our own simulated shapes: this is geometry and linear
    algebra, never a field comparison of the instruments.
    """
    bloom = _smooth_water_spectrum() - 0.0025 * gaussian_feature(620.0, 30.0)
    background = 0.4 * _smooth_water_spectrum()          # clear water, no PC dip

    def olci_line_height(spectrum: np.ndarray) -> float:
        lifted = resample_to_grid(resample(spectrum[None, :], "sentinel3_olci"), "sentinel3_olci")
        return _line_height_620(lifted[0])

    resolved = olci_line_height(bloom)
    floor = olci_line_height(background)
    diluted = olci_line_height(mix_subpixel(bloom, background, 100.0, "sentinel3_olci"))
    fill = subpixel_fill_fraction(100.0, "sentinel3_olci")

    assert resolved - floor > 0.0
    # linear mixing, linear resampling, linear line height: the signal above the
    # background scales by exactly the fill fraction
    assert diluted - floor == pytest.approx(fill * (resolved - floor), rel=1e-9)
    assert 0.0 < diluted - floor < 0.2 * (resolved - floor)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_unknown_sensor_key_raises():
    rrs = np.zeros((2, N_BANDS))
    with pytest.raises(KeyError):
        resample(rrs, "hyperion")
    with pytest.raises(TypeError):
        resample(rrs, 7)  # type: ignore[arg-type]


def test_wrong_shapes_raise():
    with pytest.raises(ValueError):
        resample(np.zeros((2, N_BANDS - 1)), "marsad_813")
    with pytest.raises(ValueError):
        resample(np.zeros((2, 3, N_BANDS)), "marsad_813")
    with pytest.raises(ValueError):
        resample_to_grid(np.zeros((2, 4)), "sentinel2_msi")


def test_band_outside_the_813_range_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        Sensor(
            key="bad",
            label="out of range",
            bands=(Band("B1", 2190.0, 180.0),),
            note="Sentinel-2 B12 is beyond the 813 SWIR limit",
        )


def test_band_with_no_grid_support_is_rejected_not_zeroed():
    """A band the grid cannot see must fail loudly, never become a zero column."""
    lonely = Sensor(
        key="lonely",
        label="unresolvably narrow band",
        bands=(Band("B1", 560.0, 30.0), Band("B2", 403.0, 0.01)),
        note="second band falls between grid points",
    )
    with pytest.raises(ValueError, match="no support"):
        resample(np.zeros((1, N_BANDS)), lonely)


def test_module_source_has_no_em_dash():
    """Repo style rule: plain hyphens only, in code and in prose."""
    import marsad.sensors as sensors_module

    banned = (chr(0x2014), chr(0x2013))  # em dash, en dash by code point
    for path in (Path(sensors_module.__file__), Path(__file__)):
        source = path.read_text(encoding="utf-8")
        for char in banned:
            assert char not in source, f"{path.name} contains a banned dash"
