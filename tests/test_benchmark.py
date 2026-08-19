"""Tests for the benchmark experiment (src/marsad/benchmark.py).

Two things are pinned here. First the CONTRACT: the result dictionary is
consumed by ``scripts/run_benchmark.py`` and by the dashboard, so its shape is
asserted key by key rather than loosely. Second the HEADLINE BEHAVIOUR, which
is the whole reason the module exists:

* in shallow turbid water, MARSAD beats both the classical operator decision
  tree on speciation and OC4 / NDCI on chlorophyll error;
* the hyperspectral instrument beats every multispectral alternative;
* speciation collapses on a sensor with no 620 nm phycocyanin band;
* a coarse sensor loses an intake-scale bloom patch inside its own pixel,
  while a fine one does not, which is the half of the sensor comparison the
  spectral-only ablation never measured;
* the 2x2 verdict puts those two axes together, and only a sensor that passes
  both can warn a desalination intake.

Sizes are deliberately tiny. One :func:`run_benchmark` call retrains Stage 1 +
Stage 2 once at full size, once per sensor, and once per distinct
(sensor, fill fraction) cell of the spatial grid, so the module-scoped fixture
runs it exactly once and every test reads the same result. The fixture also
cuts the spatial grid to two patch sizes, the intake-scale reference and a
full-fill control, which is the smallest grid with a slope in it.

Reminder, and the module under test says the same in prose: all of this is
measured on our own synthetic forward model, so these assertions check the
experiment, not the real-world skill of MARSAD on Gulf water.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from marsad.benchmark import (
    FILL_FRACTION_THRESHOLD,
    HONESTY_NOTE,
    PC_BAND_NM,
    PC_BAND_TOLERANCE_NM,
    REFERENCE_PATCH_SIZE_M,
    SHALLOW_DEPTH_M,
    SPATIAL_PATCH_SIZES_M,
    TURBID_TSS_G_M3,
    WATER_REGIMES,
    build_verdict,
    classify_regime,
    has_phycocyanin_band,
    nearest_band_distance_nm,
    run_benchmark,
    run_spatial_ablation,
)
from marsad.sensors import SENSORS, subpixel_fill_fraction

# Small enough to keep the file quick, large enough that every regime and both
# bloom classes are populated after the benchmark's per-regime top-up.
N_TRAIN, N_TEST, SEED = 260, 200, 11

CHL_KEYS = {"oc4", "oc3m", "ndci", "marsad", "n"}
SPECIATION_KEYS = {"baseline_tree", "marsad", "n"}
ABLATION_KEYS = {"label", "n_bands", "accuracy", "cyano_recall", "note"}
HEADLINE_KEYS = {
    "marsad_shallow_turbid_acc",
    "baseline_shallow_turbid_acc",
    "oc4_deep_err",
    "oc4_shallow_turbid_err",
    "hyperspectral_gain",
}

# The spatial grid the fixture runs: the intake-scale reference size the 2x2
# verdict is defined at, plus a control size at which every sensor in the
# table fills its own pixel and the spatial term is switched off. Two columns
# is the smallest grid that still has a slope, and the cost of this file is
# one model fit per distinct (sensor, fill fraction) cell.
SPATIAL_PATCH_SIZES = (REFERENCE_PATCH_SIZE_M, 1000.0)

SPATIAL_KEYS = {"patch_sizes_m", "n_train", "n_test", "sensors", "note"}
SPATIAL_SENSOR_KEYS = {"label", "gsd_m", "n_bands", "has_620nm", "by_patch_size"}
SPATIAL_CELL_KEYS = {
    "patch_size_m",
    "fill_fraction",
    "accuracy",
    "cyano_recall",
    "bloom_recall",
    "false_alarm_rate",
}
SPATIAL_CELL_METRICS = ("accuracy", "cyano_recall", "bloom_recall",
                        "false_alarm_rate")
VERDICT_BUCKETS = ("adequate_on_both", "spectral_only", "spatial_only",
                   "inadequate_on_both")
VERDICT_KEYS = {
    "reference_patch_size_m",
    "fill_fraction_threshold",
    "pc_band_nm",
    "pc_band_tolerance_nm",
    "sensors",
    "summary",
    "note",
    *VERDICT_BUCKETS,
}
VERDICT_SENSOR_KEYS = {
    "label",
    "has_620nm",
    "gsd_m",
    "fill_fraction_at_reference",
    "spectral_ok",
    "spatial_ok",
    "accuracy_at_reference",
    "cyano_recall_at_reference",
    "reason",
}

# The five v0.2 contract keys, plus the two blocks added after it. The set is
# asserted exactly rather than loosely: the dashboard reads this dictionary.
TOP_LEVEL_KEYS = {
    "chl_retrieval",
    "speciation",
    "ablation",
    "headline",
    "honesty_note",
    "spatial",
    "verdict",
}


@pytest.fixture(scope="module")
def result() -> dict:
    """One benchmark run shared by the whole module (it is the slow part)."""
    return run_benchmark(
        seed=SEED,
        n_train=N_TRAIN,
        n_test=N_TEST,
        spatial_patch_sizes_m=SPATIAL_PATCH_SIZES,
    )


# ------------------------------------------------------------ classify_regime
def test_water_regimes_are_the_four_contracted_corners():
    assert WATER_REGIMES == (
        "optically_deep",
        "shallow_clear",
        "turbid_deep",
        "shallow_turbid",
    )


def test_classify_regime_covers_every_corner_of_the_grid():
    depth = np.array([50.0, 3.0, 50.0, 3.0])
    tss = np.array([1.0, 1.0, 20.0, 20.0])
    got = classify_regime(depth, tss)
    assert got.tolist() == [
        "optically_deep",
        "shallow_clear",
        "turbid_deep",
        "shallow_turbid",
    ]
    assert got.shape == depth.shape


def test_classify_regime_thresholds_are_shallow_below_and_turbid_at_or_above():
    """Depth is strictly below 10 m; TSS is at or above 8 g/m3."""
    depth = np.array([SHALLOW_DEPTH_M, SHALLOW_DEPTH_M - 1e-6, 1.0, 1.0])
    tss = np.array([1.0, 1.0, TURBID_TSS_G_M3, TURBID_TSS_G_M3 - 1e-6])
    assert classify_regime(depth, tss).tolist() == [
        "optically_deep",
        "shallow_clear",
        "shallow_turbid",
        "shallow_clear",
    ]


def test_classify_regime_broadcasts_and_handles_non_finite_input():
    # A scalar depth against an array of TSS values.
    assert classify_regime(2.0, np.array([1.0, 99.0])).tolist() == [
        "shallow_clear",
        "shallow_turbid",
    ]
    # An unusable retrieval must not promote a pixel into the target regime.
    assert classify_regime(
        np.array([np.nan, 2.0]), np.array([99.0, np.nan])
    ).tolist() == ["turbid_deep", "shallow_clear"]


def test_classify_regime_only_ever_emits_known_regimes():
    rng = np.random.default_rng(0)
    got = classify_regime(rng.uniform(0.5, 60.0, 500), rng.lognormal(0.5, 1.2, 500))
    assert set(np.unique(got)).issubset(set(WATER_REGIMES))


# ------------------------------------------------------------------- schema
def test_result_has_exactly_the_contracted_top_level_keys(result):
    assert set(result) == TOP_LEVEL_KEYS


def test_chl_retrieval_schema(result):
    table = result["chl_retrieval"]
    assert set(table) == set(WATER_REGIMES)
    for regime, row in table.items():
        assert set(row) == CHL_KEYS, regime
        assert isinstance(row["n"], int)
        for key in ("oc4", "oc3m", "ndci", "marsad"):
            value = row[key]
            assert isinstance(value, float)
            # A median absolute log10 error is finite and non-negative.
            assert math.isfinite(value) and value >= 0.0, (regime, key)


def test_speciation_schema(result):
    table = result["speciation"]
    assert set(table) == set(WATER_REGIMES)
    for regime, row in table.items():
        assert set(row) == SPECIATION_KEYS, regime
        assert isinstance(row["n"], int)
        for key in ("baseline_tree", "marsad"):
            assert isinstance(row[key], float)
            assert 0.0 <= row[key] <= 1.0, (regime, key)


def test_ablation_schema_covers_every_sensor(result):
    table = result["ablation"]
    assert set(table) == set(SENSORS)
    for key, row in table.items():
        assert set(row) == ABLATION_KEYS, key
        assert row["label"] == SENSORS[key].label
        assert row["note"] == SENSORS[key].note
        assert row["n_bands"] == SENSORS[key].n_bands
        assert isinstance(row["accuracy"], float)
        assert isinstance(row["cyano_recall"], float)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["cyano_recall"] <= 1.0
    assert table["marsad_813"]["n_bands"] == 205


def test_headline_schema_and_agreement_with_the_tables(result):
    head = result["headline"]
    assert set(head) == HEADLINE_KEYS
    assert all(isinstance(v, float) for v in head.values())

    # The headline must be the tables, not an independently computed number.
    assert head["marsad_shallow_turbid_acc"] == result["speciation"]["shallow_turbid"]["marsad"]
    assert head["baseline_shallow_turbid_acc"] == result["speciation"]["shallow_turbid"]["baseline_tree"]
    assert head["oc4_deep_err"] == result["chl_retrieval"]["optically_deep"]["oc4"]
    assert head["oc4_shallow_turbid_err"] == result["chl_retrieval"]["shallow_turbid"]["oc4"]

    best_multispectral = max(
        row["accuracy"] for key, row in result["ablation"].items() if key != "marsad_813"
    )
    assert head["hyperspectral_gain"] == pytest.approx(
        result["ablation"]["marsad_813"]["accuracy"] - best_multispectral
    )


def test_every_regime_is_actually_measured(result):
    """No regime may be reported on zero pixels - a table of NaN proves nothing."""
    for regime in WATER_REGIMES:
        assert result["chl_retrieval"][regime]["n"] > 0, regime
        assert result["speciation"][regime]["n"] > 0, regime
        # Both tables score the same pixels.
        assert result["chl_retrieval"][regime]["n"] == result["speciation"][regime]["n"]


def test_result_is_json_serialisable(result):
    """The dashboard reads this verbatim, so it must survive a JSON round trip."""
    restored = json.loads(json.dumps(result))
    assert set(restored) == TOP_LEVEL_KEYS
    assert restored["headline"] == pytest.approx(result["headline"])


# -------------------------------------------------------- headline behaviour
def test_marsad_beats_the_operator_decision_tree_in_shallow_turbid_water(result):
    """The project's central speciation claim, in the regime it is made about."""
    row = result["speciation"]["shallow_turbid"]
    assert row["marsad"] > row["baseline_tree"]


def test_marsad_beats_oc4_and_ndci_on_chlorophyll_in_shallow_turbid_water(result):
    """Lower median absolute log10 error than both operational band ratios."""
    row = result["chl_retrieval"]["shallow_turbid"]
    assert row["marsad"] < row["oc4"]
    assert row["marsad"] < row["ndci"]
    assert row["marsad"] < row["oc3m"]


def test_hyperspectral_gain_is_positive(result):
    """205 bands beat every multispectral band set on the same architecture."""
    assert result["headline"]["hyperspectral_gain"] > 0.0
    native = result["ablation"]["marsad_813"]["accuracy"]
    for key, row in result["ablation"].items():
        if key != "marsad_813":
            assert row["accuracy"] <= native, key


def test_speciation_collapses_without_a_620_nm_band(result):
    """Sentinel-2, MODIS and Landsat carry no phycocyanin band, so cyanobacteria
    recall must fall well below the hyperspectral instrument's."""
    native_recall = result["ablation"]["marsad_813"]["cyano_recall"]
    for key in ("sentinel2_msi", "modis_aqua", "landsat8_oli"):
        assert result["ablation"][key]["cyano_recall"] < native_recall - 0.05, key


def test_oc4_is_badly_wrong_in_shallow_turbid_water(result):
    """OC4 is off by a large factor in the regime the project targets.

    Deliberately NOT asserted as "worse in shallow turbid than in deep water".
    That ordering does hold at full benchmark size, but at the tiny sizes used
    here the two regimes hold different bloom/clear mixes and the comparison
    flips on some seeds: OC4 over-reports clear water, so a shallow turbid
    sample that happens to be bloom-heavy can score better than a deep sample
    that is clear-heavy. Pinning a seed-dependent ordering would be a test
    that lies. What is stable, and what matters operationally, is that OC4 is
    far out in shallow turbid water while MARSAD is not.
    """
    head = result["headline"]
    # 0.30 log10 is a factor of two. OC4 is well past that here.
    assert head["oc4_shallow_turbid_err"] > 0.30
    assert head["oc4_deep_err"] > 0.30


# ------------------------------------------------- spatial ablation: geometry
def test_fill_fraction_orders_sensors_by_pixel_size_at_intake_scale():
    """Geometry only, no model: a coarser pixel never holds more of the patch.

    This is the spatial axis in one assertion, and it is the reason the
    spectral-only ablation flattered OLCI and MODIS. At the 100 m intake scale
    the 30 m class swallows the patch whole while a 300 m pixel holds about a
    ninth of it and a 1 km pixel a hundredth.
    """
    fills = {key: subpixel_fill_fraction(REFERENCE_PATCH_SIZE_M, key)
             for key in SENSORS}

    # Monotone in pixel size, which is the whole content of the geometry.
    by_pixel = sorted(SENSORS, key=lambda k: SENSORS[k].gsd_m)
    ordered = [fills[k] for k in by_pixel]
    assert ordered == sorted(ordered, reverse=True)

    # And the specific ordering the verdict rests on.
    assert fills["modis_aqua"] < fills["sentinel3_olci"] < fills["marsad_813"]
    assert fills["modis_aqua"] == pytest.approx(0.01)
    assert fills["sentinel3_olci"] == pytest.approx(1.0 / 9.0)
    for key in ("marsad_813", "sentinel2_msi", "landsat8_oli"):
        assert fills[key] == 1.0, key


def test_fill_fraction_ordering_holds_at_every_default_patch_size():
    """The ordering is a property of the pixel sizes, not of one lucky column."""
    for patch in SPATIAL_PATCH_SIZES_M:
        by_pixel = sorted(SENSORS, key=lambda k: SENSORS[k].gsd_m)
        ordered = [subpixel_fill_fraction(patch, k) for k in by_pixel]
        assert ordered == sorted(ordered, reverse=True), patch


def test_phycocyanin_band_presence_is_read_off_the_published_band_table():
    """The spectral axis. OLCI is the one operational sensor that passes it."""
    assert has_phycocyanin_band("sentinel3_olci")
    assert nearest_band_distance_nm("sentinel3_olci") == pytest.approx(0.0)
    # The 813 grid is contiguous, so it always has a channel on the feature.
    assert has_phycocyanin_band("marsad_813")
    assert nearest_band_distance_nm("marsad_813") <= PC_BAND_TOLERANCE_NM
    for key in ("sentinel2_msi", "modis_aqua", "landsat8_oli"):
        assert not has_phycocyanin_band(key), key
        assert nearest_band_distance_nm(key) > PC_BAND_TOLERANCE_NM, key


# --------------------------------------------------- spatial ablation: schema
def test_spatial_schema(result):
    spatial = result["spatial"]
    assert set(spatial) == SPATIAL_KEYS
    assert spatial["patch_sizes_m"] == [float(p) for p in SPATIAL_PATCH_SIZES]
    assert isinstance(spatial["n_train"], int) and spatial["n_train"] > 0
    assert isinstance(spatial["n_test"], int) and spatial["n_test"] > 0
    assert isinstance(spatial["note"], str) and spatial["note"].strip()

    assert set(spatial["sensors"]) == set(SENSORS)
    for key, row in spatial["sensors"].items():
        assert set(row) == SPATIAL_SENSOR_KEYS, key
        assert row["label"] == SENSORS[key].label
        assert row["gsd_m"] == float(SENSORS[key].gsd_m)
        assert row["n_bands"] == SENSORS[key].n_bands
        assert isinstance(row["has_620nm"], bool)
        assert row["has_620nm"] == has_phycocyanin_band(key)

        cells = row["by_patch_size"]
        # One cell per patch size, in the order the caller asked for.
        assert [c["patch_size_m"] for c in cells] == [
            float(p) for p in SPATIAL_PATCH_SIZES
        ]
        for cell in cells:
            assert set(cell) == SPATIAL_CELL_KEYS, key
            assert cell["fill_fraction"] == pytest.approx(
                subpixel_fill_fraction(cell["patch_size_m"], key)
            )
            assert 0.0 < cell["fill_fraction"] <= 1.0
            for metric in SPATIAL_CELL_METRICS:
                assert isinstance(cell[metric], float), (key, metric)
                assert 0.0 <= cell[metric] <= 1.0, (key, metric)


def test_spatial_block_is_json_serialisable(result):
    """The dashboard reads it verbatim, so it must survive a JSON round trip."""
    restored = json.loads(json.dumps(result["spatial"]))
    assert set(restored) == SPATIAL_KEYS
    assert set(restored["sensors"]) == set(SENSORS)


def test_run_spatial_ablation_stands_alone_and_draws_its_own_scenes():
    """The standalone entry point, pinned at the cheapest size that runs.

    One sensor, one patch size: a single model fit. It exists so the function
    can be called outside :func:`run_benchmark` without having to hand it a
    scene, which is how an experiment or a notebook will reach for it.
    """
    spatial = run_spatial_ablation(
        seed=SEED,
        n_train=120,
        n_test=120,
        patch_sizes_m=(REFERENCE_PATCH_SIZE_M,),
        sensor_keys=("marsad_813",),
    )
    assert set(spatial) == SPATIAL_KEYS
    assert set(spatial["sensors"]) == {"marsad_813"}
    assert spatial["n_train"] == 120 and spatial["n_test"] == 120
    (cell,) = spatial["sensors"]["marsad_813"]["by_patch_size"]
    assert cell["patch_size_m"] == REFERENCE_PATCH_SIZE_M
    assert cell["fill_fraction"] == 1.0


def test_run_spatial_ablation_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        run_spatial_ablation(patch_sizes_m=())
    with pytest.raises(ValueError):
        run_spatial_ablation(patch_sizes_m=(0.0,), sensor_keys=("marsad_813",))
    with pytest.raises(KeyError):
        run_spatial_ablation(sensor_keys=("landsat_9000",))


# ----------------------------------------------- spatial ablation: behaviour
def test_coarse_sensors_degrade_as_the_patch_shrinks_while_fine_ones_hold(result):
    """The headline behaviour of the spatial grid, in both directions.

    Below its own pixel size a sensor is measuring mostly the water around the
    patch, so its accuracy has to fall. At or above it, nothing has changed at
    all: cells with an identical fill fraction are handed bit-identical inputs
    by construction (the noise field is drawn once per scene), so the fine-GSD
    rows are asserted as EXACTLY equal rather than approximately so. That
    exactness is the control: it proves the drop on the coarse rows is the
    dilution and nothing else.
    """
    cells = {
        key: {c["patch_size_m"]: c for c in row["by_patch_size"]}
        for key, row in result["spatial"]["sensors"].items()
    }
    control = float(max(SPATIAL_PATCH_SIZES))
    intake = float(min(SPATIAL_PATCH_SIZES))

    # Coarse pixels: the patch goes sub-pixel and accuracy falls with it.
    for key in ("sentinel3_olci", "modis_aqua"):
        assert cells[key][control]["fill_fraction"] == 1.0, key
        assert cells[key][intake]["fill_fraction"] < 1.0, key
        assert (
            cells[key][intake]["accuracy"] < cells[key][control]["accuracy"] - 0.05
        ), key

    # Fine pixels: the patch fills the pixel at both sizes, so nothing moves.
    for key in ("marsad_813", "sentinel2_msi", "landsat8_oli"):
        assert cells[key][intake]["fill_fraction"] == 1.0, key
        assert cells[key][control]["fill_fraction"] == 1.0, key
        for metric in SPATIAL_CELL_METRICS:
            assert cells[key][intake][metric] == cells[key][control][metric], (
                key,
                metric,
            )


def test_the_813_arm_leads_the_spatial_grid_at_the_intake_scale(result):
    """At 100 m the hyperspectral 30 m instrument beats every alternative.

    This is the number the spectral-only ablation could not produce: against
    OLCI the spectral margin is a fraction of a percent, and once the patch is
    intake-sized the margin is large, because OLCI is measuring nine parts
    clear water to one part bloom.
    """
    intake = float(min(SPATIAL_PATCH_SIZES))
    cells = {
        key: next(c for c in row["by_patch_size"] if c["patch_size_m"] == intake)
        for key, row in result["spatial"]["sensors"].items()
    }
    native = cells["marsad_813"]["accuracy"]
    for key, cell in cells.items():
        if key != "marsad_813":
            assert cell["accuracy"] < native, key
    assert cells["sentinel3_olci"]["accuracy"] < native - 0.05


# ------------------------------------------------------------------- verdict
def test_verdict_has_one_entry_per_sensor_with_boolean_axes(result):
    verdict = result["verdict"]
    assert set(verdict) == VERDICT_KEYS
    assert verdict["reference_patch_size_m"] == REFERENCE_PATCH_SIZE_M
    assert verdict["fill_fraction_threshold"] == FILL_FRACTION_THRESHOLD
    assert verdict["pc_band_nm"] == PC_BAND_NM
    assert verdict["pc_band_tolerance_nm"] == PC_BAND_TOLERANCE_NM
    assert isinstance(verdict["summary"], str) and verdict["summary"].strip()
    assert isinstance(verdict["note"], str) and verdict["note"].strip()

    assert set(verdict["sensors"]) == set(SENSORS)
    for key, row in verdict["sensors"].items():
        assert set(row) == VERDICT_SENSOR_KEYS, key
        assert row["label"] == SENSORS[key].label
        assert row["gsd_m"] == float(SENSORS[key].gsd_m)
        # Both axes are booleans, not truthy floats: the dashboard renders them
        # as PASS / FAIL and a 0.111 would render as PASS.
        for axis in ("has_620nm", "spectral_ok", "spatial_ok"):
            assert isinstance(row[axis], bool), (key, axis)
        assert row["has_620nm"] == row["spectral_ok"] == has_phycocyanin_band(key)
        assert row["fill_fraction_at_reference"] == pytest.approx(
            subpixel_fill_fraction(REFERENCE_PATCH_SIZE_M, key)
        )
        assert row["spatial_ok"] == (
            row["fill_fraction_at_reference"] >= FILL_FRACTION_THRESHOLD
        )
        assert isinstance(row["reason"], str) and row["reason"].strip()

    # Every sensor lands in exactly one of the four corners.
    placed = [k for bucket in VERDICT_BUCKETS for k in verdict[bucket]]
    assert sorted(placed) == sorted(SENSORS)


def test_813_passes_both_axes_at_the_reference_patch_size(result):
    """The claim the whole 2x2 exists to support, at the documented reference."""
    row = result["verdict"]["sensors"]["marsad_813"]
    assert row["spectral_ok"] is True
    assert row["spatial_ok"] is True
    assert row["fill_fraction_at_reference"] == 1.0
    assert result["verdict"]["adequate_on_both"] == ["marsad_813"]


def test_the_2x2_is_the_answer_to_the_olci_question(result):
    """Each sensor in its corner, which is what makes the comparison honest.

    OLCI is the interesting one and the reason this block exists: it PASSES the
    spectral axis, so "OLCI cannot see phycocyanin" would be false, and it
    fails the spatial one at intake scale. Sentinel-2 and Landsat are the
    mirror image. MODIS fails both.
    """
    sensors_ = result["verdict"]["sensors"]
    axes = {
        key: (row["spectral_ok"], row["spatial_ok"])
        for key, row in sensors_.items()
    }
    assert axes["marsad_813"] == (True, True)
    assert axes["sentinel3_olci"] == (True, False)
    assert axes["sentinel2_msi"] == (False, True)
    assert axes["landsat8_oli"] == (False, True)
    assert axes["modis_aqua"] == (False, False)


def test_verdict_carries_the_measured_numbers_at_the_reference_patch(result):
    """The verdict is readable against the grid, not on geometry alone."""
    spatial = result["spatial"]
    assert REFERENCE_PATCH_SIZE_M in spatial["patch_sizes_m"]
    for key, row in result["verdict"]["sensors"].items():
        cell = next(
            c for c in spatial["sensors"][key]["by_patch_size"]
            if c["patch_size_m"] == REFERENCE_PATCH_SIZE_M
        )
        assert row["accuracy_at_reference"] == cell["accuracy"], key
        assert row["cyano_recall_at_reference"] == cell["cyano_recall"], key


def test_build_verdict_stands_alone_without_any_measurement():
    """Both axes are published instrument facts, so the 2x2 needs no model run."""
    verdict = build_verdict()
    assert set(verdict) == VERDICT_KEYS
    assert set(verdict["sensors"]) == set(SENSORS)
    for key, row in verdict["sensors"].items():
        assert row["accuracy_at_reference"] is None, key
        assert row["cyano_recall_at_reference"] is None, key
    assert verdict["adequate_on_both"] == ["marsad_813"]
    assert verdict["spectral_only"] == ["sentinel3_olci"]
    assert verdict["inadequate_on_both"] == ["modis_aqua"]
    assert sorted(verdict["spatial_only"]) == ["landsat8_oli", "sentinel2_msi"]


def test_verdict_summary_is_generated_from_the_booleans_not_asserted():
    """Change a pixel size and the sentence has to change with it."""
    coarse = build_verdict(reference_patch_size_m=10.0)
    assert coarse["sensors"]["marsad_813"]["spatial_ok"] is False
    assert coarse["adequate_on_both"] == []
    assert "adequate on both axes: none" in coarse["summary"]
    assert "10 m reference patch" in coarse["summary"]

    generous = build_verdict(reference_patch_size_m=5000.0)
    assert all(row["spatial_ok"] for row in generous["sensors"].values())
    assert sorted(generous["adequate_on_both"]) == ["marsad_813", "sentinel3_olci"]


# ------------------------------------------------------------------- honesty
def test_honesty_note_is_present_and_says_what_this_is(result):
    note = result["honesty_note"]
    assert isinstance(note, str) and note.strip()
    assert note == HONESTY_NOTE
    lowered = note.lower()
    assert "self-consistency" in lowered
    assert "synth" in lowered or "forward model" in lowered
    assert "never independent validation" in lowered


def test_the_spatial_and_verdict_notes_carry_the_honesty_rule(result):
    """Every public block says what it is, per docs/CONTRACTS-V2.md."""
    for note in (result["spatial"]["note"], result["verdict"]["note"]):
        assert isinstance(note, str) and note.strip()
        lowered = note.lower()
        assert "marsad.synth" in lowered or "our own" in lowered
        assert "self-consistency check" in lowered
        assert "never independent validation" in lowered
        assert "we proved" not in lowered
        assert "validated on real" not in lowered


def test_honesty_note_never_claims_real_scene_validation():
    """Forbidden phrasing per docs/CONTRACTS-V2.md, checked mechanically."""
    lowered = HONESTY_NOTE.lower()
    assert "we proved" not in lowered
    assert "validated on real" not in lowered


def test_no_em_or_en_dashes_in_the_module_or_its_script():
    """Repo style rule: plain hyphens only, in code and in prose.

    The dashes are written as escapes so this file itself stays clean.
    """
    import marsad.benchmark as module

    # .../<repo>/src/marsad/benchmark.py -> <repo>
    repo_root = Path(module.__file__).resolve().parents[2]
    targets = [Path(module.__file__), repo_root / "scripts" / "run_benchmark.py"]
    forbidden = ((0x2014, "em dash"), (0x2013, "en dash"))
    for path in targets:
        source = path.read_text(encoding="utf-8")
        for codepoint, name in forbidden:
            assert chr(codepoint) not in source, f"{name} in {path.name}"
