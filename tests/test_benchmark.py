"""Tests for the benchmark experiment (src/marsad/benchmark.py).

Two things are pinned here. First the CONTRACT: the result dictionary is
consumed by ``scripts/run_benchmark.py`` and by the dashboard, so its shape is
asserted key by key rather than loosely. Second the HEADLINE BEHAVIOUR, which
is the whole reason the module exists:

* in shallow turbid water, MARSAD beats both the classical operator decision
  tree on speciation and OC4 / NDCI on chlorophyll error;
* the hyperspectral instrument beats every multispectral alternative;
* speciation collapses on a sensor with no 620 nm phycocyanin band.

Sizes are deliberately tiny. One :func:`run_benchmark` call retrains Stage 1 +
Stage 2 once at full size and once per sensor, so the module-scoped fixture
runs it exactly once and every test reads the same result.

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
    HONESTY_NOTE,
    SHALLOW_DEPTH_M,
    TURBID_TSS_G_M3,
    WATER_REGIMES,
    classify_regime,
    run_benchmark,
)
from marsad.sensors import SENSORS

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
TOP_LEVEL_KEYS = {"chl_retrieval", "speciation", "ablation", "headline", "honesty_note"}


@pytest.fixture(scope="module")
def result() -> dict:
    """One benchmark run shared by the whole module (it is the slow part)."""
    return run_benchmark(seed=SEED, n_train=N_TRAIN, n_test=N_TEST)


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


# ------------------------------------------------------------------- honesty
def test_honesty_note_is_present_and_says_what_this_is(result):
    note = result["honesty_note"]
    assert isinstance(note, str) and note.strip()
    assert note == HONESTY_NOTE
    lowered = note.lower()
    assert "self-consistency" in lowered
    assert "synth" in lowered or "forward model" in lowered
    assert "never independent validation" in lowered


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
