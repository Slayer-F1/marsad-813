"""Smoke test for the end-to-end pipeline (src/marsad/pipeline.py).

Runs the full flow with tiny sample counts and asserts the two output files
exist and follow EXACTLY the docs/CONTRACTS.md + docs/CONTRACTS-V2.md schema
(including the v0.2 per-intake ``uncertainty`` block). Model quality is
asserted in the per-stage tests, not here.

The one exception is ``test_demo_story_seed7``, which runs at the full
``run_demo.py`` sizes because it pins the judge-facing narrative; it dominates
this file's wall clock and there is no smaller run that can pin the same story.
"""
from __future__ import annotations

import json

import joblib

from marsad import pipeline
from marsad.pipeline import HORIZON_DAYS, INTAKES, run_end_to_end
from marsad.spectra import N_BANDS

PREFIX = "window.MARSAD_DATA = "
HISTORY_DAYS = 12  # small but > 8 so the 7-day trend lookback is exercised

INTAKE_KEYS = {"name", "lat", "lon", "kind", "risk", "bloom", "uncertainty",
               "history", "forecast"}
UNCERTAINTY_KEYS = {"total", "epistemic", "confidence", "review_recommended"}

# pipeline._ROUND_DP = 4, so a rounded difference can move by at most 1e-4;
# the ordering assertions below carry that slack rather than a bare epsilon.
ROUND_SLACK = 2e-4

# Sizes for the cache tests: as small as the pipeline tolerates, because each
# cold run pays for a full Stage 1 + Stage 2 + ensemble fit.
CACHE_N_TRAIN = 150
CACHE_N_SCENE = 80
CACHE_HISTORY_DAYS = 5


def _without_timestamp(data: dict) -> dict:
    """The payload minus ``generated_utc``, which is wall-clock by design."""
    return {k: v for k, v in data.items() if k != "generated_utc"}


def _canonical(data: dict) -> str:
    """Byte-comparable rendering of a results payload (timestamp removed)."""
    return json.dumps(_without_timestamp(data), sort_keys=True)


def test_run_end_to_end_smoke(tmp_path):
    result = run_end_to_end(seed=3, outdir=tmp_path, n_train=300,
                            n_scene=150, history_days=HISTORY_DAYS)

    # --- both output files exist ------------------------------------------
    results_path = tmp_path / "outputs" / "results.json"
    datajs_path = tmp_path / "dashboard" / "data.js"
    assert results_path.exists()
    assert datajs_path.exists()

    # --- results.json parses and matches the returned dict ----------------
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data == result

    # --- data.js is a single statement wrapping the same JSON -------------
    raw = datajs_path.read_text(encoding="utf-8").strip()
    assert raw.startswith(PREFIX)
    assert raw.endswith(";")
    data_js = json.loads(raw[len(PREFIX):-1])
    assert data_js == data

    # --- top-level schema keys --------------------------------------------
    assert set(data) == {"generated_utc", "model_metrics", "intakes", "spectra_example"}
    mm = data["model_metrics"]
    assert set(mm) == {"stage1_rmse_before", "stage1_rmse_after",
                       "stage2_accuracy", "stage2_confusion", "labels"}
    assert mm["labels"] == ["no_bloom", "dinoflagellate", "cyanobacteria"]
    assert len(mm["stage2_confusion"]) == 3
    assert all(len(row) == 3 for row in mm["stage2_confusion"])

    # --- exactly the four contract intakes, in order ----------------------
    assert len(data["intakes"]) == 4
    assert [i["name"] for i in data["intakes"]] == [i["name"] for i in INTAKES]

    for intake in data["intakes"]:
        assert set(intake) == INTAKE_KEYS

        risk = intake["risk"]
        assert risk["level"] in {"GREEN", "AMBER", "RED"}
        assert 0.0 <= risk["score"] <= 1.0
        assert isinstance(risk["rationale"], list) and risk["rationale"]

        probs = intake["bloom"]["probs"]
        assert set(probs) == {"no_bloom", "dinoflagellate", "cyanobacteria"}
        assert intake["bloom"]["dominant"] in probs

        # v0.2 per-intake uncertainty block (docs/CONTRACTS-V2.md fix 1).
        unc = intake["uncertainty"]
        assert set(unc) == UNCERTAINTY_KEYS
        # Entropies are normalised by log(3), so both live in [0, 1].
        assert 0.0 <= unc["total"] <= 1.0
        assert 0.0 <= unc["epistemic"] <= 1.0
        # Mutual information can never exceed the predictive entropy it is
        # decomposed out of, and averaging over pixels preserves that.
        assert unc["epistemic"] <= unc["total"] + ROUND_SLACK
        # Top-class probability over 3 classes: at worst uniform (1/3).
        assert 1.0 / 3.0 - ROUND_SLACK <= unc["confidence"] <= 1.0
        # Must survive JSON as a real boolean, not as 1/0 (bool subclasses
        # int in Python, so this is a live failure mode for the rounder).
        assert isinstance(unc["review_recommended"], bool)

        history = intake["history"]
        assert len(history) == HISTORY_DAYS
        assert history[0]["day"] == -(HISTORY_DAYS - 1)
        assert history[-1]["day"] == 0
        assert all(0.0 <= h["score"] <= 1.0 for h in history)

        forecast = intake["forecast"]
        assert len(forecast) == 7 == HORIZON_DAYS
        assert [f["day"] for f in forecast] == list(range(1, 8))
        for step in forecast:
            assert step["lo"] <= step["score"] <= step["hi"]
            assert 0.0 <= step["lo"] and step["hi"] <= 1.0

    # --- Hatta Dam is the reservoir: no advection-driven sea bloom --------
    hatta = data["intakes"][3]
    assert hatta["kind"] == "reservoir"

    # --- example spectra span the full band grid --------------------------
    ex = data["spectra_example"]
    assert set(ex) == {"wavelength_nm", "observed", "corrected", "true"}
    assert (len(ex["wavelength_nm"]) == len(ex["observed"])
            == len(ex["corrected"]) == len(ex["true"]) == N_BANDS)

    # --- review_recommended is serialised as a JS boolean, not 1/0 --------
    assert '"review_recommended": true' in raw or '"review_recommended": false' in raw


def test_run_end_to_end_history_days_one(tmp_path):
    """Regression: history_days=1 used to crash with IndexError in the
    trend seed-search (lookback was forced to 1 on a length-1 history)."""
    result = run_end_to_end(seed=3, outdir=tmp_path, n_train=120,
                            n_scene=60, history_days=1)
    for intake in result["intakes"]:
        assert len(intake["history"]) == 1
        assert intake["history"][0]["day"] == 0


def test_model_cache_rejects_junk(tmp_path):
    """A missing, corrupt, foreign or stale artefact must be a MISS.

    The cache is an optimisation; it must never be able to crash a demo run
    or, worse, hand back models that were fitted for a different key.
    """
    path = pipeline._cache_path(tmp_path, seed=3, n_train=CACHE_N_TRAIN)

    # (a) nothing there yet
    assert pipeline._load_model_cache(path, 3, CACHE_N_TRAIN) is None

    # (b) not a joblib file at all
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a joblib artefact")
    assert pipeline._load_model_cache(path, 3, CACHE_N_TRAIN) is None

    # (c) a joblib file holding something else entirely
    joblib.dump({"hello": "world"}, path)
    assert pipeline._load_model_cache(path, 3, CACHE_N_TRAIN) is None

    # (d) right shape, wrong signature (stale recipe / other key)
    joblib.dump({
        "signature": pipeline._cache_signature(3, CACHE_N_TRAIN + 1),
        "corrector": None, "classifier": None, "ensemble": None,
        "stage1_metrics": {}, "stage2_metrics": {},
    }, path)
    assert pipeline._load_model_cache(path, 3, CACHE_N_TRAIN) is None

    # (e) right signature, but the payload does not hold real estimators
    joblib.dump({
        "signature": pipeline._cache_signature(3, CACHE_N_TRAIN),
        "corrector": None, "classifier": None, "ensemble": None,
        "stage1_metrics": {}, "stage2_metrics": {},
    }, path)
    assert pipeline._load_model_cache(path, 3, CACHE_N_TRAIN) is None


def test_model_cache_is_transparent(tmp_path, monkeypatch):
    """Caching changes the wall clock and nothing else.

    Checks, in one test so the (expensive) fits are paid for as few times as
    possible, that for a fixed seed:

    1. a cold cached run equals an uncached run byte for byte,
    2. a warm run reading the artefact equals both,
    3. the artefact really is what the warm run read (poison it and the
       poison shows up), and
    4. ``refit=True`` ignores the artefact and rewrites it.

    The ensemble is shrunk to two members for the duration: this test is
    about the cache being transparent, which is independent of how many
    members the artefact happens to hold, and member fitting is most of the
    cost of the three cold runs below.
    """
    monkeypatch.setattr(pipeline, "ENSEMBLE_MEMBERS", 2)
    kwargs = dict(seed=3, n_train=CACHE_N_TRAIN, n_scene=CACHE_N_SCENE,
                  history_days=CACHE_HISTORY_DAYS)
    cache_dir = tmp_path / "model-cache"
    path = pipeline._cache_path(cache_dir, seed=3, n_train=CACHE_N_TRAIN)

    # 1. no cache at all, then a cold run that populates the cache.
    reference = run_end_to_end(outdir=tmp_path / "nocache", **kwargs)
    cold = run_end_to_end(outdir=tmp_path / "cold", cache_dir=cache_dir, **kwargs)
    assert path.is_file()
    assert _canonical(cold) == _canonical(reference)

    # 2. warm run: same numbers, straight out of the artefact.
    warm = run_end_to_end(outdir=tmp_path / "warm", cache_dir=cache_dir, **kwargs)
    assert _canonical(warm) == _canonical(reference)

    # 3. prove the warm run really read the artefact rather than silently
    #    refitting: mark the stored holdout accuracy and watch it come back.
    payload = joblib.load(path)
    payload["stage2_metrics"]["accuracy"] = -1.0
    joblib.dump(payload, path)
    poisoned = run_end_to_end(outdir=tmp_path / "poisoned", cache_dir=cache_dir,
                              **kwargs)
    assert poisoned["model_metrics"]["stage2_accuracy"] == -1.0

    # 4. refit=True ignores the marked artefact and overwrites it, so both
    #    this run and the next warm run are back on the reference numbers.
    refitted = run_end_to_end(outdir=tmp_path / "refit", cache_dir=cache_dir,
                              refit=True, **kwargs)
    assert _canonical(refitted) == _canonical(reference)
    rewarmed = run_end_to_end(outdir=tmp_path / "rewarm", cache_dir=cache_dir,
                              **kwargs)
    assert _canonical(rewarmed) == _canonical(reference)


def test_demo_story_seed7(tmp_path):
    """Pin the judge-facing demo story (run_demo.py full defaults, seed 7).

    The DEMO_SCENARIO comment promises: Kalba RED on score alone, Hatta Dam
    RED via the precautionary AMBER->RED promotion, Khor Fakkan AMBER,
    Layyah GREEN. Pinning it here keeps the story from drifting silently
    when models, scenario numbers, or the risk policy change.
    """
    data = run_end_to_end(seed=7, outdir=tmp_path, n_train=4000, n_scene=1200)
    by_name = {i["name"]: i for i in data["intakes"]}

    kalba = by_name["Kalba"]
    assert kalba["risk"]["level"] == "RED"
    assert kalba["risk"]["score"] >= 0.65  # RED on merit, not via promotion
    assert not any("precautionary" in r for r in kalba["risk"]["rationale"])
    assert kalba["bloom"]["dominant"] == "dinoflagellate"

    hatta = by_name["Hatta Dam"]
    assert hatta["risk"]["level"] == "RED"
    assert hatta["risk"]["score"] < 0.65  # promoted borderline AMBER
    assert any("precautionary" in r for r in hatta["risk"]["rationale"])
    assert hatta["bloom"]["dominant"] == "cyanobacteria"

    assert by_name["Khor Fakkan"]["risk"]["level"] == "AMBER"
    assert by_name["Layyah"]["risk"]["level"] == "GREEN"

    # Dashboard coherence: the day-0 history point equals the card score,
    # so the chart, the dial, and the forecast origin all agree on screen.
    for intake in data["intakes"]:
        assert abs(intake["history"][-1]["score"] - intake["risk"]["score"]) < 1e-6
