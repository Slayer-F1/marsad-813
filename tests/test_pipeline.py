"""Smoke test for the end-to-end pipeline (src/marsad/pipeline.py).

Runs the full flow with tiny sample counts and asserts the two output files
exist and follow EXACTLY the docs/CONTRACTS.md schema. Model quality is
asserted in the per-stage tests, not here.
"""
from __future__ import annotations

import json

from marsad.pipeline import HORIZON_DAYS, INTAKES, run_end_to_end
from marsad.spectra import N_BANDS

PREFIX = "window.MARSAD_DATA = "
HISTORY_DAYS = 12  # small but > 8 so the 7-day trend lookback is exercised


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
        risk = intake["risk"]
        assert risk["level"] in {"GREEN", "AMBER", "RED"}
        assert 0.0 <= risk["score"] <= 1.0
        assert isinstance(risk["rationale"], list) and risk["rationale"]

        probs = intake["bloom"]["probs"]
        assert set(probs) == {"no_bloom", "dinoflagellate", "cyanobacteria"}
        assert intake["bloom"]["dominant"] in probs

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


def test_run_end_to_end_history_days_one(tmp_path):
    """Regression: history_days=1 used to crash with IndexError in the
    trend seed-search (lookback was forced to 1 on a length-1 history)."""
    result = run_end_to_end(seed=3, outdir=tmp_path, n_train=120,
                            n_scene=60, history_days=1)
    for intake in result["intakes"]:
        assert len(intake["history"]) == 1
        assert intake["history"][0]["day"] == 0


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
