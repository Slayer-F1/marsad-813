"""End-to-end orchestration of the MARSAD 813 early-warning pipeline.

Flow (see docs/CONTRACTS.md, section "pipeline.py"):

1. Generate a synthetic Gulf-water training set and split it into fit/holdout.
2. Fit Stage 1 (learned shallow-water correction) on (observed, true) pairs and
   report its holdout RMSE before/after correction.
3. Fit Stage 2 (bloom detection & speciation) on Stage-1 *corrected* spectra —
   the classifier must never see bottom/glint-contaminated spectra, exactly as
   it will not in operations.
4. Generate a fresh scene, correct it, and assign each monitored intake a slice
   of the scene: sea desalination intakes draw dinoflagellate-or-clear pixels
   (red tides are a marine phenomenon in the Gulf/Sea of Oman), while the Hatta
   Dam reservoir draws cyanobacteria-or-clear pixels (toxic cyano blooms are an
   inland freshwater phenomenon).
5. Per intake: aggregate class probabilities and chlorophyll, synthesise an
   autocorrelated 30-day risk history ending at the current score, run the
   Stage 3 drift/forecast model, and apply the Stage 4 risk policy.
6. Write ``outputs/results.json`` and ``dashboard/data.js`` with EXACTLY the
   schema in docs/CONTRACTS.md.

Geometry placeholders
---------------------
The synthetic scene has no map projection, so two quantities that in operations
come from geolocated data are taken from the fixed ``DEMO_SCENARIO`` table and
documented as placeholders:

- ``drift_toward_intake_kmday``: bloom-patch advection speed toward the intake.
  Real system: project ocean-current vectors (CMEMS/HYCOM or HF radar) onto the
  intake bearing. Reservoir gets 0.0 — a Hatta Dam bloom cannot advect anywhere.
- ``distance_km``: distance from the detected bloom patch centroid to the
  intake. Real system: geodesic distance from the classified pixel mask.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import synth
from .risk import AMBER_THRESHOLD, compute_risk_index
from .spectra import BAND_GRID
from .stage1_correction import ShallowWaterCorrector
from .stage2_classifier import BloomClassifier
from .stage3_forecast import DriftForecaster

# Monitored assets (contract-fixed). Three Gulf of Oman / Gulf coast
# desalination intakes plus one inland reservoir.
INTAKES: list[dict] = [
    {"name": "Khor Fakkan", "lat": 25.339, "lon": 56.353, "kind": "desalination_intake"},
    {"name": "Kalba", "lat": 25.074, "lon": 56.356, "kind": "desalination_intake"},
    {"name": "Layyah", "lat": 25.356, "lon": 55.386, "kind": "desalination_intake"},
    {"name": "Hatta Dam", "lat": 24.783, "lon": 56.113, "kind": "reservoir"},
]

# Demo scenario composition (PLACEHOLDER — the real system derives these from
# spatial windows around each asset, classified-pixel-mask geodesics, and
# CMEMS/HYCOM current projections). ``bloom_share`` is the target fraction of
# the intake's monitored pixels that are bloom pixels; ``distance_km`` and
# ``drift_kmday`` are the bloom-patch geometry the risk policy consumes. The
# mix is composed so the demo exercises every alert mechanism: Kalba a dense
# approaching red tide that is RED on score alone; Khor Fakkan a developing
# patch (AMBER); Layyah quiet (GREEN); Hatta Dam a borderline cyanobacteria
# case that the precautionary uncertainty rule promotes AMBER -> RED at the
# dam intake (tests/test_pipeline.py pins this story for seed 7).
DEMO_SCENARIO: dict[str, dict] = {
    "Khor Fakkan": {"bloom_share": 0.42, "distance_km": 6.5, "drift_kmday": 0.9},
    "Kalba":       {"bloom_share": 0.80, "distance_km": 2.5, "drift_kmday": 1.8},
    "Layyah":      {"bloom_share": 0.05, "distance_km": 9.0, "drift_kmday": 0.4},
    "Hatta Dam":   {"bloom_share": 0.68, "distance_km": 0.5, "drift_kmday": 0.0},
}

HORIZON_DAYS = 7          # forecast horizon fixed by the dashboard schema
HOLDOUT_FRACTION = 0.25   # fraction of the training set held out for metrics
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUND_DP = 4             # keep results.json / data.js small


def _round(obj: Any, ndigits: int = _ROUND_DP) -> Any:
    """Recursively round floats (and coerce numpy scalars) for JSON output."""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, ndigits) for v in obj]
    return obj


def _assign_intake_pixels(labels: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Assign each intake a disjoint slice of scene pixel indices.

    Physics/logic rationale: each intake monitors its own patch of water, so
    slices are disjoint. Sea intakes may only contain dinoflagellate (label 1)
    or clear (label 0) pixels; the reservoir only cyanobacteria (label 2) or
    clear. Each intake's clear/bloom pixel mix targets its
    ``DEMO_SCENARIO.bloom_share`` — the stand-in for the real spatial window
    around each asset. The rng only shuffles which pixels land where; the
    composition itself is deterministic so the demo story is reproducible.
    """
    idx_clear = rng.permutation(np.flatnonzero(labels == 0))
    idx_dino = rng.permutation(np.flatnonzero(labels == 1))
    idx_cyano = rng.permutation(np.flatnonzero(labels == 2))

    n_sea = sum(1 for i in INTAKES if i["kind"] == "desalination_intake")
    clear_chunks = np.array_split(idx_clear, len(INTAKES))
    dino_chunks = np.array_split(idx_dino, n_sea)

    assigned: list[np.ndarray] = []
    sea_i = 0
    for chunk_i, intake in enumerate(INTAKES):
        if intake["kind"] == "reservoir":
            bloom_pool = idx_cyano
            eligible = np.concatenate([idx_clear, idx_cyano])
        else:
            bloom_pool = dino_chunks[sea_i] if sea_i < len(dino_chunks) else np.array([], dtype=int)
            eligible = np.concatenate([idx_clear, idx_dino])
            sea_i += 1
        clear_pool = clear_chunks[chunk_i] if chunk_i < len(clear_chunks) else np.array([], dtype=int)

        # Hit the target bloom share exactly whenever the pools allow it:
        # either the clear pool binds (use it all, take matching blooms) or
        # the bloom pool binds (use it all, trim the clear pool to match).
        share = float(DEMO_SCENARIO[intake["name"]]["bloom_share"])
        if share <= 0.0 or len(bloom_pool) == 0:
            n_bloom, n_clear = 0, len(clear_pool)
        elif share >= 1.0:
            n_bloom, n_clear = len(bloom_pool), 0
        else:
            want = int(round(len(clear_pool) * share / (1.0 - share)))
            if want <= len(bloom_pool):
                n_bloom, n_clear = want, len(clear_pool)
            else:
                n_bloom = len(bloom_pool)
                n_clear = min(len(clear_pool),
                              int(round(n_bloom * (1.0 - share) / share)))

        pixels = np.concatenate([clear_pool[:n_clear], bloom_pool[:n_bloom]]).astype(int)
        if pixels.size == 0:  # tiny-scene guard (smoke tests): fall back to any eligible pixel
            pixels = eligible.astype(int) if eligible.size else np.arange(len(labels))
        assigned.append(pixels)
    return assigned


def run_end_to_end(seed: int = 7, outdir=None, n_train: int = 4000,
                   n_scene: int = 1200, history_days: int = 30) -> dict:
    """Train the full pipeline, assess every intake, and write the outputs.

    Parameters
    ----------
    seed : master seed; every random draw derives from it deterministically.
    outdir : base directory for ``outputs/`` and ``dashboard/``; defaults to
        the repository root so ``dashboard/index.html`` finds ``data.js``.
    n_train, n_scene, history_days : sizes, reduced by tests/--fast for speed.

    Returns
    -------
    dict with exactly the ``dashboard/data.js`` schema of docs/CONTRACTS.md
    (also serialised verbatim to ``outputs/results.json``).
    """
    rng = np.random.default_rng(seed)

    # --- 1. training data + fit/holdout split -----------------------------
    train = synth.generate_dataset(n_train, seed=seed)
    n_hold = max(1, int(round(n_train * HOLDOUT_FRACTION)))
    perm = rng.permutation(n_train)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    # --- 2. Stage 1: learned shallow-water correction ---------------------
    corrector = ShallowWaterCorrector(seed=seed)
    corrector.fit(train.rrs_observed[fit_idx], train.rrs_true[fit_idx])
    stage1_metrics = corrector.score(train.rrs_observed[hold_idx], train.rrs_true[hold_idx])

    # --- 3. Stage 2: bloom classifier, trained on CORRECTED spectra -------
    # Rationale: in operations the classifier only ever sees Stage-1 output,
    # so training on corrected spectra keeps train/serve distributions aligned.
    corrected_fit = corrector.transform(train.rrs_observed[fit_idx])
    classifier = BloomClassifier(seed=seed)
    classifier.fit(corrected_fit, train.labels[fit_idx], train.chl[fit_idx])
    corrected_hold = corrector.transform(train.rrs_observed[hold_idx])
    stage2_metrics = classifier.evaluate(corrected_hold, train.labels[hold_idx])

    # --- 4. fresh scene -> correct -> classify ----------------------------
    scene = synth.generate_dataset(n_scene, seed=seed + 1)
    scene_corrected = corrector.transform(scene.rrs_observed)
    probs_all = classifier.predict_proba(scene_corrected)
    chl_all = classifier.estimate_chl(scene_corrected)

    assign_rng = np.random.default_rng(seed + 2)
    pixel_sets = _assign_intake_pixels(scene.labels, assign_rng)

    forecaster = DriftForecaster(horizon_days=HORIZON_DAYS)
    intakes_out: list[dict] = []

    for k, (intake, pixels) in enumerate(zip(INTAKES, pixel_sets)):
        # -- aggregate Stage 2 output over the intake's pixel slice --------
        p = probs_all[pixels].mean(axis=0)
        p = p / p.sum()  # renormalise the mean of per-pixel simplex vectors
        bloom_prob = float(1.0 - p[0])
        chl = float(chl_all[pixels].mean())

        # Report the dominant *bloom* type whenever the bloom probability is
        # alert-relevant (>= AMBER threshold) — an alerted card captioned
        # "no_bloom" reads as a contradiction even when clear pixels still
        # hold a plurality of the mean probability vector.
        if bloom_prob >= AMBER_THRESHOLD:
            dominant = synth.LABELS[1 + int(np.argmax(p[1:]))]
        else:
            dominant = synth.LABELS[int(np.argmax(p))]

        # -- scenario advection + distance (real data: currents/geodesy) --
        # Drift is gated on the alert-relevant bloom probability: currents
        # only move risk toward an intake if there is a detected bloom patch
        # to move, so a quiet intake must not inherit a rising outlook from
        # the ocean current alone.
        scenario = DEMO_SCENARIO[intake["name"]]
        distance_km = float(scenario["distance_km"])
        drift_toward_intake_kmday = (
            float(scenario["drift_kmday"]) if bloom_prob >= AMBER_THRESHOLD else 0.0
        )

        # -- day-0 uncertainty: classifier AMBIGUITY, not spatial spread ---
        # mean(1 - 2|p - 0.5|) is ~0 when every pixel is confidently
        # classified and ~1 when the classifier genuinely cannot decide.
        # The per-pixel std of bloom probability is the WRONG quantity here:
        # for a confidently classified mixed slice it equals
        # sqrt(share*(1-share)) — largest exactly when the model is most
        # certain — which would invert the precautionary promotion.
        pixel_bloom_prob = 1.0 - probs_all[pixels, 0]
        ambiguity = float(np.mean(1.0 - 2.0 * np.abs(pixel_bloom_prob - 0.5)))

        # -- current risk (trend-free), then history ending at that score --
        # The trend is *defined* by the history, so the current score is first
        # computed with trend 0; the final assessment folds the recovered
        # trend back in.
        current = compute_risk_index(bloom_prob, chl, 0.0, distance_km, ambiguity)

        # Compose a history whose recent trend AGREES with the current state
        # (rising when a bloom is likely, easing when quiet) so the synthetic
        # demo tells one coherent story; a real deployment reads its archived
        # risk series instead. The bounded seed search keeps determinism.
        lookback = min(7, history_days - 1)
        hist_seed = seed + 100 + k
        if lookback < 1:
            # Degenerate single-day history: no trend is derivable from it.
            history = synth.generate_history(current.score, n_days=history_days,
                                             seed=hist_seed)
            trend_per_day = 0.0
        else:
            want_rising = bloom_prob >= 0.35
            history = None
            for off in range(50):
                cand_seed = seed + 100 + k + 1000 * off
                cand_h = synth.generate_history(current.score, n_days=history_days,
                                                seed=cand_seed)
                t = float((cand_h[-1] - cand_h[-1 - lookback]) / lookback)
                if (t > 0.002) if want_rising else (t < -0.002):
                    history, hist_seed = cand_h, cand_seed
                    break
            if history is None:
                history, hist_seed = cand_h, cand_seed
            trend_per_day = float((history[-1] - history[-1 - lookback]) / lookback)

        # -- Stage 3 forecast + final risk assessment ----------------------
        fc = forecaster.forecast(history, drift_toward_intake_kmday=drift_toward_intake_kmday)
        unc_forecast = float(np.mean(fc.hi - fc.lo))
        assessment = compute_risk_index(bloom_prob, chl, trend_per_day,
                                        distance_km, max(ambiguity, unc_forecast))

        # Re-anchor the displayed series at the FINAL assessed score so the
        # dashboard's day-0 dot, the card score, and the forecast origin all
        # agree (the first pass used the trend-free score only to break the
        # history -> trend -> score circularity). Same seed, same walk shape;
        # the recomputed trend is unchanged up to boundary reflection, so the
        # assessment itself is not recomputed.
        if abs(assessment.score - current.score) > 1e-9:
            history = synth.generate_history(assessment.score, n_days=history_days,
                                             seed=hist_seed)
            fc = forecaster.forecast(history, drift_toward_intake_kmday=drift_toward_intake_kmday)

        intakes_out.append({
            "name": intake["name"],
            "lat": intake["lat"],
            "lon": intake["lon"],
            "kind": intake["kind"],
            "risk": {
                "score": float(assessment.score),
                "level": str(assessment.level.value),
                "rationale": [str(r) for r in assessment.rationale],
            },
            "bloom": {
                "probs": {
                    "no_bloom": float(p[0]),
                    "dinoflagellate": float(p[1]),
                    "cyanobacteria": float(p[2]),
                },
                "dominant": dominant,
                "chl_mg_m3": chl,
            },
            "history": [
                {"day": int(d - (history_days - 1)), "score": float(s)}
                for d, s in enumerate(history)
            ],
            "forecast": [
                {"day": j + 1, "score": float(fc.mean[j]),
                 "lo": float(fc.lo[j]), "hi": float(fc.hi[j])}
                for j in range(HORIZON_DAYS)
            ],
        })

    # --- 5. example spectra for the dashboard chart -----------------------
    # Pick the shallowest bloom pixel: shallow + bloom maximises the visible
    # bottom/glint contamination that Stage 1 removes.
    cand = np.flatnonzero(scene.labels == 1)
    if cand.size == 0:
        cand = np.flatnonzero(scene.labels != 0)
    if cand.size == 0:
        cand = np.arange(len(scene.labels))
    example = int(cand[np.argmin(scene.depth_m[cand])])

    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_metrics": {
            "stage1_rmse_before": float(stage1_metrics["rmse_before"]),
            "stage1_rmse_after": float(stage1_metrics["rmse_after"]),
            "stage2_accuracy": float(stage2_metrics["accuracy"]),
            "stage2_confusion": [[int(v) for v in row] for row in stage2_metrics["confusion"]],
            "labels": [synth.LABELS[i] for i in (0, 1, 2)],
        },
        "intakes": intakes_out,
        "spectra_example": {
            "wavelength_nm": [float(w) for w in BAND_GRID],
            "observed": [float(v) for v in scene.rrs_observed[example]],
            "corrected": [float(v) for v in scene_corrected[example]],
            "true": [float(v) for v in scene.rrs_true[example]],
        },
    }
    data = _round(data)

    # --- 6. write outputs/results.json and dashboard/data.js --------------
    base = Path(outdir) if outdir is not None else _REPO_ROOT
    outputs_dir = base / "outputs"
    dashboard_dir = base / "dashboard"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(data, default=float)  # default=float catches stray numpy scalars
    (outputs_dir / "results.json").write_text(payload, encoding="utf-8")
    (dashboard_dir / "data.js").write_text(
        "window.MARSAD_DATA = " + payload + ";\n", encoding="utf-8"
    )
    return data
