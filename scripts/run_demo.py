#!/usr/bin/env python
"""Run the full MARSAD 813 demo pipeline and print a control-room summary.

Usage (from the repo root):

    ".venv/Scripts/python" scripts/run_demo.py [--seed N] [--fast] [--no-cache]

Trains Stage 1 (shallow-water correction), Stage 2 (bloom speciation) and the
Stage 2 uncertainty ensemble on a synthetic Gulf-water scene, assesses all four
monitored intakes, writes ``outputs/results.json`` + ``dashboard/data.js``, and
prints a per-intake risk table, the alert feed that ``scripts/serve_api.py``
would publish, and the model metrics.

``--fast`` shrinks the synthetic sample counts for a quick smoke run and writes
under ``outputs/fast-preview`` so a weaker model can never overwrite the
judge-facing dashboard. ``--no-cache`` disables the fitted-model cache and
forces a cold retrain; results are identical either way, only the wall clock
differs (see ``pipeline.run_end_to_end``).

Scientific honesty: every number printed here comes from ``marsad.synth``, our
own physics-based forward model of Gulf Case-2 water. This is a
self-consistency check against a simulation, never independent validation of
real Gulf water or evidence about how any algorithm behaves on a real scene.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marsad.alerts import alert_feed  # noqa: E402
from marsad.pipeline import run_end_to_end  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _trend_per_day(history: list[dict]) -> float:
    """Mean daily change of the risk score over (up to) the last 7 days.

    History entries are ``{"day": d, "score": s}`` with day 0 = today, so the
    7-day trend is the slope between today and 7 days ago.
    """
    lookback = max(1, min(7, len(history) - 1))
    return (history[-1]["score"] - history[-1 - lookback]["score"]) / lookback


def _print_intake_table(data: dict) -> None:
    """Per-intake risk table: level, score, class, chl, trend, uncertainty."""
    header = (f"{'Intake':<14} {'Level':<6} {'Score':>6} {'Dominant bloom':<16} "
              f"{'Chl mg/m3':>10} {'7d trend':>10} {'Unc':>6} {'Conf':>6} {'Review':>7}")
    print(header)
    print("-" * len(header))
    for intake in data["intakes"]:
        trend = _trend_per_day(intake["history"])
        unc = intake.get("uncertainty", {})
        print(f"{intake['name']:<14} "
              f"{intake['risk']['level']:<6} "
              f"{intake['risk']['score']:>6.3f} "
              f"{intake['bloom']['dominant']:<16} "
              f"{intake['bloom']['chl_mg_m3']:>10.2f} "
              f"{trend:>+9.4f}/d "
              f"{unc.get('total', float('nan')):>6.3f} "
              f"{unc.get('confidence', float('nan')):>6.3f} "
              f"{('YES' if unc.get('review_recommended') else 'no'):>7}")
    print("  Unc = ensemble predictive entropy (0 = certain, 1 = no idea); "
          "Conf = mean top-class probability;")
    print("  Review = model uncertainty above the analyst-review threshold.")


def _print_alert_feed(data: dict) -> None:
    """Print the ``GET /v1/alerts`` payload as an operator would read it.

    Same ``marsad.alerts.alert_feed`` call the API server makes, so what the
    console shows and what a subscriber receives cannot drift apart.
    """
    feed = alert_feed(data)
    counts = feed["counts"]
    print(f"\nAlert feed ({feed['source']}, data basis: {feed['data_basis']}):")
    print(f"  {counts['RED']} RED   {counts['AMBER']} AMBER   "
          f"{counts['GREEN']} GREEN   (issued {feed['generated_utc']})")
    if not feed["alerts"]:
        print("  no intake at or above AMBER - nothing to publish.")
        return
    for alert in feed["alerts"]:
        lead = alert["lead_days"]
        if lead == 0:
            lead_txt = "now"
        elif lead > 0:
            lead_txt = f"{lead}d"
        else:
            lead_txt = "none"
        print(f"  [{alert['level']:<5}] {alert['intake']:<14} "
              f"lead-to-RED {lead_txt:>4}  {alert['message']}")
        for reason in alert["rationale"]:
            print(f"{'':>26}- {reason}")


def _print_model_metrics(data: dict) -> None:
    """Stage 1 / Stage 2 holdout metrics and the confusion matrix."""
    mm = data["model_metrics"]
    improvement = 0.0
    if mm["stage1_rmse_before"] > 0:
        improvement = 100.0 * (1.0 - mm["stage1_rmse_after"] / mm["stage1_rmse_before"])
    print("\nStage 1 (shallow-water correction, holdout):")
    print(f"  RMSE before: {mm['stage1_rmse_before']:.5f}   "
          f"after: {mm['stage1_rmse_after']:.5f}   improvement: {improvement:.1f}%")
    print("Stage 2 (bloom detection & speciation, holdout):")
    print(f"  accuracy: {mm['stage2_accuracy']:.3f}")
    print("  confusion (rows = true, cols = predicted):")
    label_w = max(len(lbl) for lbl in mm["labels"])
    print(" " * (label_w + 4) + "  ".join(f"{lbl[:10]:>10}" for lbl in mm["labels"]))
    for lbl, row in zip(mm["labels"], mm["stage2_confusion"]):
        print(f"    {lbl:<{label_w}}" + "  ".join(f"{v:>10d}" for v in row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="master rng seed (default: 7)")
    parser.add_argument("--fast", action="store_true",
                        help="use small n_train/n_scene for a quick demo run")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true",
                        help="do not read or write the fitted-model cache "
                             "(forces a cold retrain; identical results)")
    args = parser.parse_args(argv)

    if args.fast:
        n_train, n_scene = 800, 300
        # A fast run trains weaker models - never let it silently replace
        # the judge-facing dashboard/data.js written by a full run.
        outdir = _REPO_ROOT / "outputs" / "fast-preview"
        cache_dir = outdir / "model-cache"
    else:
        n_train, n_scene = 4000, 1200
        outdir = None  # repo root: updates outputs/ and dashboard/data.js
        cache_dir = _REPO_ROOT / "outputs" / "model-cache"
    if args.no_cache:
        cache_dir = None

    cache_note = "disabled (--no-cache)" if cache_dir is None else str(cache_dir)
    print(f"MARSAD 813 - running end-to-end pipeline "
          f"(seed={args.seed}, n_train={n_train}, n_scene={n_scene})")
    print(f"Model cache: {cache_note}")
    data = run_end_to_end(seed=args.seed, n_train=n_train, n_scene=n_scene,
                          outdir=outdir, cache_dir=cache_dir)

    print(f"\nGenerated (UTC): {data['generated_utc']}\n")
    _print_intake_table(data)
    _print_alert_feed(data)
    _print_model_metrics(data)

    print("\nAll figures above are measured on our own physics-based synthetic "
          "Gulf-water simulation")
    print("(marsad.synth): a self-consistency check, not independent "
          "validation on real water.")

    if args.fast:
        print(f"\nFast preview written under {outdir} - "
              "the real dashboard/data.js was NOT touched.")
    else:
        print("\nOutputs written: outputs/results.json, dashboard/data.js")
        print("Open dashboard/index.html in a browser to view the control room.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
