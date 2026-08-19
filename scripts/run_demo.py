#!/usr/bin/env python
"""Run the full MARSAD 813 demo pipeline and print a control-room summary.

Usage (from the repo root):

    ".venv/Scripts/python" scripts/run_demo.py [--seed N] [--fast]

Trains Stage 1 (shallow-water correction) and Stage 2 (bloom speciation) on a
synthetic Gulf-water scene, assesses all four monitored intakes, writes
``outputs/results.json`` + ``dashboard/data.js``, and prints a per-intake risk
table plus model metrics. ``--fast`` shrinks the synthetic sample counts for a
quick smoke run (seconds instead of minutes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marsad.pipeline import run_end_to_end  # noqa: E402


def _trend_per_day(history: list[dict]) -> float:
    """Mean daily change of the risk score over (up to) the last 7 days.

    History entries are ``{"day": d, "score": s}`` with day 0 = today, so the
    7-day trend is the slope between today and 7 days ago.
    """
    lookback = max(1, min(7, len(history) - 1))
    return (history[-1]["score"] - history[-1 - lookback]["score"]) / lookback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="master rng seed (default: 7)")
    parser.add_argument("--fast", action="store_true",
                        help="use small n_train/n_scene for a quick demo run")
    args = parser.parse_args(argv)

    if args.fast:
        n_train, n_scene = 800, 300
        # A fast run trains weaker models - never let it silently replace
        # the judge-facing dashboard/data.js written by a full run.
        outdir = Path(__file__).resolve().parents[1] / "outputs" / "fast-preview"
    else:
        n_train, n_scene = 4000, 1200
        outdir = None  # repo root: updates outputs/ and dashboard/data.js

    print(f"MARSAD 813 - running end-to-end pipeline "
          f"(seed={args.seed}, n_train={n_train}, n_scene={n_scene}) ...")
    data = run_end_to_end(seed=args.seed, n_train=n_train, n_scene=n_scene,
                          outdir=outdir)

    print(f"\nGenerated (UTC): {data['generated_utc']}\n")

    # ---- per-intake risk table -------------------------------------------
    header = (f"{'Intake':<14} {'Level':<6} {'Score':>6} {'Dominant bloom':<16} "
              f"{'Chl mg/m3':>10} {'7d trend':>10}")
    print(header)
    print("-" * len(header))
    for intake in data["intakes"]:
        trend = _trend_per_day(intake["history"])
        print(f"{intake['name']:<14} "
              f"{intake['risk']['level']:<6} "
              f"{intake['risk']['score']:>6.3f} "
              f"{intake['bloom']['dominant']:<16} "
              f"{intake['bloom']['chl_mg_m3']:>10.2f} "
              f"{trend:>+9.4f}/d")

    # ---- stage metrics ----------------------------------------------------
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

    if args.fast:
        print(f"\nFast preview written under {outdir} - "
              "the real dashboard/data.js was NOT touched.")
    else:
        print("\nOutputs written: outputs/results.json, dashboard/data.js")
        print("Open dashboard/index.html in a browser to view the control room.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
