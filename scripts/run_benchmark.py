#!/usr/bin/env python
"""Run the MARSAD 813 benchmark and print the comparison tables.

Usage (from the repo root):

    ".venv/Scripts/python" scripts/run_benchmark.py [--seed N] [--fast]

This is the experiment behind the project's central claim: standard satellite
chlorophyll algorithms lose their footing in shallow turbid Gulf-coastal water
while the MARSAD two-stage approach does not, and speciation collapses on any
sensor that cannot see the 620 nm phycocyanin band. It prints three tables -
chlorophyll retrieval error by water regime, speciation accuracy by water
regime, and the hyperspectral sensor ablation - then the headline numbers.

A full run trains Stage 1 + Stage 2 once at full size plus once per sensor and
takes a couple of minutes. ``--fast`` shrinks every sample count for a quick
smoke run and, like ``scripts/run_demo.py``, writes into
``outputs/fast-preview/`` so a weaker model can never silently replace the
judge-facing ``dashboard/benchmark.js``.

SCIENTIFIC HONESTY: every number printed here is measured on marsad.synth, our
own physics-based forward model, so this is a self-consistency check against a
simulation, never independent validation on real Gulf water. The note is
printed with the results for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Allow running the script directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from marsad.benchmark import WATER_REGIMES, run_benchmark  # noqa: E402
from marsad.sensors import SENSORS  # noqa: E402

#: A sensor can only see phycocyanin if it has a band this close to 620 nm.
#: Half a typical ocean-colour band width: further away and the 620 nm
#: absorption is averaged into the surrounding continuum.
_PC_BAND_TOLERANCE_NM = 10.0

_ROUND_DP = 4  # keep benchmark.json / benchmark.js small and readable
_WRAP = 78

# Human-readable regime names for the printed tables. The keys stay machine
# readable in the JSON; only the console gets prose.
_REGIME_LABELS = {
    "optically_deep": "Optically deep",
    "shallow_clear": "Shallow, clear",
    "turbid_deep": "Deep, turbid",
    "shallow_turbid": "Shallow + turbid",
}


def _round(obj: Any, ndigits: int = _ROUND_DP) -> Any:
    """Recursively round floats and coerce numpy scalars for JSON output."""
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, ndigits) for v in obj]
    return obj


def _rule(width: int) -> str:
    return "-" * width


def _has_620_band(sensor_key: str) -> bool:
    """True when the sensor carries a band close enough to see phycocyanin."""
    centers = SENSORS[sensor_key].centers_nm
    return bool(np.min(np.abs(centers - 620.0)) <= _PC_BAND_TOLERANCE_NM)


def _print_chl_table(result: dict) -> None:
    """Chlorophyll retrieval error by water regime (lower is better)."""
    print("\n1. CHLOROPHYLL RETRIEVAL - median absolute log10 error, lower is better")
    print("   Baselines run on the RAW observed spectra (what an operator has);")
    print("   MARSAD runs on Stage-1-corrected spectra (what MARSAD has).")
    print("   0.30 means typically a factor of 2 out; 1.00 means an order of magnitude.")
    header = (f"   {'Water regime':<18}{'n':>6}{'OC4':>9}{'OC3M':>9}"
              f"{'NDCI':>9}{'MARSAD':>9}{'best baseline / MARSAD':>25}")
    print("\n" + header)
    print("   " + _rule(len(header) - 3))
    for regime in WATER_REGIMES:
        row = result["chl_retrieval"][regime]
        best = min(row["oc4"], row["oc3m"], row["ndci"])
        factor = best / row["marsad"] if row["marsad"] > 0 else float("inf")
        print(f"   {_REGIME_LABELS[regime]:<18}{row['n']:>6}{row['oc4']:>9.3f}"
              f"{row['oc3m']:>9.3f}{row['ndci']:>9.3f}{row['marsad']:>9.3f}"
              f"{factor:>24.1f}x")


def _print_speciation_table(result: dict) -> None:
    """Speciation accuracy by water regime (higher is better)."""
    print("\n2. SPECIATION - accuracy over {no_bloom, dinoflagellate, cyanobacteria}")
    print("   Baseline = the classical operator decision tree (NDCI chl threshold,")
    print("   then 620 nm phycocyanin line height). MARSAD = Stage 2.")
    header = (f"   {'Water regime':<18}{'n':>6}{'baseline tree':>16}"
              f"{'MARSAD':>10}{'gain':>10}")
    print("\n" + header)
    print("   " + _rule(len(header) - 3))
    for regime in WATER_REGIMES:
        row = result["speciation"][regime]
        gain = row["marsad"] - row["baseline_tree"]
        print(f"   {_REGIME_LABELS[regime]:<18}{row['n']:>6}"
              f"{row['baseline_tree']:>16.3f}{row['marsad']:>10.3f}{gain:>+10.3f}")


def _print_ablation_table(result: dict) -> None:
    """Same architecture, different band sets."""
    print("\n3. HYPERSPECTRAL ABLATION - same Stage 1 + Stage 2, degraded band sets")
    print("   Each row retrains the identical architecture on spectra resampled to")
    print("   that sensor and lifted back to 205 inputs, so only the spectral")
    print("   information differs, never the model.")
    header = (f"   {'Sensor':<32}{'bands':>7}{'accuracy':>10}{'cyano recall':>14}")
    print("\n" + header)
    print("   " + _rule(len(header) - 3))
    for key, row in result["ablation"].items():
        print(f"   {row['label']:<32}{row['n_bands']:>7}{row['accuracy']:>10.3f}"
              f"{row['cyano_recall']:>14.3f}")
    print("\n   Capability notes (read every accuracy together with its physics):")
    for row in result["ablation"].values():
        head = f"   - {row['label']}: "
        for i, line in enumerate(textwrap.wrap(row["note"], _WRAP - len(head))):
            print((" " * len(head) if i else head) + line)


def _print_headline(result: dict) -> None:
    """The five numbers the pitch actually quotes."""
    h = result["headline"]
    print("\nHEADLINE")
    print(_rule(_WRAP))
    spec_gain = h["marsad_shallow_turbid_acc"] - h["baseline_shallow_turbid_acc"]
    print(f"  Speciation in SHALLOW TURBID water (the Gulf-coast failure regime):")
    print(f"    classical operator decision tree : {h['baseline_shallow_turbid_acc']:.3f}")
    print(f"    MARSAD Stage 1 + Stage 2         : {h['marsad_shallow_turbid_acc']:.3f}"
          f"   ({spec_gain:+.3f})")
    print("\n  OC4 chlorophyll error, deep water vs shallow turbid water"
          " (median |log10|):")
    print(f"    optically deep                   : {h['oc4_deep_err']:.3f}"
          f"   (a factor of {10 ** h['oc4_deep_err']:.1f} out)")
    print(f"    shallow + turbid                 : {h['oc4_shallow_turbid_err']:.3f}"
          f"   (a factor of {10 ** h['oc4_shallow_turbid_err']:.1f} out)")
    marsad_st = result["chl_retrieval"]["shallow_turbid"]["marsad"]
    print(f"    MARSAD, shallow + turbid         : {marsad_st:.3f}"
          f"   (a factor of {10 ** marsad_st:.1f} out)")
    print("\n  Hyperspectral gain (813 accuracy - best multispectral accuracy):")
    print(f"    {h['hyperspectral_gain']:+.3f}")
    best_key = max(
        (k for k in result["ablation"] if k != "marsad_813"),
        key=lambda k: result["ablation"][k]["accuracy"],
    )
    print(f"    best multispectral alternative   : {result['ablation'][best_key]['label']}")

    # The single gain number hides the shape of the result, so split the field
    # by the physical property that actually decides speciation.
    native = result["ablation"]["marsad_813"]
    blind = [(k, row) for k, row in result["ablation"].items()
             if k != "marsad_813" and not _has_620_band(k)]
    if blind:
        best_blind_key, best_blind = max(blind, key=lambda kv: kv[1]["accuracy"])
        print(f"\n  Against the sensors with NO 620 nm band (no phycocyanin route),"
              f"\n  best of which is {best_blind['label']}:")
        print(f"    accuracy      {best_blind['accuracy']:.3f} -> {native['accuracy']:.3f}"
              f"   ({native['accuracy'] - best_blind['accuracy']:+.3f})")
        print(f"    cyano recall  {best_blind['cyano_recall']:.3f} -> "
              f"{native['cyano_recall']:.3f}"
              f"   ({native['cyano_recall'] - best_blind['cyano_recall']:+.3f})")
    print("\n  Read the gain with the per-sensor table, never on its own: Sentinel-3")
    print("  OLCI is the one operational sensor that does carry a 620 nm band, and in")
    print("  our forward model it recovers most of the speciation signal, so the")
    print("  margin over it is small. The large, robust collapse is on the rest.")


def _print_honesty(result: dict) -> None:
    print("\nWHAT THIS IS AND IS NOT")
    print(_rule(_WRAP))
    for line in textwrap.wrap(result["honesty_note"], _WRAP):
        print("  " + line)


def _write_outputs(result: dict, base: Path) -> tuple[Path, Path]:
    """Write ``outputs/benchmark.json`` and ``dashboard/benchmark.js``."""
    payload_obj = dict(result)
    payload_obj["generated_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    payload = json.dumps(_round(payload_obj), default=float)

    outputs_dir = base / "outputs"
    dashboard_dir = base / "dashboard"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    json_path = outputs_dir / "benchmark.json"
    js_path = dashboard_dir / "benchmark.js"
    json_path.write_text(payload, encoding="utf-8")
    js_path.write_text("window.MARSAD_BENCHMARK = " + payload + ";\n", encoding="utf-8")
    return json_path, js_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=11,
                        help="master rng seed (default: 11)")
    parser.add_argument("--fast", action="store_true",
                        help="small sample counts for a quick smoke run; writes "
                             "under outputs/fast-preview/ instead of the real outputs")
    args = parser.parse_args(argv)

    if args.fast:
        n_train, n_test = 500, 400
        base = _REPO_ROOT / "outputs" / "fast-preview"
    else:
        n_train, n_test = 4000, 2000
        base = _REPO_ROOT

    print("MARSAD 813 - benchmark: do standard algorithms actually fail here?")
    print(_rule(_WRAP))
    print(f"seed={args.seed}  n_train={n_train}  n_test={n_test}"
          f"{'  [FAST PREVIEW]' if args.fast else ''}")
    print("Training Stage 1 + Stage 2 once at full size and once per sensor ...")

    result = run_benchmark(seed=args.seed, n_train=n_train, n_test=n_test)

    _print_chl_table(result)
    _print_speciation_table(result)
    _print_ablation_table(result)
    _print_headline(result)
    _print_honesty(result)

    json_path, js_path = _write_outputs(result, base)
    print("\nOutputs written:")
    print(f"  {json_path.relative_to(_REPO_ROOT)}")
    print(f"  {js_path.relative_to(_REPO_ROOT)}")
    if args.fast:
        print("\nFast preview: the judge-facing dashboard/benchmark.js was NOT touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
