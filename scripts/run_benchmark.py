#!/usr/bin/env python
"""Run the MARSAD 813 benchmark and print the comparison tables.

Usage (from the repo root):

    ".venv/Scripts/python" scripts/run_benchmark.py [--seed N] [--fast]

This is the experiment behind the project's central claim: standard satellite
chlorophyll algorithms lose their footing in shallow turbid Gulf-coastal water
while the MARSAD two-stage approach does not, and a sensor needs BOTH a 620 nm
phycocyanin band and a pixel small enough to hold an intake-scale bloom patch
before it can warn a desalination intake. It prints five tables - chlorophyll
retrieval error by water regime, speciation accuracy by water regime, the
hyperspectral (spectral) sensor ablation, the spatial ablation of accuracy
against bloom patch size, and the 2x2 verdict that puts the two axes together
- then the headline numbers.

The spatial table is what makes the sensor comparison honest in both
directions. Sentinel-3 OLCI really does carry a 620 nm band, so on band sets
alone the margin over it is small and a judge is right to ask why not just use
free daily OLCI; at 300 m ground sampling a 100 m patch fills about a ninth of
one of its pixels, which is the part a spectral-only ablation never measured.

A full run trains Stage 1 + Stage 2 once at full size, once per sensor, and
once per distinct (sensor, fill fraction) cell of the spatial grid, and takes a
few minutes. ``--fast`` cuts BOTH the patch list and every sample count for a
quick smoke run and, like ``scripts/run_demo.py``, writes into
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Allow running the script directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from marsad.benchmark import (  # noqa: E402
    SPATIAL_PATCH_SIZES_M,
    SPATIAL_TEST_CAP,
    SPATIAL_TRAIN_CAP,
    WATER_REGIMES,
    has_phycocyanin_band,
    run_benchmark,
)

#: Patch sizes for ``--fast``. The reference size of the 2x2 verdict has to
#: stay in the list or the verdict loses its measured columns; 1000 m is the
#: control, the size at which every sensor in the table fills its own pixel
#: and the spatial term is switched off. Two columns is the smallest grid
#: that still shows a slope.
_FAST_PATCH_SIZES_M = (100.0, 1000.0)

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


def _fmt_patch(size_m: float) -> str:
    """Patch size as a short column heading, e.g. ``"100 m"``."""
    return f"{size_m:.0f} m"


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


def _print_spatial_grid(result: dict, metric: str, title: str, note: str) -> None:
    """One sensor-by-patch-size table of the spatial ablation."""
    spatial = result["spatial"]
    sizes = spatial["patch_sizes_m"]
    header = (f"   {'Sensor':<32}{'GSD':>8}"
              + "".join(f"{_fmt_patch(s):>9}" for s in sizes))
    print(f"\n   {title}")
    print("   " + note)
    print("\n" + header)
    print("   " + _rule(len(header) - 3))
    for row in spatial["sensors"].values():
        cells = "".join(
            f"{cell[metric]:>9.3f}" for cell in row["by_patch_size"]
        )
        print(f"   {row['label']:<32}{row['gsd_m']:>6.0f} m{cells}")


def _print_spatial_table(result: dict) -> None:
    """The spatial ablation: accuracy against bloom patch size, per sensor."""
    spatial = result["spatial"]
    print("\n4. SPATIAL ABLATION - same Stage 1 + Stage 2, bloom patch diluted"
          " into the pixel")
    print("   Each bloom pixel is mixed with the clear-water background of its own")
    print("   scene at the fill fraction a patch of that size gets in one pixel of")
    print("   that sensor, THEN degraded to that sensor's band set. Columns are the")
    print("   bloom patch side length in metres; a coarse sensor loses the patch in")
    print(f"   its own pixel. n_train={spatial['n_train']} per cell,"
          f" n_test={spatial['n_test']}.")

    _print_spatial_grid(
        result, "fill_fraction",
        "4a. FILL FRACTION - share of one pixel the patch covers (geometry only)",
        "min(1, (patch / GSD)^2): halving the patch quarters the fill.",
    )
    _print_spatial_grid(
        result, "accuracy",
        "4b. SPECIATION ACCURACY - the headline metric of this table",
        "over {no_bloom, dinoflagellate, cyanobacteria}; higher is better.",
    )
    _print_spatial_grid(
        result, "cyano_recall",
        "4c. CYANOBACTERIA RECALL - the phycocyanin-dependent class",
        "of the pixels that ARE cyanobacteria, the share that were found.",
    )
    _print_spatial_grid(
        result, "bloom_recall",
        "4d. BLOOM DETECTION RECALL - any bloom found, species ignored",
        "read it with 4b: see the caveat below before quoting this row.",
    )
    print("\n   Caveat on 4d, stated because it shapes how the row reads: the")
    print("   background is the mean of the scene's bloom-free pixels, so a heavily")
    print("   diluted bloom pixel lands closer to that mean than a real bloom-free")
    print("   pixel does, and 'unnaturally average' becomes a detectable cue in")
    print("   itself. Bloom recall can therefore stay high in cells where speciation")
    print("   has already collapsed. Accuracy (4b) and cyanobacteria recall (4c) are")
    print("   the metrics to quote at low fill fractions, not 4d.")


def _print_verdict_table(result: dict) -> None:
    """The 2x2: is this sensor adequate spectrally, spatially, both or neither?"""
    verdict = result["verdict"]
    ref = verdict["reference_patch_size_m"]
    thr = verdict["fill_fraction_threshold"]
    print(f"\n5. THE 2x2 VERDICT - can this sensor warn an intake about a"
          f" {ref:.0f} m patch?")
    print(f"   SPECTRAL axis: a band within {verdict['pc_band_tolerance_nm']:.0f} nm"
          f" of {verdict['pc_band_nm']:.0f} nm, the phycocyanin")
    print("                  absorption that is the only pigment route to"
          " cyanobacteria.")
    print(f"   SPATIAL axis:  a {ref:.0f} m patch fills at least {thr:.2f} of one"
          " pixel, so the patch")
    print("                  dominates its own measurement.")
    print("   Both axes come from published band tables and ground sampling"
          " distances,")
    print("   not from our simulation. The one assumption is the 813 pixel size.")
    header = (f"   {'Sensor':<32}{'620 nm':>8}{'GSD':>8}"
              f"{'fill':>8}{'SPECTRAL':>10}{'SPATIAL':>9}")
    print("\n" + header)
    print("   " + _rule(len(header) - 3))
    for row in verdict["sensors"].values():
        print(f"   {row['label']:<32}"
              f"{'yes' if row['has_620nm'] else 'no':>8}"
              f"{row['gsd_m']:>6.0f} m"
              f"{row['fill_fraction_at_reference']:>8.3f}"
              f"{'PASS' if row['spectral_ok'] else 'FAIL':>10}"
              f"{'PASS' if row['spatial_ok'] else 'FAIL':>9}")
    print("\n   Why each sensor lands where it does:")
    for row in verdict["sensors"].values():
        head = f"   - {row['label']}: "
        for i, line in enumerate(textwrap.wrap(row["reason"], _WRAP - len(head))):
            print((" " * len(head) if i else head) + line)


def _print_two_axis_conclusion(result: dict) -> None:
    """The corrected conclusion, generated from the measured table."""
    verdict = result["verdict"]
    spatial = result["spatial"]
    sizes = spatial["patch_sizes_m"]
    ref = verdict["reference_patch_size_m"]
    control = max(sizes)

    print("\n  THE OLCI QUESTION - why not just use free daily Sentinel-3 OLCI?")
    for line in textwrap.wrap(verdict["summary"], _WRAP - 4):
        print("    " + line)

    print(f"\n    Measured, same architecture throughout: a bloom patch shrunk from"
          f"\n    {control:.0f} m (every sensor resolves it) to the {ref:.0f} m"
          " intake scale.")
    header = (f"    {'Sensor':<32}{'acc ' + _fmt_patch(control):>12}"
              f"{'acc ' + _fmt_patch(ref):>12}{'drop':>8}"
              f"{'cyano ' + _fmt_patch(ref):>14}")
    print("\n" + header)
    print("    " + _rule(len(header) - 4))
    for key, row in spatial["sensors"].items():
        cells = {c["patch_size_m"]: c for c in row["by_patch_size"]}
        if control not in cells or ref not in cells:
            continue
        hi, lo = cells[control], cells[ref]
        print(f"    {row['label']:<32}{hi['accuracy']:>12.3f}{lo['accuracy']:>12.3f}"
              f"{lo['accuracy'] - hi['accuracy']:>+8.3f}"
              f"{lo['cyano_recall']:>14.3f}")

    # Generated from the buckets, never asserted: if a band table or a ground
    # sampling distance changes, this sentence changes with it rather than
    # going quietly out of date.
    def _named(bucket: str) -> list[str]:
        return [verdict["sensors"][k]["label"] for k in verdict[bucket]]

    clauses = []
    for bucket, one, many in (
        ("spatial_only",
         "resolves the patch and cannot see the pigment",
         "resolve the patch and cannot see the pigment"),
        ("spectral_only",
         "sees the pigment and cannot resolve the patch",
         "see the pigment and cannot resolve the patch"),
        ("inadequate_on_both", "can do neither", "can do neither"),
    ):
        names = _named(bucket)
        if names:
            clauses.append(", ".join(names) + " " + (one if len(names) == 1 else many))
    both = _named("adequate_on_both")
    if both and clauses:
        subject = "the sensor" if len(both) == 1 else "the sensors"
        verb = "is" if len(both) == 1 else "are"
        closing = (
            "Neither axis is sufficient on its own, which is the whole point: "
            + "; ".join(clauses)
            + f". On this table {subject} adequate on BOTH axes at the "
            f"{ref:.0f} m intake scale {verb}: " + ", ".join(both) + "."
        )
    elif both:
        closing = (
            f"Every sensor in this table is adequate on both axes at the "
            f"{ref:.0f} m intake scale, so the two-axis argument does not "
            "separate them here. Do not quote it as if it did."
        )
    else:
        closing = (
            f"On this table NO sensor is adequate on both axes at the {ref:.0f} m "
            "intake scale, MARSAD 813 included. That is the measurement; do not "
            "quote a two-axis advantage the numbers do not show."
        )
    print()
    for line in textwrap.wrap(closing, _WRAP - 4):
        print("    " + line)


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
             if k != "marsad_813" and not has_phycocyanin_band(k)]
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
    print("  The gain above is a SPECTRAL number and it is not the whole comparison;")
    print("  the spatial axis below is the half it leaves out.")
    _print_two_axis_conclusion(result)


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

    # --fast has to cut the spatial grid as well as the sample counts: the
    # grid costs one model fit per distinct (sensor, fill fraction) cell, so
    # halving the patch list is worth as much as halving the pixel counts.
    if args.fast:
        n_train, n_test = 500, 400
        patch_sizes = _FAST_PATCH_SIZES_M
        spatial_n_train, spatial_n_test = 250, 300
        base = _REPO_ROOT / "outputs" / "fast-preview"
    else:
        n_train, n_test = 4000, 2000
        patch_sizes = SPATIAL_PATCH_SIZES_M
        spatial_n_train, spatial_n_test = SPATIAL_TRAIN_CAP, SPATIAL_TEST_CAP
        base = _REPO_ROOT

    print("MARSAD 813 - benchmark: do standard algorithms actually fail here?")
    print(_rule(_WRAP))
    print(f"seed={args.seed}  n_train={n_train}  n_test={n_test}"
          f"{'  [FAST PREVIEW]' if args.fast else ''}")
    print("spatial grid: patch sizes "
          + ", ".join(_fmt_patch(p) for p in patch_sizes)
          + f"  (n_train={spatial_n_train}, n_test={spatial_n_test} per cell)")
    print("Training Stage 1 + Stage 2 once at full size, once per sensor, and")
    print("once per distinct (sensor, fill fraction) cell of the spatial grid ...")

    started = time.perf_counter()
    result = run_benchmark(
        seed=args.seed,
        n_train=n_train,
        n_test=n_test,
        spatial_patch_sizes_m=patch_sizes,
        spatial_n_train=spatial_n_train,
        spatial_n_test=spatial_n_test,
    )
    elapsed = time.perf_counter() - started

    _print_chl_table(result)
    _print_speciation_table(result)
    _print_ablation_table(result)
    _print_spatial_table(result)
    _print_verdict_table(result)
    _print_headline(result)
    _print_honesty(result)

    print(f"\nWall clock: {elapsed:.1f} s"
          f"{' (fast preview)' if args.fast else ' (full run)'}")

    json_path, js_path = _write_outputs(result, base)
    print("\nOutputs written:")
    print(f"  {json_path.relative_to(_REPO_ROOT)}")
    print(f"  {js_path.relative_to(_REPO_ROOT)}")
    if args.fast:
        print("\nFast preview: the judge-facing dashboard/benchmark.js was NOT touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
