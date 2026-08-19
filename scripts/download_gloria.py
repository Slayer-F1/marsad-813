#!/usr/bin/env python
"""Pointers for swapping MARSAD's synthetic training data for real data.

This is a documented STUB: it performs NO network access by default. It prints
step-by-step instructions for the two real datasets the pipeline is designed to
ingest, and creates the destination directory so the rest of the codebase has a
stable path to look for real spectra.

Datasets
--------
1. GLORIA — globally representative in-situ hyperspectral Rrs + water-quality
   dataset (7,572 stations, 350–900 nm at 1 nm; chl-a, TSS, aCDOM labels).
   Lehmann et al. (2023), Scientific Data.
   DOI:  doi:10.1594/PANGAEA.948492
   URL:  https://doi.pangaea.de/10.1594/PANGAEA.948492
   Use:  resample its 1-nm Rrs onto ``spectra.BAND_GRID`` (400–1700 nm; bands
         beyond 900 nm padded with the synthetic SWIR model) and replace/augment
         ``synth.generate_dataset`` training pairs with real (Rrs, chl) rows.

2. NASA PACE OCI Level-2 ocean colour — real hyperspectral satellite granules
   (~5 nm, 340–890 nm) over the Gulf / Sea of Oman, the operational analogue
   of the Arab 813 instrument.
   Portal:   https://oceancolor.gsfc.nasa.gov/data/pace/
   Search:   https://search.earthdata.nasa.gov/search?q=PACE_OCI_L2
   Access:   requires a (free) NASA Earthdata login; the ``earthaccess`` Python
             package automates authentication + granule download.
   Use:  run Stage 1 + Stage 2 inference on real granules for the demo scene
         instead of ``synth.generate_dataset``.

Usage
-----
    ".venv/Scripts/python" scripts/download_gloria.py [--dest data/gloria]
"""
from __future__ import annotations

import argparse
from pathlib import Path

GLORIA_DOI = "doi:10.1594/PANGAEA.948492"
GLORIA_URL = "https://doi.pangaea.de/10.1594/PANGAEA.948492"
PACE_PORTAL = "https://oceancolor.gsfc.nasa.gov/data/pace/"
PACE_SEARCH = "https://search.earthdata.nasa.gov/search?q=PACE_OCI_L2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print real-data download instructions (no network).")
    parser.add_argument("--dest", type=Path, default=Path("data") / "gloria",
                        help="destination directory for the datasets (default: data/gloria)")
    args = parser.parse_args(argv)

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    print("MARSAD 813 - real-data acquisition (stub; nothing is downloaded)")
    print("=" * 66)
    print(f"Destination directory (created): {dest.resolve()}\n")

    print("[1] GLORIA in-situ hyperspectral dataset")
    print(f"    DOI: {GLORIA_DOI}")
    print(f"    URL: {GLORIA_URL}")
    print("    Steps:")
    print("      1. Open the PANGAEA landing page above.")
    print("      2. Download 'GLORIA-2022.zip' (Rrs spectra + lab measurements).")
    print(f"      3. Unzip into: {dest / 'GLORIA-2022'}")
    print("      4. Resample the 1-nm Rrs columns onto marsad.spectra.BAND_GRID.\n")

    print("[2] NASA PACE OCI Level-2 granules (Gulf / Sea of Oman)")
    print(f"    Portal: {PACE_PORTAL}")
    print(f"    Search: {PACE_SEARCH}")
    print("    Steps:")
    print("      1. Create a free NASA Earthdata login.")
    print("      2. pip install earthaccess (NOT done by this repo's venv).")
    print("      3. Search short_name='PACE_OCI_L2_AOP' with a bounding box of")
    print("         roughly lon 55-57 E, lat 24-26 N (Sharjah coast + Hatta).")
    print(f"      4. Save granules under: {dest / 'pace_oci_l2'}\n")

    # TODO(real-data): implement the actual downloads behind an explicit
    # --download flag, e.g. urllib.request for the PANGAEA zip and
    # earthaccess.download() for PACE granules. Deliberately NOT implemented:
    # the hackathon demo must run fully offline and reproducibly.
    print("No network calls were made. See TODO notes in this script for where")
    print("real download logic would live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
