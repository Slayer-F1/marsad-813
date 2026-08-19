# MARSAD 813 - Roadmap: PRD weeks → repo milestones

This maps the week-by-week plan in the PRD ("MARSAD 813 - Concept & Execution
Plan", parent directory) onto concrete deliverables in this repository. Dates are
the PRD's; the PoC deadline is **11 Oct 23:59**.

## DONE - v0.1 scaffold (this repo, Now → 16 Aug window)

Delivered by the current scaffold, running entirely on physics-based synthetic
Gulf spectra:

- **Band grid** - `src/marsad/spectra.py`: 205 bands, 400–1700 nm, diagnostic
  wavelengths (443, 555, 620, 665, 675, 708, 1600 nm). Single source of truth,
  designed to be swapped for the real 813 band centres.
- **Synthetic Gulf scene generator** - `src/marsad/synth.py`: chl-a absorption,
  red-edge, phycocyanin, sediment backscatter, depth-attenuated bottom
  reflectance, sunglint, sensor noise; three-class labels (no_bloom /
  dinoflagellate / cyanobacteria).
- **Synthetic pipeline v0.1** - Stage 1 learned shallow-water correction
  (`stage1_correction.py`), Stage 2 bloom classifier + chl regression
  (`stage2_classifier.py`), Stage 3 damped-trend forecast with uncertainty bands
  (`stage3_forecast.py`), per-intake risk policy (`risk.py`), end-to-end
  orchestration over 4 UAE intakes/reservoirs (`pipeline.py`) writing
  `outputs/results.json` + `dashboard/data.js`.
- **Dashboard** - `dashboard/index.html`: self-contained control-room view
  (risk cards, history + forecast bands, before/after spectra, metrics).
- **Tests** - `tests/`: per-module seeded fast tests (Stage 1 RMSE improvement,
  Stage 2 accuracy, forecast band ordering, risk monotonicity, pipeline smoke).
- **Docs** - `docs/CONTRACTS.md` (frozen APIs), `README.md`,
  `docs/ACTION-CHECKLIST.md` (human deadlines).

The human-only admin items for this window (briefing 11 Aug, training application
16 Aug, letter, recruiting) live in `docs/ACTION-CHECKLIST.md`.

## 17–30 Aug - Data assembly → GLORIA ingest

PRD: GLORIA, PACE, Sentinel archive over UAE coast; historical bloom event list;
AOI definition (Khor Fakkan/Kalba east coast + one Gulf-coast intake + one inland
dam - matching `pipeline.INTAKES`).

Repo milestones:

- Promote `scripts/download_gloria.py` from documented stub to a working
  downloader (GLORIA, PANGAEA doi:10.1594/PANGAEA.948492; PACE URLs).
- New ingest code under `src/marsad/`: load GLORIA spectra + chl/TSS/phycocyanin
  labels, resample onto `spectra.BAND_GRID` (400–900 nm coverage; SWIR padded),
  store under `data/`.
- Retrain Stage 2 on GLORIA-real spectra with synthetic augmentation; report the
  synthetic-vs-real accuracy gap honestly in the dashboard metrics.
- Compile the historical bloom event list (MOCCAE records, published 2008–09
  Gulf of Oman studies) as a versioned file under `data/`.

## 31 Aug – 13 Sep - Stage 1 on real scenes + 813 simulation fidelity

PRD: Stage 1 correction model + baseline retrievals (chlorophyll, turbidity) on
EnMAP/PRISMA Gulf scenes; first 813 scenes via GIQ.

Repo milestones:

- EnMAP/PRISMA scene reader; **813-band simulation fidelity**: resample real
  scenes onto the 205-band 400–1700 nm grid (the PRD's band-subset simulation of
  813) so the whole pipeline runs unchanged on real imagery.
- Fit/evaluate `ShallowWaterCorrector` on real coastal pixels; keep the synthetic
  generator as the fallback and regression baseline.
- Baseline chlorophyll and turbidity retrievals on corrected scenes for sanity
  checks against published Gulf values.
- If GIQ delivers first 813 scenes: swap real band centres into
  `spectra.py` and rerun the suite.

## 14–27 Sep - Stage 2 transfer learning + hindcast (registration ≤ 25 Sep)

PRD: Stage 2 classifier with GLORIA transfer learning; hindcast test on a
documented bloom event; **complete registration before 25 Sep** (human action -
see `docs/ACTION-CHECKLIST.md`).

Repo milestones:

- Stage 2 trained GLORIA-first, fine-tuned on Gulf scenes.
- **Hindcast validation on the 2008–09 Cochlodinium event**: run the pipeline on
  archive scenes/records around the documented event and show the risk index
  flags it days before reported impact; write the result to `outputs/` as the
  headline validation artifact.

## 28 Sep – 5 Oct - Stage 3 fusion + GIQ deployment

PRD: Stage 3 fusion + forecast; per-intake risk index; GIQ dashboard MVP.

Repo milestones:

- Stage 3 driven by real Sentinel-2/-3 daily watch-layer time series instead of
  `synth.generate_history`; keep the `Forecast` contract unchanged.
- Per-intake risk indices computed from real inputs end-to-end.
- **GIQ deployment**: package `dashboard/` for GIQ hosting; alert feed per
  intake/reservoir; uncertainty shown per prediction.

## 6–11 Oct - Validation write-up, demo video, submission

PRD: validation write-up, demo video, submission package; **submit PoC by
11 Oct 23:59**.

Repo milestones:

- Validation write-up (hindcast results, Stage 1 RMSE gains on real scenes,
  synthetic-vs-real gaps stated openly) under `docs/`.
- **Demo video** walking the dashboard through a bloom scenario.
- Frozen submission tag: pinned outputs, reproducible `run_demo.py`, packaged
  per the platform's submission requirements.
