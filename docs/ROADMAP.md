# MARSAD 813 - Roadmap: PRD weeks to repo milestones

This maps the week-by-week plan in the PRD ("MARSAD 813 - Concept & Execution
Plan", parent directory) onto concrete deliverables in this repository. Dates are
the PRD's; the PoC deadline is **11 Oct 23:59**.

Status legend: **DONE** = in the repo and covered by tests; **NEXT** = the work
actually blocking the next milestone; everything else is planned.

---

## DONE - v0.1 scaffold (Now to 16 Aug window)

The three-stage pipeline, running entirely on physics-based synthetic Gulf spectra:

- **Band grid** - `src/marsad/spectra.py`: 205 bands, 400-1700 nm, diagnostic
  wavelengths (443, 555, 620, 665, 675, 708, 1600 nm). Single source of truth,
  designed to be swapped for the real 813 band centres.
- **Synthetic Gulf scene generator** - `src/marsad/synth.py`: chl-a absorption,
  red edge, phycocyanin, sediment backscatter, depth-attenuated bottom
  reflectance, sunglint, sensor noise; three-class labels (no_bloom /
  dinoflagellate / cyanobacteria).
- **Pipeline** - Stage 1 learned shallow-water correction (`stage1_correction.py`),
  Stage 2 bloom classifier + chl regression (`stage2_classifier.py`), Stage 3
  damped-trend forecast with uncertainty bands (`stage3_forecast.py`), per-intake
  risk policy (`risk.py`), end-to-end orchestration over 4 UAE intakes and
  reservoirs (`pipeline.py`) writing `outputs/results.json` + `dashboard/data.js`.
- **Dashboard** - `dashboard/index.html`: self-contained control-room view
  (risk cards, history + forecast bands, before/after spectra, metrics).
- **Docs** - `docs/CONTRACTS.md` (frozen APIs), `README.md`,
  `docs/ACTION-CHECKLIST.md` (human deadlines).

## DONE - v0.2 evidence layer (16-19 Aug)

v0.1 could not answer "compared to what?". v0.2 exists to answer it, and to make
every claim reproducible from one command. All figures below are measured on
`marsad.synth`, our own forward model, so they are a self-consistency check
against a simulation and never independent validation on real Gulf water. That
rule is binding repo-wide and is written into `docs/CONTRACTS-V2.md`.

- **Baseline comparison** - `src/marsad/baselines.py`: OC4, OC3M, NDCI and
  NDCI-chlorophyll, the Gitelson red/NIR ratio, 620 nm phycocyanin line height,
  a Nechad-style turbidity proxy, and the classical operator decision tree, each
  cited to its source paper and run as an operator would run it today.
- **The benchmark** - `src/marsad/benchmark.py` + `scripts/run_benchmark.py`:
  water-regime split (optically deep / shallow clear / turbid deep / shallow
  turbid), chlorophyll retrieval error and speciation accuracy per regime.
  Measured at seed 11, n_train=4000, n_test=2000, 135.5 s: in shallow turbid
  water the operator decision tree scores 0.474 speciation accuracy against
  MARSAD's 0.993 (+0.520), and median absolute log10 chlorophyll error is OC4
  0.706 / OC3M 0.769 / NDCI 0.621 against MARSAD 0.041, a factor of 15. OC4
  degrades from 0.530 in optically deep water to 0.706 shallow turbid and 0.982
  shallow clear.
- **Hyperspectral (spectral) ablation** - `src/marsad/sensors.py` (published band
  tables and Gaussian SRF resampling for Sentinel-2 MSI, Sentinel-3 OLCI, MODIS
  Aqua, Landsat-8 OLI) driving the ablation arm of the benchmark. Same
  architecture, degraded band sets: 813 (205 bands) 0.960 accuracy / 0.929
  cyanobacteria recall, OLCI 0.955 / 0.912, MODIS 0.886 / 0.804, Landsat-8
  0.815 / 0.657, Sentinel-2 0.803 / 0.619. `hyperspectral_gain` over the best
  multispectral sensor is only +0.005 because OLCI carries a 620 nm band; against
  the best sensor with **no** 620 nm band (MODIS) the gain is +0.073 accuracy and
  +0.124 cyanobacteria recall. Both numbers ship together, always.
- **Spatial ablation and the 2x2 verdict** - `benchmark.run_spatial_ablation` and
  `benchmark.build_verdict`, on top of `sensors.subpixel_fill_fraction` and
  `sensors.mix_subpixel`. **This is what closed the OLCI question**, and it was
  the largest gap in the v0.2 evidence layer: the spectral ablation silently
  assumed every sensor RESOLVES the bloom patch, so a +0.005 margin over free
  daily OLCI was the only thing a judge could see, and the obvious follow-up
  question had no answer in the repo. It does now, and the fix cuts in both
  directions rather than only ours. Each bloom pixel is diluted into one sensor
  pixel by linear sub-pixel mixing with the clear-water background of its own
  scene, with the instrument's 1-2 % noise re-applied to the mixed pixel, before
  the band-set degradation; the architecture is retrained per (sensor, patch
  size) cell. Measured at the 100 m intake scale: 813 0.943 accuracy / 0.899
  cyanobacteria recall, Sentinel-2 0.804 / 0.630, Landsat-8 0.797 / 0.603, OLCI
  0.748 / 0.310, MODIS 0.687 / 0.247. At the 3000 m control, where every sensor
  resolves the patch, OLCI is 0.930 / 0.867 and MODIS 0.855 / 0.712, so shrinking
  the patch to intake scale costs OLCI -0.182 accuracy and -0.557 cyanobacteria
  recall while the 30 m and 20 m sensors do not move at all. The 813 margin over
  OLCI therefore goes from +0.005 on band sets alone to **+0.195 accuracy and
  +0.589 cyanobacteria recall**, and OLCI drops below Sentinel-2, a sensor with
  no 620 nm band at all: its pigment advantage is erased by its pixel size. The
  2x2 verdict states the conclusion in the form a judge can check line by line -
  MODIS fails both axes (fill 0.010, no band nearer than 25 nm to 620 nm), OLCI
  passes spectrally (0 nm offset) and fails spatially (fill 0.111), Sentinel-2
  and Landsat-8 pass spatially (fill 1.000) and fail spectrally (45 nm and 35 nm
  offsets), and MARSAD 813 is the only sensor adequate on both. Both axes are read
  off published band tables and published ground sampling distances; the
  accuracies are simulation, and the single assumption in the table is the 813
  ground sampling distance (see the open item below).
- **Per-prediction uncertainty** - `src/marsad/uncertainty.py`: bagged
  `EnsembleClassifier`, predictive entropy split into epistemic and aleatoric and
  normalised to [0, 1], expected calibration error, reliability curve, and a
  `review_queue` that routes ambiguous pixels to a human analyst. Surfaced per
  intake in `outputs/results.json` and `dashboard/data.js`, and never used to
  downgrade a risk level.
- **Hindcast harness** - `src/marsad/hindcast.py`: citable `BloomEvent` records
  (2008-2009 *Cochlodinium polykrikoides*, Sea of Oman, Richlen et al. 2010; plus
  one inland cyanobacteria case marked ILLUSTRATIVE), simulated daily timelines,
  and lead-time scoring. At seed 5 the harness gives 19 days of warning on the
  simulated 2008-2009 timeline with 0 false-alarm days, and 16 days on the
  illustrative reservoir case. The point of the module is its shape: swap in
  archived scenes and rerun unchanged.
- **Alert API** - `src/marsad/alerts.py` + `scripts/serve_api.py`: one alert per
  monitored asset with level, score, bloom class, chlorophyll, lead days to RED,
  operator one-liner and rationale, served over stdlib `http.server` on
  `/health`, `/v1/alerts`, `/v1/intakes`, `/v1/intakes/<name>`, `/v1/metrics`.
  Every payload carries `data_basis: synthetic physics-based simulation`.
- **Pipeline caching** - `pipeline.run_end_to_end(cache_dir=..., refit=...)` plus
  `scripts/run_demo.py --no-cache`, so repeated demo runs are fast without ever
  changing results for a given seed.
- **CI** - `.github/workflows/ci.yml`: `pytest -q` (180 tests, about 2 minutes) on push and pull
  request, and a guard that fails the build on any em-dash in a tracked file or
  a commit message.

Human-only admin for the August window (briefing 11 Aug, training application
16 Aug, recommendation letter, recruiting) lives in `docs/ACTION-CHECKLIST.md`.

### OPEN ITEM carried by the spatial ablation - the assumed 813 GSD (due 11 Oct)

The spatial half of the sensor argument rests on one number we do not have:
`sensors.ASSUMED_813_GSD_M = 30.0`. It is an **assumption**, not a published
specification. The real 813 ground sampling distance was not in the public domain
when this was written, so the instrument was placed in the class it most resembles
(EnMAP 30 m, PRISMA 30 m), the constant was made greppable, and the assumption is
restated in the 813 sensor note that every ablation table prints, in the verdict
note, and in the README.

What has to happen before the 11 Oct submission, in this order:

1. **Get the published figure** from GIQ (the same request that covers the real
   band centres for `spectra.py`; ask for both in one message rather than twice).
2. **Replace the constant in that one place** and rerun `run_benchmark.py`. Nothing
   else in the code changes: `subpixel_fill_fraction`, the grid, the verdict and
   the printed conclusion all read the constant, and the verdict prose is generated
   from the booleans it computes rather than asserted, so the tables and the
   sentences move together.
3. **Re-read the verdict before requoting it.** The 2x2 as it stands puts 813 in
   the "adequate on both axes" cell because a 100 m patch fills a 30 m pixel
   completely. The fill fraction is quadratic, so the conclusion holds for any
   published GSD up to about 141 m (fill 1.000 at 100 m, 0.50 at about 141 m) and
   flips beyond it, where 813 would land in the same cell as OLCI. If the published
   figure is coarser than that, the honest move is to say so and rebuild the pitch
   around the spectral axis plus the tasking argument, not to retune
   `REFERENCE_PATCH_SIZE_M` until the answer comes back the way we like it.
4. **Update the three places that quote 30 m in prose**: the 813 sensor note in
   `sensors.py`, README section 3b, and this item.

Until step 1 lands, every table that reports a spatial result must keep labelling
the 813 pixel an assumption. `tests/test_sensors.py` asserts that the note carries
that label, so removing it fails CI rather than slipping through review.

---

## NEXT - 17-30 Aug: real spectra enter the repo (GLORIA ingest)

PRD: GLORIA, PACE, Sentinel archive over the UAE coast; historical bloom event
list; AOI definition (Khor Fakkan / Kalba east coast + one Gulf-coast intake +
one inland dam, matching `pipeline.INTAKES`).

The single blocking dependency for everything after this point is that no real
spectrum has entered the repo yet. Order of work:

1. **Promote `scripts/download_gloria.py`** from documented stub to a working
   downloader (GLORIA, PANGAEA doi:10.1594/PANGAEA.948492; NASA PACE OCI L2
   granules). Cache under `data/raw/`, which `.gitignore` already excludes.
2. **New `src/marsad/gloria.py` ingest module** behind a contract in the same
   style as the others: load GLORIA Rrs plus chlorophyll / TSS / phycocyanin
   labels, resample onto `spectra.BAND_GRID` with `sensors.resample_to_grid`
   (GLORIA covers roughly 400-900 nm; the SWIR tail is padded and that padding
   must be flagged in the returned metadata, never silently zeroed).
3. **Retrain Stage 2 GLORIA-first**, synthetic-augmented, and report the
   synthetic-versus-real accuracy gap in `dashboard/data.js` as its own metric.
   A drop is expected and publishing it is the point.
4. **Rerun the benchmark on GLORIA spectra.** `benchmark.run_benchmark` already
   takes its data from one generator call; the honest version of the headline is
   the one where OC4 and the operator decision tree are scored against in-situ
   chlorophyll, not against our own simulator. This is the first result in the
   project that is not self-referential.
5. **Historical bloom event file** under `data/` (MOCCAE records, published
   2008-2009 Gulf of Oman studies), feeding `hindcast.DOCUMENTED_EVENTS` from
   data rather than from a literal in the module.

Exit criterion: `run_benchmark.py --source gloria` produces a table whose
baselines are scored against measured chlorophyll.

## 31 Aug to 13 Sep - Stage 1 on real scenes + 813 simulation fidelity

PRD: Stage 1 correction model and baseline retrievals (chlorophyll, turbidity) on
EnMAP / PRISMA Gulf scenes; first 813 scenes via GIQ.

- **Scene reader** for EnMAP L2A and PRISMA over the AOI, resampled onto the
  205-band grid with `sensors.resample` / `resample_to_grid`. This is the
  band-subset simulation of 813 the PRD calls for, and it is why `sensors.py`
  was written before any scene existed.
- **Fit and evaluate `ShallowWaterCorrector` on real coastal pixels.** The
  synthetic generator stays as the regression baseline: if Stage 1 stops
  improving RMSE on synthetic data after retraining on real scenes, that is a
  bug and CI should catch it.
- **Baseline retrievals on real corrected scenes** (chlorophyll, turbidity)
  compared with published Gulf values as an external sanity check.
- **If GIQ delivers first 813 scenes**: replace the uniform grid in `spectra.py`
  with the real band centres and rerun the whole suite. Every module reads the
  grid from that one module, so the change is one file plus a test refresh. Ask
  for the **ground sampling distance in the same message**: it settles
  `sensors.ASSUMED_813_GSD_M` and therefore the spatial half of the sensor
  argument (see the OPEN ITEM under v0.2). Getting the two numbers on separate
  round trips is the easiest way to have one of them still missing in October.

Risk to watch: shallow-water truth is scarce. If real-scene Stage 1 underperforms,
the documented fallback is to restrict alerts to optically deep pixels near the
intakes and to say so on the dashboard rather than to quietly widen the error bars.

## 14-27 Sep - Stage 2 transfer learning + the real hindcast (registration by 25 Sep)

PRD: Stage 2 with GLORIA transfer learning; hindcast test on a documented bloom
event; **complete registration before 25 Sep** (human action, see
`docs/ACTION-CHECKLIST.md`).

- **Stage 2 trained GLORIA-first, fine-tuned on Gulf scenes**, with the
  uncertainty ensemble refitted on the same data so calibration is measured on
  real spectra (`uncertainty.expected_calibration_error`). Epistemic uncertainty
  is the metric that should visibly drop as real training water arrives.
- **The real-scene hindcast of the 2008-2009 event.** This is the headline
  validation artifact of the whole project. Concretely: pull the archived scenes
  covering the Sea of Oman and the UAE east coast for the 45 days before the
  reported November 2008 impact, push them through the unchanged pipeline, and
  score the alert sequence with `hindcast.evaluate_lead_time`.
  - Sensor reality check: 2008 predates Sentinel-3 OLCI (2016) and EnMAP (2022),
    so the historical window is MODIS Aqua and MERIS. MERIS carried a 620 nm
    band and was operating until April 2012, which makes it the only archive
    with the phycocyanin channel over that event. Budget time for MERIS L2 and
    accept that the 2008 hindcast will be multispectral: it validates detection
    and lead time, not the hyperspectral speciation advantage.
  - Present-day proof of the hyperspectral path needs a *recent* Gulf bloom with
    EnMAP or PACE coverage. Identify a candidate date from MOCCAE records during
    this window; it is the second validation artifact, not a substitute for the
    first.
  - Replace `simulate_event_timeseries` with the archive reader and delete the
    simulated path from the reported result. Until that happens, no lead-time
    number from this repo may be described as validated.

Exit criterion: `outputs/hindcast.json` contains a lead-time number derived from
archived imagery, with the honesty note rewritten to describe what it now is.

## 28 Sep to 5 Oct - Stage 3 fusion + GIQ deployment

PRD: Stage 3 fusion and forecast; per-intake risk index; GIQ dashboard MVP.

- **Stage 3 on a real daily watch layer**: Sentinel-2 MSI and Sentinel-3 OLCI
  time series over each intake replacing `synth.generate_history`, with the
  `Forecast` contract unchanged so `risk.py` and `alerts.py` need no edit.
- **Per-intake risk indices computed from real inputs end to end.**
- **GIQ deployment**: package `dashboard/` for GIQ hosting and stand the alert
  API up behind it. Deployment work to scope early, because it is the item most
  likely to be discovered late:
  - decide whether GIQ serves static assets only (then the API needs its own
    host) or can run a Python process;
  - `dashboard/data.js` and `dashboard/benchmark.js` are file-protocol friendly
    by design, so a static deployment already works and the API is additive;
  - if the API is exposed beyond localhost it needs a read-only bind, a rate
    limit and a stated retention policy. The current server is explicitly a
    demonstration endpoint and must not be pointed at the public internet as is.
- **Uncertainty shown per prediction** is already in the payload; the remaining
  work is the dashboard treatment of the review queue.

## 6-11 Oct - Validation write-up, demo video, submission

PRD: validation write-up, demo video, submission package; **submit the PoC by
11 Oct 23:59**.

- **Validation write-up** under `docs/`: the real-scene hindcast result, Stage 1
  RMSE gains on real scenes, the GLORIA-versus-synthetic accuracy gap, and both
  ablations. Written so that a reader can tell at a glance which numbers came from
  measurements and which from the simulator.
- **Close the 813 GSD open item, or state it as open on the slide.** By this point
  `sensors.ASSUMED_813_GSD_M` is either the published figure or still an
  assumption. Either is presentable; an unlabelled assumption is not. The two-axis
  verdict is the strongest sensor argument in the submission and it is also the
  one resting on that constant, so it is the first thing a sharp judge will probe.
- **Demo video** (target 3 minutes, storyboard drafted by 6 Oct so recording is
  not the thing that slips): the problem in 20 seconds with the dated 2008-2009
  shutdown; the dashboard with an intake going AMBER then RED; the benchmark
  table showing what the operator's current algorithm would have said about the
  same water; the lead-time chart; one closing slide on the honesty boundary.
  Record with the pipeline output frozen, never live.
- **Frozen submission tag**: pinned `outputs/`, a reproducible `run_demo.py` and
  `run_benchmark.py` at a stated seed, green CI, and the package assembled per
  the platform's submission requirements.

Hard blockers to keep visible for the whole of October: registration completed
before 25 Sep, and at least one real-scene result in the repo. A polished demo of
a simulation is a weaker submission than a rough demo of a measurement.
