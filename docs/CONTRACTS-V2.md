# MARSAD 813 - v0.2 Module Contracts (authoritative)

Extends `docs/CONTRACTS.md` (v0.1, still binding for existing modules). Same rules:
numpy / scikit-learn / stdlib ONLY; `src/marsad/spectra.py` owns the band grid
(`N_BANDS = 205`, `BAND_GRID` 400-1700 nm); labels `0 = no_bloom`,
`1 = dinoflagellate`, `2 = cyanobacteria`; all randomness through
`np.random.default_rng(seed)`.

## Scientific honesty rule (BINDING on every module and every doc string here)

The synthetic scene generator (`synth.py`) is **our own forward model**. Any
benchmark run on it is a **self-consistency check against a physics-based
simulation**, NEVER independent validation. Every public result - docstrings,
printed output, dashboard text, README - must say so in plain words. Permitted
phrasing: "on our physics-based Gulf-water simulation, consistent with the
Case-2 water literature". Forbidden: "we proved standard algorithms fail on
Gulf water", or any wording implying real-scene validation. Real validation is
the hindcast on documented events once GLORIA/PACE/813 data lands.

---

## `src/marsad/sensors.py` - multispectral simulation & spectral resampling

Why: lets us (a) ingest any real sensor onto the 813 grid later, and (b) run the
ablation that decides whether hyperspectral is the *enabling* sensor or a
passenger. A sensor is two things here, a band set and a pixel size, and both
are part of the data structure because a band that resolves a pigment line is
useless if the patch carrying that line is a hundredth of a pixel.

```python
@dataclass(frozen=True)
class Band:
    name: str
    center_nm: float
    fwhm_nm: float

ASSUMED_813_GSD_M: float = 30.0
    # ASSUMPTION, NOT A PUBLISHED SPECIFICATION. The real 813 ground sampling
    # distance is not public while this hackathon runs, so the instrument is
    # placed in the class it most resembles (EnMAP 30 m, PRISMA 30 m). It is a
    # module constant so it is greppable, appears in the sensor note every
    # ablation table prints, and can be replaced in exactly ONE place when GIQ
    # publishes the real figure. Never quote it as a specification.

@dataclass(frozen=True)
class Sensor:
    key: str
    label: str            # human-readable, e.g. "Sentinel-2 MSI"
    bands: tuple[Band, ...]
    note: str             # one-line honest capability note (see below)
    gsd_m: float = ASSUMED_813_GSD_M
        # Ground sampling distance in metres for the WATER-relevant bands, i.e.
        # the side of one pixel on the sea surface. Defaulted so that any
        # ad-hoc Sensor built in a test behaves like the native instrument
        # instead of silently acquiring a coarse pixel. Must be positive and
        # finite; __post_init__ rejects anything else.
    @property
    def n_bands(self) -> int
    @property
    def centers_nm(self) -> np.ndarray

SENSORS: dict[str, Sensor]   # keys below, all band centres within 400-1700 nm

def resample(rrs: np.ndarray, sensor: Sensor | str) -> np.ndarray
    # (n, N_BANDS) -> (n, sensor.n_bands); Gaussian SRF per band, normalised,
    # integrated over BAND_GRID. Bands whose SRF has ~no support on the grid
    # must be dropped from the Sensor definition, not silently zeroed.

def resample_to_grid(rrs_sensor: np.ndarray, sensor: Sensor | str) -> np.ndarray
    # (n, sensor.n_bands) -> (n, N_BANDS) by linear interp + edge hold.
    # This is the "what a multispectral sensor can tell you about the full
    # spectrum" upsampling used by the ablation so every model sees N_BANDS
    # inputs and only the INFORMATION differs, not the array width.

def subpixel_fill_fraction(patch_size_m: float, sensor: Sensor | str) -> float
    # min(1.0, (patch_size_m / sensor.gsd_m) ** 2), a float in (0, 1].
    # The quadratic is the point: halving the patch quarters the fill, so
    # coarse pixels lose intake-scale features very fast. Best case by
    # construction (square patch, pixel-aligned, wholly inside one pixel), so
    # a real patch of the same area fills its brightest pixel LESS than this.
    # Raises ValueError on a non-positive or non-finite patch size.

def mix_subpixel(rrs_target, rrs_background, patch_size_m, sensor) -> np.ndarray
    # Linear spectral mixing f * target + (1 - f) * background with f from
    # subpixel_fill_fraction. (n, N_BANDS) in, (n, N_BANDS) out; a single
    # (N_BANDS,) target returns a single spectrum; a single background
    # broadcasts against every target. At f == 1 the target is returned
    # unchanged. Ignores adjacency effects and the point-spread function,
    # both of which would smear a small bright patch further into its
    # background, so this is the CONSERVATIVE choice: any 813 advantage it
    # shows is a lower bound on what a full radiative-transfer treatment
    # would give.
```

Required `SENSORS` keys and real band sets (centre/FWHM nm, water-relevant,
clipped to 400-1700), each with its published water-band ground sampling
distance:

- `"marsad_813"` - the native 205-band grid (identity resample must round-trip
  to within 1e-12), `gsd_m = ASSUMED_813_GSD_M`. note: "205 contiguous bands,
  400-1700 nm" plus the explicit statement that the 30 m pixel is an assumption
  and not a published figure.
- `"sentinel2_msi"` - 443/20, 490/65, 560/35, 665/30, 705/15, 740/15, 783/20,
  842/115, 865/20, 945/20, 1375/30, 1610/90; `gsd_m = 20.0`, because the
  red-edge bands B5-B7 that carry the bloom signal are 20 m and not the 10 m of
  B2-B4/B8. note must state: no band at 620 nm, so phycocyanin is not directly
  observable.
- `"sentinel3_olci"` - 400/15, 412.5/10, 442.5/10, 490/10, 510/10, 560/10,
  620/10, 665/10, 673.75/7.5, 681.25/7.5, 708.75/10, 753.75/7.5, 778.75/15,
  865/20, 885/10, 900/10, 1020/40; `gsd_m = 300.0`. note must be HONEST: OLCI
  *does* carry a 620 nm band, but one band at 300 m ground sampling cannot
  separate phycocyanin absorption from co-varying sediment/chl absorption, and
  at 300 m a 100 m intake-scale patch fills only about a ninth of one pixel -
  this nuance is a credibility point, do not overstate it in either direction.
- `"modis_aqua"` - 412/15, 443/10, 469/20, 488/10, 531/10, 547/10, 555/20,
  645/50, 667/10, 678/10, 748/10, 859/35, 869/15, 1240/20, 1640/50;
  `gsd_m = 1000.0`. note: no 620 nm band; 1 km pixels miss intake-scale patches.
- `"landsat8_oli"` - 443/16, 482/60, 561/57, 655/37, 865/28, 1609/85;
  `gsd_m = 30.0`. note: 6 usable water bands; blue-green ratio only, but the
  30 m pixel is the one thing OLI does bring to this comparison.

## `src/marsad/baselines.py` - standard operational algorithms (the comparison)

Literature algorithms as they would actually be run by an operator today.
Each takes Rrs `(n, N_BANDS)` on the 813 grid and returns `(n,)`.

```python
def oc4_chl(rrs) -> np.ndarray
    # NASA OC4 4th-order polynomial band ratio (SeaWiFS coefficients
    # a = [0.3272, -2.9940, 2.7218, -1.2259, -0.5683]);
    # R = log10(max(Rrs443, Rrs490, Rrs510) / Rrs555). The open-ocean
    # workhorse - and the algorithm Case-2 water is known to break.
def oc3m_chl(rrs) -> np.ndarray
    # MODIS OC3M, a = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280],
    # R = log10(max(Rrs443, Rrs488) / Rrs547).
def ndci(rrs) -> np.ndarray            # (Rrs708 - Rrs665) / (Rrs708 + Rrs665)
def ndci_chl(rrs) -> np.ndarray        # Mishra & Mishra 2012:
    #   chl = 14.039 + 86.115*NDCI + 194.325*NDCI**2
def red_nir_ratio(rrs) -> np.ndarray   # Gitelson 2-band Rrs708/Rrs665
def phycocyanin_line_height(rrs) -> np.ndarray
    # baseline-subtracted 620 nm absorption depth: linear baseline from
    # 600 -> 650 nm evaluated at 620, minus Rrs620. Positive = PC absorption.
    # This is the ONLY classical route to cyanobacteria speciation and it
    # needs a 620 nm band to exist at all.
def turbidity_proxy(rrs) -> np.ndarray # Nechad-style, Rrs665-driven
def classify_baseline(rrs, chl_threshold=10.0, pc_lh_threshold=0.0005) -> np.ndarray
    # (n,) labels in {0,1,2} using the classical operator decision tree:
    # bloom if ndci_chl >= chl_threshold; then cyanobacteria if
    # phycocyanin_line_height >= pc_lh_threshold else dinoflagellate.
    # This is what MARSAD's Stage 2 must beat.
```

All functions clip to finite, non-negative outputs where physically required
(chl >= 0) and must not emit warnings or NaN on zero/negative Rrs - guard the
divisions. Docstrings cite the source algorithm and state the honesty rule.

## `src/marsad/uncertainty.py` - per-prediction uncertainty & calibration

PRD requirement: "Uncertainty shown per prediction (judges reward honesty;
operators require it)."

```python
class EnsembleClassifier:
    """Bagged ensemble of BloomClassifier for epistemic uncertainty."""
    def __init__(self, n_members: int = 5, seed: int = 0): ...
    def fit(self, rrs, labels, chl) -> "EnsembleClassifier"
        # each member fits a bootstrap resample with its own seed
    def predict_proba(self, rrs) -> np.ndarray        # (n,3) ensemble mean
    def predict(self, rrs) -> np.ndarray              # (n,)
    def estimate_chl(self, rrs) -> np.ndarray         # (n,) ensemble mean
    def uncertainty(self, rrs) -> dict[str, np.ndarray]
        # {"total": predictive entropy of the mean (aleatoric+epistemic),
        #  "epistemic": mutual information = total - mean(member entropies),
        #  "aleatoric": total - epistemic,
        #  "confidence": max mean probability}
        # entropies in NATS, normalised to [0,1] by dividing by log(3).
    def save(self, path) / @classmethod load(cls, path)

def expected_calibration_error(probs, labels, n_bins: int = 10) -> float
def reliability_curve(probs, labels, n_bins: int = 10) -> dict
    # {"bin_confidence": [...], "bin_accuracy": [...], "bin_count": [...]}
    # empty bins omitted; used by the dashboard reliability plot.
def review_queue(unc: dict, threshold: float = 0.35) -> np.ndarray
    # boolean (n,): predictions whose "total" uncertainty exceeds threshold and
    # should be routed to a human analyst rather than auto-alerted.
```

## `src/marsad/benchmark.py` - the experiment that proves the claim

```python
WATER_REGIMES: tuple[str, ...] = ("optically_deep", "shallow_clear",
                                  "turbid_deep", "shallow_turbid")

def classify_regime(depth_m, tss) -> np.ndarray   # (n,) of WATER_REGIMES strings
    # shallow if depth_m < 10; turbid if tss >= 8 g/m3. "shallow_turbid" is
    # the Gulf-coast failure regime the whole project targets.

def run_benchmark(seed: int = 11, n_train: int = 4000, n_test: int = 2000, *,
                  spatial_patch_sizes_m: Sequence[float] = SPATIAL_PATCH_SIZES_M,
                  spatial_n_train: int = SPATIAL_TRAIN_CAP,
                  spatial_n_test: int = SPATIAL_TEST_CAP,
                  reference_patch_size_m: float = REFERENCE_PATCH_SIZE_M,
                  fill_fraction_threshold: float = FILL_FRACTION_THRESHOLD) -> dict
```
Returns EXACTLY these five keys, unchanged since v0.2, plus the two additive
blocks `"spatial"` and `"verdict"` specified in the next section. Nothing in the
original five moved or changed meaning:
```python
{
  "chl_retrieval": {            # median absolute log10 error, per regime
     "<regime>": {"oc4": f, "oc3m": f, "ndci": f, "marsad": f, "n": int}, ...},
  "speciation": {               # accuracy per regime
     "<regime>": {"baseline_tree": f, "marsad": f, "n": int}, ...},
  "ablation": {                 # SAME MARSAD architecture, different sensors
     "<sensor_key>": {"label": str, "n_bands": int, "accuracy": f,
                      "cyano_recall": f, "note": str}, ...},
  "headline": {"marsad_shallow_turbid_acc": f, "baseline_shallow_turbid_acc": f,
               "oc4_deep_err": f, "oc4_shallow_turbid_err": f,
               "hyperspectral_gain": f},   # 813 acc - best multispectral acc
  "honesty_note": str,          # the Scientific honesty rule, one sentence
}
```
Method: generate train/test; fit Stage 1 + Stage 2 on train; for each test
pixel compute baseline retrievals on the RAW observed spectra (that is what an
operator has) and MARSAD retrievals on Stage-1-corrected spectra; group by
regime. Ablation: for each sensor in `SENSORS`, `resample` the observed
spectra to that sensor then `resample_to_grid` back, run the SAME Stage
1 + Stage 2 training/eval on those degraded spectra, and report accuracy plus
per-class recall for cyanobacteria (the phycocyanin-dependent class).
`hyperspectral_gain` = marsad_813 accuracy - best non-813 sensor accuracy.

In every ablation arm, spectral and spatial alike, the Stage 1 target stays the
FULL-RESOLUTION clean spectrum. The product Stage 2 consumes does not change
when the input sensor gets coarser, and scoring a degraded arm against a
degraded target would redefine the deliverable per arm and hide the exact
information loss the ablation exists to measure.

### The spatial axis (additive: `"spatial"` and `"verdict"`)

Why this exists, stated plainly because it is a correction to our own earlier
result: the spectral ablation silently assumes every sensor RESOLVES the bloom
patch. That assumption is false at the scale that decides whether an intake
gets a warning. OLCI pixels are 300 m and MODIS ocean-colour pixels are 1 km,
so an intake-scale patch is sub-pixel and its signal reaches the instrument
already diluted with the surrounding clear water. `hyperspectral_gain` is a
SPECTRAL number and it is not the whole comparison; it must never be quoted as
if it were. Adding the spatial term is what makes the sensor comparison honest
in BOTH directions, and it is the reason the answer to "why not just use free
daily OLCI?" is not "because it cannot see phycocyanin", which would be false,
but "because at 300 m it sees an intake-scale patch diluted about nine to one".

```python
SPATIAL_PATCH_SIZES_M: tuple[float, ...] = (50.0, 100.0, 300.0, 1000.0, 3000.0)
    # Bloom patch side lengths. 3000 m is the CONTROL column: every sensor in
    # the table resolves it, so the spatial term is switched off there and only
    # the band set is left. Quote a drop only against that control.
REFERENCE_PATCH_SIZE_M: float = 100.0     # intake scale; operational, retunable
FILL_FRACTION_THRESHOLD: float = 0.5      # patch must outweigh its background
PC_BAND_NM: float = 620.0                 # the phycocyanin absorption
PC_BAND_TOLERANCE_NM: float = 10.0        # about half an ocean-colour band width
SPATIAL_NOISE_FRAC: tuple[float, float] = (0.01, 0.02)   # matches synth.py
SPATIAL_TRAIN_CAP: int = 900              # per GRID CELL, not per sensor
SPATIAL_TEST_CAP: int = 1200              # keeps cyano recall off a few dozen px
SPATIAL_NOTE: str                         # method + honesty rule, one paragraph
VERDICT_NOTE: str                         # which parts are fact, which simulation

def nearest_band_distance_nm(sensor, target_nm: float = PC_BAND_NM) -> float
    # nm from target_nm to the closest band centre. A sensor cannot measure an
    # absorption it has no band on top of: resample_to_grid draws a straight
    # line across the gap, so a 620 nm dip between a 560 and a 665 nm band is
    # not attenuated, it is ABSENT.
def has_phycocyanin_band(sensor) -> bool
    # nearest_band_distance_nm(sensor) <= PC_BAND_TOLERANCE_NM. Of the
    # operational ocean-colour sensors only Sentinel-3 OLCI passes.

def run_spatial_ablation(seed: int = 11, n_train: int = SPATIAL_TRAIN_CAP,
                         n_test: int = SPATIAL_TEST_CAP,
                         patch_sizes_m: Sequence[float] = SPATIAL_PATCH_SIZES_M,
                         sensor_keys: Sequence[str] | None = None,
                         train: synth.SynthDataset | None = None,
                         test: synth.SynthDataset | None = None) -> dict
```
Returns:
```python
{
  "patch_sizes_m": [f, ...],    # in the order given
  "n_train": int, "n_test": int,            # pixels per CELL
  "sensors": {"<sensor_key>": {
      "label": str, "gsd_m": f, "n_bands": int, "has_620nm": bool,
      "by_patch_size": [{"patch_size_m": f, "fill_fraction": f, "accuracy": f,
                         "cyano_recall": f, "bloom_recall": f,
                         "false_alarm_rate": f}, ...]}},   # same order
  "note": SPATIAL_NOTE,
}
```
Method per cell, and the ORDER is the physical one: (1) take the clear-water
background as the mean observed spectrum of that scene's own `no_bloom` pixels;
(2) mix every BLOOM pixel into one sensor pixel with `sensors.mix_subpixel` at
this patch size and re-apply the instrument's 1-2 % multiplicative noise to the
MIXED pixel; (3) degrade to the sensor's band set and lift back to the 813 grid;
(4) refit Stage 1 + Stage 2 from scratch and score. Only bloom pixels are mixed:
depth and sediment vary on kilometre scales, survive a coarse pixel, and are not
the claim under test. Cells with an identical fill fraction get bit-identical
inputs, because the noise field is drawn once per scene, so the fit is computed
once and reused; that is memoisation, not approximation.

Re-applying the noise after mixing is NOT cosmetic and must not be removed.
Linear mixing against a constant background is exactly invertible, and the
standardisation in front of both stages would undo the fill factor on its own,
so without a noise floor that stays put while the patch signal shrinks this grid
would report that pixel size does not matter. The instrument's noise attaches to
the PIXEL, not to the patch. The price is one extra noise realisation relative
to the spectral-only `"ablation"` block, so the two blocks must NOT be compared
cell for cell; comparisons live inside the grid.

`"bloom_recall"` carries a documented caveat and must not be quoted at low fill
fractions: because the background is a constant scene mean, a heavily diluted
bloom pixel lands closer to that mean than a genuine `no_bloom` pixel does, and
"unnaturally average" becomes a cue in itself. Accuracy and `"cyano_recall"` are
the metrics that carry the result there, and they are what the verdict uses.

```python
def build_verdict(spatial: dict | None = None,
                  reference_patch_size_m: float = REFERENCE_PATCH_SIZE_M,
                  fill_fraction_threshold: float = FILL_FRACTION_THRESHOLD,
                  sensor_keys: Sequence[str] | None = None) -> dict
```
Returns the 2x2 that answers the OLCI question:
```python
{
  "reference_patch_size_m": f, "fill_fraction_threshold": f,
  "pc_band_nm": f, "pc_band_tolerance_nm": f,
  "sensors": {"<sensor_key>": {
      "label": str, "has_620nm": bool, "gsd_m": f,
      "fill_fraction_at_reference": f, "spectral_ok": bool, "spatial_ok": bool,
      "accuracy_at_reference": f | None,        # None when not measured
      "cyano_recall_at_reference": f | None,
      "reason": str}},
  "adequate_on_both": [key, ...], "spectral_only": [key, ...],
  "spatial_only": [key, ...], "inadequate_on_both": [key, ...],
  "summary": str, "note": VERDICT_NOTE,
}
```
Both axes are read off published instrument tables, not out of our simulation:
SPECTRAL from the band table, SPATIAL from the ground sampling distance. The
single assumption in the whole table is the 813 GSD
(`sensors.ASSUMED_813_GSD_M`), flagged as an assumption wherever it appears.
What the simulation supplies is `accuracy_at_reference` and
`cyano_recall_at_reference`, attached from `spatial` when the reference patch
size is in its grid. `summary`, `reason` and the four buckets are GENERATED from
the booleans actually computed and must never be hard-coded: if a band table or
a ground sampling distance changes, the prose changes with it. The same rule
binds `scripts/run_benchmark.py`, whose closing conclusion is generated from the
buckets and states explicitly when the table separates no sensors at all.

`scripts/run_benchmark.py` prints five tables in this order: chlorophyll
retrieval by regime, speciation by regime, the spectral ablation, the spatial
ablation (fill fraction, accuracy, cyanobacteria recall, bloom detection recall
with its caveat), and the 2x2 verdict, then the headline, the generated OLCI
conclusion, the honesty note and the wall clock. `--fast` cuts the patch list to
`(100.0, 1000.0)` as well as the sample counts, and writes under
`outputs/fast-preview/` so a weak model can never replace the judge-facing
`dashboard/benchmark.js`. The reference patch size must stay in any shortened
patch list or the verdict loses its measured columns.

## `src/marsad/hindcast.py` - event-based validation harness

```python
@dataclass
class BloomEvent:
    name: str; region: str; impact_date: str      # ISO date of reported impact
    source: str                                    # literature/agency citation
    onset_lead_days: int                           # bloom onset before impact

DOCUMENTED_EVENTS: tuple[BloomEvent, ...]
    # At minimum the 2008-09 Cochlodinium polykrikoides Gulf of Oman event
    # (impact 2008-11-15, desalination plant shutdowns, Sea of Oman / UAE east
    # coast) with its literature citation, plus one inland cyanobacteria case
    # marked clearly as ILLUSTRATIVE if not documented.

def simulate_event_timeseries(event, n_days=45, seed=0) -> dict
    # {"days": (n,) ints relative to impact (negative = before),
    #  "rrs_observed": (n, N_BANDS), "labels": (n,), "chl": (n,)}
    # A bloom that develops from clear water to peak intensity, crossing
    # operational thresholds some days before `impact_date`.

def evaluate_lead_time(alert_levels: list[str], days: np.ndarray) -> dict
    # {"first_amber_day": int|None, "first_red_day": int|None,
    #  "lead_days_amber": int|None, "lead_days_red": int|None,
    #  "false_alarm_days": int}   # alerts fired while true label is no_bloom
    # lead_days = -first_alert_day (days of warning before impact).

def run_hindcast(seed: int = 5, n_train: int = 3000) -> dict
    # {"events": [{"event": {...}, "metrics": {...}}, ...], "honesty_note": str}
```
The honesty note must state these are **simulated** event timelines shaped to
published event descriptions, and that replacing `simulate_event_timeseries`
with archived Sentinel/EnMAP scenes over the documented dates is the real
validation step (and is on the roadmap).

## `src/marsad/alerts.py` - alert feed & API payloads

```python
@dataclass
class Alert:
    intake: str; level: str; score: float; issued_utc: str
    bloom_class: str; chl_mg_m3: float; lead_days: int
    message: str                                   # operator-facing one-liner
    rationale: list[str]
    def to_dict(self) -> dict

def alerts_from_results(results: dict, min_level: str = "AMBER") -> list[Alert]
    # results = the pipeline's data dict. Emits one Alert per intake at or
    # above min_level. lead_days = first forecast day whose score crosses the
    # RED threshold (0 if already RED, -1 if never within the horizon).

def alert_feed(results: dict) -> dict
    # {"generated_utc": str, "source": "MARSAD 813",
    #  "data_basis": "synthetic physics-based simulation",
    #  "counts": {"RED": int, "AMBER": int, "GREEN": int},
    #  "alerts": [alert.to_dict(), ...]}
```

`scripts/serve_api.py`: stdlib `http.server` only, `--port` (default 8813),
`--results` path. Routes: `GET /health`, `GET /v1/alerts` (alert_feed),
`GET /v1/intakes` (all intake records), `GET /v1/intakes/<name>`,
`GET /v1/metrics` (model_metrics). JSON, CORS `*`, 404 JSON body on unknown
route. Prints the route table on start. NO third-party web framework.

## Fixes required in existing modules

1. `src/marsad/pipeline.py` - model caching. Add
   `run_end_to_end(..., cache_dir=None, refit=False)`: when `cache_dir` is
   given, persist the fitted Stage 1 + Stage 2 to it keyed by
   `(seed, n_train)` and reuse unless `refit=True`. Must not change results
   for a given seed. Also add `"uncertainty"` per intake in the output dict:
   `{"total": f, "epistemic": f, "confidence": f, "review_recommended": bool}`
   sourced from `EnsembleClassifier` - and add the same key to the
   `dashboard/data.js` schema.
2. `src/marsad/stage2_classifier.py` - no API change, but `fit` must accept
   `sample_weight=None` pass-through so the ensemble can bootstrap.
3. `scripts/run_demo.py` - add `--no-cache`, and print the alert feed summary.

## Tests

One `tests/test_<module>.py` per new module: seeded, fast (single-digit
seconds each), asserting shapes, ranges, and the headline behaviour -
in particular:
- `sensors`: identity round-trip for `marsad_813`; resample reduces band
  count; Sentinel-2 has no band within 10 nm of 620; OLCI does. Spatial half:
  every sensor declares a positive GSD, the documented values are asserted
  against the published figures, the 813 note labels its GSD an ASSUMPTION,
  `subpixel_fill_fraction` falls as the square of the size ratio and ranks the
  sensors by pixel size, mixing at full fill returns the target exactly, mixing
  a tiny patch returns almost the background, mixed output stays between its
  two inputs elementwise, and sub-pixel dilution measurably shrinks the 620 nm
  line OLCI would otherwise see.
- `baselines`: OC4 recovers chl on CLEAN deep-water spectra but degrades on
  contaminated shallow-turbid ones; phycocyanin line height is higher for
  cyanobacteria than dinoflagellate spectra; no NaN/warnings on zero input.
- `uncertainty`: entropies in [0,1]; epistemic <= total; confident inputs give
  low uncertainty; ECE in [0,1]; `review_queue` flags the ambiguous rows.
- `benchmark`: schema exactly as specified; `marsad` beats baselines in
  `shallow_turbid`; `hyperspectral_gain > 0`. Spatial half: the `"spatial"` and
  `"verdict"` blocks match the schema above and stay JSON-serialisable;
  `run_spatial_ablation` and `build_verdict` each stand alone without the other;
  coarse sensors degrade as the patch shrinks while the 20 m and 30 m sensors do
  not move at all; the 813 arm leads the grid at the intake scale; the four
  buckets partition the sensors and the summary sentence is generated from the
  booleans rather than asserted; and no module or script in the pair contains an
  em-dash or an en-dash.
- `hindcast`: lead days positive for the documented event; false alarms
  counted; schema.
- `alerts`: feed schema; min_level filtering; RED always emitted.

Run from repo root: `".venv/Scripts/python" -m pytest -q`.
