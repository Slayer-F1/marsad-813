# MARSAD 813 - v0.2 Module Contracts (authoritative)

Extends `docs/CONTRACTS.md` (v0.1, still binding for existing modules). Same rules:
numpy / scikit-learn / stdlib ONLY; `src/marsad/spectra.py` owns the band grid
(`N_BANDS = 205`, `BAND_GRID` 400–1700 nm); labels `0 = no_bloom`,
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
ablation that proves hyperspectral is the *enabling* sensor, not a passenger.

```python
@dataclass(frozen=True)
class Band:
    name: str
    center_nm: float
    fwhm_nm: float

@dataclass(frozen=True)
class Sensor:
    key: str
    label: str            # human-readable, e.g. "Sentinel-2 MSI"
    bands: tuple[Band, ...]
    note: str             # one-line honest capability note (see below)
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
```

Required `SENSORS` keys and real band sets (centre/FWHM nm, water-relevant,
clipped to 400–1700):

- `"marsad_813"` - the native 205-band grid (identity resample must round-trip
  to within 1e-12). note: "205 contiguous bands, 400-1700 nm".
- `"sentinel2_msi"` - 443/20, 490/65, 560/35, 665/30, 705/15, 740/15, 783/20,
  842/115, 865/20, 945/20, 1375/30, 1610/90. note must state: no band at 620 nm,
  so phycocyanin is not directly observable.
- `"sentinel3_olci"` - 400/15, 412.5/10, 442.5/10, 490/10, 510/10, 560/10,
  620/10, 665/10, 673.75/7.5, 681.25/7.5, 708.75/10, 753.75/7.5, 778.75/15,
  865/20, 885/10, 900/10, 1020/40. note must be HONEST: OLCI *does* carry a
  620 nm band, but one band at 300 m ground sampling cannot separate
  phycocyanin absorption from co-varying sediment/chl absorption - this
  nuance is a credibility point, do not overstate.
- `"modis_aqua"` - 412/15, 443/10, 469/20, 488/10, 531/10, 547/10, 555/20,
  645/50, 667/10, 678/10, 748/10, 859/35, 869/15, 1240/20, 1640/50.
  note: no 620 nm band; 1 km pixels miss intake-scale patches.
- `"landsat8_oli"` - 443/16, 482/60, 561/57, 655/37, 865/28, 1609/85.
  note: 6 usable water bands; blue-green ratio only.

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

def run_benchmark(seed: int = 11, n_train: int = 4000, n_test: int = 2000) -> dict
```
Returns EXACTLY:
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
`hyperspectral_gain` = marsad_813 accuracy − best non-813 sensor accuracy.

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
  count; Sentinel-2 has no band within 10 nm of 620; OLCI does.
- `baselines`: OC4 recovers chl on CLEAN deep-water spectra but degrades on
  contaminated shallow-turbid ones; phycocyanin line height is higher for
  cyanobacteria than dinoflagellate spectra; no NaN/warnings on zero input.
- `uncertainty`: entropies in [0,1]; epistemic <= total; confident inputs give
  low uncertainty; ECE in [0,1]; `review_queue` flags the ambiguous rows.
- `benchmark`: schema exactly as specified; `marsad` beats baselines in
  `shallow_turbid`; `hyperspectral_gain > 0`.
- `hindcast`: lead days positive for the documented event; false alarms
  counted; schema.
- `alerts`: feed schema; min_level filtering; RED always emitted.

Run from repo root: `".venv/Scripts/python" -m pytest -q`.
