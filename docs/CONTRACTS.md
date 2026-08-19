# MARSAD 813 — Module Contracts (authoritative)

Every module implements EXACTLY these public APIs. `src/marsad/spectra.py` already
exists and is the single source of truth for the band grid (`N_BANDS = 205`,
`BAND_GRID` = 400–1700 nm). Dependencies allowed: numpy, scikit-learn, stdlib only
(no torch, no pandas, no matplotlib).

## Global conventions

- Spectra are remote-sensing reflectance Rrs, shape `(n, N_BANDS)`, float64,
  typical magnitude 0–0.05 sr⁻¹ (contaminated spectra may exceed this).
- Class labels: `0 = no_bloom`, `1 = dinoflagellate` (red tide, Karenia/Cochlodinium),
  `2 = cyanobacteria` (toxic, inland reservoirs, phycocyanin at 620 nm).
- All randomness goes through `numpy.random.default_rng(seed)`; every public
  function that uses randomness takes a `seed` parameter.
- Risk scores are floats in [0, 1].

## `src/marsad/synth.py` — synthetic Gulf-water scene generator

```python
LABELS: dict[int, str]  # {0: "no_bloom", 1: "dinoflagellate", 2: "cyanobacteria"}

@dataclass
class SynthDataset:
    rrs_observed: np.ndarray  # (n, N_BANDS) contaminated: bottom + sediment + glint + noise
    rrs_true: np.ndarray      # (n, N_BANDS) clean at-surface Rrs (the correction target)
    labels: np.ndarray        # (n,) int in {0,1,2}
    chl: np.ndarray           # (n,) chlorophyll-a mg/m3 (lognormal-ish, blooms high)
    tss: np.ndarray           # (n,) total suspended sediment g/m3
    depth_m: np.ndarray       # (n,) water depth; shallow pixels get bottom reflectance
    phycocyanin: np.ndarray   # (n,) pigment conc., >0 mainly for label 2

def generate_dataset(n_samples: int, seed: int = 0, shallow_fraction: float = 0.5) -> SynthDataset
def generate_history(current_score: float, n_days: int = 30, seed: int = 0) -> np.ndarray
    # (n_days,) autocorrelated daily risk-score history in [0,1] ENDING AT current_score
    # (bounded random walk backwards in time; last element == current_score).
```

Physics to encode in `generate_dataset` (all built from `spectra.gaussian_feature`):
- Base water: blue-green Rrs shape decaying to ~0 beyond ~900 nm; SWIR ≈ 0.
- Chl-a: absorption dips at 443 and 675 nm scaling with log-chl; green peak at 555;
  dense blooms add a red-edge peak at ~708 nm.
- Cyanobacteria (label 2): additional phycocyanin absorption dip at 620 nm.
- TSS: broad backscatter lift, larger at shorter wavelengths, slope varies.
- Shallow bottom (`rrs_observed` only): sandy bottom reflectance added as
  `R_bottom * exp(-2 * Kd * depth)`, Kd higher when chl/tss high.
- Sunglint (`rrs_observed` only): spectrally flat offset, nonzero in SWIR.
- Sensor noise: ~1–2 % multiplicative Gaussian on `rrs_observed`.
- Dinoflagellate blooms occur in shallow AND deep sea pixels; cyanobacteria
  pixels are inland (treat as shallow, low TSS variability).

## `src/marsad/stage1_correction.py` — learned shallow-water correction

```python
class ShallowWaterCorrector:
    def __init__(self, hidden=(128, 64, 128), max_iter=300, seed: int = 0): ...
    def fit(self, rrs_observed, rrs_true) -> "ShallowWaterCorrector"
    def transform(self, rrs_observed) -> np.ndarray          # corrected Rrs
    def score(self, rrs_observed, rrs_true) -> dict          # {"rmse_before": f, "rmse_after": f}
    def save(self, path) / @classmethod load(cls, path)      # joblib
```
Implementation — RESIDUAL formulation (load-bearing): the MLP regresses the
**contamination** `observed − true` and `transform` returns
`observed − predicted_contamination`. It must NOT regress the clean spectrum
directly: direct regression repaints narrow pigment lines (620 nm phycocyanin)
with their conditional mean and was measured to cut Stage-2 accuracy from
~0.97 to ~0.83 despite excellent RMSE. StandardScaler on inputs and
contamination targets around a multi-output `MLPRegressor` (one output per
band). `rmse_after` must beat `rmse_before` on held-out synthetic data.

## `src/marsad/stage2_classifier.py` — bloom detection & speciation

```python
class BloomClassifier:
    def __init__(self, seed: int = 0): ...
    def fit(self, rrs, labels, chl) -> "BloomClassifier"
    def predict_proba(self, rrs) -> np.ndarray   # (n, 3) columns in label order 0,1,2
    def predict(self, rrs) -> np.ndarray         # (n,) argmax labels
    def estimate_chl(self, rrs) -> np.ndarray    # (n,) mg/m3, regression head
    def evaluate(self, rrs, labels) -> dict      # {"accuracy": f, "confusion": 3x3 int list}
    def save(self, path) / @classmethod load(cls, path)
```
Implementation: engineered band-ratio features (443/555, 620 line depth, 665/708
NDCI, red-edge height) CONCATENATED with the full spectrum; sklearn classifier
(MLP or gradient boosting) + separate regressor for log-chl.

## `src/marsad/stage3_forecast.py` — drift & forecast

```python
@dataclass
class Forecast:
    mean: np.ndarray  # (horizon,) risk scores in [0,1]
    lo: np.ndarray    # (horizon,) 10th percentile
    hi: np.ndarray    # (horizon,) 90th percentile
    method: str

class DriftForecaster:
    def __init__(self, horizon_days: int = 7): ...
    def forecast(self, history: np.ndarray, drift_toward_intake_kmday: float = 0.0) -> Forecast
```
Implementation: damped-trend exponential smoothing on the history + an advection
bump proportional to positive `drift_toward_intake_kmday`; uncertainty band widens
with sqrt(lead time); clip to [0, 1]; `hi >= mean >= lo` elementwise.

## `src/marsad/risk.py` — per-intake risk policy

```python
class RiskLevel(str, Enum): GREEN = "GREEN"; AMBER = "AMBER"; RED = "RED"

@dataclass
class RiskAssessment:
    score: float            # [0,1]
    level: RiskLevel
    rationale: list[str]    # human-readable reasons, shown on the dashboard

def compute_risk_index(bloom_prob: float, chl_mg_m3: float, trend_per_day: float,
                       distance_km: float, uncertainty: float) -> RiskAssessment
```
`bloom_prob` = 1 − P(no_bloom). `uncertainty` is ASSESSMENT uncertainty:
max(mean normalized forecast hi-lo band, classifier mean ambiguity
`1 − 2|p − 0.5|`) — never a spatial bloom/clear mix fraction, which is
maximal precisely when the classifier is most confident about a mixed scene.
Mark the weights/thresholds block with a
`# --- TEAM DECISION ---` comment: this is operator policy, meant to be tuned.
Defaults: score = weighted blend (prob dominates), proximity boost < 5 km,
positive trend boost; RED ≥ 0.65, AMBER ≥ 0.35; high uncertainty can promote
AMBER→RED near intakes (precautionary) — never silently downgrade.

## `src/marsad/pipeline.py` — end-to-end orchestration

```python
INTAKES: list[dict]  # exactly these four:
# {"name": "Khor Fakkan", "lat": 25.339, "lon": 56.353, "kind": "desalination_intake"}
# {"name": "Kalba",       "lat": 25.074, "lon": 56.356, "kind": "desalination_intake"}
# {"name": "Layyah",      "lat": 25.356, "lon": 55.386, "kind": "desalination_intake"}
# {"name": "Hatta Dam",   "lat": 24.783, "lon": 56.113, "kind": "reservoir"}

def run_end_to_end(seed: int = 7, outdir=None, n_train: int = 4000,
                   n_scene: int = 1200, history_days: int = 30) -> dict
```
Steps: train/holdout split → fit Stage 1 on (observed, true) → fit Stage 2 on
**corrected** spectra → report stage1 rmse + stage2 accuracy on holdout →
assign each intake a slice of a fresh scene (Hatta Dam gets cyano-or-clear
pixels only; sea intakes dino-or-clear) → per intake: mean probs, chl estimate,
`synth.generate_history` ending at current score, Stage 3 forecast, risk
assessment → write `outputs/results.json` AND `dashboard/data.js`.

`dashboard/data.js` is a single statement `window.MARSAD_DATA = {...};` with EXACTLY:

```json
{
  "generated_utc": "ISO-8601 string",
  "model_metrics": {
    "stage1_rmse_before": 0.0, "stage1_rmse_after": 0.0,
    "stage2_accuracy": 0.0, "stage2_confusion": [[0,0,0],[0,0,0],[0,0,0]],
    "labels": ["no_bloom", "dinoflagellate", "cyanobacteria"]
  },
  "intakes": [{
    "name": "", "lat": 0.0, "lon": 0.0, "kind": "",
    "risk": {"score": 0.0, "level": "GREEN|AMBER|RED", "rationale": [""]},
    "bloom": {"probs": {"no_bloom": 0.0, "dinoflagellate": 0.0, "cyanobacteria": 0.0},
               "dominant": "", "chl_mg_m3": 0.0},
    "history": [{"day": -29, "score": 0.0}],
    "forecast": [{"day": 1, "score": 0.0, "lo": 0.0, "hi": 0.0}]
  }],
  "spectra_example": {"wavelength_nm": [], "observed": [], "corrected": [], "true": []}
}
```

`scripts/run_demo.py`: argparse (`--seed`, `--fast`), calls `run_end_to_end`,
prints a readable per-intake summary table + metrics to stdout.
`scripts/download_gloria.py`: documented stub with the real GLORIA (PANGAEA
doi:10.1594/PANGAEA.948492) and NASA PACE URLs, `--dest` arg, clear TODO notes —
no network call executed by default.

## `dashboard/index.html` — self-contained control-room dashboard

Reads `<script src="data.js">` (works from file://, no CDN, no external fonts).
Vanilla JS + inline SVG/canvas. Dark control-room aesthetic, MARSAD (مَرصَد)
branding, English primary. Sections: (1) header with generated time + overall
status; (2) one risk card per intake — level colour, score dial, dominant bloom
type, chl, rationale list; (3) history + 7-day forecast chart with lo/hi
uncertainty band per intake (selectable); (4) Stage 1 before/after spectra chart
with key wavelengths annotated; (5) model metrics incl. confusion matrix.
Handle missing `data.js` with a visible "run scripts/run_demo.py first" notice.

## Tests

Each `tests/test_<module>.py`: fast (seconds, small n), seeded, assert shapes,
value ranges, and the module's headline behaviour (stage1 improves RMSE,
stage2 accuracy > 0.8 on easy synthetic holdout, forecast bands ordered,
risk levels monotone in probability, pipeline smoke test writes both outputs).
Run from repo root: `".venv/Scripts/python" -m pytest tests/test_X.py -q`
(conftest.py handles the import path).
