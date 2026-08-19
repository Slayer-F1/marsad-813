"""The experiment that turns MARSAD's central claim into a measured result.

The claim under test is operational, not academic: *standard satellite
chlorophyll algorithms fail in shallow turbid Gulf-coastal water, and the
MARSAD two-stage approach does not.* This module measures it three ways.

1. **Chlorophyll retrieval error by water regime** (:func:`run_benchmark`,
   ``"chl_retrieval"``). OC4, OC3M and NDCI are run exactly as an operator
   would run them: on the RAW observed spectra, because raw radiometry after
   ordinary atmospheric correction is what an operator actually has in hand.
   MARSAD is run on Stage-1-corrected spectra, because that is what MARSAD
   actually has in hand. Errors are grouped by the four water regimes of
   :data:`WATER_REGIMES`, so the failure is localised to the regime the
   project targets instead of being averaged away over a whole scene.

2. **Speciation accuracy by water regime** (``"speciation"``). The classical
   operator decision tree (:func:`marsad.baselines.classify_baseline`:
   NDCI-chlorophyll threshold, then 620 nm phycocyanin line height) against
   MARSAD Stage 2, same regime grouping.

3. **The hyperspectral ablation** (``"ablation"``). The *same* Stage 1 +
   Stage 2 architecture is retrained from scratch on spectra degraded to each
   multispectral band set (:mod:`marsad.sensors`), and scored on overall
   accuracy plus recall of the cyanobacteria class. Cyanobacteria recall is
   the sharp end: it depends on a phycocyanin absorption feature at 620 nm,
   so a sensor with no band there cannot see the pigment at all, whatever its
   spatial resolution. ``hyperspectral_gain`` is the accuracy of the 205-band
   813 instrument minus the best multispectral alternative.

Why the regime split is the whole point
---------------------------------------
Blue-green band ratios such as OC4 assume Case-1 water, where every optical
property co-varies with phytoplankton. Two Gulf-coastal realities break that
assumption in opposite directions: a bright sandy bottom in a few metres of
water adds reflectance across the visible, and resuspended mineral sediment
adds backscatter weighted to the blue-green. Both move the band ratio without
any change in chlorophyll. Averaged over a large scene the effect looks like
scatter; split by regime it is a systematic bias that lands precisely on the
shallow turbid pixels where desalination intakes sit.

Fairness of the comparison
--------------------------
Three choices keep this from being a straw man and each is deliberate.

* The baselines get the raw observation, not a deliberately corrupted one.
  They are published algorithms run as published, coefficients unchanged.
* Every ablation arm gets the same architecture, the same training-set size,
  the same seed, and ``N_BANDS`` inputs after
  :func:`marsad.sensors.resample_to_grid`, so only the spectral *information*
  differs and an accuracy drop can never be blamed on model capacity.
* Every arm is held to the SAME Stage 1 product specification: recover the
  clean at-surface Rrs on the full 813 analysis grid, because that is what
  Stage 2 consumes whatever the input sensor was. Scoring each arm against
  its own blurred copy of the truth would quietly redefine the deliverable
  per sensor, which is the one thing an ablation must not do.

What the ablation actually shows, stated honestly
-------------------------------------------------
The collapse is on the sensors with no 620 nm band. Sentinel-2, MODIS and
Landsat lose a large fraction of cyanobacteria recall because the phycocyanin
feature is interpolated straight over and is simply not in their data.
Sentinel-3 OLCI is the interesting case and the honest one: it is the single
operational ocean-colour sensor that does carry a 620 nm band, so in our
forward model it recovers most of the speciation signal. The margin over it,
which is what ``hyperspectral_gain`` reports because OLCI is the best
multispectral arm, is SMALL: a few tenths of a percent to a couple of percent
of accuracy, and of the same order as the run-to-run scatter of the
simulation. That margin comes from the continuum sampling Stage 1 needs to
separate bottom and sediment, not from the pigment band, which OLCI has.

So ``hyperspectral_gain`` must never be quoted on its own. The defensible
sentence is: "far better than the Sentinel-2 / MODIS / Landsat class, which
carry no 620 nm band and lose roughly a third of cyanobacteria recall;
marginally better than OLCI, whose remaining limitation is spatial, 300 m
pixels over an intake, and is therefore not something this spectral-only
ablation measures at all."

SCIENTIFIC HONESTY (binding, docs/CONTRACTS-V2.md)
--------------------------------------------------
Every spectrum here comes from :mod:`marsad.synth`, which is OUR OWN forward
model. This benchmark is therefore a self-consistency check against a
physics-based simulation, consistent with the Case-2 water literature, and it
is NEVER independent validation of how these algorithms behave on real Gulf
water. It shows that the failure modes the literature reports for band ratios
in optically complex water are present in our simulation and that our
architecture survives them there. Reporting any number below as "we proved
standard algorithms fail on Gulf water" would be false. Real validation is
the archived-scene hindcast (:mod:`marsad.hindcast`) once GLORIA / PACE / 813
scenes are in hand, and it has not been done yet.
"""
from __future__ import annotations

import numpy as np

from marsad import baselines, sensors, synth
from marsad.stage1_correction import ShallowWaterCorrector
from marsad.stage2_classifier import BloomClassifier

__all__ = [
    "WATER_REGIMES",
    "SHALLOW_DEPTH_M",
    "TURBID_TSS_G_M3",
    "ABLATION_TRAIN_CAP",
    "HONESTY_NOTE",
    "classify_regime",
    "run_benchmark",
]

#: The four optical water regimes the benchmark reports separately.
#: ``"shallow_turbid"`` is the Gulf-coast failure regime the whole project
#: targets: optically shallow enough for the sea floor to reflect into the
#: sensor AND turbid enough for mineral backscatter to dominate the blue.
WATER_REGIMES: tuple[str, ...] = (
    "optically_deep",
    "shallow_clear",
    "turbid_deep",
    "shallow_turbid",
)

# --- TEAM DECISION ---
# Regime boundaries. These are operational definitions, not physical
# constants, and they are meant to be tuned per water body.
#
# 10 m: the depth at which the two-way attenuation exp(-2 * Kd * z) of a sandy
# bottom drops below the radiometric noise floor for typical Gulf-coastal Kd
# (~0.1-0.3 m^-1 in the visible). Below it the bottom is a real term in the
# observed reflectance; above it the water is optically deep and the bottom
# contributes nothing measurable.
#
# 8 g/m3 TSS: roughly where mineral backscatter starts to dominate the
# blue-green signal in Case-2 coastal water and blue-green ratios lose their
# footing. Gulf-coast resuspension events run well past this.
SHALLOW_DEPTH_M: float = 10.0
TURBID_TSS_G_M3: float = 8.0
# --- END TEAM DECISION ---

#: Ablation arms are trained on at most this many pixels. Retraining Stage 1 +
#: Stage 2 once per sensor is the dominant cost of the whole benchmark, so the
#: cap keeps a full run to a few minutes. Every arm, the 813 arm included, is
#: trained at the SAME size, which is what makes the comparison valid: the cap
#: changes all arms equally and therefore cannot manufacture a gain.
ABLATION_TRAIN_CAP: int = 1200

#: Retrieved and reference chlorophyll are floored here (mg m^-3) before the
#: log10 error, so a retrieval of exactly zero produces a large finite error
#: instead of -inf. Three orders of magnitude below the clearest oligotrophic
#: water, so it never touches a physically meaningful value.
_CHL_FLOOR_MG_M3: float = 1e-3

#: Target minimum pixels per regime in the test scene (see
#: :func:`_stratified_test_scene`). A median over a handful of pixels is
#: noise, and deep turbid water is genuinely rare in the forward model, so
#: every regime is topped up to at least this many pixels before anything is
#: reported about it.
_MIN_REGIME_PIXELS: int = 60
_TOPUP_CHUNK: int = 4000
_TOPUP_MAX_ROUNDS: int = 8

HONESTY_NOTE: str = (
    "Every number in this benchmark is measured on marsad.synth, our own "
    "physics-based forward model of Gulf water, so it is a self-consistency "
    "check against a simulation - consistent with the Case-2 water literature "
    "on band-ratio failure in optically complex water - and never independent "
    "validation of how these algorithms behave on real Gulf scenes, which "
    "requires the archived-scene hindcast on GLORIA / PACE / 813 data and has "
    "not been done yet."
)


def classify_regime(depth_m: np.ndarray, tss: np.ndarray) -> np.ndarray:
    """Label each pixel with its optical water regime.

    Parameters
    ----------
    depth_m:
        (n,) water depth in metres. Shallow when below
        :data:`SHALLOW_DEPTH_M`, i.e. shallow enough for bottom reflectance to
        reach the sensor through the two-way diffuse path.
    tss:
        (n,) total suspended sediment in g m^-3. Turbid at or above
        :data:`TURBID_TSS_G_M3`, i.e. enough mineral backscatter to dominate
        the blue-green bands that OCx band ratios rely on.

    Returns
    -------
    np.ndarray
        (n,) array of strings drawn from :data:`WATER_REGIMES`. The two
        stressors are independent, so the four regimes are the four corners of
        the (shallow, turbid) grid and ``"shallow_turbid"`` is the corner where
        both failure mechanisms act at once.

    Both inputs are broadcast against each other, so scalars are accepted, and
    non-finite values fall on the deep/clear side rather than raising: an
    unusable depth retrieval should not silently promote a pixel into the
    regime we are trying to prove something about.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    turb = np.asarray(tss, dtype=np.float64)
    shallow = np.isfinite(depth) & (depth < SHALLOW_DEPTH_M)
    turbid = np.isfinite(turb) & (turb >= TURBID_TSS_G_M3)
    code = shallow.astype(np.int64) + 2 * turbid.astype(np.int64)
    return np.asarray(WATER_REGIMES, dtype="<U16")[code]


# --------------------------------------------------------------------- helpers


def _median_abs_log10_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Median absolute log10 error between retrieved and reference chl.

    Chlorophyll spans three orders of magnitude between oligotrophic Gulf
    water and a surface bloom, so error is only meaningful in log space: a
    30 mg m^-3 miss is nothing on a 200 mg m^-3 slick and a total failure on
    0.5 mg m^-3 background water. The MEDIAN, not the mean, because band-ratio
    polynomials produce heavy-tailed outliers by construction (a 4th-order fit
    evaluated outside its calibration range), and one such pixel would
    otherwise decide the whole regime's score. 0.3 means "typically a factor
    of two out"; 1.0 means "typically an order of magnitude out".
    """
    p = np.maximum(np.asarray(pred, dtype=np.float64), _CHL_FLOOR_MG_M3)
    t = np.maximum(np.asarray(true, dtype=np.float64), _CHL_FLOOR_MG_M3)
    return float(np.median(np.abs(np.log10(p) - np.log10(t))))


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of pixels whose predicted class equals the true class."""
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def _recall(pred: np.ndarray, true: np.ndarray, cls: int) -> float:
    """Recall of one class: of the pixels that ARE ``cls``, how many were found.

    Recall rather than accuracy or precision because the operational cost is
    asymmetric: a missed toxic cyanobacteria bloom reaches the intake, a false
    one costs an analyst ten minutes. Returns 0.0 when the class is absent
    from the reference labels, which is the conservative reading of "we have
    no evidence this sensor can find them".
    """
    mask = np.asarray(true) == cls
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.asarray(pred)[mask] == cls))


def _concat_datasets(parts: list[synth.SynthDataset]) -> synth.SynthDataset:
    """Row-concatenate synthetic scenes into one dataset."""
    if len(parts) == 1:
        return parts[0]
    return synth.SynthDataset(
        rrs_observed=np.concatenate([p.rrs_observed for p in parts], axis=0),
        rrs_true=np.concatenate([p.rrs_true for p in parts], axis=0),
        labels=np.concatenate([p.labels for p in parts]),
        chl=np.concatenate([p.chl for p in parts]),
        tss=np.concatenate([p.tss for p in parts]),
        depth_m=np.concatenate([p.depth_m for p in parts]),
        phycocyanin=np.concatenate([p.phycocyanin for p in parts]),
    )


def _stratified_test_scene(n_test: int, seed: int) -> tuple[synth.SynthDataset, int]:
    """Draw a test scene, then top it up so every regime is measurable.

    A natural draw from :func:`marsad.synth.generate_dataset` reproduces the
    real asymmetry of the coast: optically deep water offshore is not
    resuspension-prone, so deep *turbid* pixels are genuinely rare (a fraction
    of a percent). That is physically right and statistically useless - a
    median over three pixels is noise - so after the natural draw this
    function keeps drawing extra scenes, biased toward the deficient regimes
    via ``shallow_fraction``, and appends ONLY pixels from regimes that are
    still short of :data:`_MIN_REGIME_PIXELS`.

    This is stratified sampling and it is safe here for one specific reason:
    every per-regime statistic the benchmark reports is CONDITIONAL on the
    regime, so changing how many pixels of each regime are in the sample
    changes the precision of each estimate but not its expectation. Nothing in
    the output is a scene-wide average over regimes. The one pooled number
    that does exist, ablation accuracy, is computed on the first ``n_test``
    rows only, which is why the base count is returned alongside the scene.

    Returns
    -------
    (dataset, n_base)
        The scene, and the number of leading rows that came from the natural
        unstratified draw.
    """
    n_base = int(n_test)
    parts = [synth.generate_dataset(n_base, seed=seed)]
    target = max(1, min(_MIN_REGIME_PIXELS, n_base))

    for round_i in range(_TOPUP_MAX_ROUNDS):
        counts = {r: 0 for r in WATER_REGIMES}
        for regime, count in zip(*np.unique(
            classify_regime(
                np.concatenate([p.depth_m for p in parts]),
                np.concatenate([p.tss for p in parts]),
            ),
            return_counts=True,
        )):
            counts[str(regime)] = int(count)

        deficit = {r: target - counts[r] for r in WATER_REGIMES if counts[r] < target}
        if not deficit:
            break

        # Bias the extra draw toward whichever side is short. Deep pixels only
        # appear when shallow_fraction is low, and label-2 (inland) pixels are
        # always shallow, so this is the only lever that moves the mix.
        worst = min(deficit, key=lambda r: counts[r])
        shallow_fraction = 1.0 if worst.startswith("shallow") else 0.0
        extra = synth.generate_dataset(
            _TOPUP_CHUNK, seed=seed + 1000 + round_i, shallow_fraction=shallow_fraction
        )
        extra_regime = classify_regime(extra.depth_m, extra.tss)

        keep = np.zeros(extra_regime.size, dtype=bool)
        for regime, needed in deficit.items():
            idx = np.flatnonzero(extra_regime == regime)[:needed]
            keep[idx] = True
        if not keep.any():
            continue
        parts.append(
            synth.SynthDataset(
                rrs_observed=extra.rrs_observed[keep],
                rrs_true=extra.rrs_true[keep],
                labels=extra.labels[keep],
                chl=extra.chl[keep],
                tss=extra.tss[keep],
                depth_m=extra.depth_m[keep],
                phycocyanin=extra.phycocyanin[keep],
            )
        )

    scene = _concat_datasets(parts)
    present = set(np.unique(classify_regime(scene.depth_m, scene.tss)).tolist())
    missing = [r for r in WATER_REGIMES if r not in present]
    if missing:  # pragma: no cover - needs a pathological forward model
        raise RuntimeError(
            f"regimes {missing} produced no pixels after {_TOPUP_MAX_ROUNDS} "
            "top-up draws; the synthetic generator or the regime thresholds "
            "have changed and the benchmark can no longer report them."
        )
    return scene, n_base


def _fit_pipeline(
    rrs_observed: np.ndarray,
    rrs_true: np.ndarray,
    labels: np.ndarray,
    chl: np.ndarray,
    seed: int,
) -> tuple[ShallowWaterCorrector, BloomClassifier]:
    """Fit Stage 1 then Stage 2, exactly as :mod:`marsad.pipeline` does.

    Stage 2 is trained on Stage-1 *corrected* spectra because corrected
    spectra are the only thing it ever sees in operations; training it on raw
    ones would leave the train and serve distributions misaligned. Every
    ablation arm calls this same function, which is what makes "the SAME
    architecture" a fact about the code rather than a promise in prose.
    """
    corrector = ShallowWaterCorrector(seed=seed).fit(rrs_observed, rrs_true)
    classifier = BloomClassifier(seed=seed).fit(
        corrector.transform(rrs_observed), labels, chl
    )
    return corrector, classifier


def _degrade(rrs: np.ndarray, sensor_key: str) -> np.ndarray:
    """Round-trip spectra through a sensor's band set, back onto the 813 grid.

    ``resample`` integrates the spectrum against each band's Gaussian spectral
    response (what that radiometer would record); ``resample_to_grid``
    interpolates those band values back to all ``N_BANDS`` channels. The array
    width is unchanged, so the ablation models differ only in the information
    their inputs carry. A narrow feature between two bands, such as the 620 nm
    phycocyanin dip on Sentinel-2, is interpolated straight over and is simply
    gone.
    """
    return sensors.resample_to_grid(sensors.resample(rrs, sensor_key), sensor_key)


# ------------------------------------------------------------------ experiment


def run_benchmark(seed: int = 11, n_train: int = 4000, n_test: int = 2000) -> dict:
    """Run the full comparison and return the contracted result dictionary.

    Method
    ------
    1. Generate a training scene and a test scene (:func:`_stratified_test_scene`).
    2. Fit Stage 1 on (observed, true) pairs and Stage 2 on corrected spectra.
    3. Score the published baselines on the RAW observed test spectra, which
       is what an operator has, and MARSAD on Stage-1-corrected ones, which is
       what MARSAD has. Group both by water regime.
    4. Retrain the same architecture once per sensor on spectra degraded to
       that sensor's band set, against the same full-resolution Stage 1
       target, and score accuracy plus cyanobacteria recall.

    Parameters
    ----------
    seed:
        Master seed. The test scene uses ``seed + 1`` so it is an independent
        draw, never a re-draw of the training pixels.
    n_train:
        Training pixels for Stage 1 + Stage 2. The ablation arms use
        ``min(n_train, ABLATION_TRAIN_CAP)`` pixels each, identically.
    n_test:
        Pixels in the natural test draw, before the per-regime top-up.

    Returns
    -------
    dict
        Exactly the five keys of docs/CONTRACTS-V2.md: ``chl_retrieval``,
        ``speciation``, ``ablation``, ``headline`` and ``honesty_note``.
        Errors in ``chl_retrieval`` are median absolute log10 errors (lower is
        better), values in ``speciation`` and ``ablation`` are fractions in
        [0, 1] (higher is better), and ``hyperspectral_gain`` is a difference
        of accuracies that is positive when the 205-band instrument beats
        every multispectral alternative.

    Read ``honesty_note`` before quoting any number: all of it is measured on
    our own forward model and is a self-consistency check, not validation.
    """
    train = synth.generate_dataset(int(n_train), seed=seed)
    test, n_base = _stratified_test_scene(int(n_test), seed=seed + 1)

    # --- 1. the MARSAD pipeline, trained once on full-resolution spectra ----
    corrector, classifier = _fit_pipeline(
        train.rrs_observed, train.rrs_true, train.labels, train.chl, seed
    )
    corrected_test = corrector.transform(test.rrs_observed)
    marsad_chl = classifier.estimate_chl(corrected_test)
    marsad_labels = classifier.predict(corrected_test)

    # --- 2. the baselines, on the RAW observation an operator would have ----
    # No shallow-water correction, no learning: published coefficients applied
    # to the measured spectrum. That is the comparison that means something.
    raw = test.rrs_observed
    chl_estimates = {
        "oc4": baselines.oc4_chl(raw),
        "oc3m": baselines.oc3m_chl(raw),
        "ndci": baselines.ndci_chl(raw),
        "marsad": marsad_chl,
    }
    baseline_labels = baselines.classify_baseline(raw)

    # --- 3. group by water regime ------------------------------------------
    regimes = classify_regime(test.depth_m, test.tss)
    chl_retrieval: dict[str, dict] = {}
    speciation: dict[str, dict] = {}
    for regime in WATER_REGIMES:
        mask = regimes == regime
        n_regime = int(np.count_nonzero(mask))
        chl_retrieval[regime] = {
            name: _median_abs_log10_error(values[mask], test.chl[mask])
            for name, values in chl_estimates.items()
        }
        chl_retrieval[regime]["n"] = n_regime
        speciation[regime] = {
            "baseline_tree": _accuracy(baseline_labels[mask], test.labels[mask]),
            "marsad": _accuracy(marsad_labels[mask], test.labels[mask]),
            "n": n_regime,
        }

    # --- 4. the hyperspectral ablation --------------------------------------
    # Pooled over the natural draw only (rows [:n_base]), so the per-regime
    # top-up cannot tilt a scene-level accuracy.
    n_ab = min(int(n_train), ABLATION_TRAIN_CAP)
    ab_labels = test.labels[:n_base]
    ablation: dict[str, dict] = {}
    for key, sensor in sensors.SENSORS.items():
        ab_train_obs = _degrade(train.rrs_observed[:n_ab], key)
        # The Stage 1 target is the FULL-RESOLUTION clean spectrum for every
        # arm, because the product Stage 2 consumes does not change when the
        # input sensor does. Scoring a coarse arm against a coarse target
        # would redefine the deliverable per sensor and hide the information
        # loss the ablation exists to measure.
        ab_corrector, ab_classifier = _fit_pipeline(
            ab_train_obs, train.rrs_true[:n_ab],
            train.labels[:n_ab], train.chl[:n_ab], seed,
        )
        ab_pred = ab_classifier.predict(
            ab_corrector.transform(_degrade(test.rrs_observed[:n_base], key))
        )
        ablation[key] = {
            "label": sensor.label,
            "n_bands": int(sensor.n_bands),
            "accuracy": _accuracy(ab_pred, ab_labels),
            "cyano_recall": _recall(ab_pred, ab_labels, 2),
            "note": sensor.note,
        }

    # --- 5. headline numbers -------------------------------------------------
    native_acc = ablation["marsad_813"]["accuracy"]
    multispectral = [v["accuracy"] for k, v in ablation.items() if k != "marsad_813"]
    best_multispectral = max(multispectral) if multispectral else native_acc

    headline = {
        "marsad_shallow_turbid_acc": speciation["shallow_turbid"]["marsad"],
        "baseline_shallow_turbid_acc": speciation["shallow_turbid"]["baseline_tree"],
        "oc4_deep_err": chl_retrieval["optically_deep"]["oc4"],
        "oc4_shallow_turbid_err": chl_retrieval["shallow_turbid"]["oc4"],
        "hyperspectral_gain": float(native_acc - best_multispectral),
    }

    return {
        "chl_retrieval": chl_retrieval,
        "speciation": speciation,
        "ablation": ablation,
        "headline": headline,
        "honesty_note": HONESTY_NOTE,
    }
