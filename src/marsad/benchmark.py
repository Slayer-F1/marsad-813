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

4. **The spatial ablation** (``"spatial"``) and the **two-axis verdict**
   (``"verdict"``). A band set is only half of what an instrument can see;
   the other half is the size of its pixel. Measurement 3 on its own
   silently assumes every sensor resolves the bloom patch, and that is
   false at the scale which decides whether an intake gets a warning: OLCI
   pixels are 300 m and MODIS ocean-colour pixels are 1 km, so an
   intake-scale patch is sub-pixel and reaches the instrument already mixed
   with the clear water around it. :func:`run_spatial_ablation` dilutes each
   bloom pixel into one sensor pixel before degrading it to that sensor's
   band set, then retrains and rescores the same architecture, so the table
   is accuracy against patch size per sensor. :func:`build_verdict` reduces
   the two axes, a 620 nm band and a pixel small enough to hold the patch,
   to the 2x2 that answers the OLCI question.

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

That last clause described v0.2 and is no longer where the analysis stops.
The ``"spatial"`` block measures exactly the term the spectral ablation
leaves out, and the ``"verdict"`` block reports the two axes together, which
is what makes the sensor comparison honest in BOTH directions: OLCI really
does carry the pigment band we need, and a 100 m patch really does fill only
about a ninth of one of its pixels.

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

from collections.abc import Sequence

import numpy as np

from marsad import baselines, sensors, synth
from marsad.spectra import N_BANDS
from marsad.stage1_correction import ShallowWaterCorrector
from marsad.stage2_classifier import BloomClassifier

__all__ = [
    "WATER_REGIMES",
    "SHALLOW_DEPTH_M",
    "TURBID_TSS_G_M3",
    "ABLATION_TRAIN_CAP",
    "SPATIAL_PATCH_SIZES_M",
    "SPATIAL_TRAIN_CAP",
    "SPATIAL_TEST_CAP",
    "REFERENCE_PATCH_SIZE_M",
    "FILL_FRACTION_THRESHOLD",
    "PC_BAND_NM",
    "PC_BAND_TOLERANCE_NM",
    "HONESTY_NOTE",
    "SPATIAL_NOTE",
    "VERDICT_NOTE",
    "classify_regime",
    "nearest_band_distance_nm",
    "has_phycocyanin_band",
    "run_spatial_ablation",
    "build_verdict",
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


# --- TEAM DECISION ---
# The spatial ablation grid, in metres of bloom-patch side length.
#
# 50 m is below every pixel in the table except the 30 m class, so it is the
# column where even Landsat and the 813 assumption start to be tested. 100 m is
# the intake scale this project is built around. 300 m and 1000 m are exactly
# the OLCI and MODIS pixel sizes, so those columns sit on the "patch just fills
# the pixel" boundary for those two sensors and make the transition legible
# rather than hidden between decades. 3000 m is a regional slick that every
# sensor in the table resolves outright, and it is the control column: the
# spatial term is switched off there and only the band set is left.
SPATIAL_PATCH_SIZES_M: tuple[float, ...] = (50.0, 100.0, 300.0, 1000.0, 3000.0)

# Reference patch size for the 2x2 verdict: the size of bloom patch a
# desalination intake actually has to be warned about. Intake structures and
# the water parcel drawn through them are tens to a couple of hundred metres
# across, so 100 m is the round figure in the middle of that range. This is an
# operational definition, not a physical constant, and it is meant to be
# retuned per site.
REFERENCE_PATCH_SIZE_M: float = 100.0

# A sensor counts as spatially adequate when the patch fills at least this much
# of one pixel. At 0.5 the patch contributes more to the pixel's spectrum than
# its surroundings do; below it the pixel is mostly background water and the
# bloom is a minority term in its own measurement.
FILL_FRACTION_THRESHOLD: float = 0.5

# The phycocyanin absorption feature, and how close a band has to sit to see
# it. 10 nm is about half a typical ocean-colour band width: further away and
# the 620 nm absorption is averaged into the surrounding continuum rather than
# measured.
PC_BAND_NM: float = 620.0
PC_BAND_TOLERANCE_NM: float = 10.0

# Multiplicative sensor-noise level re-applied to the mixed pixel, matching the
# 1-2 % per-pixel noise that marsad.synth already puts on rrs_observed. See
# _dilute for why re-applying it is not optional.
SPATIAL_NOISE_FRAC: tuple[float, float] = (0.01, 0.02)
# --- END TEAM DECISION ---

#: Training pixels per cell of the spatial grid, reduced from
#: :data:`ABLATION_TRAIN_CAP` because the spatial ablation is a GRID: every
#: sensor is retrained at every patch size, so the cost grows with the area of
#: the table rather than the length of the sensor list. 900 is adequate for
#: what this grid measures, for two reasons. First, the quantity of interest is
#: the DIFFERENCE between cells of one row, and every cell is trained at
#: exactly the same size, so the cap moves all cells together and cannot
#: manufacture the ordering the grid exists to measure. Second, the effect
#: being measured is enormous next to the training-size effect: diluting a
#: patch to one percent of a MODIS pixel costs tens of points of accuracy,
#: while the training-size difference between this cap and the 1200-pixel
#: spectral ablation cap moves the 813 arm by a fraction of a point.
SPATIAL_TRAIN_CAP: int = 900

#: Test pixels per cell of the spatial grid. Recall of the cyanobacteria class
#: is estimated on roughly a quarter of these, so 1200 keeps that estimate on a
#: few hundred pixels rather than a few dozen.
SPATIAL_TEST_CAP: int = 1200

SPATIAL_NOTE: str = (
    "Spatial ablation: each bloom pixel is diluted into one sensor pixel by "
    "linear sub-pixel mixing with the clear-water background of its own scene "
    "before the spectra are degraded to that sensor's band set, so a coarse "
    "instrument is scored on the mixed signal it would actually receive from "
    "an intake-scale patch instead of on a patch it is assumed to resolve. "
    "Both the patch spectra and the background come from marsad.synth, our own "
    "forward model, so this is a self-consistency check against a physics-based "
    "simulation, consistent with the Case-2 water literature, and never "
    "independent validation on real Gulf scenes."
)

VERDICT_NOTE: str = (
    "Two axes decide whether a sensor can warn a desalination intake about an "
    "intake-scale bloom. SPECTRAL: does it carry a band within "
    "{tol:.0f} nm of the {pc:.0f} nm phycocyanin absorption, the only pigment "
    "route to cyanobacteria. SPATIAL: does a patch of the reference size fill "
    "at least the threshold fraction of one of its pixels. The spectral axis "
    "is read off the published band table and the spatial axis off the "
    "published ground sampling distance, so both are facts about the "
    "instruments rather than results of our simulation; the one exception is "
    "the 813 ground sampling distance, which is an explicit assumption. What "
    "our simulation supplies is the accuracy each combination actually "
    "achieves, and that part is a self-consistency check on our own forward "
    "model, never independent validation on real Gulf scenes."
).format(tol=PC_BAND_TOLERANCE_NM, pc=PC_BAND_NM)


def nearest_band_distance_nm(
    sensor: sensors.Sensor | str, target_nm: float = PC_BAND_NM
) -> float:
    """Distance in nm from ``target_nm`` to the closest band centre of a sensor.

    The spectral half of the verdict in one number. A sensor cannot measure an
    absorption feature it has no band on top of: the interpolation in
    :func:`marsad.sensors.resample_to_grid` draws a straight line across the
    gap, so a 620 nm dip lying between a 560 nm and a 665 nm band is not
    attenuated, it is absent.
    """
    sen = sensors.SENSORS[sensor] if isinstance(sensor, str) else sensor
    return float(np.min(np.abs(sen.centers_nm - float(target_nm))))


def has_phycocyanin_band(sensor: sensors.Sensor | str) -> bool:
    """True when the sensor carries a band close enough to see phycocyanin.

    "Close enough" is :data:`PC_BAND_TOLERANCE_NM` from :data:`PC_BAND_NM`. Of
    the operational ocean-colour sensors in :data:`marsad.sensors.SENSORS` only
    Sentinel-3 OLCI passes, which is exactly why the spectral ablation on its
    own cannot answer "why not just use free daily OLCI?" and why the spatial
    axis has to be measured alongside it.
    """
    return bool(nearest_band_distance_nm(sensor) <= PC_BAND_TOLERANCE_NM)


def _bloom_recall(pred: np.ndarray, true: np.ndarray) -> float:
    """Of the pixels carrying ANY bloom, the fraction flagged as some bloom.

    Coarser than :func:`_recall` on the cyanobacteria class, and deliberately
    so: this is the detection question, "is there a bloom heading for the
    intake at all", stripped of the speciation question, "which one". A sensor
    can pass this and still fail the operator, because the response to a toxic
    cyanobacteria bloom is not the response to a dinoflagellate one.
    """
    mask = np.asarray(true) != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.asarray(pred)[mask] != 0))


def _false_alarm_rate(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of genuinely bloom-free pixels that were flagged as blooming.

    Reported next to :func:`_bloom_recall` so that a high recall can be read
    for what it is. Recall on its own is trivially gamed by flagging
    everything, and this pair is the check on that. It also documents something
    the measurement actually shows: the diluted arms keep their bloom recall
    high WITHOUT paying in false alarms, which is not the ordinary
    recall-against-precision trade but the constant-background artifact
    described in :func:`_clear_water_background`. Either way, a bloom recall of
    1.000 at a fill fraction of 0.03 is not evidence of a working sensor;
    accuracy and cyanobacteria recall are.
    """
    mask = np.asarray(true) == 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.asarray(pred)[mask] != 0))


def _slice_dataset(dataset: synth.SynthDataset, n_rows: int) -> synth.SynthDataset:
    """First ``n_rows`` pixels of a scene, or the scene itself if it is shorter."""
    n = int(n_rows)
    if n >= dataset.labels.size:
        return dataset
    return synth.SynthDataset(
        rrs_observed=dataset.rrs_observed[:n],
        rrs_true=dataset.rrs_true[:n],
        labels=dataset.labels[:n],
        chl=dataset.chl[:n],
        tss=dataset.tss[:n],
        depth_m=dataset.depth_m[:n],
        phycocyanin=dataset.phycocyanin[:n],
    )


def _clear_water_background(
    rrs_observed: np.ndarray, labels: np.ndarray, what: str
) -> np.ndarray:
    """Mean observed spectrum of the scene's own bloom-free pixels.

    Why this is the right background, rather than an idealised clear-water
    spectrum out of a textbook: the water that shares a pixel with an
    intake-scale bloom patch IS the surrounding water of the same body, with
    the same depth distribution, the same suspended sediment, the same sun
    glint and the same atmospheric residual. The scene's own ``no_bloom``
    pixels are precisely our sample of that water, so mixing into their mean
    dilutes the bloom towards its real neighbourhood. Importing an external
    clear-water spectrum would change the water body at the same time as the
    patch size, and the two effects could no longer be told apart.

    The MEAN rather than a per-pixel draw, for two reasons. It makes the
    dilution deterministic, so a cell of the grid is a function of the patch
    size and the band set alone. And it gives every sensor the identical
    background, so no arm can win or lose on a luckier neighbourhood.

    The cost of that choice is stated plainly here because it shapes the
    numbers, and it flatters the coarse sensors rather than us. A constant
    background means a heavily diluted bloom pixel lands very close to the
    scene mean, closer than a typical genuine ``no_bloom`` pixel, whose depth
    and sediment vary. A classifier can learn that "unnaturally average" is
    itself a cue. That is why the diluted arms hold BLOOM DETECTION recall at
    or near 1.000, and hold it without any rise in false alarms, in the same
    cells where cyanobacteria recall and overall accuracy have already
    collapsed: Sentinel-3 OLCI at a fill fraction of 0.028 keeps a bloom recall
    of 1.000 while its cyanobacteria recall falls to about 0.28. Bloom recall
    must therefore not be quoted at low fill fractions. Accuracy and
    cyanobacteria recall are the metrics that carry the result there, and they
    are the ones the verdict and the headline use.
    """
    mask = np.asarray(labels) == 0
    if not np.any(mask):
        raise ValueError(
            f"the {what} scene has no no_bloom pixels, so it carries no "
            "clear-water background to dilute a patch into; enlarge the scene "
            "or lower the bloom fraction"
        )
    return np.asarray(rrs_observed, dtype=np.float64)[mask].mean(axis=0)


def _instrument_noise(n_rows: int, seed: int) -> np.ndarray:
    """(n_rows, N_BANDS) multiplicative sensor-noise field, drawn once per scene.

    Drawn once and reused for every cell of the grid, so that the only things
    changing from cell to cell are the fill fraction and the band set. Two
    cells with the same fill fraction therefore receive bit-identical inputs,
    which is what makes the memoisation in :func:`run_spatial_ablation` an
    exact reuse rather than an approximation.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n_rows)
    lo, hi = SPATIAL_NOISE_FRAC
    sigma = rng.uniform(lo, hi, size=n)[:, None]
    return 1.0 + sigma * rng.standard_normal((n, N_BANDS))


def _dilute(
    rrs_observed: np.ndarray,
    labels: np.ndarray,
    background: np.ndarray,
    patch_size_m: float,
    sensor_key: str,
    noise: np.ndarray,
) -> np.ndarray:
    """Mix each bloom pixel into one sensor pixel, then re-apply sensor noise.

    Only pixels that carry a bloom are mixed. The bloom is the localised
    feature here: a patch of order a hundred metres across sitting in water
    that is otherwise ordinary. The properties that make a bloom-free pixel
    what it is, depth over a sandy bottom and a resuspended sediment plume,
    vary on kilometre scales instead, so they survive a coarse pixel and there
    is nothing to dilute them into. Mixing them towards the scene mean as well
    would model a coarse sensor as also blind to bathymetry and turbidity,
    which is not the claim under test.

    Re-applying the instrument noise afterwards is NOT cosmetic, it is what
    makes the experiment measure anything at all. Linear mixing against a
    constant background is exactly invertible, ``target = (mixed - (1 - f) *
    background) / f``, and the standardisation in front of both stages would
    undo the factor ``f`` on its own. Without a noise floor that stays put
    while the patch signal shrinks, a fill fraction of 0.0025 would cost
    nothing and this grid would report that pixel size does not matter. The
    physical statement behind the fix is simple: the instrument's noise
    attaches to the PIXEL, not to the patch, so a patch contributing one
    percent of the pixel signal is measured against the full pixel's noise. The
    level, 1-2 % multiplicative, is the level marsad.synth already uses.

    The price is that a full-fill cell of this grid carries one more noise
    realisation than the corresponding row of the spectral-only ``"ablation"``
    block, so the two blocks must not be compared cell for cell. Every cell of
    THIS grid carries that same extra realisation, so comparisons inside the
    grid, which is where the result lives, stay controlled.
    """
    out = np.array(rrs_observed, dtype=np.float64, copy=True)
    bloom = np.asarray(labels) != 0
    if np.any(bloom):
        out[bloom] = sensors.mix_subpixel(
            out[bloom], background, patch_size_m, sensor_key
        )
    return np.clip(out * noise, 0.0, None)


def run_spatial_ablation(
    seed: int = 11,
    n_train: int = SPATIAL_TRAIN_CAP,
    n_test: int = SPATIAL_TEST_CAP,
    patch_sizes_m: Sequence[float] = SPATIAL_PATCH_SIZES_M,
    sensor_keys: Sequence[str] | None = None,
    train: synth.SynthDataset | None = None,
    test: synth.SynthDataset | None = None,
) -> dict:
    """Retrain the MARSAD architecture once per (sensor, bloom patch size) cell.

    The spectral ablation in :func:`run_benchmark` silently assumes every
    sensor resolves the bloom patch. That assumption is false at the scale that
    decides whether an intake gets a warning: OLCI pixels are 300 m and MODIS
    ocean-colour pixels are 1 km, so a 100 m patch is sub-pixel and reaches the
    instrument already mixed with the clear water around it. This function adds
    the missing term, and it cuts in both directions. It is the reason the
    honest answer to "why not just use free daily OLCI?" is not "because it
    cannot see phycocyanin", which would be false, but "because at 300 m it
    sees an intake-scale patch diluted about nine to one".

    Method, per cell
    ----------------
    1. Take the clear-water background as the mean observed spectrum of that
       scene's own bloom-free pixels (:func:`_clear_water_background`).
    2. Dilute every bloom pixel into one sensor pixel with
       :func:`marsad.sensors.mix_subpixel` at this patch size, then re-apply
       the instrument's 1-2 % noise to the mixed pixel (:func:`_dilute`).
    3. Degrade the result to the sensor's band set and lift it back onto the
       813 grid (:func:`_degrade`), so the model input width never changes.
    4. Refit Stage 1 + Stage 2 from scratch (:func:`_fit_pipeline`) against the
       SAME full-resolution clean Stage 1 target every other arm is held to,
       and score the test scene the same way.

    The order matters and is the physical one: the patch is mixed into the
    pixel at the sea surface, and the spectral response function integrates
    whatever arrives at the aperture afterwards.

    The Stage 1 target stays the unmixed, full-resolution clean spectrum for
    every cell, exactly as in the spectral ablation. The product Stage 2
    consumes does not change when the pixel gets coarser, and scoring a diluted
    arm against a diluted target would quietly redefine the deliverable per
    cell and hide the very information loss this grid exists to measure.

    Cells whose fill fraction is identical receive bit-identical inputs by
    construction, because the noise field is drawn once per scene, so the fit
    is computed once and reused. That is memoisation, not approximation: on the
    default grid it turns 25 model fits into 10.

    Parameters
    ----------
    seed:
        Master seed. Scenes generated here use ``seed`` and ``seed + 1``,
        matching :func:`run_benchmark`, and the noise fields use offsets of it.
    n_train, n_test:
        Pixels per cell. Defaults :data:`SPATIAL_TRAIN_CAP` and
        :data:`SPATIAL_TEST_CAP`; when ``train`` or ``test`` is supplied these
        act as caps on the supplied scene instead.
    patch_sizes_m:
        Bloom patch side lengths in metres. Default
        :data:`SPATIAL_PATCH_SIZES_M`.
    sensor_keys:
        Subset of :data:`marsad.sensors.SENSORS` to run. Default: all of them.
    train, test:
        Optional pre-generated scenes, so :func:`run_benchmark` can score this
        grid on the very same pixels as the spectral ablation instead of on an
        independent draw.

    Returns
    -------
    dict
        ``{"patch_sizes_m": [...], "n_train": int, "n_test": int,
        "sensors": {key: {"label", "gsd_m", "n_bands", "has_620nm",
        "by_patch_size": [{"patch_size_m", "fill_fraction", "accuracy",
        "cyano_recall", "bloom_recall", "false_alarm_rate"}, ...]}},
        "note": str}``. ``by_patch_size`` follows the order of
        ``patch_sizes_m``.

    Honesty (binding, docs/CONTRACTS-V2.md): every spectrum mixed here, patch
    and background alike, comes from :mod:`marsad.synth`, our own forward
    model. This grid is a self-consistency check against a physics-based
    simulation, consistent with the Case-2 water literature, and it is never
    independent validation of how OLCI or MODIS behave on real Gulf water. The
    ground sampling distances are published instrument specifications, with the
    single documented exception of the 813 figure
    (:data:`marsad.sensors.ASSUMED_813_GSD_M`, an assumption); the accuracies
    are simulation.
    """
    keys = tuple(sensor_keys) if sensor_keys is not None else tuple(sensors.SENSORS)
    if not keys:
        raise ValueError("sensor_keys is empty: nothing to ablate")
    for key in keys:
        if key not in sensors.SENSORS:
            raise KeyError(
                f"unknown sensor key {key!r}; known keys: {sorted(sensors.SENSORS)}"
            )

    sizes = tuple(float(p) for p in patch_sizes_m)
    if not sizes:
        raise ValueError("patch_sizes_m is empty: nothing to ablate")
    for size in sizes:
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError(
                f"patch_sizes_m must be positive, finite metres, got {size!r}"
            )

    train_scene = (
        _slice_dataset(train, int(n_train))
        if train is not None
        else synth.generate_dataset(int(n_train), seed=seed)
    )
    test_scene = (
        _slice_dataset(test, int(n_test))
        if test is not None
        else synth.generate_dataset(int(n_test), seed=seed + 1)
    )
    n_tr = int(train_scene.labels.size)
    n_te = int(test_scene.labels.size)

    background_train = _clear_water_background(
        train_scene.rrs_observed, train_scene.labels, "training"
    )
    background_test = _clear_water_background(
        test_scene.rrs_observed, test_scene.labels, "test"
    )
    noise_train = _instrument_noise(n_tr, seed=seed + 7001)
    noise_test = _instrument_noise(n_te, seed=seed + 7002)

    out: dict[str, dict] = {}
    for key in keys:
        sensor = sensors.SENSORS[key]
        by_fill: dict[float, dict] = {}
        rows: list[dict] = []
        for size in sizes:
            fill = sensors.subpixel_fill_fraction(size, sensor)
            scored = by_fill.get(fill)
            if scored is None:
                x_train = _degrade(
                    _dilute(
                        train_scene.rrs_observed,
                        train_scene.labels,
                        background_train,
                        size,
                        key,
                        noise_train,
                    ),
                    key,
                )
                x_test = _degrade(
                    _dilute(
                        test_scene.rrs_observed,
                        test_scene.labels,
                        background_test,
                        size,
                        key,
                        noise_test,
                    ),
                    key,
                )
                corrector, classifier = _fit_pipeline(
                    x_train,
                    train_scene.rrs_true,
                    train_scene.labels,
                    train_scene.chl,
                    seed,
                )
                pred = classifier.predict(corrector.transform(x_test))
                scored = {
                    "accuracy": _accuracy(pred, test_scene.labels),
                    "cyano_recall": _recall(pred, test_scene.labels, 2),
                    "bloom_recall": _bloom_recall(pred, test_scene.labels),
                    "false_alarm_rate": _false_alarm_rate(pred, test_scene.labels),
                }
                by_fill[fill] = scored
            rows.append(
                {"patch_size_m": float(size), "fill_fraction": float(fill), **scored}
            )
        out[key] = {
            "label": sensor.label,
            "gsd_m": float(sensor.gsd_m),
            "n_bands": int(sensor.n_bands),
            "has_620nm": has_phycocyanin_band(sensor),
            "by_patch_size": rows,
        }

    return {
        "patch_sizes_m": [float(s) for s in sizes],
        "n_train": n_tr,
        "n_test": n_te,
        "sensors": out,
        "note": SPATIAL_NOTE,
    }


def build_verdict(
    spatial: dict | None = None,
    reference_patch_size_m: float = REFERENCE_PATCH_SIZE_M,
    fill_fraction_threshold: float = FILL_FRACTION_THRESHOLD,
    sensor_keys: Sequence[str] | None = None,
) -> dict:
    """Classify every sensor on the spectral axis and on the spatial axis.

    This is the 2x2 that actually answers "why not just use free daily
    Sentinel-3 OLCI?", a question the spectral ablation on its own invites and
    cannot settle. Both axes are read off published instrument tables rather
    than out of our simulation:

    * SPECTRAL: does the sensor carry a band within
      :data:`PC_BAND_TOLERANCE_NM` of the :data:`PC_BAND_NM` phycocyanin
      absorption, the only pigment route to separating cyanobacteria from
      other blooms (:func:`has_phycocyanin_band`).
    * SPATIAL: does a patch of ``reference_patch_size_m`` fill at least
      ``fill_fraction_threshold`` of one pixel
      (:func:`marsad.sensors.subpixel_fill_fraction`).

    The single assumption in the whole table is the 813 ground sampling
    distance (:data:`marsad.sensors.ASSUMED_813_GSD_M`, 30 m), which is not a
    published figure and is flagged as an assumption wherever it appears.

    When ``spatial`` is a result of :func:`run_spatial_ablation` that includes
    the reference patch size, the measured accuracy and cyanobacteria recall at
    that patch size are attached to each sensor, so the verdict can be read
    against the numbers instead of being taken on geometry alone. Those two
    fields are ``None`` when no such measurement is available.

    Returns
    -------
    dict
        ``{"reference_patch_size_m", "fill_fraction_threshold", "pc_band_nm",
        "pc_band_tolerance_nm", "sensors": {key: {"label", "has_620nm",
        "gsd_m", "fill_fraction_at_reference", "spectral_ok", "spatial_ok",
        "accuracy_at_reference", "cyano_recall_at_reference", "reason"}},
        "adequate_on_both", "spectral_only", "spatial_only",
        "inadequate_on_both", "summary", "note"}``.

    The ``summary`` sentence is generated from the booleans actually computed,
    never asserted, so if a band table or a ground sampling distance changes
    the sentence changes with it.
    """
    keys = tuple(sensor_keys) if sensor_keys is not None else tuple(sensors.SENSORS)
    reference = float(reference_patch_size_m)
    threshold = float(fill_fraction_threshold)

    measured: dict[str, dict] = {}
    if spatial:
        for key, row in spatial.get("sensors", {}).items():
            for cell in row.get("by_patch_size", ()):
                if float(cell["patch_size_m"]) == reference:
                    measured[key] = cell
                    break

    buckets: dict[str, list[str]] = {
        "adequate_on_both": [],
        "spectral_only": [],
        "spatial_only": [],
        "inadequate_on_both": [],
    }
    out: dict[str, dict] = {}
    for key in keys:
        sensor = sensors.SENSORS[key]
        gap = nearest_band_distance_nm(sensor)
        fill = sensors.subpixel_fill_fraction(reference, sensor)
        spectral_ok = bool(gap <= PC_BAND_TOLERANCE_NM)
        spatial_ok = bool(fill >= threshold)

        if spectral_ok:
            spectral_txt = (
                f"carries a band {gap:.0f} nm from the {PC_BAND_NM:.0f} nm "
                "phycocyanin line, so speciation is spectrally possible"
            )
        else:
            spectral_txt = (
                f"has no band nearer than {gap:.0f} nm to {PC_BAND_NM:.0f} nm, "
                "so phycocyanin is interpolated straight over and cyanobacteria "
                "cannot be separated on pigment absorption"
            )
        if spatial_ok:
            spatial_txt = (
                f"a {sensor.gsd_m:.0f} m pixel is filled {fill:.2f} by a "
                f"{reference:.0f} m patch, so the patch dominates its own "
                "measurement"
            )
        else:
            spatial_txt = (
                f"a {sensor.gsd_m:.0f} m pixel is filled only {fill:.3f} by a "
                f"{reference:.0f} m patch, so the reading is dominated by the "
                "surrounding water"
            )
        if spectral_ok and spatial_ok:
            headline, bucket = "adequate on both axes", "adequate_on_both"
        elif spectral_ok:
            headline, bucket = "spectrally adequate, spatially not", "spectral_only"
        elif spatial_ok:
            headline, bucket = "spatially adequate, spectrally not", "spatial_only"
        else:
            headline, bucket = "inadequate on both axes", "inadequate_on_both"
        buckets[bucket].append(key)

        connector = "and" if spectral_ok == spatial_ok else "but"
        cell = measured.get(key)
        out[key] = {
            "label": sensor.label,
            "has_620nm": spectral_ok,
            "gsd_m": float(sensor.gsd_m),
            "fill_fraction_at_reference": float(fill),
            "spectral_ok": spectral_ok,
            "spatial_ok": spatial_ok,
            "accuracy_at_reference": (
                float(cell["accuracy"]) if cell is not None else None
            ),
            "cyano_recall_at_reference": (
                float(cell["cyano_recall"]) if cell is not None else None
            ),
            "reason": f"{headline}: {spectral_txt}, {connector} {spatial_txt}.",
        }

    def _named(bucket: str) -> str:
        names = [out[k]["label"] for k in buckets[bucket]]
        return ", ".join(names) if names else "none"

    summary = (
        f"At a {reference:.0f} m reference patch and a {threshold:.2f} fill "
        f"threshold: adequate on both axes: {_named('adequate_on_both')}; "
        "spectrally adequate but too coarse to resolve the patch: "
        f"{_named('spectral_only')}; fine enough to resolve the patch but blind "
        f"to {PC_BAND_NM:.0f} nm: {_named('spatial_only')}; inadequate on both "
        f"axes: {_named('inadequate_on_both')}."
    )

    return {
        "reference_patch_size_m": reference,
        "fill_fraction_threshold": threshold,
        "pc_band_nm": PC_BAND_NM,
        "pc_band_tolerance_nm": PC_BAND_TOLERANCE_NM,
        "sensors": out,
        **buckets,
        "summary": summary,
        "note": VERDICT_NOTE,
    }


# ------------------------------------------------------------------ experiment


def run_benchmark(
    seed: int = 11,
    n_train: int = 4000,
    n_test: int = 2000,
    *,
    spatial_patch_sizes_m: Sequence[float] = SPATIAL_PATCH_SIZES_M,
    spatial_n_train: int = SPATIAL_TRAIN_CAP,
    spatial_n_test: int = SPATIAL_TEST_CAP,
    reference_patch_size_m: float = REFERENCE_PATCH_SIZE_M,
    fill_fraction_threshold: float = FILL_FRACTION_THRESHOLD,
) -> dict:
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
    5. Run the spatial ablation (:func:`run_spatial_ablation`) on the same
       pixels, which dilutes each bloom pixel into one sensor pixel before
       the band-set degradation, and reduce the spectral and spatial axes
       to the 2x2 verdict (:func:`build_verdict`).

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
    spatial_patch_sizes_m:
        Bloom patch side lengths in metres for the spatial grid. Default
        :data:`SPATIAL_PATCH_SIZES_M`. Shrink it to make a run cheap: the
        grid costs one model fit per distinct (sensor, fill fraction) pair.
    spatial_n_train, spatial_n_test:
        Caps on the pixels each spatial cell trains and scores on. The
        spatial grid reuses the scenes generated above rather than drawing
        its own, so its cells sit on the same pixels as the spectral
        ablation and the two tables are directly comparable.
    reference_patch_size_m, fill_fraction_threshold:
        The two documented thresholds behind the 2x2 verdict: the
        intake-scale patch a sensor is asked to resolve, and the share of
        one pixel it has to fill to count as resolved.

    Returns
    -------
    dict
        The five keys of docs/CONTRACTS-V2.md unchanged: ``chl_retrieval``,
        ``speciation``, ``ablation``, ``headline`` and ``honesty_note``,
        plus two additive blocks introduced after v0.2, ``spatial`` and
        ``verdict``. Nothing in the original five moved or changed meaning.
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

    # --- 6. the spatial ablation and the two-axis verdict -------------------
    # Scored on the SAME scenes as everything above, capped, and on the
    # natural test draw only (rows [:n_base]) for the same reason the
    # spectral ablation uses it: the per-regime top-up must not tilt a
    # scene-level accuracy.
    spatial = run_spatial_ablation(
        seed=seed,
        n_train=min(int(n_train), int(spatial_n_train)),
        n_test=min(int(n_base), int(spatial_n_test)),
        patch_sizes_m=spatial_patch_sizes_m,
        train=train,
        test=_slice_dataset(test, n_base),
    )
    verdict = build_verdict(
        spatial=spatial,
        reference_patch_size_m=reference_patch_size_m,
        fill_fraction_threshold=fill_fraction_threshold,
    )

    return {
        "chl_retrieval": chl_retrieval,
        "speciation": speciation,
        "ablation": ablation,
        "headline": headline,
        "honesty_note": HONESTY_NOTE,
        "spatial": spatial,
        "verdict": verdict,
    }
