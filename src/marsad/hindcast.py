"""Event-based hindcast harness - the framework for MARSAD's real validation.

The operational question this module answers is: **how many days of warning
would MARSAD have given before a documented harmful-algal-bloom impact?**
Lead time - days between the first alert and the reported impact date - is the
headline metric a desalination-intake operator cares about. Per-pixel accuracy
on a random scene is not: a model that is right on average but silent until the
day the plant shuts down is operationally worthless.

SCIENTIFIC HONESTY (binding, see docs/CONTRACTS-V2.md)
------------------------------------------------------
The daily spectra used here are NOT archived satellite observations. They are
SIMULATED with :mod:`marsad.synth`, which is our own physics-based forward
model, and are merely *shaped* to published descriptions of the documented
events (bloom type, onset timing relative to the reported impact). Running
MARSAD over them is therefore a self-consistency check against our own
simulation - never independent validation of real Gulf water - and the lead
times reported below are as much a property of the simulated timeline as of
the model.

The real validation step is to replace :func:`simulate_event_timeseries` with
archived scenes over the documented dates and coordinates (Sentinel-3 OLCI and
Sentinel-2 MSI for the historical window, EnMAP / PACE for present-day
hyperspectral coverage, GLORIA in-situ matchups for the water-leaving
reference) and then to rerun :func:`run_hindcast` unchanged. That swap is the
whole point of the module's structure. It is on the roadmap and it is NOT done
yet. No result from this module may be reported as "MARSAD was validated on
the 2008 Gulf of Oman red tide".

Design
------
* :class:`BloomEvent` is a documented, citable event record.
* :func:`simulate_event_timeseries` turns one event into a daily time series:
  a bloom that grows from clear water (logistic / Verhulst growth, nutrient
  limited at the top) and progressively fills the pixel, mixed with the local
  background water by a linear sub-pixel areal mixing model - the standard
  ocean-colour treatment of a partially covered pixel.
* :func:`evaluate_lead_time` scores an alert sequence the way an operator
  would: first warning, first red alert, days of warning, and how often the
  system cried wolf while the water was genuinely clear.
* :func:`run_hindcast` wires the trained Stage 1 + Stage 2 pipeline and the
  risk policy over every documented event.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np

from marsad import synth
from marsad.risk import compute_risk_index
from marsad.spectra import N_BANDS
from marsad.stage1_correction import ShallowWaterCorrector
from marsad.stage2_classifier import BloomClassifier

HONESTY_NOTE = (
    "Event timelines are SIMULATED with marsad.synth, our own physics-based "
    "forward model, and only shaped to published descriptions of the "
    "documented events (bloom type and onset timing before the reported "
    "impact date). This is a self-consistency check against our own "
    "simulation, not independent validation on real Gulf water. The actual "
    "validation step - replacing simulate_event_timeseries with archived "
    "Sentinel-3 OLCI / Sentinel-2 MSI and EnMAP / PACE scenes over the "
    "documented dates and rerunning this harness unchanged - is on the "
    "roadmap and has NOT been done yet."
)

# --- TEAM DECISION ---
# Hindcast scenario constants. These are operator/scenario assumptions, not
# retrieved quantities: the hindcast asks "would we have raised the alarm on
# the water that actually reached this intake", so the detected patch is
# treated as local throughout the window and its distance is not itself being
# hindcast. Tune with the plant, not with the model.
PATCH_DISTANCE_KM = 3.0        # bloom patch assumed inside the intake's 5 km ring
TREND_LOOKBACK_DAYS = 3        # window for the daily risk-trend estimate
HINDCAST_WINDOW_DAYS = 45      # days of imagery ending on the impact date
DAILY_NOISE_FRACTION = 0.01    # 1 % per-acquisition multiplicative noise
PATCHINESS_SIGMA = 0.12        # lognormal day-to-day jitter on bloom coverage
# --- END TEAM DECISION ---

_ALERT_ORDER = {"GREEN": 0, "AMBER": 1, "RED": 2}

# Keywords that mark an event as an inland cyanobacteria case (label 2).
# Everything else is treated as a marine dinoflagellate red tide (label 1),
# which is the class of every documented Gulf / Sea of Oman event we cite.
_CYANO_KEYWORDS = (
    "cyano", "microcystis", "phycocyanin", "reservoir", "lake", "dam",
    "inland", "freshwater",
)


@dataclass
class BloomEvent:
    """One documented harmful-algal-bloom event with a reported impact.

    Attributes
    ----------
    name:
        Event name. Any event that is not a properly documented, citable
        incident MUST carry the word ILLUSTRATIVE here.
    region:
        Water body and affected assets.
    impact_date:
        ISO date (YYYY-MM-DD) of the reported operational impact - the day the
        hindcast counts back from.
    source:
        Literature or agency citation. The credibility of the whole hindcast
        rests on this string being real and checkable.
    onset_lead_days:
        Days between local bloom onset and the reported impact. This is a
        MODELLING choice consistent with the published description of the
        event, not an instrument-derived onset timestamp.
    """

    name: str
    region: str
    impact_date: str
    source: str
    onset_lead_days: int

    def __post_init__(self) -> None:
        # Fail loudly on a malformed record rather than silently hindcasting
        # against a date nobody can look up.
        date.fromisoformat(self.impact_date)
        self.onset_lead_days = int(self.onset_lead_days)
        if self.onset_lead_days < 1:
            raise ValueError("onset_lead_days must be >= 1")


DOCUMENTED_EVENTS: tuple[BloomEvent, ...] = (
    BloomEvent(
        name="2008-2009 Cochlodinium polykrikoides red tide, Sea of Oman",
        region=(
            "Gulf of Oman / Sea of Oman - UAE east coast "
            "(Fujairah, Khor Fakkan, Kalba) and Strait of Hormuz"
        ),
        impact_date="2008-11-15",
        source=(
            "Richlen, M.L., Morton, S.L., Jamali, E.A., Rajan, A. and "
            "Anderson, D.M. (2010), 'The catastrophic 2008-2009 red tide in "
            "the Arabian Gulf region, with observations on the identification "
            "and phylogeny of the fish-killing dinoflagellate Cochlodinium "
            "polykrikoides', Harmful Algae 9(2), 163-172, "
            "doi:10.1016/j.hal.2009.08.013. The bloom fouled intake filters "
            "and forced reverse-osmosis desalination plants on the UAE east "
            "coast to cut output or shut down, alongside mass fish kills and "
            "reef damage. NOTE: 2008-11-15 is a representative date inside "
            "the reported autumn-2008 shutdown period, not a single "
            "agency-confirmed timestamp - published accounts give the period, "
            "not per-plant hours."
        ),
        onset_lead_days=28,
    ),
    BloomEvent(
        name=(
            "ILLUSTRATIVE: inland reservoir cyanobacteria bloom "
            "(Hatta Dam analogue)"
        ),
        region=(
            "Inland freshwater reservoir, UAE (Hatta Dam analogue) - "
            "ILLUSTRATIVE, not a reported incident"
        ),
        impact_date="2026-07-20",
        source=(
            "ILLUSTRATIVE SCENARIO - NO citation, because we know of no "
            "documented UAE reservoir cyanobacteria shutdown to cite. The "
            "timeline is shaped after published Microcystis drinking-water "
            "events elsewhere (for example the Toledo, Ohio 'do not drink' "
            "advisory of 2-4 August 2014 on western Lake Erie). It exists to "
            "exercise the phycocyanin / cyanobacteria path of the pipeline "
            "end to end. It is NOT evidence of anything and must never be "
            "presented as a validated case."
        ),
        onset_lead_days=18,
    ),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _event_bloom_class(event: BloomEvent) -> int:
    """Bloom class (1 dinoflagellate, 2 cyanobacteria) implied by an event.

    :class:`BloomEvent` deliberately carries no class field - the class is a
    property of the organism named in the citation - so it is inferred from
    the event text: inland/freshwater wording means the cyanobacteria class
    (the phycocyanin-at-620-nm path), anything else means a marine
    dinoflagellate red tide.
    """
    text = f"{event.name} {event.region}".lower()
    return 2 if any(word in text for word in _CYANO_KEYWORDS) else 1


def _normalise_level(level) -> str:
    """Accept a ``RiskLevel``, its value, or a plain string; return the name."""
    raw = getattr(level, "value", level)
    text = str(raw).strip().upper()
    if text not in _ALERT_ORDER:
        raise ValueError(
            f"unknown alert level {level!r}; expected one of {sorted(_ALERT_ORDER)}"
        )
    return text


def _round(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate_event_timeseries(event: BloomEvent, n_days: int = 45,
                              seed: int = 0) -> dict:
    """Simulate the daily spectra an operator would have seen before an event.

    NOT archived imagery. The spectra come from :mod:`marsad.synth`, our own
    forward model, and are shaped to the published description of ``event``.
    Replacing this function with archived Sentinel/EnMAP scenes over
    ``event.impact_date`` is the real validation step (see module docstring).

    Model
    -----
    The window ends on the impact date (``day = 0``); ``day = -k`` is *k* days
    of warning. Before local onset (``day < -event.onset_lead_days``) the site
    is genuinely clear water: each day draws an independent background pixel
    from the synthetic no-bloom population, so the pre-onset period carries
    realistic turbid/shallow/glint variability and a false alarm there is a
    real false alarm, not a modelling artefact.

    From onset onward a bloom fills the pixel following logistic (Verhulst)
    growth - exponential division limited by nutrient depletion, the standard
    phytoplankton growth curve - with lognormal day-to-day jitter for
    patchiness and advection. The observed spectrum is the linear sub-pixel
    areal mixture of that day's background water and a dense-bloom end member
    of the event's class, which is the standard ocean-colour treatment of a
    partially covered pixel; chlorophyll mixes by the same areal fraction.

    Parameters
    ----------
    event:
        The documented event to reconstruct.
    n_days:
        Length of the daily window ending on the impact date (>= 5).
    seed:
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``{"days": (n,) int array relative to impact (negative = before),
        "rrs_observed": (n, N_BANDS), "labels": (n,), "chl": (n,)}``, where
        ``labels`` is the TRUE state (0 before onset, the event's bloom class
        from onset onward) used to score false alarms.
    """
    n_days = int(n_days)
    if n_days < 5:
        raise ValueError("n_days must be >= 5 to contain a usable window")

    rng = np.random.default_rng(seed)
    days = np.arange(-(n_days - 1), 1, dtype=int)
    bloom_class = _event_bloom_class(event)

    # Keep at least two clear days inside the window so the false-alarm rate
    # is defined even for events whose published onset predates the window.
    onset = int(np.clip(event.onset_lead_days, 1, n_days - 3))

    # Background and bloom end members come from the committed forward model
    # so the hindcast sees exactly the contamination physics (bottom, TSS,
    # glint, noise) that Stage 1 was trained to remove.
    pool_size = max(400, 12 * n_days)
    pool = synth.generate_dataset(pool_size, seed=seed + 991)
    clear_idx = np.flatnonzero(pool.labels == 0)
    bloom_idx = np.flatnonzero(pool.labels == bloom_class)
    if clear_idx.size == 0 or bloom_idx.size == 0:
        raise RuntimeError("synthetic pool lacks the classes needed for this event")

    # One independent background pixel per day (day-to-day water variability).
    bg = rng.choice(clear_idx, size=n_days, replace=clear_idx.size < n_days)
    bg_rrs = pool.rrs_observed[bg]
    bg_chl = pool.chl[bg]

    # Dense-bloom end member: the 75th percentile of the class chlorophyll
    # distribution - a firmly dense bloom, not a freak outlier.
    order = bloom_idx[np.argsort(pool.chl[bloom_idx])]
    peak_i = int(order[min(order.size - 1, int(0.75 * order.size))])
    bloom_rrs = pool.rrs_observed[peak_i]
    bloom_chl = float(pool.chl[peak_i])

    # Logistic coverage: 0 before onset, rising to full cover on impact day.
    frac = np.zeros(n_days, dtype=float)
    growing = days >= -onset
    t = (days[growing] + onset + 1.0) / (onset + 1.0)  # (0, 1]
    k_growth, t_mid = 9.0, 0.5
    raw = 1.0 / (1.0 + np.exp(-k_growth * (t - t_mid)))
    raw0 = 1.0 / (1.0 + np.exp(k_growth * t_mid))
    raw1 = 1.0 / (1.0 + np.exp(-k_growth * (1.0 - t_mid)))
    frac[growing] = (raw - raw0) / (raw1 - raw0)
    # Patchiness / advection: the slick is not a smooth ramp day to day.
    frac[growing] *= np.exp(PATCHINESS_SIGMA * rng.standard_normal(int(growing.sum())))
    frac = np.clip(frac, 0.0, 1.0)

    mix = frac[:, None]
    rrs = (1.0 - mix) * bg_rrs + mix * bloom_rrs[None, :]
    # Each day is a separate acquisition: add its own small sensor noise.
    rrs = rrs * (1.0 + DAILY_NOISE_FRACTION * rng.standard_normal(rrs.shape))
    rrs = np.clip(rrs, 0.0, None)

    chl = (1.0 - frac) * bg_chl + frac * bloom_chl
    labels = np.where(days >= -onset, bloom_class, 0).astype(int)

    if rrs.shape != (n_days, N_BANDS):
        raise RuntimeError(f"simulated window has shape {rrs.shape}, expected "
                           f"{(n_days, N_BANDS)} on the 813 band grid")

    return {
        "days": days,
        "rrs_observed": np.asarray(rrs, dtype=np.float64),
        "labels": labels,
        "chl": np.asarray(chl, dtype=np.float64),
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def evaluate_lead_time(alert_levels: list[str], days: np.ndarray,
                       labels: np.ndarray | None = None) -> dict:
    """Score an alert sequence the way an intake operator would read it.

    Lead time is the headline operational metric: days of warning between the
    first alert and the reported impact (``day = 0``). An alert is any level
    at or above AMBER - AMBER is when the operator starts pre-treating and
    staging, which is exactly the moment the warning became useful, so a RED
    day also counts as an amber-level crossing.

    Parameters
    ----------
    alert_levels:
        Per-day levels, ``"GREEN" | "AMBER" | "RED"`` (``RiskLevel`` members
        are accepted too), in the same order as ``days``.
    days:
        Day index relative to the impact date; negative = before impact.
    labels:
        Optional per-day TRUE labels (0 = no_bloom). Needed to count false
        alarms; without them no day can honestly be called a false alarm and
        ``false_alarm_days`` is reported as 0.

    Returns
    -------
    dict
        ``{"first_amber_day", "first_red_day", "lead_days_amber",
        "lead_days_red", "false_alarm_days"}``. Day and lead entries are
        ``None`` when the level never fired - a system that stays silent has
        no lead time, and reporting 0 would read as "warned on the day".
        ``lead_days = -first_alert_day``, so a value <= 0 means the alert came
        on or after the impact date: too late to act on.

    Notes
    -----
    The lead times count the first alert of ANY kind, including one raised
    while the water was still clear, so they must always be read together with
    ``false_alarm_days``. :func:`run_hindcast` reports the stricter
    ``lead_days_true_alert`` (first alert raised while a bloom was genuinely
    present) alongside them for exactly this reason.
    """
    levels = [_normalise_level(x) for x in alert_levels]
    day_arr = np.asarray(days).astype(int).ravel()
    if len(levels) != day_arr.size:
        raise ValueError(
            f"alert_levels ({len(levels)}) and days ({day_arr.size}) must match"
        )

    ranks = np.array([_ALERT_ORDER[x] for x in levels], dtype=int)
    amber = ranks >= _ALERT_ORDER["AMBER"]
    red = ranks >= _ALERT_ORDER["RED"]

    first_amber = int(day_arr[amber][0]) if amber.any() else None
    first_red = int(day_arr[red][0]) if red.any() else None

    false_alarm_days = 0
    if labels is not None:
        lab = np.asarray(labels).astype(int).ravel()
        if lab.size != day_arr.size:
            raise ValueError(
                f"labels ({lab.size}) and days ({day_arr.size}) must match"
            )
        false_alarm_days = int(np.count_nonzero(amber & (lab == 0)))

    return {
        "first_amber_day": first_amber,
        "first_red_day": first_red,
        "lead_days_amber": None if first_amber is None else int(-first_amber),
        "lead_days_red": None if first_red is None else int(-first_red),
        "false_alarm_days": false_alarm_days,
    }


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def run_hindcast(seed: int = 5, n_train: int = 3000) -> dict:
    """Run the trained pipeline over every documented event and score lead time.

    Trains Stage 1 (shallow-water correction) and Stage 2 (bloom detection and
    speciation) on a fresh synthetic scene, then for each event walks the
    simulated daily window through correction, classification and the
    :mod:`marsad.risk` policy, and scores the resulting alert sequence with
    :func:`evaluate_lead_time`.

    The risk trend fed to the policy is the slope of the system's own recent
    risk scores (a two-pass computation: trend-free scores first, then the
    trend folded back in), matching :mod:`marsad.pipeline`. Assessment
    uncertainty is the classifier ambiguity ``1 - 2|p_bloom - 0.5|``.

    Parameters
    ----------
    seed:
        Master seed; every draw derives from it deterministically.
    n_train:
        Training pixels for Stage 1 + Stage 2 (tests use a few hundred).

    Returns
    -------
    dict
        ``{"events": [{"event": {...}, "metrics": {...}}, ...],
        "honesty_note": str}``, JSON-serialisable. ``metrics`` carries the
        :func:`evaluate_lead_time` keys plus the daily series and summary
        numbers. Read ``honesty_note`` before quoting any number from it: the
        timelines are simulated, so this is a self-consistency check, not
        validation on real scenes.
    """
    train = synth.generate_dataset(int(n_train), seed=seed)
    corrector = ShallowWaterCorrector(seed=seed)
    corrector.fit(train.rrs_observed, train.rrs_true)
    # Stage 2 is trained on CORRECTED spectra because that is all it ever sees
    # in operations (same convention as marsad.pipeline).
    classifier = BloomClassifier(seed=seed)
    classifier.fit(corrector.transform(train.rrs_observed), train.labels, train.chl)

    events_out: list[dict] = []
    for k, event in enumerate(DOCUMENTED_EVENTS):
        ts = simulate_event_timeseries(event, n_days=HINDCAST_WINDOW_DAYS,
                                       seed=seed + 10 + k)
        days = ts["days"]
        true_labels = ts["labels"]

        corrected = corrector.transform(ts["rrs_observed"])
        probs = classifier.predict_proba(corrected)
        chl_est = classifier.estimate_chl(corrected)
        bloom_prob = 1.0 - probs[:, 0]
        # Ambiguity, not spread: ~0 when the classifier is decided either way.
        ambiguity = 1.0 - 2.0 * np.abs(bloom_prob - 0.5)

        # Pass 1: trend-free scores (the trend is defined by this series).
        base = np.array([
            compute_risk_index(float(bp), float(c), 0.0, PATCH_DISTANCE_KM,
                               float(u)).score
            for bp, c, u in zip(bloom_prob, chl_est, ambiguity)
        ])
        # Pass 2: fold the recovered daily trend back into the assessment.
        trend = np.zeros_like(base)
        for i in range(1, base.size):
            j = max(0, i - TREND_LOOKBACK_DAYS)
            trend[i] = (base[i] - base[j]) / (i - j)

        assessments = [
            compute_risk_index(float(bp), float(c), float(tr),
                               PATCH_DISTANCE_KM, float(u))
            for bp, c, tr, u in zip(bloom_prob, chl_est, trend, ambiguity)
        ]
        levels = [a.level.value for a in assessments]
        scores = np.array([a.score for a in assessments])

        metrics = evaluate_lead_time(levels, days, true_labels)

        bloom_mask = true_labels != 0
        alerted = np.array([_ALERT_ORDER[x] >= 1 for x in levels], dtype=bool)
        mean_bloom_probs = probs[bloom_mask].mean(axis=0)

        # Honest companion to lead_days_amber: the contracted lead time counts
        # the FIRST alert of any kind, so a false alarm on clear water would
        # inflate it. This one counts only the first alert raised while a
        # bloom was genuinely present. Read the pair together.
        true_alert = alerted & bloom_mask
        first_true = int(days[true_alert][0]) if true_alert.any() else None

        metrics.update({
            "first_true_alert_day": first_true,
            "lead_days_true_alert": None if first_true is None else int(-first_true),
            "n_days": int(days.size),
            "n_clear_days": int(np.count_nonzero(~bloom_mask)),
            "onset_day": int(days[bloom_mask][0]),
            "peak_risk_score": _round(scores.max()),
            "peak_chl_est_mg_m3": _round(chl_est.max(), 3),
            "true_chl_peak_mg_m3": _round(float(ts["chl"].max()), 3),
            "bloom_day_detection_rate": _round(
                float(np.mean(alerted[bloom_mask])) if bloom_mask.any() else 0.0),
            "true_class": synth.LABELS[_event_bloom_class(event)],
            "detected_class": synth.LABELS[1 + int(np.argmax(mean_bloom_probs[1:]))],
            "series": {
                "days": [int(d) for d in days],
                "true_label": [int(v) for v in true_labels],
                "bloom_prob": [_round(v) for v in bloom_prob],
                "chl_est_mg_m3": [_round(v, 3) for v in chl_est],
                "risk_score": [_round(v) for v in scores],
                "alert_level": list(levels),
            },
        })

        events_out.append({"event": asdict(event), "metrics": metrics})

    return {"events": events_out, "honesty_note": HONESTY_NOTE}


__all__ = [
    "BloomEvent",
    "DOCUMENTED_EVENTS",
    "HONESTY_NOTE",
    "evaluate_lead_time",
    "run_hindcast",
    "simulate_event_timeseries",
]
