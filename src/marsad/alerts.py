"""Alert feed and API payloads - the operator-facing end of the pipeline.

The PRD promises a "dashboard + alert API" as the delivery mechanism: the
science only has value if a desalination plant operator receives a short,
actionable statement before the bloom reaches the intake. This module turns
the pipeline's per-intake risk assessments (``pipeline.run_end_to_end``
output, also serialised to ``outputs/results.json``) into alert records that
``scripts/serve_api.py`` serves as JSON.

Operational rationale
---------------------
* One alert per monitored asset, never per pixel: the unit of operator action
  is "this intake", not "this 30 m pixel".
* ``lead_days`` is the headline operational number - how many days of warning
  the plant gets before the risk score crosses the RED threshold. Zero means
  the asset is already RED (act now); -1 means no RED crossing is forecast
  inside the horizon, so the alert is informational rather than a countdown.
* The alert level comes straight from ``risk.compute_risk_index`` and is never
  recomputed here. Alert policy lives in one place (``risk.py``), so tuning
  the plant's probability-of-detection vs false-alarm trade-off does not need
  a second edit in the delivery layer.
* Every alert carries the risk rationale strings unchanged, because an
  operator who cannot see why the system alerted will not trust it twice.

Scientific honesty (binding, see docs/CONTRACTS-V2.md)
------------------------------------------------------
The scenes behind these alerts come from ``synth.py``, which is our own
physics-based forward model of Gulf Case-2 water. Everything this module emits
is therefore a **synthetic physics-based simulation**, not an observation of
real water and not independent validation of anything. The feed states this in
its ``data_basis`` field so no downstream consumer can mistake a demo alert for
a real one; real validation is the hindcast on documented events once
GLORIA/PACE/813 data lands.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .risk import RED_THRESHOLD, RiskLevel

# Delivery-layer constants. The alert LEVELS and their thresholds are risk.py's
# policy; only the wording and the feed envelope are decided here.
SOURCE = "MARSAD 813"
DATA_BASIS = "synthetic physics-based simulation"

# Ordering used by the ``min_level`` filter. GREEN < AMBER < RED.
LEVEL_ORDER: dict[str, int] = {
    RiskLevel.GREEN.value: 0,
    RiskLevel.AMBER.value: 1,
    RiskLevel.RED.value: 2,
}

# Operator-facing wording for the class labels of synth.LABELS.
_CLASS_PHRASE: dict[str, str] = {
    "no_bloom": "no bloom detected",
    "dinoflagellate": "dinoflagellate bloom (red tide)",
    "cyanobacteria": "cyanobacteria bloom (toxic, phycocyanin)",
}

# lead_days sentinel: no RED crossing anywhere in the forecast horizon.
NO_CROSSING = -1


@dataclass
class Alert:
    """One operator-facing alert for one monitored asset.

    Attributes
    ----------
    intake : monitored asset name, e.g. "Khor Fakkan".
    level : "GREEN" / "AMBER" / "RED" as decided by ``risk.compute_risk_index``.
    score : risk score in [0, 1] backing that level.
    issued_utc : ISO-8601 UTC timestamp of the analysis this alert reports on
        (the pipeline run), NOT the moment the feed happened to be requested.
    bloom_class : dominant class label ("no_bloom"/"dinoflagellate"/"cyanobacteria").
    chl_mg_m3 : estimated chlorophyll-a concentration for the asset's pixels.
    lead_days : days of warning before the forecast crosses the RED threshold;
        0 if the asset is already RED, -1 if no crossing inside the horizon.
    message : single-line operator summary (asset, class, level, lead time).
    rationale : the risk-policy reasons, passed through unchanged.
    """

    intake: str
    level: str
    score: float
    issued_utc: str
    bloom_class: str
    chl_mg_m3: float
    lead_days: int
    message: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable dict (plain str/float/int/list only)."""
        return {
            "intake": str(self.intake),
            "level": str(self.level),
            "score": round(float(self.score), 4),
            "issued_utc": str(self.issued_utc),
            "bloom_class": str(self.bloom_class),
            "chl_mg_m3": round(float(self.chl_mg_m3), 3),
            "lead_days": int(self.lead_days),
            "message": str(self.message),
            "rationale": [str(r) for r in self.rationale],
        }


def _level_str(level: Any) -> str:
    """Normalise a level given as str or RiskLevel to its canonical string."""
    value = getattr(level, "value", level)
    text = str(value).strip().upper()
    if text not in LEVEL_ORDER:
        raise ValueError(
            f"unknown alert level {level!r}; expected one of {sorted(LEVEL_ORDER)}"
        )
    return text


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lead_days_to_red(intake: dict) -> int:
    """Days of warning before this asset's forecast crosses the RED threshold.

    Returns 0 when the asset is already RED (including a precautionary
    AMBER -> RED promotion, whose score can sit below ``RED_THRESHOLD`` - the
    operator is on alert now either way), the ``day`` of the first forecast
    entry whose mean score reaches ``risk.RED_THRESHOLD``, or ``NO_CROSSING``
    (-1) when no such day exists inside the forecast horizon.

    The mean forecast track is used rather than its upper band: the band is
    already folded into the level through the precautionary uncertainty rule
    in ``risk.compute_risk_index``, and counting the 90th-percentile track here
    would double-count the same uncertainty into the lead time.
    """
    risk = intake.get("risk", {})
    level = str(risk.get("level", RiskLevel.GREEN.value)).upper()
    score = float(risk.get("score", 0.0))
    if level == RiskLevel.RED.value or score >= RED_THRESHOLD:
        return 0

    for step in intake.get("forecast", []) or []:
        if float(step.get("score", 0.0)) >= RED_THRESHOLD:
            return int(step.get("day", NO_CROSSING))
    return NO_CROSSING


def _format_message(intake_name: str, level: str, bloom_class: str, score: float,
                    chl_mg_m3: float, lead_days: int, horizon_days: int) -> str:
    """Build the one-line operator summary carried by every alert.

    Contains the four things a control-room operator needs before opening the
    dashboard: which asset, what is blooming, how bad it is now, and how long
    they have before it is RED.
    """
    phrase = _CLASS_PHRASE.get(bloom_class, str(bloom_class).replace("_", " "))
    head = (f"{level} at {intake_name}: {phrase}, risk {score:.2f}, "
            f"chl-a {chl_mg_m3:.1f} mg/m3")
    if lead_days == 0:
        tail = "RED threshold already reached, act on the intake now"
    elif lead_days > 0:
        day_word = "day" if lead_days == 1 else "days"
        tail = f"forecast reaches RED in {lead_days} {day_word}"
    else:
        tail = f"no RED crossing within the {horizon_days}-day forecast"
    return f"{head} - {tail}."


def _alert_from_intake(intake: dict, issued_utc: str) -> Alert:
    """Build one Alert from one ``results["intakes"]`` record."""
    risk = intake.get("risk", {})
    bloom = intake.get("bloom", {})
    forecast = intake.get("forecast", []) or []

    name = str(intake.get("name", "unknown intake"))
    level = _level_str(risk.get("level", RiskLevel.GREEN.value))
    score = float(risk.get("score", 0.0))
    bloom_class = str(bloom.get("dominant", "no_bloom"))
    chl = float(bloom.get("chl_mg_m3", 0.0))
    lead = lead_days_to_red(intake)

    rationale = [str(r) for r in risk.get("rationale", [])]
    # v0.2 pipeline attaches per-intake uncertainty; when it flags a review the
    # operator must see that on the alert itself, not only on the dashboard.
    unc = intake.get("uncertainty")
    if isinstance(unc, dict) and unc.get("review_recommended"):
        rationale.append(
            "model uncertainty above the review threshold - confirm with a "
            "human analyst before acting")

    message = _format_message(name, level, bloom_class, score, chl, lead,
                              horizon_days=len(forecast))
    return Alert(intake=name, level=level, score=score, issued_utc=issued_utc,
                 bloom_class=bloom_class, chl_mg_m3=chl, lead_days=lead,
                 message=message, rationale=rationale)


def _intake_records(results: dict) -> list[dict]:
    """Validate and return ``results["intakes"]`` with an actionable error."""
    if not isinstance(results, dict) or "intakes" not in results:
        raise ValueError(
            "results dict has no 'intakes' key: pass the output of "
            "marsad.pipeline.run_end_to_end (or the parsed "
            "outputs/results.json written by scripts/run_demo.py)")
    intakes = results["intakes"]
    if not isinstance(intakes, Iterable) or isinstance(intakes, (str, bytes)):
        raise ValueError("results['intakes'] must be a list of intake records")
    return list(intakes)


def alerts_from_results(results: dict, min_level: str = "AMBER") -> list[Alert]:
    """Emit one Alert per monitored asset at or above ``min_level``.

    Parameters
    ----------
    results : the pipeline data dict (``run_end_to_end`` return value or the
        parsed ``outputs/results.json``).
    min_level : "GREEN", "AMBER" or "RED" (a ``RiskLevel`` is also accepted).
        Default "AMBER": GREEN assets are healthy and are reported through the
        feed counts rather than as alerts, so the feed stays actionable.

    Returns
    -------
    list[Alert] sorted most urgent first: level descending, then shortest
    lead time to RED, then score descending. A control room reads top down.
    """
    floor = LEVEL_ORDER[_level_str(min_level)]
    issued_utc = str(results.get("generated_utc") or _now_utc())

    alerts = [
        _alert_from_intake(intake, issued_utc)
        for intake in _intake_records(results)
        if LEVEL_ORDER[_level_str(intake.get("risk", {}).get("level", "GREEN"))] >= floor
    ]
    # Sort key: urgency. lead_days -1 (no crossing) must rank AFTER any real
    # countdown, so it is mapped to +infinity rather than compared as -1.
    alerts.sort(key=lambda a: (
        -LEVEL_ORDER[a.level],
        float("inf") if a.lead_days < 0 else a.lead_days,
        -a.score,
    ))
    return alerts


def alert_feed(results: dict) -> dict:
    """Full alert feed payload served by ``GET /v1/alerts``.

    Returns EXACTLY::

        {"generated_utc": str, "source": "MARSAD 813",
         "data_basis": "synthetic physics-based simulation",
         "counts": {"RED": int, "AMBER": int, "GREEN": int},
         "alerts": [alert.to_dict(), ...]}

    ``counts`` covers EVERY monitored asset (GREEN included) so a dashboard can
    show "1 RED, 1 AMBER, 2 GREEN" while ``alerts`` carries only the records
    that need an operator response. ``data_basis`` is not decoration: this feed
    is produced from our own physics-based simulation of Gulf Case-2 water, so
    it is a self-consistency demonstration and never an observation of a real
    bloom.
    """
    intakes = _intake_records(results)
    counts = {RiskLevel.RED.value: 0, RiskLevel.AMBER.value: 0, RiskLevel.GREEN.value: 0}
    for intake in intakes:
        counts[_level_str(intake.get("risk", {}).get("level", "GREEN"))] += 1

    return {
        # The analysis timestamp, not the request time: a feed that reports
        # "now" while serving a three-day-old pipeline run would misrepresent
        # how fresh the assessment is.
        "generated_utc": str(results.get("generated_utc") or _now_utc()),
        "source": SOURCE,
        "data_basis": DATA_BASIS,
        "counts": counts,
        "alerts": [a.to_dict() for a in alerts_from_results(results)],
    }
