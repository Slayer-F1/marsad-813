"""Tests for the event-based hindcast harness (src/marsad/hindcast.py).

Seeded and small: the one full `run_hindcast` call uses a few hundred training
pixels so the file stays inside a few seconds. Model quality is asserted in the
per-stage tests; what matters here is the operational contract - schema, a
positive lead time on the documented event, false alarms counted on the clear
pre-onset days, and no crash when the system never alerts.

Reminder (the module says the same in prose): the timelines under test are
SIMULATED by our own forward model, so these assertions check the harness, not
the real-world skill of MARSAD.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from marsad.hindcast import (
    DOCUMENTED_EVENTS,
    BloomEvent,
    evaluate_lead_time,
    run_hindcast,
    simulate_event_timeseries,
)
from marsad.risk import RiskLevel
from marsad.spectra import N_BANDS

LEAD_KEYS = {"first_amber_day", "first_red_day", "lead_days_amber",
             "lead_days_red", "false_alarm_days"}
EVENT_KEYS = {"name", "region", "impact_date", "source", "onset_lead_days"}

# The documented Gulf event is the headline case and must come first.
DOCUMENTED = DOCUMENTED_EVENTS[0]


# --------------------------------------------------------------- event records
def test_documented_events_are_well_formed():
    assert isinstance(DOCUMENTED_EVENTS, tuple) and DOCUMENTED_EVENTS
    for event in DOCUMENTED_EVENTS:
        assert isinstance(event, BloomEvent)
        assert event.onset_lead_days >= 1
        assert len(event.source) > 40, "a citation must actually cite something"
        # Honesty gate: an event without a real citation must be flagged as
        # ILLUSTRATIVE in BOTH its name and its source, so it can never be
        # quoted as evidence by accident.
        if "ILLUSTRATIVE" in event.source.upper():
            assert "ILLUSTRATIVE" in event.name.upper()


def test_cochlodinium_gulf_of_oman_event_is_present():
    """The 2008-2009 Sea of Oman red tide that shut UAE east-coast plants."""
    assert DOCUMENTED.impact_date == "2008-11-15"
    assert "oman" in DOCUMENTED.region.lower()
    text = f"{DOCUMENTED.name} {DOCUMENTED.source}".lower()
    assert "cochlodinium" in text
    assert "richlen" in text and "doi:" in text
    assert "ILLUSTRATIVE" not in DOCUMENTED.name.upper()


# ------------------------------------------------------------------ simulation
def test_simulate_event_timeseries_schema_and_growth():
    n_days = 45
    ts = simulate_event_timeseries(DOCUMENTED, n_days=n_days, seed=1)

    assert set(ts) == {"days", "rrs_observed", "labels", "chl"}
    days, rrs, labels, chl = ts["days"], ts["rrs_observed"], ts["labels"], ts["chl"]

    # Window ends on the impact date; day -k means k days of warning.
    assert days.shape == (n_days,)
    assert np.issubdtype(days.dtype, np.integer)
    assert days[-1] == 0
    assert np.all(np.diff(days) == 1)

    assert rrs.shape == (n_days, N_BANDS)
    assert np.all(np.isfinite(rrs)) and np.all(rrs >= 0.0)
    assert chl.shape == (n_days,) and labels.shape == (n_days,)

    # True state: clear water before local onset, the event's class after it.
    onset = DOCUMENTED.onset_lead_days
    clear = days < -onset
    bloom = ~clear
    assert clear.sum() >= 2, "need clear days to score false alarms against"
    assert set(np.unique(labels[clear])) == {0}
    assert set(np.unique(labels[bloom])) == {1}, "marine event = dinoflagellate"

    # The bloom actually develops: background chl, dense bloom by impact day.
    assert chl[clear].mean() < 5.0
    assert chl[-1] > 10.0
    assert chl[bloom].mean() > chl[clear].mean()
    # Clear days are independent water samples, not one frozen spectrum.
    assert rrs[clear].std(axis=0).max() > 0.0


def test_simulate_event_timeseries_is_seeded_and_class_aware():
    a = simulate_event_timeseries(DOCUMENTED, n_days=20, seed=4)
    b = simulate_event_timeseries(DOCUMENTED, n_days=20, seed=4)
    c = simulate_event_timeseries(DOCUMENTED, n_days=20, seed=5)
    assert np.array_equal(a["rrs_observed"], b["rrs_observed"])
    assert not np.array_equal(a["rrs_observed"], c["rrs_observed"])

    # An inland reservoir event exercises the cyanobacteria class instead.
    inland = simulate_event_timeseries(DOCUMENTED_EVENTS[1], n_days=25, seed=2)
    assert set(np.unique(inland["labels"])) == {0, 2}

    # A window shorter than the published onset still leaves clear days.
    short = simulate_event_timeseries(DOCUMENTED, n_days=8, seed=6)
    assert (short["labels"] == 0).sum() >= 2
    assert (short["labels"] != 0).sum() >= 1


# ----------------------------------------------------------------- lead time
def test_evaluate_lead_time_scores_a_known_sequence():
    days = np.array([-5, -4, -3, -2, -1, 0])
    levels = ["GREEN", "GREEN", "AMBER", "AMBER", "RED", "RED"]
    labels = np.array([0, 0, 0, 1, 1, 1])  # bloom truly starts on day -2

    out = evaluate_lead_time(levels, days, labels)
    assert set(out) == LEAD_KEYS
    assert out["first_amber_day"] == -3
    assert out["first_red_day"] == -1
    assert out["lead_days_amber"] == 3      # days of warning before impact
    assert out["lead_days_red"] == 1
    # The amber on day -3 fired while the water was genuinely clear.
    assert out["false_alarm_days"] == 1

    # Without truth labels no day can honestly be called a false alarm.
    assert evaluate_lead_time(levels, days)["false_alarm_days"] == 0


def test_evaluate_lead_time_handles_a_silent_system():
    days = np.arange(-9, 1)
    out = evaluate_lead_time(["GREEN"] * 10, days, np.zeros(10, dtype=int))
    assert set(out) == LEAD_KEYS
    assert out["first_amber_day"] is None and out["first_red_day"] is None
    # None, not 0: a system that never warns has no lead time.
    assert out["lead_days_amber"] is None and out["lead_days_red"] is None
    assert out["false_alarm_days"] == 0


def test_evaluate_lead_time_input_handling():
    days = np.array([-2, -1, 0])
    # RiskLevel members are accepted as well as plain strings.
    out = evaluate_lead_time([RiskLevel.GREEN, RiskLevel.AMBER, RiskLevel.RED],
                             days, np.array([0, 1, 1]))
    assert out["lead_days_amber"] == 1 and out["lead_days_red"] == 0

    # A RED day also counts as the amber-level crossing (first warning).
    only_red = evaluate_lead_time(["GREEN", "GREEN", "RED"], days)
    assert only_red["first_amber_day"] == 0

    with pytest.raises(ValueError):
        evaluate_lead_time(["GREEN", "AMBER"], days)
    with pytest.raises(ValueError):
        evaluate_lead_time(["GREEN", "AMBER", "PUCE"], days)
    with pytest.raises(ValueError):
        evaluate_lead_time(["GREEN"] * 3, days, np.zeros(2, dtype=int))


# ------------------------------------------------------------------- harness
@pytest.fixture(scope="module")
def hindcast_result():
    """One small hindcast run shared by the schema tests (keeps the file fast)."""
    return run_hindcast(seed=11, n_train=200)


def test_run_hindcast_schema(hindcast_result):
    assert set(hindcast_result) == {"events", "honesty_note"}
    events = hindcast_result["events"]
    assert len(events) == len(DOCUMENTED_EVENTS)

    for entry in events:
        assert set(entry) == {"event", "metrics"}
        assert set(entry["event"]) == EVENT_KEYS
        metrics = entry["metrics"]
        assert LEAD_KEYS <= set(metrics)
        for key in ("first_amber_day", "first_red_day",
                    "lead_days_amber", "lead_days_red"):
            assert metrics[key] is None or isinstance(metrics[key], int)
        assert isinstance(metrics["false_alarm_days"], int)
        assert 0 <= metrics["false_alarm_days"] <= metrics["n_clear_days"]
        assert 0.0 <= metrics["bloom_day_detection_rate"] <= 1.0

        series = metrics["series"]
        assert {len(v) for v in series.values()} == {metrics["n_days"]}
        assert set(series["alert_level"]) <= {"GREEN", "AMBER", "RED"}

    # The whole result must survive a round trip to the API / dashboard.
    assert json.loads(json.dumps(hindcast_result)) == hindcast_result


def test_run_hindcast_warns_before_the_documented_impact(hindcast_result):
    """Headline operational metric: days of warning before the reported impact."""
    entry = hindcast_result["events"][0]
    assert entry["event"]["impact_date"] == "2008-11-15"
    metrics = entry["metrics"]

    assert metrics["lead_days_amber"] is not None, "system never warned at all"
    assert metrics["lead_days_amber"] > 0, "warning must precede the impact date"
    assert metrics["first_amber_day"] < 0
    # The stricter companion metric: warned while a bloom was really there.
    assert metrics["lead_days_true_alert"] is not None
    assert metrics["lead_days_true_alert"] > 0
    assert metrics["lead_days_true_alert"] <= metrics["lead_days_amber"]
    assert metrics["detected_class"] in {"dinoflagellate", "cyanobacteria"}


def test_run_hindcast_honesty_note(hindcast_result):
    note = hindcast_result["honesty_note"].lower()
    assert "simulated" in note
    assert "not independent validation" in note or "self-consistency" in note
    assert "roadmap" in note
    # House style: plain hyphens only, never an em-dash or en-dash (escaped
    # here so this file stays free of the characters it forbids).
    assert "\u2014" not in hindcast_result["honesty_note"]
    assert "\u2013" not in hindcast_result["honesty_note"]
