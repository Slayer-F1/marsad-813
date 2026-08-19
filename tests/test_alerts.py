"""Tests for the alert feed (src/marsad/alerts.py) and the stdlib alert API.

All fixtures here are small handcrafted results dicts, never a pipeline run:
these tests pin the DELIVERY contract (feed schema, filtering, lead-time
semantics, HTTP surface), and the science is tested in the stage-level tests.
The data in the fixtures is invented for the test and, like every MARSAD demo
payload, describes a synthetic physics-based simulation rather than real water.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from marsad.alerts import (
    DATA_BASIS,
    LEVEL_ORDER,
    NO_CROSSING,
    SOURCE,
    Alert,
    alert_feed,
    alerts_from_results,
    lead_days_to_red,
)
from marsad.risk import RED_THRESHOLD

# scripts/ is not a package; add it to the path the way the script itself is run.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import serve_api  # noqa: E402

ALERT_KEYS = {"intake", "level", "score", "issued_utc", "bloom_class",
              "chl_mg_m3", "lead_days", "message", "rationale"}
FEED_KEYS = {"generated_utc", "source", "data_basis", "counts", "alerts"}
GENERATED_UTC = "2026-08-19T06:00:00+00:00"


def _intake(name: str, level: str, score: float, dominant: str, chl: float,
            forecast_scores: list[float], **extra) -> dict:
    """One ``results["intakes"]`` record in the pipeline's schema."""
    record = {
        "name": name,
        "lat": 25.0,
        "lon": 56.0,
        "kind": "reservoir" if name == "Hatta Dam" else "desalination_intake",
        "risk": {"score": score, "level": level,
                 "rationale": [f"bloom probability {int(score * 100)}%"]},
        "bloom": {
            "probs": {"no_bloom": 1.0 - score, "dinoflagellate": score, "cyanobacteria": 0.0},
            "dominant": dominant,
            "chl_mg_m3": chl,
        },
        "history": [{"day": -1, "score": max(score - 0.05, 0.0)},
                    {"day": 0, "score": score}],
        "forecast": [{"day": d + 1, "score": s, "lo": max(s - 0.1, 0.0),
                      "hi": min(s + 0.1, 1.0)}
                     for d, s in enumerate(forecast_scores)],
    }
    record.update(extra)
    return record


@pytest.fixture()
def results() -> dict:
    """Handcrafted results dict exercising every alert path.

    Kalba is RED on score alone; Hatta Dam is RED below RED_THRESHOLD via the
    precautionary promotion in risk.py; Khor Fakkan is an AMBER whose forecast
    reaches RED on day 3; Fujairah is an AMBER that never crosses inside the
    horizon; Layyah is GREEN and must not produce an alert by default.
    """
    return {
        "generated_utc": GENERATED_UTC,
        "model_metrics": {
            "stage1_rmse_before": 0.0102, "stage1_rmse_after": 0.0031,
            "stage2_accuracy": 0.94,
            "stage2_confusion": [[30, 1, 0], [2, 28, 1], [0, 1, 29]],
            "labels": ["no_bloom", "dinoflagellate", "cyanobacteria"],
        },
        "intakes": [
            _intake("Khor Fakkan", "AMBER", 0.52, "dinoflagellate", 18.4,
                    [0.55, 0.60, 0.68, 0.72, 0.74, 0.75, 0.75]),
            _intake("Kalba", "RED", 0.71, "dinoflagellate", 38.2,
                    [0.73, 0.76, 0.78, 0.79, 0.80, 0.80, 0.81]),
            _intake("Layyah", "GREEN", 0.08, "no_bloom", 1.1,
                    [0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12]),
            _intake("Fujairah", "AMBER", 0.40, "dinoflagellate", 9.7,
                    [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47]),
            # RED by precautionary promotion: level RED with score < threshold.
            _intake("Hatta Dam", "RED", 0.60, "cyanobacteria", 24.8,
                    [0.61, 0.62, 0.62, 0.63, 0.63, 0.64, 0.64],
                    uncertainty={"total": 0.41, "epistemic": 0.12,
                                 "confidence": 0.58, "review_recommended": True}),
        ],
        "spectra_example": {"wavelength_nm": [], "observed": [],
                            "corrected": [], "true": []},
    }


# --------------------------------------------------------------------------
# alerts_from_results
# --------------------------------------------------------------------------

def test_alerts_from_handcrafted_results(results):
    alerts = alerts_from_results(results)
    assert all(isinstance(a, Alert) for a in alerts)
    # GREEN excluded by the AMBER default; the other four are emitted.
    names = [a.intake for a in alerts]
    assert "Layyah" not in names
    assert set(names) == {"Kalba", "Hatta Dam", "Khor Fakkan", "Fujairah"}
    # Most urgent first: both REDs ahead of both AMBERs.
    assert [a.level for a in alerts] == ["RED", "RED", "AMBER", "AMBER"]
    # Every alert carries the risk rationale through unchanged.
    kalba = next(a for a in alerts if a.intake == "Kalba")
    assert kalba.rationale == results["intakes"][1]["risk"]["rationale"]
    assert kalba.issued_utc == GENERATED_UTC
    assert kalba.bloom_class == "dinoflagellate"
    assert kalba.chl_mg_m3 == pytest.approx(38.2)


def test_min_level_filtering(results):
    red_only = alerts_from_results(results, min_level="RED")
    assert {a.intake for a in red_only} == {"Kalba", "Hatta Dam"}
    assert all(a.level == "RED" for a in red_only)

    everything = alerts_from_results(results, min_level="GREEN")
    assert len(everything) == len(results["intakes"])
    assert "Layyah" in {a.intake for a in everything}

    default = alerts_from_results(results)
    assert default == alerts_from_results(results, min_level="AMBER")
    # RED is never filtered out, whatever the floor.
    for floor in LEVEL_ORDER:
        assert "Kalba" in {a.intake for a in alerts_from_results(results, floor)}

    with pytest.raises(ValueError):
        alerts_from_results(results, min_level="PURPLE")


def test_missing_intakes_key_is_an_actionable_error():
    with pytest.raises(ValueError, match="run_end_to_end"):
        alerts_from_results({"generated_utc": GENERATED_UTC})


# --------------------------------------------------------------------------
# lead_days semantics
# --------------------------------------------------------------------------

def test_lead_days_already_red_is_zero(results):
    by_name = {a.intake: a for a in alerts_from_results(results)}
    # RED on score alone.
    assert by_name["Kalba"].lead_days == 0
    # RED by precautionary promotion, score BELOW the RED threshold: the
    # operator is on alert now, so the countdown is still zero.
    assert results["intakes"][4]["risk"]["score"] < RED_THRESHOLD
    assert by_name["Hatta Dam"].lead_days == 0


def test_lead_days_counts_first_forecast_crossing(results):
    by_name = {a.intake: a for a in alerts_from_results(results)}
    # Khor Fakkan's forecast is [0.55, 0.60, 0.68, ...]; 0.68 on day 3 is the
    # first value at or above the RED threshold.
    assert by_name["Khor Fakkan"].lead_days == 3
    assert results["intakes"][0]["forecast"][2]["score"] >= RED_THRESHOLD
    assert results["intakes"][0]["forecast"][1]["score"] < RED_THRESHOLD


def test_lead_days_never_crosses_is_minus_one(results):
    by_name = {a.intake: a for a in alerts_from_results(results)}
    assert by_name["Fujairah"].lead_days == NO_CROSSING == -1
    # An empty forecast cannot show a crossing either.
    assert lead_days_to_red({"risk": {"level": "AMBER", "score": 0.4},
                             "forecast": []}) == NO_CROSSING


def test_urgency_sort_puts_the_shortest_countdown_first(results):
    ambers = [a for a in alerts_from_results(results) if a.level == "AMBER"]
    # Khor Fakkan (3 days to RED) outranks Fujairah (no crossing), even though
    # "no crossing" is encoded as -1.
    assert [a.intake for a in ambers] == ["Khor Fakkan", "Fujairah"]


# --------------------------------------------------------------------------
# Alert.to_dict / message
# --------------------------------------------------------------------------

def test_to_dict_round_trips_through_json(results):
    for alert in alerts_from_results(results, min_level="GREEN"):
        payload = alert.to_dict()
        assert set(payload) == ALERT_KEYS
        assert json.loads(json.dumps(payload)) == payload
        assert isinstance(payload["lead_days"], int)
        assert isinstance(payload["score"], float)
        assert isinstance(payload["rationale"], list)


def test_message_is_a_useful_operator_one_liner(results):
    by_name = {a.intake: a for a in alerts_from_results(results)}

    kalba = by_name["Kalba"].message
    assert "Kalba" in kalba and "RED" in kalba and "dinoflagellate" in kalba
    assert "already" in kalba.lower()

    khor = by_name["Khor Fakkan"].message
    assert "Khor Fakkan" in khor and "AMBER" in khor
    assert "3 days" in khor

    fujairah = by_name["Fujairah"].message
    assert "no RED crossing" in fujairah and "7-day" in fujairah

    hatta = by_name["Hatta Dam"].message
    assert "cyanobacteria" in hatta
    # The v0.2 uncertainty block routes this one to a human analyst.
    assert any("human analyst" in r for r in by_name["Hatta Dam"].rationale)

    for alert in by_name.values():
        # A one-liner is one line, and carries asset + level + class + verdict.
        assert "\n" not in alert.message
        assert alert.intake in alert.message and alert.level in alert.message
        # House style: plain hyphens only, no em/en dashes in operator text.
        assert chr(0x2014) not in alert.message and chr(0x2013) not in alert.message


# --------------------------------------------------------------------------
# alert_feed
# --------------------------------------------------------------------------

def test_feed_schema_is_exactly_as_contracted(results):
    feed = alert_feed(results)
    assert set(feed) == FEED_KEYS
    assert feed["generated_utc"] == GENERATED_UTC
    assert feed["source"] == SOURCE == "MARSAD 813"
    # Honesty rule: the feed must mark itself as simulation output.
    assert feed["data_basis"] == DATA_BASIS == "synthetic physics-based simulation"
    assert set(feed["counts"]) == {"RED", "AMBER", "GREEN"}
    assert feed["counts"] == {"RED": 2, "AMBER": 2, "GREEN": 1}
    assert sum(feed["counts"].values()) == len(results["intakes"])
    assert all(set(a) == ALERT_KEYS for a in feed["alerts"])
    # GREEN assets are counted but not alerted.
    assert len(feed["alerts"]) == 4
    assert json.loads(json.dumps(feed)) == feed


# --------------------------------------------------------------------------
# scripts/serve_api.py
# --------------------------------------------------------------------------

@pytest.fixture()
def api(results):
    """Run the alert API on an ephemeral port in a background thread."""
    server = serve_api.make_server(results, port=0, host="127.0.0.1", quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "alert API thread did not shut down cleanly"


def _get(url: str) -> tuple[int, dict, dict]:
    """GET a URL, returning (status, headers, parsed JSON body)."""
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 404/400 bodies are JSON too
        return exc.code, dict(exc.headers), json.loads(exc.read().decode("utf-8"))


def test_api_health_and_alerts(api, results):
    status, headers, health = _get(f"{api}/health")
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "application/json" in headers["Content-Type"]
    assert health["status"] == "ok"
    assert health["data_basis"] == DATA_BASIS
    assert health["intakes"] == 5

    status, headers, feed = _get(f"{api}/v1/alerts")
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert set(feed) == FEED_KEYS
    # The wire payload is byte-for-byte what alert_feed() produces in process.
    assert feed == alert_feed(results)
    assert feed["counts"]["RED"] == 2
    assert feed["alerts"][0]["level"] == "RED"


def test_api_alerts_min_level_query(api):
    status, _, feed = _get(f"{api}/v1/alerts?min_level=RED")
    assert status == 200
    assert {a["intake"] for a in feed["alerts"]} == {"Kalba", "Hatta Dam"}
    # Counts still cover every asset, so the dashboard tally does not change.
    assert feed["counts"] == {"RED": 2, "AMBER": 2, "GREEN": 1}

    status, _, bad = _get(f"{api}/v1/alerts?min_level=PURPLE")
    assert status == 400 and bad["error"] == "bad_request"


def test_api_intakes_collection_and_url_decoded_name(api):
    status, _, listing = _get(f"{api}/v1/intakes")
    assert status == 200
    assert listing["count"] == 5
    assert [i["name"] for i in listing["intakes"]][0] == "Khor Fakkan"

    # The percent-encoded space must resolve to the real asset name.
    status, _, one = _get(f"{api}/v1/intakes/Khor%20Fakkan")
    assert status == 200
    assert one["intake"]["name"] == "Khor Fakkan"
    assert one["intake"]["risk"]["level"] == "AMBER"
    assert one["data_basis"] == DATA_BASIS

    # Hand-typed variants resolve to the same asset.
    for variant in ("khor-fakkan", "khor_fakkan", "KHOR%20FAKKAN"):
        status, _, alt = _get(f"{api}/v1/intakes/{variant}")
        assert status == 200 and alt["intake"]["name"] == "Khor Fakkan"

    status, _, missing = _get(f"{api}/v1/intakes/Atlantis")
    assert status == 404
    assert missing["error"] == "not_found"
    assert "Khor Fakkan" in missing["available"]


def test_api_metrics_and_unknown_route(api):
    status, _, metrics = _get(f"{api}/v1/metrics")
    assert status == 200
    assert metrics["model_metrics"]["stage2_accuracy"] == pytest.approx(0.94)
    assert metrics["model_metrics"]["labels"][2] == "cyanobacteria"

    status, headers, body = _get(f"{api}/v1/nope")
    assert status == 404
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert body["error"] == "not_found"
    assert any("/v1/alerts" in r for r in body["routes"])


def test_missing_results_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(serve_api.ResultsUnavailableError) as exc:
        serve_api.load_results(tmp_path / "results.json")
    message = str(exc.value)
    assert "scripts/run_demo.py" in message
    assert "results.json" in message

    broken = tmp_path / "results.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(serve_api.ResultsUnavailableError, match="not valid JSON"):
        serve_api.load_results(broken)
