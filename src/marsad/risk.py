"""Per-intake risk policy - turn model outputs into an operator alert.

Combines the Stage 2 bloom probability and chlorophyll estimate with the
Stage 3 trend, the distance between the detected bloom and the intake,
and the forecast uncertainty into a single risk score in [0, 1] plus a
GREEN/AMBER/RED level and concrete human-readable rationale strings for
the control-room dashboard.

Design rationale
----------------
* The bloom probability dominates the score: a confident detection is an
  alert regardless of context, so a certain bloom alone must be able to
  reach RED.
* Chlorophyll, positive trend, and proximity are corroborating boosters
  with saturating (clipped-linear) responses - each can escalate a
  borderline case but none can create an alert out of nothing.
* The proximity boost is gated by the bloom probability: "3 km from the
  intake" only matters if there is actually a bloom at 3 km.
* Precautionary asymmetry: high forecast uncertainty close to an intake
  promotes AMBER to RED (better a false alarm than a missed intake
  shutdown), and uncertainty NEVER downgrades a level.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass
class RiskAssessment:
    score: float            # [0,1]
    level: RiskLevel
    rationale: list[str]    # human-readable reasons, shown on the dashboard


# --- TEAM DECISION ---
# Operator-tunable alert policy. These weights and thresholds set the
# probability-of-detection vs false-alarm trade-off for the desalination
# operators: raising RED_THRESHOLD or the saturation constants means
# fewer alarms but a higher chance of missing a real intake event.
# They are policy, not physics - tune them with the plant, not the model.
W_PROB = 0.70                    # bloom probability dominates the blend
W_CHL = 0.10                     # chlorophyll-a corroboration weight
W_TREND = 0.10                   # rising-risk (positive trend) boost weight
W_PROX = 0.10                    # proximity-to-intake boost weight
CHL_SATURATION_MG_M3 = 30.0      # chl term saturates at dense-bloom levels
TREND_SATURATION_PER_DAY = 0.05  # trend term saturates at +0.05 risk/day
PROXIMITY_RADIUS_KM = 5.0        # proximity boost applies inside this radius
AMBER_THRESHOLD = 0.35           # score >= 0.35 -> AMBER
RED_THRESHOLD = 0.65             # score >= 0.65 -> RED
UNCERTAINTY_PROMOTION = 0.20     # uncertainty >= this near an intake
#                                  promotes AMBER -> RED (precautionary).
#                                  Calibrated to the Stage-3 forecaster: its
#                                  mean 7-day hi-lo band is ~0.22-0.28 on
#                                  realistic histories, so 0.20 means "the
#                                  forecast cannot rule out RED"; an earlier
#                                  0.30 gate was above every real band and
#                                  never fired on forecast uncertainty alone.
PROMOTION_MARGIN = 0.10          # ...but only for BORDERLINE ambers, i.e.
#                                  score >= RED_THRESHOLD - PROMOTION_MARGIN.
#                                  Without this gate every mid-AMBER near an
#                                  intake becomes RED and the RED level stops
#                                  carrying information.
# --- END TEAM DECISION ---


def compute_risk_index(bloom_prob: float, chl_mg_m3: float, trend_per_day: float,
                       distance_km: float, uncertainty: float) -> RiskAssessment:
    """Compute the per-intake risk score, alert level, and rationale.

    Parameters
    ----------
    bloom_prob : float
        1 - P(no_bloom) from Stage 2, in [0, 1].
    chl_mg_m3 : float
        Estimated chlorophyll-a concentration (mg/m3), >= 0.
    trend_per_day : float
        Recent risk-score change per day (Stage 3); only positive
        (worsening) trends add risk.
    distance_km : float
        Distance from the detected bloom patch to the intake (km).
    uncertainty : float
        Assessment uncertainty in [0, 1]: the max of the mean normalized
        forecast hi-lo band width and the classifier's mean ambiguity
        (1 - 2|p - 0.5|). Used only for the precautionary AMBER->RED
        promotion. NOT a spatial bloom/clear mix fraction - a confidently
        classified half-bloom scene is not "uncertain".

    Returns
    -------
    RiskAssessment
        score in [0, 1], GREEN/AMBER/RED level, and concrete rationale
        strings for the dashboard.
    """
    p = min(max(float(bloom_prob), 0.0), 1.0)
    chl = max(float(chl_mg_m3), 0.0)
    trend = float(trend_per_day)
    dist = max(float(distance_km), 0.0)
    unc = min(max(float(uncertainty), 0.0), 1.0)

    # Saturating (clipped-linear) corroboration terms in [0, 1].
    chl_term = min(chl / CHL_SATURATION_MG_M3, 1.0)
    trend_term = min(max(trend, 0.0) / TREND_SATURATION_PER_DAY, 1.0)
    prox_frac = max((PROXIMITY_RADIUS_KM - dist) / PROXIMITY_RADIUS_KM, 0.0)
    prox_term = prox_frac * p  # proximity only matters if a bloom is likely

    score = (W_PROB * p + W_CHL * chl_term
             + W_TREND * trend_term + W_PROX * prox_term)
    score = min(max(score, 0.0), 1.0)

    if score >= RED_THRESHOLD:
        level = RiskLevel.RED
    elif score >= AMBER_THRESHOLD:
        level = RiskLevel.AMBER
    else:
        level = RiskLevel.GREEN

    rationale: list[str] = [f"bloom probability {p * 100:.0f}%"]
    if chl >= 1.0:
        dense = " (dense bloom)" if chl >= 20.0 else ""
        rationale.append(f"chlorophyll-a {chl:.1f} mg/m3{dense}")
    if trend > 0.0:
        rationale.append(f"risk trend +{trend:.3f}/day and rising")
    elif trend < 0.0:
        rationale.append(f"risk trend {trend:.3f}/day, easing")
    if dist < PROXIMITY_RADIUS_KM and p >= 0.5:
        approach = " and approaching" if trend > 0.0 else ""
        rationale.append(f"bloom {dist:.1f} km from intake{approach}")

    # Precautionary promotion: high uncertainty close to an intake turns a
    # BORDERLINE AMBER (within PROMOTION_MARGIN of the RED threshold) into
    # RED.  Promotion only ever RAISES the level - uncertainty never
    # silently downgrades an alert.
    if (level is RiskLevel.AMBER
            and score >= RED_THRESHOLD - PROMOTION_MARGIN
            and unc >= UNCERTAINTY_PROMOTION
            and dist < PROXIMITY_RADIUS_KM):
        level = RiskLevel.RED
        rationale.append(
            f"assessment uncertainty {unc * 100:.0f}% within "
            f"{dist:.1f} km of intake - precautionary promotion AMBER to RED")

    if level is RiskLevel.GREEN:
        rationale.append("no significant bloom threat to this intake")

    return RiskAssessment(score=float(score), level=level, rationale=rationale)
