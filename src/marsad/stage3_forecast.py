"""Stage 3 - bloom-risk drift & forecast.

Turns the daily risk-score history of one intake into a short-range
probabilistic forecast. The model is deliberately small and transparent:

* **Damped-trend exponential smoothing** (Holt's linear method with a
  damping factor ``PHI < 1``, implemented directly in numpy). Bloom risk
  is strongly autocorrelated on daily scales - blooms build and decay
  over days - so a level + trend model captures the local dynamics,
  while damping keeps multi-day extrapolations from running away
  (a three-day rise does not mean the risk rises forever).
* **Advection bump**: surface currents pushing a bloom patch toward an
  intake raise the encounter risk roughly in proportion to the
  cumulative displacement ``drift * lead`` (km). Only movement *toward*
  the intake (positive ``drift_toward_intake_kmday``) adds risk;
  movement away is treated conservatively as "no extra risk", never as
  negative risk.
* **Uncertainty band**: forecast errors are assumed to accumulate like a
  random walk, so the 10th–90th percentile half-width grows with
  ``sqrt(lead)``, scaled by the one-step-ahead residual spread measured
  on the history itself.

All outputs are clipped to the valid risk range [0, 1] and satisfy
``lo <= mean <= hi`` elementwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Smoothing constants (fixed, not fitted - the histories are short daily
# series and fixed constants keep the forecaster deterministic and fast).
_ALPHA = 0.5    # level smoothing: how fast the level tracks new observations
_BETA = 0.3     # trend smoothing: how fast the slope estimate adapts
_PHI = 0.85     # per-day trend damping (0 < PHI < 1)

# Uncertainty band: +/- z for the 10th/90th percentiles of a normal.
_Z_10_90 = 1.2815515655446004
# Floor on the one-step residual spread - a perfectly flat history still
# carries irreducible observation/model uncertainty.
_SIGMA_FLOOR = 0.02

# Advection term: extra risk saturates once the cumulative displacement
# reaches the patch/intake interaction scale (~10 km for Gulf coastal cells).
_ADVECTION_WEIGHT = 0.25     # maximum extra risk attributable to advection
_ADVECTION_SCALE_KM = 10.0   # displacement (km) at which the bump saturates


@dataclass
class Forecast:
    """Probabilistic risk forecast for one intake.

    Attributes
    ----------
    mean : np.ndarray
        (horizon,) forecast risk scores in [0, 1], lead days 1..horizon.
    lo : np.ndarray
        (horizon,) 10th percentile, ``lo <= mean`` elementwise.
    hi : np.ndarray
        (horizon,) 90th percentile, ``hi >= mean`` elementwise.
    method : str
        Human-readable identifier of the forecasting method.
    """

    mean: np.ndarray  # (horizon,) risk scores in [0,1]
    lo: np.ndarray    # (horizon,) 10th percentile
    hi: np.ndarray    # (horizon,) 90th percentile
    method: str


class DriftForecaster:
    """Damped-trend exponential smoothing + advection forecaster.

    Parameters
    ----------
    horizon_days : int
        Number of daily lead steps to forecast (default 7).
    """

    def __init__(self, horizon_days: int = 7):
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        self.horizon_days = int(horizon_days)

    def forecast(self, history: np.ndarray,
                 drift_toward_intake_kmday: float = 0.0) -> Forecast:
        """Forecast the next ``horizon_days`` daily risk scores.

        Parameters
        ----------
        history : np.ndarray
            (n_days,) daily risk scores in [0, 1], oldest first, ending at
            the current day. At least one observation is required.
        drift_toward_intake_kmday : float
            Advection speed of the bloom patch toward the intake in km/day.
            Positive values (approaching) add risk; negative values
            (receding) add nothing - we never *reduce* forecast risk on
            the basis of currents, which is the precautionary choice for
            an early-warning product.

        Returns
        -------
        Forecast
            mean/lo/hi arrays of shape (horizon_days,), clipped to [0, 1]
            with ``lo <= mean <= hi`` elementwise.
        """
        h = np.clip(np.asarray(history, dtype=np.float64).ravel(), 0.0, 1.0)
        if h.size == 0:
            raise ValueError("history must contain at least one observation")

        # --- damped-trend exponential smoothing (Holt, damped) ------------
        # State: level l_t, trend b_t.  Recursions:
        #   l_t = alpha*y_t + (1-alpha)*(l_{t-1} + phi*b_{t-1})
        #   b_t = beta*(l_t - l_{t-1}) + (1-beta)*phi*b_{t-1}
        # h-step forecast: l_n + (phi + phi^2 + ... + phi^h) * b_n
        level = float(h[0])
        trend = float(h[1] - h[0]) if h.size >= 2 else 0.0
        residuals: list[float] = []
        for y in h[1:]:
            one_step = level + _PHI * trend
            residuals.append(float(y) - one_step)
            new_level = _ALPHA * float(y) + (1.0 - _ALPHA) * one_step
            trend = _BETA * (new_level - level) + (1.0 - _BETA) * _PHI * trend
            level = new_level

        lead = np.arange(1, self.horizon_days + 1, dtype=np.float64)
        # Damped cumulative trend multiplier: phi + phi^2 + ... + phi^lead.
        phi_pows = np.cumsum(_PHI ** lead)
        mean = level + phi_pows * trend

        # --- advection bump ----------------------------------------------
        # Cumulative displacement toward the intake after `lead` days is
        # drift * lead (km); risk contribution rises linearly with it and
        # saturates at the interaction scale.  Negative drift adds nothing.
        drift = max(float(drift_toward_intake_kmday), 0.0)
        bump = _ADVECTION_WEIGHT * np.minimum(
            drift * lead / _ADVECTION_SCALE_KM, 1.0)
        mean = mean + bump

        # --- uncertainty band --------------------------------------------
        # Random-walk error accumulation: half-width ~ sigma1 * sqrt(lead).
        sigma1 = max(float(np.std(residuals)) if residuals else 0.0,
                     _SIGMA_FLOOR)
        half_width = _Z_10_90 * sigma1 * np.sqrt(lead)
        lo = mean - half_width
        hi = mean + half_width

        # Clip to the valid risk range; clipping is monotone so the
        # lo <= mean <= hi ordering survives, but enforce it explicitly.
        mean = np.clip(mean, 0.0, 1.0)
        lo = np.minimum(np.clip(lo, 0.0, 1.0), mean)
        hi = np.maximum(np.clip(hi, 0.0, 1.0), mean)

        return Forecast(mean=mean, lo=lo, hi=hi,
                        method="damped_trend_exponential_smoothing+advection")
