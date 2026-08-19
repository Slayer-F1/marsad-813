"""Stage 1 - learned shallow-water spectral correction.

Physics rationale
-----------------
Over optically shallow Gulf water the at-sensor remote-sensing reflectance is
contaminated by three additive/multiplicative effects that a purely analytic
atmospheric correction does not remove:

* **Bottom reflectance** - sandy sea-floor light escaping the water column,
  ``R_bottom * exp(-2 * Kd * depth)``: a broad, smooth lift across the
  VIS/NIR that mimics sediment and biases every band-ratio bloom index.
* **Sunglint** - specular sky/sun reflection at the surface: a spectrally
  flat offset that is the *only* signal left in the SWIR (water itself is
  ~black beyond ~1300 nm), so the SWIR bands carry the information needed
  to estimate and subtract it.
* **Sensor noise** - ~1–2 % multiplicative Gaussian.

These effects are non-linear functions of unobserved state (depth, Kd,
bottom albedo, glint geometry), but they leave a joint spectral fingerprint
across the full 205-band grid. A multi-output MLP regressor can therefore
learn to estimate them from (observed, clean) training pairs, using e.g.
the SWIR floor to infer the glint offset and the shape of the NIR shoulder
to infer the bottom term.

Implementation - RESIDUAL formulation
-------------------------------------
The network predicts the **contamination** ``Rrs_observed - Rrs_true`` and
the correction is ``observed - predicted_contamination`` - it does NOT
regress the clean spectrum directly. This matters: a direct
``observed -> clean`` regression minimises average per-band error and
therefore repaints narrow diagnostic pigment features (e.g. the ~5e-4 sr^-1
phycocyanin line at 620 nm) with their conditional mean, silently erasing
exactly the information Stage 2 speciation needs. Measured on synthetic
holdout data, direct regression cut downstream classification from 0.96 to
0.83 despite a 98 % RMSE improvement; the residual form reaches 0.97
because the contamination is spectrally smooth (easy to regress) while
every fine feature the network does not model passes through untouched.

``StandardScaler`` on the inputs *and* on the contamination targets wrapped
around a multi-output ``MLPRegressor`` (one output per band). Rrs magnitudes
are O(1e-2) sr^-1, so standardising both sides keeps the MSE loss and the
adam step size well-conditioned. Early stopping keeps fit time modest.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .spectra import N_BANDS

__all__ = ["ShallowWaterCorrector"]


def _validate_spectra(x: np.ndarray, name: str) -> np.ndarray:
    """Coerce to float64 (n, N_BANDS) and fail loudly on wrong shapes."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_BANDS:
        raise ValueError(
            f"{name} must have shape (n, {N_BANDS}); got {arr.shape}"
        )
    return arr


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error over all samples and bands (sr^-1)."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


class ShallowWaterCorrector:
    """Learned correction mapping contaminated Rrs to clean at-surface Rrs.

    Parameters
    ----------
    hidden:
        MLP hidden layer sizes. The default (128, 64, 128) is a bottleneck
        (denoising-autoencoder-like) shape: the 64-unit waist forces the
        network to summarise the contamination state (glint level, bottom
        contribution) rather than memorise per-band offsets.
    max_iter:
        Cap on adam epochs; combined with early stopping this keeps
        training to seconds on hackathon-scale data.
    seed:
        Seeds MLP weight init and the internal early-stopping split.
    """

    def __init__(self, hidden=(128, 64, 128), max_iter=300, seed: int = 0):
        self.hidden = tuple(hidden)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.seed,
        )
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, rrs_observed, rrs_true) -> "ShallowWaterCorrector":
        """Learn the observed -> true mapping from paired spectra.

        Both arrays must be (n, N_BANDS); rows are paired pixels of the
        same scene (contaminated measurement, clean correction target).
        """
        x = _validate_spectra(rrs_observed, "rrs_observed")
        y = _validate_spectra(rrs_true, "rrs_true")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"paired sample counts differ: {x.shape[0]} != {y.shape[0]}"
            )
        # Residual target: regress the contamination, not the clean spectrum
        # (see module docstring - preserves narrow pigment lines).
        contamination = x - y
        xs = self._x_scaler.fit_transform(x)
        ys = self._y_scaler.fit_transform(contamination)
        self._mlp.fit(xs, ys)
        self._fitted = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, rrs_observed) -> np.ndarray:
        """Return corrected (clean-water estimate) Rrs, shape (n, N_BANDS)."""
        if not self._fitted:
            raise RuntimeError("ShallowWaterCorrector must be fit() before transform()")
        x = _validate_spectra(rrs_observed, "rrs_observed")
        pred_scaled = self._mlp.predict(self._x_scaler.transform(x))
        contamination = self._y_scaler.inverse_transform(pred_scaled)
        return np.asarray(x - contamination, dtype=np.float64)

    # ---------------------------------------------------------------- score
    def score(self, rrs_observed, rrs_true) -> dict:
        """RMSE of the raw observation vs. of the corrected spectra.

        Returns ``{"rmse_before": float, "rmse_after": float}`` where
        *before* compares observed against true (how bad the contamination
        is) and *after* compares ``transform(observed)`` against true (how
        much of it the model removed). ``rmse_after < rmse_before`` is the
        headline success criterion of Stage 1.
        """
        x = _validate_spectra(rrs_observed, "rrs_observed")
        y = _validate_spectra(rrs_true, "rrs_true")
        corrected = self.transform(x)
        return {
            "rmse_before": _rmse(x, y),
            "rmse_after": _rmse(corrected, y),
        }

    # ------------------------------------------------------------- persist
    def save(self, path) -> None:
        """Serialise the fitted corrector (scalers + MLP) with joblib."""
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path) -> "ShallowWaterCorrector":
        """Load a corrector previously written by :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(
                f"{path} does not contain a {cls.__name__} (got {type(obj).__name__})"
            )
        return obj
