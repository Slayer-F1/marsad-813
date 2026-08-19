"""Stage 2 - bloom detection & speciation from corrected Rrs spectra.

Classifies each pixel spectrum into the three MARSAD classes
(``0 = no_bloom``, ``1 = dinoflagellate``, ``2 = cyanobacteria``) and
regresses chlorophyll-a concentration, using physically motivated band
features concatenated with the full 205-band spectrum.

Physics behind the engineered features
--------------------------------------
- **443/555 blue-green ratio** - chlorophyll-a's Soret band absorbs at
  443 nm while productive water still reflects around the 555 nm green
  peak, so the ratio drops as biomass rises (the classic OCx logic).
- **620 nm line depth** - phycocyanin, the marker pigment of toxic
  cyanobacteria, absorbs at 620 nm. Depth of the spectrum below a linear
  continuum drawn between ~600 and ~650 nm isolates that pigment and is
  what separates cyanobacteria from dinoflagellates.
- **NDCI (708-665)/(708+665)** - the Normalized Difference Chlorophyll
  Index: dense blooms push a reflectance peak near 708 nm while chl-a
  red absorption at 665-675 nm deepens, so NDCI grows with bloom density.
- **Red-edge line height at 708 nm** - height of Rrs(708) above the
  665-750 nm chord; a positive fluorescence/scattering peak appears only
  for dense surface blooms.
- **Green peak line height at 555 nm** - height of Rrs(555) above the
  490-665 nm chord; sensitive to overall in-water scattering by cells.

The raw spectrum is appended so the learner can exploit any residual
shape information the hand-built indices miss.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marsad.spectra import BAND_GRID, N_BANDS, band_index

N_CLASSES = 3  # 0 = no_bloom, 1 = dinoflagellate, 2 = cyanobacteria

_EPS = 1e-9  # guards ratios; Rrs magnitudes are ~1e-3..1e-2 sr^-1


def _as_spectra(rrs: np.ndarray) -> np.ndarray:
    """Validate/coerce input to a float64 ``(n, N_BANDS)`` array."""
    arr = np.atleast_2d(np.asarray(rrs, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != N_BANDS:
        raise ValueError(
            f"expected spectra of shape (n, {N_BANDS}), got {arr.shape}"
        )
    return arr


def _line_height(rrs: np.ndarray, lo_nm: float, mid_nm: float, hi_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed height of Rrs(mid) above the straight-line continuum lo->hi.

    Continuum removal: pigment absorption shows up as a *negative* height
    (a dip below the chord), scattering/fluorescence peaks as positive.
    Returns ``(height, continuum)`` for each spectrum.
    """
    i_lo, i_mid, i_hi = band_index(lo_nm), band_index(mid_nm), band_index(hi_nm)
    w = (BAND_GRID[i_mid] - BAND_GRID[i_lo]) / (BAND_GRID[i_hi] - BAND_GRID[i_lo])
    continuum = (1.0 - w) * rrs[:, i_lo] + w * rrs[:, i_hi]
    return rrs[:, i_mid] - continuum, continuum


def engineer_features(rrs: np.ndarray) -> np.ndarray:
    """Engineered band features concatenated with the full spectrum.

    Returns ``(n, 5 + N_BANDS)`` float64: the five diagnostic indices
    described in the module docstring followed by the raw 205 bands.
    Non-finite values (possible on pathological inputs) are zeroed so the
    downstream estimators never see NaN/inf.
    """
    rrs = _as_spectra(rrs)
    r443 = rrs[:, band_index(443.0)]
    r555 = rrs[:, band_index(555.0)]
    r665 = rrs[:, band_index(665.0)]
    r708 = rrs[:, band_index(708.0)]

    # Blue-green ratio: falls with increasing chl-a absorption at 443 nm.
    ratio_443_555 = r443 / (r555 + _EPS)

    # Phycocyanin 620 nm line depth vs its 600/650 nm neighbours,
    # normalised by the continuum so it is insensitive to brightness.
    pc_height, pc_continuum = _line_height(rrs, 600.0, 620.0, 650.0)
    pc_depth_620 = -pc_height / (pc_continuum + _EPS)  # >0 when absorbing

    # NDCI: red-edge peak vs chl-a red absorption, bounded in [-1, 1].
    ndci = (r708 - r665) / (r708 + r665 + _EPS)

    # Red-edge peak height above the 665-750 nm chord (dense blooms only).
    rededge_height, _ = _line_height(rrs, 665.0, 708.0, 750.0)

    # Green peak height above the 490-665 nm chord (cell scattering).
    green_peak, _ = _line_height(rrs, 490.0, 555.0, 665.0)

    engineered = np.column_stack(
        [ratio_443_555, pc_depth_620, ndci, rededge_height, green_peak]
    )
    features = np.concatenate([engineered, rrs], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class BloomClassifier:
    """Bloom detection & speciation with a chlorophyll regression head.

    Two sklearn pipelines share the same feature representation
    (:func:`engineer_features`):

    - a ``StandardScaler + HistGradientBoostingClassifier`` for the three
      classes (gradient boosting handles the mixed-scale ratio/spectrum
      feature vector well and is fast on the ~210-dim input), and
    - a ``StandardScaler + HistGradientBoostingRegressor`` trained on
      ``log1p(chl)`` - chlorophyll is lognormal-ish across bloom/no-bloom
      water, so regressing in log space keeps the loss balanced instead of
      being dominated by the densest blooms.

    ``predict_proba`` always returns columns in label order ``0, 1, 2``
    even when the training set lacked some classes (degenerate fits are
    handled by mapping the fitted ``classes_`` into fixed columns and, for
    a single-class fit, bypassing sklearn entirely).
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._clf = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gbc",
                    HistGradientBoostingClassifier(
                        max_iter=150, random_state=self.seed
                    ),
                ),
            ]
        )
        self._reg = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gbr",
                    HistGradientBoostingRegressor(
                        max_iter=150, random_state=self.seed
                    ),
                ),
            ]
        )
        self._single_class: int | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, rrs: np.ndarray, labels: np.ndarray, chl: np.ndarray) -> "BloomClassifier":
        """Fit classifier and chl regressor on (corrected) Rrs spectra.

        Parameters
        ----------
        rrs : (n, N_BANDS) remote-sensing reflectance.
        labels : (n,) ints in {0, 1, 2}.
        chl : (n,) chlorophyll-a in mg/m3 (regressed as log1p).
        """
        rrs = _as_spectra(rrs)
        labels = np.asarray(labels).astype(int).ravel()
        chl = np.asarray(chl, dtype=np.float64).ravel()
        if not (len(labels) == len(chl) == rrs.shape[0]):
            raise ValueError("rrs, labels and chl must have matching lengths")

        X = engineer_features(rrs)

        seen = np.unique(labels)
        if seen.size == 1:
            # Degenerate fit: sklearn classifiers cannot fit a single
            # class; remember it and predict it with probability 1.
            self._single_class = int(seen[0])
        else:
            self._single_class = None
            self._clf.fit(X, labels)

        self._reg.fit(X, np.log1p(np.clip(chl, 0.0, None)))
        self._fitted = True
        return self

    # -------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BloomClassifier is not fitted; call fit() first")

    def predict_proba(self, rrs: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ALWAYS in label order 0, 1, 2.

        Classes absent from training get probability 0; rows sum to 1.
        """
        self._check_fitted()
        X = engineer_features(rrs)
        out = np.zeros((X.shape[0], N_CLASSES), dtype=np.float64)

        if self._single_class is not None:
            out[:, self._single_class] = 1.0
            return out

        raw = self._clf.predict_proba(X)
        for j, cls in enumerate(np.asarray(self._clf.classes_, dtype=int)):
            out[:, cls] = raw[:, j]

        # Defensive renormalisation (no-op when all mass was mapped).
        totals = out.sum(axis=1, keepdims=True)
        np.divide(out, totals, out=out, where=totals > 0)
        return out

    def predict(self, rrs: np.ndarray) -> np.ndarray:
        """Hard labels: argmax over :meth:`predict_proba` columns."""
        return np.argmax(self.predict_proba(rrs), axis=1)

    def estimate_chl(self, rrs: np.ndarray) -> np.ndarray:
        """Chlorophyll-a estimate in mg/m3 (inverse of the log1p target)."""
        self._check_fitted()
        X = engineer_features(rrs)
        return np.clip(np.expm1(self._reg.predict(X)), 0.0, None)

    # ------------------------------------------------------------- evaluate
    def evaluate(self, rrs: np.ndarray, labels: np.ndarray) -> dict:
        """Accuracy and 3x3 confusion matrix (rows = true, cols = predicted).

        Returns plain Python types (float / nested int lists) so the result
        is directly JSON-serialisable for the dashboard.
        """
        labels = np.asarray(labels).astype(int).ravel()
        pred = self.predict(rrs)
        accuracy = float(np.mean(pred == labels))
        cm = confusion_matrix(labels, pred, labels=list(range(N_CLASSES)))
        return {
            "accuracy": accuracy,
            "confusion": [[int(v) for v in row] for row in cm],
        }

    # ------------------------------------------------------------ persist
    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted model (both heads) with joblib."""
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BloomClassifier":
        """Load a model previously written by :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj
