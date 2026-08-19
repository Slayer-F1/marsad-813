"""Per-prediction uncertainty and calibration for MARSAD Stage 2.

Operational rationale
---------------------
A desalination operator cannot act on a bare class label. Shutting an
intake or switching to reservoir supply costs money, so the alert has to
carry how sure the model is and, crucially, *why* it is unsure. This
module answers that with a bagged ensemble of :class:`BloomClassifier`
and the standard information-theoretic split of predictive uncertainty
(Depeweg et al. 2018; Houlsby et al. 2011 "BALD"; Gal 2016):

- **total** = H(mean of member posteriors), the predictive entropy. All
  the uncertainty the system has about this pixel.
- **epistemic** = mutual information between the prediction and the model
  parameters = H(mean posterior) - mean of member entropies. Model
  ignorance: the members disagree, so this pixel is unlike the training
  water. It shrinks with more (or more diverse) training data.
- **aleatoric** = total - epistemic. Irreducible ambiguity: the members
  agree that the spectrum itself does not determine the class, e.g. a
  faint 620 nm phycocyanin dip that is equally consistent with a mixed
  dinoflagellate/sediment scene.
- **confidence** = the largest mean class probability.

Entropies are computed in NATS and normalised to [0, 1] by dividing by
log(3), the entropy of a uniform posterior over the three MARSAD classes
(0 = no_bloom, 1 = dinoflagellate, 2 = cyanobacteria). 0 means a
one-hot posterior, 1 means "no idea".

The split is what makes the number actionable: high *epistemic*
uncertainty is a data-coverage problem and the pixel should go to a human
analyst (see :func:`review_queue`); high *aleatoric* uncertainty will not
improve with retraining and needs a different observation, e.g. a field
sample or a higher-resolution overpass.

Calibration is measured with the expected calibration error and the
reliability curve of Naeini et al. (2015) and Guo et al. (2017): among
pixels the model calls 80 % likely, roughly 80 % should actually be that
class. An uncalibrated confidence is worse than no confidence at all,
because operators will learn to distrust the whole feed.

Ensembling follows bagging (Breiman 1996) as used for uncertainty by
Lakshminarayanan et al. (2017): every member fits an independent
bootstrap resample of the training set under its own seed.

Scientific honesty
------------------
Any calibration or uncertainty figure this project reports on data from
``marsad.synth`` is measured on OUR OWN forward model, so it is a
self-consistency check against a physics-based simulation, never
independent validation of real Gulf water. Honest calibration numbers
require the hindcast on archived GLORIA / PACE / 813 scenes over
documented bloom events.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np

from marsad.spectra import N_BANDS
from marsad.stage2_classifier import N_CLASSES, BloomClassifier

# Entropy of the uniform posterior over the 3 MARSAD classes, in nats.
# Dividing by it maps every entropy onto [0, 1].
MAX_ENTROPY_NATS = float(np.log(N_CLASSES))

#: Default routing threshold on normalised total uncertainty. Above it a
#: prediction is queued for a human analyst instead of auto-alerting.
#: --- TEAM DECISION --- operator policy, tune against analyst workload.
REVIEW_THRESHOLD = 0.35


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_probs(probs: np.ndarray) -> np.ndarray:
    """Validate/coerce a probability matrix to float64 ``(n, N_CLASSES)``.

    Negative dust is clipped and rows are renormalised, so slightly
    off-normal input from an upstream model is tolerated; genuinely
    malformed input (wrong width, non-finite, all-zero row) raises.
    """
    arr = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
        raise ValueError(
            f"expected probabilities of shape (n, {N_CLASSES}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("probabilities must be finite")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every probability row must have positive total mass")
    return arr / totals


def _normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats over the last axis, divided by log(3).

    Uses the ``0 * log(0) = 0`` convention, so a one-hot posterior gives
    exactly 0 and a uniform one exactly 1.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    terms = np.where(p > 0.0, p * np.log(np.where(p > 0.0, p, 1.0)), 0.0)
    return -terms.sum(axis=-1) / MAX_ENTROPY_NATS


def _bin_stats(
    probs: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence/accuracy/count per equal-width confidence bin.

    Bins partition [0, 1] into ``n_bins`` equal widths on the top-class
    probability; EMPTY BINS ARE DROPPED. Returns
    ``(mean_confidence, accuracy, count)``.
    """
    probs = _as_probs(probs)
    labels = np.asarray(labels).astype(int).ravel()
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("probs and labels must have matching lengths")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    # floor(conf * n_bins) with conf == 1.0 folded into the last bin.
    bin_id = np.clip((confidence * n_bins).astype(int), 0, n_bins - 1)

    counts = np.bincount(bin_id, minlength=n_bins)
    conf_sum = np.bincount(bin_id, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(bin_id, weights=correct, minlength=n_bins)

    keep = counts > 0
    counts_kept = counts[keep].astype(np.int64)
    return (
        conf_sum[keep] / counts_kept,
        acc_sum[keep] / counts_kept,
        counts_kept,
    )


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------

class EnsembleClassifier:
    """Bagged ensemble of :class:`BloomClassifier` for epistemic uncertainty.

    Each member is fitted on an independent bootstrap resample of the
    training set (draw n rows with replacement, Breiman 1996) and gets its
    own ``seed``, so the boosting heads are decorrelated as well. Where
    the members agree, the ensemble is interpolating water it has seen;
    where they disagree, the pixel is outside what MARSAD has been shown,
    which is exactly the epistemic term in :meth:`uncertainty`.

    Cost note (why the resample is materialised rather than passed as
    weights): ``BloomClassifier.fit`` accepts ``sample_weight``, and
    bootstrap multiplicities are mathematically the same thing, but
    sklearn's ``HistGradientBoosting*`` drops its constant-hessian fast
    path as soon as sample weights are supplied. Measured on this repo at
    n = 4000: about 7.5 s per member with an indexed resample versus
    about 32 s with weights. Materialising a (4000, 205) float64 view
    costs ~6.5 MB, so we buy the speed with memory we have. 5 members on
    4000 spectra therefore fits in roughly 40 s single-threaded, which is
    what the demo and a nightly retrain can afford; ``n_members`` is the
    dial if that budget changes.
    """

    def __init__(self, n_members: int = 5, seed: int = 0):
        if int(n_members) < 1:
            raise ValueError("n_members must be >= 1")
        self.n_members = int(n_members)
        self.seed = int(seed)
        self.members: list[BloomClassifier] = []
        self._fitted = False

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _bootstrap_index(
        rng: np.random.Generator, n: int, labels: np.ndarray
    ) -> np.ndarray:
        """Indices of one bootstrap resample: n draws with replacement.

        Any class that happens to be missed entirely is given one row
        back, taken from the most over-represented class. A member that
        never sees, say, cyanobacteria would report a structurally
        impossible posterior rather than genuine model disagreement, and
        would inflate the epistemic term for reasons that have nothing to
        do with the water.
        """
        idx = rng.integers(0, n, size=n)
        classes = np.unique(labels)
        counts = {int(c): int(np.count_nonzero(labels[idx] == c)) for c in classes}
        for cls in classes:
            if counts[int(cls)] > 0:
                continue
            donor = max(counts, key=lambda c: counts[c])
            position = int(np.flatnonzero(labels[idx] == donor)[0])
            idx[position] = rng.choice(np.flatnonzero(labels == cls))
            counts[donor] -= 1
            counts[int(cls)] = 1
        return idx

    def fit(
        self, rrs: np.ndarray, labels: np.ndarray, chl: np.ndarray
    ) -> "EnsembleClassifier":
        """Fit every member on its own bootstrap resample of the training set.

        Parameters
        ----------
        rrs : (n, N_BANDS) corrected remote-sensing reflectance.
        labels : (n,) ints in {0, 1, 2}.
        chl : (n,) chlorophyll-a in mg/m3.
        """
        rrs = np.atleast_2d(np.asarray(rrs, dtype=np.float64))
        if rrs.ndim != 2 or rrs.shape[1] != N_BANDS:
            raise ValueError(
                f"expected spectra of shape (n, {N_BANDS}), got {rrs.shape}"
            )
        labels = np.asarray(labels).astype(int).ravel()
        chl = np.asarray(chl, dtype=np.float64).ravel()
        if not (len(labels) == len(chl) == rrs.shape[0]):
            raise ValueError("rrs, labels and chl must have matching lengths")

        rng = np.random.default_rng(self.seed)
        n = rrs.shape[0]
        members: list[BloomClassifier] = []
        for m in range(self.n_members):
            idx = self._bootstrap_index(rng, n, labels)
            member = BloomClassifier(seed=self.seed + m)
            member.fit(rrs[idx], labels[idx], chl[idx])
            members.append(member)

        self.members = members
        self._fitted = True
        return self

    # -------------------------------------------------------------- predict
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "EnsembleClassifier is not fitted; call fit() first"
            )

    def member_probas(self, rrs: np.ndarray) -> np.ndarray:
        """Per-member class probabilities, ``(n_members, n, 3)``."""
        self._check_fitted()
        return np.stack([m.predict_proba(rrs) for m in self.members], axis=0)

    def predict_proba(self, rrs: np.ndarray) -> np.ndarray:
        """Ensemble mean class probabilities, ``(n, 3)``, rows summing to 1.

        Columns are in label order 0, 1, 2, inherited from
        :meth:`BloomClassifier.predict_proba`.
        """
        return self.member_probas(rrs).mean(axis=0)

    def predict(self, rrs: np.ndarray) -> np.ndarray:
        """Hard labels: argmax of the ensemble mean posterior."""
        return np.argmax(self.predict_proba(rrs), axis=1)

    def estimate_chl(self, rrs: np.ndarray) -> np.ndarray:
        """Ensemble mean chlorophyll-a estimate in mg/m3, ``(n,)``."""
        self._check_fitted()
        est = np.stack([m.estimate_chl(rrs) for m in self.members], axis=0)
        return np.clip(est.mean(axis=0), 0.0, None)

    def uncertainty(self, rrs: np.ndarray) -> dict[str, np.ndarray]:
        """Per-prediction uncertainty decomposition.

        Returns a dict of ``(n,)`` arrays:

        ``"total"``
            Predictive entropy of the ensemble mean posterior
            (aleatoric + epistemic), normalised to [0, 1].
        ``"epistemic"``
            Mutual information = total - mean of the member entropies.
            Members disagreeing about this pixel. Tiny negative values
            from floating-point cancellation are clamped to 0.
        ``"aleatoric"``
            total - epistemic: ambiguity the members agree on.
        ``"confidence"``
            Largest mean class probability, in [1/3, 1].
        """
        member = self.member_probas(rrs)
        mean_probs = member.mean(axis=0)

        total = np.clip(_normalised_entropy(mean_probs), 0.0, 1.0)
        mean_member_entropy = _normalised_entropy(member).mean(axis=0)
        # Jensen guarantees H(mean) >= mean(H); the clip only removes
        # float cancellation noise on near-identical members.
        epistemic = np.clip(total - mean_member_entropy, 0.0, None)
        epistemic = np.minimum(epistemic, total)
        aleatoric = np.clip(total - epistemic, 0.0, 1.0)
        confidence = mean_probs.max(axis=1)

        return {
            "total": total,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "confidence": confidence,
        }

    # ------------------------------------------------------------- persist
    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted ensemble (all members) with joblib."""
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnsembleClassifier":
        """Load an ensemble previously written by :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error of the top-class confidence.

    Naeini et al. (2015), popularised by Guo et al. (2017): bin the
    predictions by confidence, then average ``|accuracy - confidence|``
    over bins weighted by bin population. 0 is perfect calibration, 1 is
    the pathological case of full confidence in always-wrong predictions.

    Operationally this is the number that decides whether an operator can
    read "0.9" as "9 times out of 10".
    """
    conf, acc, count = _bin_stats(probs, labels, n_bins)
    if count.size == 0:
        return 0.0
    weights = count / count.sum()
    return float(np.clip(np.sum(weights * np.abs(acc - conf)), 0.0, 1.0))


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """Reliability (calibration) curve for the dashboard plot.

    Returns ``{"bin_confidence": [...], "bin_accuracy": [...],
    "bin_count": [...]}`` with one entry per NON-EMPTY confidence bin, as
    plain Python lists so the result drops straight into ``data.js``.
    A perfectly calibrated model plots on the diagonal
    ``bin_accuracy == bin_confidence``; points below the diagonal are
    overconfidence, the failure mode that erodes operator trust fastest.
    """
    conf, acc, count = _bin_stats(probs, labels, n_bins)
    return {
        "bin_confidence": [float(v) for v in conf],
        "bin_accuracy": [float(v) for v in acc],
        "bin_count": [int(v) for v in count],
    }


def review_queue(unc: dict, threshold: float = REVIEW_THRESHOLD) -> np.ndarray:
    """Boolean mask of predictions that should go to a human analyst.

    Parameters
    ----------
    unc : the dict returned by :meth:`EnsembleClassifier.uncertainty`.
    threshold : normalised total uncertainty above which the automatic
        alert is withheld. Default 0.35, roughly the entropy of a
        posterior that puts ~0.8 on its top class.

    True means "do not auto-alert on this pixel, show it to an analyst".
    Withholding is deliberately the only automated action here: MARSAD
    never silently downgrades a risk level on the grounds of uncertainty,
    it just refuses to speak for the model where the model is guessing.
    """
    if "total" not in unc:
        raise KeyError(
            "review_queue expects the dict from EnsembleClassifier.uncertainty, "
            "with a 'total' key"
        )
    total = np.atleast_1d(np.asarray(unc["total"], dtype=np.float64))
    return total > float(threshold)
