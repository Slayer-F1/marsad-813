"""Synthetic Gulf-water hyperspectral scene generator for MARSAD 813.

Generates paired (contaminated, clean) remote-sensing reflectance Rrs spectra
on the 813 band grid (:mod:`marsad.spectra`), with per-pixel biogeochemistry
(chlorophyll-a, phycocyanin, suspended sediment, depth) and three classes:

* ``0 no_bloom``        - oligotrophic/mesotrophic Gulf water,
* ``1 dinoflagellate``  - red-tide blooms (Karenia/Cochlodinium), shallow AND
  deep sea pixels,
* ``2 cyanobacteria``   - toxic inland-reservoir blooms, always shallow,
  carrying the phycocyanin marker pigment.

Physics encoded (all spectral features built from
:func:`marsad.spectra.gaussian_feature`):

* **Base water** - blue-bright Rrs decaying exponentially with wavelength so
  reflectance is ~0 beyond ~900 nm and effectively black in the SWIR (water
  absorption dominates there).
* **Chlorophyll-a** - absorption dips at 443 nm (Soret band) and 675 nm (red
  band) whose depth scales with log-chl (pigment packaging makes absorption
  sublinear in concentration); a green reflectance peak near 555 nm; and for
  dense surface blooms a red-edge peak near 708 nm (scattering by the cell
  slick where water absorption already dominates).
* **Phycocyanin** - an extra absorption dip at 620 nm, essentially only for
  cyanobacteria pixels: it is the diagnostic that separates label 2 from a
  spectrally similar dinoflagellate bloom.
* **TSS** - broad mineral backscatter lift, stronger at short wavelengths
  (power-law slope varies per pixel), rolled off in the NIR/SWIR by water
  absorption.
* **Shallow bottom** (``rrs_observed`` only) - sandy bottom reflectance
  attenuated by the two-way diffuse path, ``R_bottom * exp(-2 * Kd * depth)``,
  with Kd increased by chl and TSS (more absorbing/scattering water hides the
  bottom faster). Beyond ~15 m the term is below the sensor noise floor, so it
  is only applied to shallow pixels.
* **Sunglint** (``rrs_observed`` only) - spectrally flat specular offset,
  nonzero even in the SWIR (which is why the SWIR-dark band is the classic
  glint probe).
* **Sensor noise** - 1-2 % multiplicative Gaussian on ``rrs_observed``.

The design goal is the project's core asymmetry: on *clean* spectra the three
classes are statistically separable (a linear classifier exceeds 0.85 holdout
accuracy), while bottom + sediment + glint contamination genuinely corrupts
naive band ratios on shallow/turbid pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marsad.spectra import BAND_GRID, N_BANDS, gaussian_feature

LABELS: dict[int, str] = {0: "no_bloom", 1: "dinoflagellate", 2: "cyanobacteria"}

# Scene class mixture (kept away from extreme imbalance so every synthetic
# scene contains all three regimes).
_CLASS_PROBS = (0.40, 0.32, 0.28)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically plain logistic; used for smooth 'dense bloom' onsets."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass
class SynthDataset:
    """One synthetic scene: contaminated + clean spectra and per-pixel truth."""

    rrs_observed: np.ndarray  # (n, N_BANDS) contaminated: bottom + sediment + glint + noise
    rrs_true: np.ndarray      # (n, N_BANDS) clean at-surface Rrs (the correction target)
    labels: np.ndarray        # (n,) int in {0,1,2}
    chl: np.ndarray           # (n,) chlorophyll-a mg/m3 (lognormal-ish, blooms high)
    tss: np.ndarray           # (n,) total suspended sediment g/m3
    depth_m: np.ndarray       # (n,) water depth; shallow pixels get bottom reflectance
    phycocyanin: np.ndarray   # (n,) pigment conc., >0 mainly for label 2


def generate_dataset(n_samples: int, seed: int = 0, shallow_fraction: float = 0.5) -> SynthDataset:
    """Generate a synthetic Gulf-water scene of ``n_samples`` pixels.

    Parameters
    ----------
    n_samples:
        Number of pixels.
    seed:
        Seed for ``numpy.random.default_rng`` (all randomness flows through it).
    shallow_fraction:
        Probability that a *sea* pixel (labels 0/1) is optically shallow.
        Cyanobacteria pixels (label 2) are inland-reservoir pixels and are
        always shallow, so the realised shallow fraction is >= this value.

    Returns
    -------
    SynthDataset
        ``rrs_true`` is the clean at-surface Rrs (Stage 1 regression target);
        ``rrs_observed`` adds bottom reflectance (shallow pixels), sunglint and
        1-2 % multiplicative sensor noise.
    """
    rng = np.random.default_rng(seed)
    n = int(n_samples)
    lam = BAND_GRID  # (B,) nm

    # ------------------------------------------------------------------ labels
    labels = rng.choice(3, size=n, p=_CLASS_PROBS)
    is_no = labels == 0
    is_dino = labels == 1
    is_cyano = labels == 2

    # ------------------------------------------------- biogeochemical state
    # Chl-a is lognormal within class: background water ~0.1-5 mg/m3, blooms
    # tens of mg/m3 (Gulf red tides routinely exceed 20 mg/m3).
    chl = np.empty(n, dtype=float)
    chl[is_no] = rng.lognormal(np.log(0.8), 0.7, is_no.sum())
    chl[is_dino] = rng.lognormal(np.log(25.0), 0.6, is_dino.sum())
    chl[is_cyano] = rng.lognormal(np.log(20.0), 0.6, is_cyano.sum())

    # Phycocyanin tracks biomass only for cyanobacteria (marker pigment).
    phycocyanin = np.zeros(n, dtype=float)
    phycocyanin[is_cyano] = chl[is_cyano] * rng.uniform(0.35, 0.85, is_cyano.sum())

    # Depth: sea pixels are shallow with probability `shallow_fraction`;
    # inland reservoirs (cyanobacteria) are always shallow.
    shallow = rng.random(n) < shallow_fraction
    shallow |= is_cyano
    depth_shallow = rng.uniform(0.8, 8.0, n)
    depth_deep = rng.uniform(15.0, 60.0, n)
    depth_m = np.where(shallow, depth_shallow, depth_deep)

    # TSS: lognormal; coastal shallow sea water is resuspension-prone (higher
    # and more variable), inland reservoirs are quiescent (low variability).
    tss = rng.lognormal(np.log(1.5), 0.8, n)
    coastal = shallow & ~is_cyano
    tss[coastal] *= rng.uniform(1.5, 4.0, coastal.sum())
    tss[is_cyano] = rng.lognormal(np.log(2.0), 0.25, is_cyano.sum())

    # -------------------------------------------------------- clean spectrum
    # Pigment forcing on a log scale (absorption saturates with packaging):
    # chl 0.1 mg/m3 -> 0, chl 100 mg/m3 -> 1.
    chlf = np.clip((np.log10(chl) + 1.0) / 3.0, 0.0, 1.0)
    pcf = phycocyanin / (phycocyanin + 15.0)  # saturating phycocyanin forcing

    # Base water: blue-bright exponential decay -> ~0 beyond ~900 nm, SWIR black.
    base_amp = rng.uniform(0.004, 0.010, n)
    decay_nm = rng.uniform(180.0, 320.0, n)
    base = base_amp[:, None] * np.exp(-(lam - 400.0)[None, :] / decay_nm[:, None])

    # TSS backscatter: power-law lift (steeper in the blue, per-pixel slope),
    # multiplied by an NIR/SWIR envelope because water absorption removes any
    # water-leaving signal at long wavelengths regardless of turbidity.
    slope = rng.uniform(0.5, 1.5, n)
    tss_amp = 0.0045 * tss / (tss + 4.0)
    nir_env = np.exp(-(((lam - 400.0) / 450.0) ** 2))
    tss_lift = (
        tss_amp[:, None]
        * (400.0 / lam)[None, :] ** slope[:, None]
        * nir_env[None, :]
    )

    # Diagnostic spectral features (unit-amplitude Gaussians on the band grid).
    g443 = gaussian_feature(443.0, 40.0)   # chl-a Soret absorption
    g555 = gaussian_feature(555.0, 70.0)   # green reflectance peak
    g620 = gaussian_feature(620.0, 30.0)   # phycocyanin absorption
    g675 = gaussian_feature(675.0, 30.0)   # chl-a red absorption
    g708 = gaussian_feature(708.0, 22.0)   # red-edge peak of dense blooms

    # Absorption dips act multiplicatively on the upwelling light (pigments
    # attenuate whatever light the water column would otherwise reflect).
    dips = (
        (0.55 * chlf)[:, None] * g443
        + (0.38 * chlf)[:, None] * g675
        + (0.50 * pcf)[:, None] * g620
    )
    transmit = np.clip(1.0 - dips, 0.05, None)

    # Red edge only appears for dense surface blooms (smooth onset ~25 mg/m3).
    rededge_amp = 0.0040 * _sigmoid((chl - 25.0) / 12.0)

    rrs_true = (
        (base + tss_lift) * transmit
        + (0.0035 * chlf)[:, None] * g555
        + rededge_amp[:, None] * g708
    )
    rrs_true = np.clip(rrs_true, 0.0, None)

    # ------------------------------------------------------- contamination
    # Sandy bottom: bright quasi-linear albedo ramp; the water column hides it
    # with the two-way diffuse attenuation exp(-2 * Kd * depth). Kd has the
    # pure-water spectral shape (small in the visible, huge beyond ~730 nm)
    # plus flat bio-optical terms that grow with chl and TSS.
    albedo = rng.uniform(0.12, 0.35, n)
    bottom_shape = 0.6 + 0.4 * (lam - 400.0) / 1300.0
    kd = (
        (0.045 + 2.5 * _sigmoid((lam - 730.0) / 55.0))[None, :]
        + (0.035 * np.log1p(chl) + 0.03 * np.log1p(tss))[:, None]
    )
    bottom = (
        (albedo / np.pi)[:, None]
        * bottom_shape[None, :]
        * np.exp(-2.0 * kd * depth_m[:, None])
    )
    # At 15+ m the exponential puts the bottom term below the noise floor;
    # restrict it to shallow pixels so deep observed/true spectra converge.
    bottom *= shallow[:, None]

    # Sunglint: spectrally flat specular offset, present in the SWIR too.
    glint = np.minimum(rng.exponential(0.0008, n), 0.006)

    # Sensor noise: 1-2 % multiplicative Gaussian (per-pixel noise level).
    noise_sigma = rng.uniform(0.01, 0.02, n)
    noise = 1.0 + noise_sigma[:, None] * rng.standard_normal((n, N_BANDS))

    rrs_observed = np.clip((rrs_true + bottom + glint[:, None]) * noise, 0.0, None)

    return SynthDataset(
        rrs_observed=rrs_observed,
        rrs_true=rrs_true,
        labels=labels,
        chl=chl,
        tss=tss,
        depth_m=depth_m,
        phycocyanin=phycocyanin,
    )


def generate_history(current_score: float, n_days: int = 30, seed: int = 0) -> np.ndarray:
    """Autocorrelated daily risk-score history ending exactly at ``current_score``.

    A bounded AR(1) random walk is unrolled *backwards in time* from the
    present: deviations from today's score follow ``d[t-1] = phi * d[t] + eps``
    (phi = 0.88, sigma = 0.045), so the history is smooth day to day, wanders
    plausibly on ~weekly scales, and by construction its last element equals
    ``current_score`` exactly. Excursions outside [0, 1] are reflected at the
    boundaries (a risk index saturates rather than clips into flat runs).

    Parameters
    ----------
    current_score:
        Today's risk score; clipped into [0, 1] before use.
    n_days:
        History length (must be >= 1); element ``-1`` is today.
    seed:
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Shape ``(n_days,)`` float array in [0, 1] with
        ``history[-1] == current_score`` (after clipping to [0, 1]).
    """
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    rng = np.random.default_rng(seed)
    current = float(np.clip(current_score, 0.0, 1.0))

    phi, sigma = 0.88, 0.045
    dev = np.empty(n_days, dtype=float)
    dev[-1] = 0.0
    for t in range(n_days - 2, -1, -1):
        dev[t] = phi * dev[t + 1] + rng.normal(0.0, sigma)

    hist = current + dev
    # Reflect into [0, 1] (valid for excursions within [-1, 2], far beyond the
    # AR(1) stationary range of ~ +/- 3 * sigma / sqrt(1 - phi^2) ~ 0.28).
    hist = np.abs(hist)
    hist = 1.0 - np.abs(1.0 - hist)
    hist = np.clip(hist, 0.0, 1.0)
    hist[-1] = current  # exact endpoint, regardless of float reflection
    return hist
