"""Standard operational HAB algorithms - the comparison MARSAD 813 must beat.

Everything in this module is a published band-ratio or line-height algorithm
exactly as a water-quality operator would run it today: no learning, no
shallow-water correction, one scene in, one number per pixel out. Each
function takes remote-sensing reflectance ``Rrs`` on the 813 band grid
(:mod:`marsad.spectra`), shape ``(n, N_BANDS)`` in sr^-1, and returns shape
``(n,)``.

Algorithms and sources
----------------------
- **OC4** (:func:`oc4_chl`) - O'Reilly et al. 1998, *Ocean color chlorophyll
  algorithms for SeaWiFS*, JGR 103(C11):24937-24953; 4th-order polynomial in
  the log maximum band ratio, NASA OBPG SeaWiFS coefficients. The open-ocean
  workhorse, and the algorithm Case-2 (coastal, optically complex) water is
  known to break: bottom reflectance, mineral sediment and sunglint all move
  the blue-green ratio without any change in chlorophyll.
- **OC3M** (:func:`oc3m_chl`) - O'Reilly et al. 2000, SeaWiFS Postlaunch
  Technical Report Series Vol. 11; the MODIS-Aqua 3-band variant.
- **NDCI** (:func:`ndci`, :func:`ndci_chl`) - Mishra & Mishra 2012, Remote
  Sensing of Environment 117:394-406. Red / red-edge index built for turbid
  productive water, where the blue-green ratio has already failed.
- **Two-band NIR-red ratio** (:func:`red_nir_ratio`) - Gitelson et al. 2008 /
  Dall'Olmo & Gitelson 2005; Rrs(708)/Rrs(665) tracks the red-edge peak of
  dense surface blooms.
- **Phycocyanin line height** (:func:`phycocyanin_line_height`) - continuum
  removal at the 620 nm phycocyanin absorption band, in the spirit of the
  cyanobacterial line-height indices of Simis et al. 2005 and Wynne et al.
  2008. This is the only classical route to cyanobacteria speciation, and it
  cannot exist at all on a sensor without a band near 620 nm.
- **Turbidity** (:func:`turbidity_proxy`) - Nechad et al. 2010, RSE
  114:854-866, single-band semi-analytical form, red-band calibration as
  reported by Dogliotti et al. 2015, RSE 156:157-168.
- **Operator decision tree** (:func:`classify_baseline`) - bloom threshold on
  NDCI-derived chlorophyll, then phycocyanin line height to separate
  cyanobacteria from dinoflagellates. This is today's manual workflow, and it
  is what MARSAD's Stage 2 must beat.

Numerical policy
----------------
Real and simulated Rrs can be zero (deep SWIR bands) or negative (atmospheric
over-correction). Every division and logarithm here is guarded so that no
input can produce NaN, inf, or a numpy warning: reflectances entering a ratio
are floored at :data:`_RRS_FLOOR`, sums entering a normalised difference are
tested before dividing, and retrieved chlorophyll is clipped to the
non-negative, finite range set by :data:`_LOG10_CHL_LIMITS`. Band centres are
the nominal literature wavelengths mapped to the nearest 813 band (grid
spacing ~6.4 nm, so the mismatch is at most ~3 nm), which is exactly the
resampling an operator performs when moving an algorithm between sensors.

Scientific honesty
------------------
These baselines are the reference MARSAD 813 is measured against. Any score
obtained by running them on :mod:`marsad.synth` output is a self-consistency
check against our own physics-based forward model, consistent with the Case-2
water literature - never independent validation on real Gulf water. Real
validation is the hindcast on documented events once GLORIA / PACE / 813
scenes are in hand.
"""
from __future__ import annotations

import numpy as np

from marsad.spectra import BAND_GRID, N_BANDS, band_index

__all__ = [
    "OC4_COEFFS",
    "OC3M_COEFFS",
    "NDCI_CHL_COEFFS",
    "NECHAD_A_FNU",
    "NECHAD_C",
    "oc4_chl",
    "oc3m_chl",
    "ndci",
    "ndci_chl",
    "red_nir_ratio",
    "phycocyanin_line_height",
    "turbidity_proxy",
    "classify_baseline",
]

# --- algorithm coefficients (ascending polynomial order, as published) ------

#: NASA OBPG OC4 (SeaWiFS) coefficients for log10(chl) in log10 band ratio.
OC4_COEFFS: tuple[float, ...] = (0.3272, -2.9940, 2.7218, -1.2259, -0.5683)
#: NASA OBPG OC3M (MODIS-Aqua) coefficients, same polynomial form.
OC3M_COEFFS: tuple[float, ...] = (0.2424, -2.7423, 1.8017, 0.0015, -1.2280)
#: Mishra & Mishra 2012 NDCI -> chl-a coefficients (mg m^-3).
NDCI_CHL_COEFFS: tuple[float, ...] = (14.039, 86.115, 194.325)
#: Nechad/Dogliotti red-band turbidity calibration: A in FNU, C dimensionless.
NECHAD_A_FNU: float = 228.1
NECHAD_C: float = 0.1641

# --- numerical guards ------------------------------------------------------

# Water-leaving Rrs in the visible is ~1e-3 sr^-1, so a 1e-6 sr^-1 floor sits
# three orders of magnitude below any real signal: it never perturbs a valid
# retrieval and only bites on zero or negative (over-corrected) input, where
# it keeps ratios finite instead of dividing by zero.
_RRS_FLOOR = 1e-6
# Saturation on the retrieved log10(chl). Band-ratio polynomials are 4th
# order and explode outside their calibration range; clipping to
# 1e-3..1e4 mg m^-3 keeps outputs finite and safely brackets every physically
# meaningful concentration (open-ocean desert to surface scum).
_LOG10_CHL_LIMITS = (-3.0, 4.0)
# The Nechad form diverges as rho_w -> C; hold just short of the pole so a
# saturated pixel returns a large finite turbidity instead of inf.
_NECHAD_RHO_MAX = 0.95 * NECHAD_C


def _as_spectra(rrs: np.ndarray) -> np.ndarray:
    """Coerce input to a finite float64 ``(n, N_BANDS)`` array.

    A single spectrum of shape ``(N_BANDS,)`` is accepted and promoted.
    Non-finite samples (a real possibility in L2 products with masked or
    saturated pixels) are replaced by zero, so downstream guards see a plain
    "no signal" band rather than a NaN that would silently poison a median.
    """
    arr = np.atleast_2d(np.asarray(rrs, dtype=np.float64))
    if arr.ndim != 2 or arr.shape[1] != N_BANDS:
        raise ValueError(
            f"expected spectra of shape (n, {N_BANDS}), got {arr.shape}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _band(rrs: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """Column of ``rrs`` at the 813 band nearest ``wavelength_nm``."""
    return rrs[:, band_index(wavelength_nm)]


def _floored(x: np.ndarray) -> np.ndarray:
    """Clip a reflectance column to the strictly positive retrieval floor."""
    return np.maximum(x, _RRS_FLOOR)


def _chl_from_band_ratio(log_ratio: np.ndarray, coeffs: tuple[float, ...]) -> np.ndarray:
    """Evaluate an OCx polynomial: ``chl = 10 ** sum(a_i * R**i)``."""
    log_chl = np.polynomial.polynomial.polyval(log_ratio, coeffs)
    log_chl = np.clip(log_chl, _LOG10_CHL_LIMITS[0], _LOG10_CHL_LIMITS[1])
    return np.clip(10.0**log_chl, 0.0, None)


def _line_height(
    rrs: np.ndarray, lo_nm: float, mid_nm: float, hi_nm: float
) -> np.ndarray:
    """Depth of ``Rrs(mid)`` below the straight continuum ``lo -> hi``.

    Continuum removal is the classical way to isolate a narrow pigment
    absorption feature from the broad brightness of the water: a positive
    return means the spectrum dips below the chord (absorption), a negative
    one means it peaks above it (scattering or fluorescence). Interpolation
    uses the actual 813 band centres, not the nominal wavelengths.
    """
    i_lo, i_mid, i_hi = band_index(lo_nm), band_index(mid_nm), band_index(hi_nm)
    span = BAND_GRID[i_hi] - BAND_GRID[i_lo]
    w = (BAND_GRID[i_mid] - BAND_GRID[i_lo]) / span
    continuum = (1.0 - w) * rrs[:, i_lo] + w * rrs[:, i_hi]
    return continuum - rrs[:, i_mid]


def oc4_chl(rrs: np.ndarray) -> np.ndarray:
    """NASA OC4 chlorophyll-a (mg m^-3) from the maximum blue-green ratio.

    ``R = log10(max(Rrs443, Rrs490, Rrs510) / Rrs555)`` fed to the 4th-order
    polynomial of O'Reilly et al. 1998 with the OBPG SeaWiFS coefficients
    :data:`OC4_COEFFS`. Physically: chlorophyll-a absorbs strongly in the blue
    (Soret band) while the green stays comparatively bright, so the blue-green
    ratio collapses as biomass rises. Taking the maximum of three blue-green
    bands keeps the ratio in a well-conditioned range across three orders of
    magnitude of chlorophyll.

    The assumption underneath is Case-1 water, where every optical property
    co-varies with phytoplankton. In Case-2 Gulf-coastal water it fails in the
    direction that matters operationally: bright sandy bottom, resuspended
    sediment and sunglint all add signal in the green and red, pushing the
    ratio down and the retrieval up, so clear shallow water can be reported as
    a bloom while a genuine bloom over sand can be diluted away.

    Returns non-negative finite values; zero or negative input reflectance is
    floored rather than divided by, so no NaN, inf or warning is produced.
    """
    rrs = _as_spectra(rrs)
    blue = np.maximum.reduce(
        [_band(rrs, 443.0), _band(rrs, 490.0), _band(rrs, 510.0)]
    )
    green = _band(rrs, 555.0)
    log_ratio = np.log10(_floored(blue) / _floored(green))
    return _chl_from_band_ratio(log_ratio, OC4_COEFFS)


def oc3m_chl(rrs: np.ndarray) -> np.ndarray:
    """NASA OC3M (MODIS-Aqua) chlorophyll-a (mg m^-3).

    Same empirical family as :func:`oc4_chl` with two blue bands instead of
    three: ``R = log10(max(Rrs443, Rrs488) / Rrs547)`` and coefficients
    :data:`OC3M_COEFFS` (O'Reilly et al. 2000). Included because MODIS-Aqua is
    the sensor most Gulf monitoring programmes actually operate on, so it is
    the number an operator is most likely to be looking at today.

    On the 813 grid (~6.4 nm bands) the nominal 488 nm and 490 nm centres fall
    in the same band, so OC3M differs from OC4 here through its coefficients
    and its 547 nm reference band rather than through the blue bands.

    Same numerical guarantees as :func:`oc4_chl`.
    """
    rrs = _as_spectra(rrs)
    blue = np.maximum(_band(rrs, 443.0), _band(rrs, 488.0))
    green = _band(rrs, 547.0)
    log_ratio = np.log10(_floored(blue) / _floored(green))
    return _chl_from_band_ratio(log_ratio, OC3M_COEFFS)


def ndci(rrs: np.ndarray) -> np.ndarray:
    """Normalised Difference Chlorophyll Index, ``(Rrs708 - Rrs665)/(sum)``.

    Mishra & Mishra 2012. Dense blooms push a reflectance peak to ~708 nm
    (cell scattering where water absorption already suppresses the background)
    while chlorophyll-a red absorption near 665-675 nm deepens, so the index
    rises with bloom density. Because both bands sit in the red / red-edge,
    NDCI is far less sensitive to blue-scattering sediment than a blue-green
    ratio - which is why it, not OCx, is the turbid-water standard.

    Negative reflectance is floored at zero (reflectance cannot be negative)
    before the difference; if both bands carry no signal the index is 0. The
    result is finite and clipped to [-1, 1].
    """
    rrs = _as_spectra(rrs)
    r665 = np.maximum(_band(rrs, 665.0), 0.0)
    r708 = np.maximum(_band(rrs, 708.0), 0.0)
    total = r665 + r708
    valid = total > _RRS_FLOOR
    safe_total = np.where(valid, total, 1.0)
    out = np.where(valid, (r708 - r665) / safe_total, 0.0)
    return np.clip(out, -1.0, 1.0)


def ndci_chl(rrs: np.ndarray) -> np.ndarray:
    """Chlorophyll-a (mg m^-3) from NDCI, Mishra & Mishra 2012.

    ``chl = 14.039 + 86.115 * NDCI + 194.325 * NDCI**2`` (:data:`NDCI_CHL_COEFFS`).

    Two operational caveats worth stating plainly, because they drive the
    baseline decision tree in :func:`classify_baseline`. First, the quadratic
    was calibrated on productive turbid water and has a floor near
    14 mg m^-3 at NDCI = 0, so it cannot report oligotrophic concentrations at
    all. Second, it is a parabola: applied outside its calibration range
    (strongly negative NDCI, i.e. clear water where the red edge is dark) it
    turns back upward and reports high chlorophyll for clear water. Both are
    real failure modes of running the published formula as published, and we
    keep the formula as published.

    Output is clipped to non-negative and is finite for any input.
    """
    return np.clip(
        np.polynomial.polynomial.polyval(ndci(rrs), NDCI_CHL_COEFFS), 0.0, None
    )


def red_nir_ratio(rrs: np.ndarray) -> np.ndarray:
    """Gitelson two-band NIR-red ratio ``Rrs708 / Rrs665``.

    Dall'Olmo & Gitelson 2005; Gitelson et al. 2008. The simplest surface-bloom
    indicator: values below ~1 mean the red edge is darker than the red band
    (ordinary water), values above ~1 mean a scattering surface bloom. Used
    here as an independent bloom-intensity baseline alongside NDCI.

    The denominator is floored at the retrieval floor and the numerator at
    zero, so the ratio is always finite and non-negative.
    """
    rrs = _as_spectra(rrs)
    r665 = _floored(_band(rrs, 665.0))
    r708 = np.maximum(_band(rrs, 708.0), 0.0)
    return np.clip(r708 / r665, 0.0, None)


def phycocyanin_line_height(rrs: np.ndarray) -> np.ndarray:
    """Depth of the 620 nm phycocyanin absorption below its 600-650 continuum.

    A straight baseline is drawn from Rrs(600) to Rrs(650), evaluated at
    620 nm, and Rrs(620) subtracted from it: positive means the spectrum dips
    at 620 nm, the signature of phycocyanin, the accessory pigment carried by
    cyanobacteria and by nothing else in this problem. Line-height continuum
    removal in the spirit of Simis et al. 2005 and the cyanobacterial index of
    Wynne et al. 2008; units are sr^-1, the same as Rrs.

    This is the only classical route to cyanobacteria speciation, and it is
    purely a function of having a band at 620 nm: on Sentinel-2 or MODIS,
    which carry none, the feature is not merely noisy, it is unobservable.
    That is the ablation argument for a hyperspectral instrument, and it is
    also why the number is small - a few 1e-4 sr^-1 - and therefore fragile in
    the presence of bottom reflectance and sediment, which tilt the same
    continuum.

    Pure differencing, so no division and no NaN for any input.
    """
    return _line_height(_as_spectra(rrs), 600.0, 620.0, 650.0)


def turbidity_proxy(rrs: np.ndarray) -> np.ndarray:
    """Single-band red turbidity (FNU), Nechad-style, driven by Rrs(665).

    Semi-analytical single-band form of Nechad et al. 2010 with the red-band
    calibration of Dogliotti et al. 2015::

        T = A * rho_w / (1 - rho_w / C),   rho_w = pi * Rrs(665)

    Note the calibration coefficients :data:`NECHAD_A_FNU` and
    :data:`NECHAD_C` are the published 645 nm red-band pair applied at 665 nm,
    so this is a proxy with the right shape and roughly the right scale, not a
    calibrated turbidity product; treat the numbers as relative.

    Physically, red reflectance is dominated by particle backscatter because
    water absorption removes the water-leaving path radiance, so a single red
    band tracks suspended particle load; the hyperbolic term accounts for the
    saturation of reflectance at high load. In MARSAD this is a context
    variable, not a bloom detector: it is what tells an operator whether an
    elevated chlorophyll retrieval might just be sediment.

    Negative reflectance is floored at zero and ``rho_w`` is held just short of
    the pole at ``C``, so a saturated pixel returns a large finite turbidity
    rather than inf. Output is non-negative.
    """
    rrs = _as_spectra(rrs)
    rho_w = np.pi * np.maximum(_band(rrs, 665.0), 0.0)
    rho_w = np.minimum(rho_w, _NECHAD_RHO_MAX)
    return np.clip(NECHAD_A_FNU * rho_w / (1.0 - rho_w / NECHAD_C), 0.0, None)


def classify_baseline(
    rrs: np.ndarray,
    chl_threshold: float = 10.0,
    pc_lh_threshold: float = 0.0005,
) -> np.ndarray:
    """Classical operator decision tree over the baseline indices.

    The workflow a monitoring analyst runs today, encoded literally:

    1. bloom if ``ndci_chl >= chl_threshold`` (mg m^-3), else ``0 no_bloom``;
    2. within a bloom, ``2 cyanobacteria`` if the 620 nm phycocyanin line
       height reaches ``pc_lh_threshold`` (sr^-1), else ``1 dinoflagellate``.

    Defaults follow common practice: 10 mg m^-3 is a widely used bloom-alert
    chlorophyll level, and 5e-4 sr^-1 is a conservative detection level for the
    620 nm feature at typical ocean-colour magnitudes. Both are policy knobs,
    not physics.

    Returns ``(n,)`` int labels in ``{0, 1, 2}`` matching
    ``marsad.synth.LABELS``. This is the accuracy MARSAD's Stage 2 must beat,
    and - per the project honesty rule - beating it on our own synthetic scenes
    is a self-consistency check against our forward model, not evidence about
    real Gulf water.
    """
    rrs = _as_spectra(rrs)
    bloom = ndci_chl(rrs) >= float(chl_threshold)
    cyano = phycocyanin_line_height(rrs) >= float(pc_lh_threshold)
    labels = np.zeros(rrs.shape[0], dtype=np.int64)
    labels[bloom] = 1
    labels[bloom & cyano] = 2
    return labels
