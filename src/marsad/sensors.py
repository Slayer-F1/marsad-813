"""Multispectral sensor simulation and spectral resampling.

This module does two load-bearing jobs for MARSAD 813.

1. **Ingest path.** Real instruments deliver a handful of wide bands, not the
   205 contiguous channels of the 813 grid. :func:`resample` projects an
   813-grid spectrum onto any sensor's bands, and :func:`resample_to_grid`
   lifts sensor bands back onto the 813 grid, so archived Sentinel / MODIS /
   Landsat scenes can be pushed through the same pipeline once real data lands.

2. **The hyperspectral ablation.** Running the *same* MARSAD architecture on
   spectra that have been degraded to Sentinel-2, OLCI, MODIS or Landsat band
   sets is how we test whether hyperspectral sampling is the enabling sensor or
   merely a passenger. Because both directions of the round trip keep the array
   width at ``N_BANDS``, only the spectral *information* changes, never the
   model input dimension, so an accuracy drop cannot be blamed on architecture.

Radiometry: a band value is the spectral-response-weighted mean of Rrs over the
band, which is what a radiometer with that response function records. Each
response is modelled as a Gaussian of the published FWHM, sampled on
``BAND_GRID`` and normalised to unit sum (the grid is uniform, so a normalised
sum and a normalised integral are the same thing). Because the 813 grid steps
about 6.4 nm, a band narrower than that collapses towards nearest-neighbour
sampling. That is an honest limit of our band grid, not of the instrument.

Band tables are published centre / FWHM values clipped to the 400-1700 nm
range of the 813 instrument, so bands beyond 1700 nm are simply absent
(Sentinel-2 B12 at 2190 nm, Landsat-8 B7 at 2201 nm, MODIS band 7 at 2130 nm).

Sources for the band tables:

* Sentinel-2 MSI (S2A): ESA MSI band table; Drusch et al. 2012,
  Remote Sensing of Environment 120, 25-36.
* Sentinel-3 OLCI: ESA OLCI bands Oa1-Oa21; Donlon et al. 2012,
  Remote Sensing of Environment 120, 37-57.
* MODIS Aqua: NASA MODIS spectral band specification, ocean-colour and land
  bands; Esaias et al. 1998, IEEE TGRS 36, 1250-1265.
* Landsat-8 OLI: USGS Landsat 8 band designations; Roy et al. 2014,
  Remote Sensing of Environment 145, 154-172.

Scientific honesty rule (CONTRACTS-V2, binding): the spectra resampled here
come from ``synth.py``, which is our own forward model. An ablation that
degrades those simulated spectra to a Sentinel-2 or MODIS band set measures how
much information that band set carries *in our simulation*. It is a
self-consistency check against a physics-based simulation, consistent with the
Case-2 water literature, and it is never a measurement of how those instruments
perform on real Gulf water. Only the archived-scene hindcast can claim that.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .spectra import BAND_GRID, N_BANDS

__all__ = ["Band", "Sensor", "SENSORS", "resample", "resample_to_grid"]

# FWHM -> sigma for a Gaussian spectral response.
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.35482...

_GRID_LO = float(BAND_GRID[0])
_GRID_HI = float(BAND_GRID[-1])
_GRID_STEP = float(BAND_GRID[1] - BAND_GRID[0])

# A band whose unnormalised Gaussian response sums to less than this over
# BAND_GRID cannot be measured on our grid at all. Normalising it would divide
# by ~0 and manufacture a band out of numerical noise, so we refuse instead.
_MIN_SRF_SUPPORT = 1e-6


@dataclass(frozen=True)
class Band:
    """One spectral channel: centre wavelength and full width at half maximum.

    ``name`` is the operator-facing designation used by the agency that flies
    the instrument (``"B4"``, ``"Oa7"``, ``"band 13"``), so a reader can check
    our table against the published one.
    """

    name: str
    center_nm: float
    fwhm_nm: float


@dataclass(frozen=True)
class Sensor:
    """A band set plus an honest one-line note on what it can and cannot see.

    ``note`` is deliberately part of the data structure: the ablation table in
    ``benchmark.py`` prints it next to each accuracy number so that a drop in
    performance is always read together with the physical reason for it (no
    620 nm band, 1 km pixels, and so on) rather than as a bare score.
    """

    key: str
    label: str
    bands: tuple[Band, ...]
    note: str

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError(f"sensor {self.key!r} has no bands")
        for band in self.bands:
            if not np.isfinite(band.center_nm) or not np.isfinite(band.fwhm_nm):
                raise ValueError(f"{self.key}/{band.name}: non-finite band definition")
            if band.fwhm_nm <= 0.0:
                raise ValueError(f"{self.key}/{band.name}: FWHM must be positive")
            if not (_GRID_LO - 1e-9 <= band.center_nm <= _GRID_HI + 1e-9):
                raise ValueError(
                    f"{self.key}/{band.name}: centre {band.center_nm} nm is outside "
                    f"the 813 range {_GRID_LO:.0f}-{_GRID_HI:.0f} nm. Bands outside "
                    "the range must be dropped from the sensor definition."
                )

    @property
    def n_bands(self) -> int:
        """Number of bands in this sensor's table (after 400-1700 nm clipping)."""
        return len(self.bands)

    @property
    def centers_nm(self) -> np.ndarray:
        """(n_bands,) band centre wavelengths in nm, in table order."""
        return np.asarray([b.center_nm for b in self.bands], dtype=float)


def _native_bands() -> tuple[Band, ...]:
    """The 813 grid itself, as a Sensor band table (FWHM = grid spacing)."""
    return tuple(
        Band(name=f"B{i + 1:03d}", center_nm=float(w), fwhm_nm=_GRID_STEP)
        for i, w in enumerate(BAND_GRID)
    )


SENSORS: dict[str, Sensor] = {
    "marsad_813": Sensor(
        key="marsad_813",
        label="MARSAD 813 imaging spectrometer",
        bands=_native_bands(),
        note="205 contiguous bands, 400-1700 nm",
    ),
    # ESA Sentinel-2A MSI. B12 (2190 nm) is beyond the 813 range and is dropped.
    "sentinel2_msi": Sensor(
        key="sentinel2_msi",
        label="Sentinel-2 MSI",
        bands=(
            Band("B1", 443.0, 20.0),      # coastal aerosol
            Band("B2", 490.0, 65.0),      # blue
            Band("B3", 560.0, 35.0),      # green
            Band("B4", 665.0, 30.0),      # red
            Band("B5", 705.0, 15.0),      # red edge 1
            Band("B6", 740.0, 15.0),      # red edge 2
            Band("B7", 783.0, 20.0),      # red edge 3
            Band("B8", 842.0, 115.0),     # wide NIR
            Band("B8A", 865.0, 20.0),     # narrow NIR
            Band("B9", 945.0, 20.0),      # water vapour
            Band("B10", 1375.0, 30.0),    # cirrus
            Band("B11", 1610.0, 90.0),    # SWIR 1
        ),
        note=(
            "12 bands inside 400-1700 nm at 10-60 m, but no band at 620 nm, so "
            "phycocyanin is not directly observable and cyanobacteria cannot be "
            "separated from other blooms on pigment absorption."
        ),
    ),
    # ESA Sentinel-3 OLCI, bands Oa1-Oa21. The O2 A-band trio (Oa13-Oa15,
    # 761.25/764.375/767.5 nm) and the 940 nm water-vapour band Oa20 are
    # atmospheric sounding channels, not water-leaving radiance bands, so they
    # are not part of this water-relevant table.
    "sentinel3_olci": Sensor(
        key="sentinel3_olci",
        label="Sentinel-3 OLCI",
        bands=(
            Band("Oa1", 400.0, 15.0),
            Band("Oa2", 412.5, 10.0),
            Band("Oa3", 442.5, 10.0),
            Band("Oa4", 490.0, 10.0),
            Band("Oa5", 510.0, 10.0),
            Band("Oa6", 560.0, 10.0),
            Band("Oa7", 620.0, 10.0),     # phycocyanin marker
            Band("Oa8", 665.0, 10.0),
            Band("Oa9", 673.75, 7.5),
            Band("Oa10", 681.25, 7.5),    # chlorophyll fluorescence peak
            Band("Oa11", 708.75, 10.0),   # red edge
            Band("Oa12", 753.75, 7.5),
            Band("Oa16", 778.75, 15.0),
            Band("Oa17", 865.0, 20.0),
            Band("Oa18", 885.0, 10.0),
            Band("Oa19", 900.0, 10.0),
            Band("Oa21", 1020.0, 40.0),
        ),
        note=(
            "OLCI does carry a 620 nm band, the only operational ocean-colour "
            "sensor that does, but one 10 nm band at 300 m ground sampling "
            "cannot robustly separate phycocyanin absorption from co-varying "
            "sediment and chlorophyll absorption in the same pixel."
        ),
    ),
    # NASA MODIS Aqua, ocean-colour and land bands within 400-1700 nm.
    "modis_aqua": Sensor(
        key="modis_aqua",
        label="MODIS Aqua",
        bands=(
            Band("band 8", 412.0, 15.0),
            Band("band 9", 443.0, 10.0),
            Band("band 3", 469.0, 20.0),
            Band("band 10", 488.0, 10.0),
            Band("band 11", 531.0, 10.0),
            Band("band 12", 547.0, 10.0),
            Band("band 4", 555.0, 20.0),
            Band("band 1", 645.0, 50.0),
            Band("band 13", 667.0, 10.0),
            Band("band 14", 678.0, 10.0),   # fluorescence
            Band("band 15", 748.0, 10.0),
            Band("band 2", 859.0, 35.0),
            Band("band 16", 869.0, 15.0),
            Band("band 5", 1240.0, 20.0),
            Band("band 6", 1640.0, 50.0),
        ),
        note=(
            "No 620 nm band, so no phycocyanin route, and 1 km ocean-colour "
            "pixels average intake-scale patches away before they can be seen."
        ),
    ),
    # USGS Landsat-8 OLI reflective bands within 400-1700 nm. The 590 nm
    # panchromatic band, the 1373 nm cirrus band and B7 (2201 nm) are excluded:
    # the first two are not water-leaving radiance products, the third is out of
    # the 813 range.
    "landsat8_oli": Sensor(
        key="landsat8_oli",
        label="Landsat-8 OLI",
        bands=(
            Band("B1", 443.0, 16.0),      # coastal aerosol
            Band("B2", 482.0, 60.0),      # blue
            Band("B3", 561.0, 57.0),      # green
            Band("B4", 655.0, 37.0),      # red
            Band("B5", 865.0, 28.0),      # NIR
            Band("B6", 1609.0, 85.0),     # SWIR 1
        ),
        note=(
            "6 usable water bands, blue-green ratio only: no 620 nm band and no "
            "red-edge band, so neither speciation nor dense-bloom red edge is "
            "measurable, whatever the 30 m pixel size allows spatially."
        ),
    ),
}


def _resolve(sensor: Sensor | str) -> Sensor:
    """Accept either a Sensor or a SENSORS key."""
    if isinstance(sensor, Sensor):
        return sensor
    if isinstance(sensor, str):
        try:
            return SENSORS[sensor]
        except KeyError:
            raise KeyError(
                f"unknown sensor key {sensor!r}; known keys: {sorted(SENSORS)}"
            ) from None
    raise TypeError(f"sensor must be a Sensor or a str key, got {type(sensor).__name__}")


def _is_native_grid(sensor: Sensor) -> bool:
    """True when the sensor's centres are the 813 grid itself."""
    return sensor.n_bands == N_BANDS and bool(
        np.allclose(sensor.centers_nm, BAND_GRID, rtol=0.0, atol=1e-9)
    )


@lru_cache(maxsize=None)
def _srf_matrix(sensor: Sensor) -> np.ndarray:
    """(n_bands, N_BANDS) matrix of normalised Gaussian spectral responses.

    Row ``j`` holds the weights that turn a spectrum on ``BAND_GRID`` into the
    reading of band ``j``: a Gaussian centred on ``center_nm`` with the
    published FWHM, sampled on the grid and divided by its own sum so the band
    reports a weighted mean reflectance rather than an unnormalised integral.

    An exact-grid sensor (the 813 instrument itself) is the identity: it *is*
    the grid, so resampling it onto its own bands must return the input
    unchanged. Convolving it with a Gaussian of its own band width would smear
    exactly the narrow pigment lines (620 nm phycocyanin, 675 nm chlorophyll)
    that the instrument exists to resolve.

    Raises ValueError when a band has effectively no response on the grid; such
    a band has to be removed from the Sensor definition rather than silently
    contributing a zero column to every downstream model.
    """
    if _is_native_grid(sensor):
        srf = np.eye(N_BANDS, dtype=float)
        srf.flags.writeable = False
        return srf

    rows = np.empty((sensor.n_bands, N_BANDS), dtype=float)
    for j, band in enumerate(sensor.bands):
        sigma = band.fwhm_nm / _FWHM_TO_SIGMA
        weights = np.exp(-0.5 * ((BAND_GRID - band.center_nm) / sigma) ** 2)
        total = float(weights.sum())
        if total < _MIN_SRF_SUPPORT:
            raise ValueError(
                f"{sensor.key}/{band.name}: spectral response has no support on "
                f"the {_GRID_LO:.0f}-{_GRID_HI:.0f} nm grid (weight sum "
                f"{total:.3e}). Drop this band from the Sensor definition."
            )
        rows[j] = weights / total
    rows.flags.writeable = False
    return rows


def _as_2d(arr: np.ndarray, n_expected: int, what: str) -> tuple[np.ndarray, bool]:
    """Coerce to float64 (n, n_expected); report whether input was a single row."""
    out = np.asarray(arr, dtype=float)
    was_1d = out.ndim == 1
    if was_1d:
        out = out[None, :]
    if out.ndim != 2 or out.shape[1] != n_expected:
        raise ValueError(
            f"{what} must have shape (n, {n_expected}) or ({n_expected},), "
            f"got {np.shape(arr)}"
        )
    return out, was_1d


def resample(rrs: np.ndarray, sensor: Sensor | str) -> np.ndarray:
    """Project 813-grid spectra onto a sensor's bands.

    Parameters
    ----------
    rrs
        (n, N_BANDS) remote-sensing reflectance on ``BAND_GRID``. A single
        (N_BANDS,) spectrum is accepted and returns a 1-D result.
    sensor
        A :class:`Sensor` or a key of :data:`SENSORS`.

    Returns
    -------
    (n, sensor.n_bands) band-averaged reflectance, that is, what the sensor
    would report if it viewed the same water. Weights are non-negative and sum
    to one per band, so a non-negative spectrum stays non-negative and a flat
    spectrum is reproduced exactly at every band.

    The ``"marsad_813"`` sensor is the identity to within floating point: the
    813 grid resampled onto itself is the same spectrum.
    """
    sen = _resolve(sensor)
    arr, was_1d = _as_2d(rrs, N_BANDS, "rrs")
    out = arr @ _srf_matrix(sen).T
    return out[0] if was_1d else out


def resample_to_grid(rrs_sensor: np.ndarray, sensor: Sensor | str) -> np.ndarray:
    """Lift sensor bands back onto the 813 grid by linear interpolation.

    Piecewise-linear between band centres, edge hold outside them (the value of
    the first / last band is carried to 400 nm and to 1700 nm). This is the
    generous reading of a multispectral measurement: everything the band set
    determines about the full spectrum, and nothing more. Whatever the bands do
    not sample gets filled in by a straight line, which is exactly why a
    sensor with no 620 nm band cannot show a phycocyanin dip: the dip is
    interpolated away between the green and red bands.

    The ablation uses this so that every model sees ``N_BANDS`` inputs and only
    the information content differs, never the array width.

    Parameters
    ----------
    rrs_sensor
        (n, sensor.n_bands) band values, for example the output of
        :func:`resample`. A single (n_bands,) row is accepted and returns 1-D.
    sensor
        A :class:`Sensor` or a key of :data:`SENSORS`.

    Returns
    -------
    (n, N_BANDS) reflectance on ``BAND_GRID``.
    """
    sen = _resolve(sensor)
    arr, was_1d = _as_2d(rrs_sensor, sen.n_bands, "rrs_sensor")

    centers = sen.centers_nm
    order = np.argsort(centers, kind="stable")
    x = centers[order]
    y = arr[:, order]

    if x.size == 1:
        # One band determines a constant spectrum, nothing to interpolate.
        out = np.repeat(y, N_BANDS, axis=1)
        return out[0] if was_1d else out

    # Bracketing interval per grid wavelength, then clip the interpolation
    # weight to [0, 1] so extrapolation beyond the outer bands becomes an
    # edge hold instead of a runaway straight line.
    idx = np.clip(np.searchsorted(x, BAND_GRID, side="right") - 1, 0, x.size - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    span = x1 - x0
    w = np.where(span > 0.0, (BAND_GRID - x0) / np.where(span > 0.0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)

    y0 = y[:, idx]
    y1 = y[:, idx + 1]
    out = y0 + (y1 - y0) * w[None, :]
    return out[0] if was_1d else out
