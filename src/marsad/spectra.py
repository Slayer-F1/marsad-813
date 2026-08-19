"""Band definitions for the 813 hyperspectral instrument simulation.

The Arab 813 satellite carries a ~205-band imaging spectrometer spanning
400-1700 nm (VNIR + SWIR). Exact band centres are not public during the
hackathon, so the whole codebase treats this uniform grid as the single
source of truth; swap in the real centres here when GIQ publishes them.
"""
from __future__ import annotations

import numpy as np

N_BANDS = 205
BAND_GRID = np.linspace(400.0, 1700.0, N_BANDS)  # nm, ~6.4 nm spacing

# Diagnostic wavelengths (nm) used across the pipeline.
KEY_WAVELENGTHS = {
    "chl_absorption_blue": 443.0,  # chlorophyll-a Soret absorption
    "green_peak": 555.0,           # reflectance peak of productive water
    "phycocyanin": 620.0,          # cyanobacteria marker pigment
    "ndci_red": 665.0,
    "chl_absorption_red": 675.0,   # chlorophyll-a red absorption
    "ndci_rededge": 708.0,         # near-IR peak of dense surface blooms
    "swir_dark": 1600.0,           # water is ~black here; glint/atmosphere probe
}


def band_index(wavelength_nm: float) -> int:
    """Nearest band index for a wavelength in nm."""
    return int(np.argmin(np.abs(BAND_GRID - wavelength_nm)))


def band_slice(lo_nm: float, hi_nm: float) -> slice:
    """Contiguous band slice covering [lo_nm, hi_nm] inclusive."""
    lo, hi = band_index(lo_nm), band_index(hi_nm)
    return slice(min(lo, hi), max(lo, hi) + 1)


def gaussian_feature(center_nm: float, fwhm_nm: float, amplitude: float = 1.0) -> np.ndarray:
    """Gaussian spectral feature sampled on BAND_GRID.

    Used by the synthetic scene generator to place pigment absorption and
    fluorescence/backscatter peaks at diagnostic wavelengths.
    """
    sigma = fwhm_nm / 2.3548200450309493  # FWHM -> sigma
    return amplitude * np.exp(-0.5 * ((BAND_GRID - center_nm) / sigma) ** 2)
