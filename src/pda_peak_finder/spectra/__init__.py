"""UV spectral analysis: peak apex spectrum extraction and lambda-max.

Given a :class:`~pda_peak_finder.models.PDAData` (the raw 3D absorbance
surface) and a :class:`~pda_peak_finder.models.PeakTable` (RT/height/area
already computed by peak detection), this module fills in the two fields
that require looking at the wavelength axis: ``Peak.spectrum`` (the UV
spectrum at the peak apex) and ``Peak.lambda_max`` (wavelength of maximum
absorbance in that spectrum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from ..errors import DataValidationError
from ..models import PDAData, Peak, PeakTable, UVSpectrum


@dataclass
class SpectrumConfig:
    """Options controlling how apex spectra are extracted and analyzed."""

    apex_average_scans: int = 0
    smoothing_window: int | None = None
    smoothing_polyorder: int = 3
    lambda_search_range: tuple[float, float] | None = None
    baseline_subtract: bool = False


def _apex_scan_index(data: PDAData, peak: Peak) -> int:
    """Resolve the scan index for a peak's apex.

    Uses ``peak.apex_index`` directly when it is a valid index into
    ``data.times``; otherwise falls back to the scan nearest ``peak.apex_time``
    (mirroring :meth:`PDAData.spectrum_at`).
    """
    n = len(data.times)
    idx = peak.apex_index
    if idx is not None and 0 <= idx < n:
        return int(idx)
    return int(np.argmin(np.abs(data.times - peak.apex_time)))


def extract_peak_spectrum(
    data: PDAData, peak: Peak, config: SpectrumConfig | None = None
) -> UVSpectrum:
    """Extract the UV spectrum at a peak's apex.

    The spectrum is taken at the scan nearest ``peak.apex_time`` (using
    ``peak.apex_index`` when it is a valid index), optionally averaged over
    neighboring scans, smoothed over the wavelength axis, and baseline
    subtracted, per ``config``.
    """
    config = config or SpectrumConfig()
    idx = _apex_scan_index(data, peak)

    if config.apex_average_scans > 0:
        lo = max(0, idx - config.apex_average_scans)
        hi = min(len(data.times) - 1, idx + config.apex_average_scans)
        values = data.absorbance[lo : hi + 1, :].mean(axis=0)
    else:
        values = data.absorbance[idx, :].copy()

    if config.smoothing_window is not None:
        values = savgol_filter(
            values,
            window_length=config.smoothing_window,
            polyorder=config.smoothing_polyorder,
        )

    if config.baseline_subtract:
        values = values - values.min()

    apex_time = float(data.times[idx])
    return UVSpectrum(
        wavelengths=data.wavelengths,
        values=values,
        time=apex_time,
        label=f"Peak apex RT {apex_time:.3f} min",
        injection_id=data.metadata.injection_id,
    )


def compute_lambda_max(
    spectrum: UVSpectrum, search_range: tuple[float, float] | None = None
) -> float:
    """Wavelength of maximum absorbance, optionally restricted to a window."""
    if search_range is not None:
        lo, hi = search_range
        mask = (spectrum.wavelengths >= lo) & (spectrum.wavelengths <= hi)
        if not mask.any():
            raise DataValidationError(f"no wavelength points in {lo:g}-{hi:g} nm")
        wavelengths = spectrum.wavelengths[mask]
        values = spectrum.values[mask]
    else:
        wavelengths = spectrum.wavelengths
        values = spectrum.values

    idx = int(np.argmax(values))
    return float(wavelengths[idx])


def annotate_peaks(
    data: PDAData, table: PeakTable, config: SpectrumConfig | None = None
) -> PeakTable:
    """Fill ``.spectrum`` and ``.lambda_max`` on every peak in ``table``.

    Mutates the peaks in place (and returns the same table) so callers can
    chain this onto whatever produced the table.
    """
    config = config or SpectrumConfig()
    for peak in table.peaks:
        spectrum = extract_peak_spectrum(data, peak, config)
        peak.spectrum = spectrum
        peak.lambda_max = compute_lambda_max(spectrum, config.lambda_search_range)
    return table


def absorbance_at(spectrum: UVSpectrum, wavelength: float) -> float:
    """Absorbance at the wavelength nearest ``wavelength`` in the spectrum."""
    idx = int(np.argmin(np.abs(spectrum.wavelengths - wavelength)))
    return float(spectrum.values[idx])


def filter_peaks_by_absorbance(
    table: PeakTable,
    wavelength: float,
    *,
    min_absorbance: float = 0.0,
    min_fraction: float = 0.0,
) -> PeakTable:
    """Drop peaks that barely absorb at a monitoring wavelength.

    A peak is kept when the absorbance of its apex spectrum at ``wavelength``
    is at least ``min_absorbance`` (absolute, AU) AND at least
    ``min_fraction`` of that spectrum's maximum absorbance. Peaks that have
    no extracted spectrum are conservatively kept (annotate first to filter
    them). Returns a NEW PeakTable; the input is not modified.

    Use this to ignore peaks with no / almost no absorption at, e.g., 230 nm.
    """
    kept: list[Peak] = []
    for peak in table.peaks:
        if peak.spectrum is None:
            kept.append(peak)
            continue
        a = absorbance_at(peak.spectrum, wavelength)
        spectrum_max = float(np.max(peak.spectrum.values))
        threshold = max(min_absorbance, min_fraction * spectrum_max)
        if a >= threshold:
            kept.append(peak)
    # Renumber kept peaks so peak_id stays contiguous (P001, P002, ...).
    for i, peak in enumerate(kept, start=1):
        peak.peak_id = f"P{i:03d}"
    return PeakTable(
        peaks=kept,
        injection_id=table.injection_id,
        source_label=table.source_label,
    )


__all__ = [
    "SpectrumConfig",
    "extract_peak_spectrum",
    "compute_lambda_max",
    "annotate_peaks",
    "absorbance_at",
    "filter_peaks_by_absorbance",
]
