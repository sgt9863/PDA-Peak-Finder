"""Chromatographic peak detection.

Turns a :class:`~pda_peak_finder.models.Chromatogram` (typically the MaxPlot
trace) into a :class:`~pda_peak_finder.models.PeakTable`. Detection itself is
a thin, well-understood wrapper around ``scipy.signal``:

* optional Savitzky-Golay smoothing before peak finding,
* :func:`scipy.signal.find_peaks` for apex candidates (height / prominence /
  minimum separation),
* :func:`scipy.signal.peak_widths` twice per peak — once at
  ``fwhm_rel_height`` for FWHM, once at ``bounds_rel_height`` for the
  integration window — followed by trapezoidal integration of the raw signal
  inside that window.

Nothing here does baseline correction or spectral work; that is the job of
other modules that consume the resulting :class:`Peak` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from ..models import Chromatogram, Peak, PeakTable


@dataclass
class PeakDetectionConfig:
    """Tunable parameters for :func:`detect_peaks`.

    All time-like parameters are in minutes, matching the package-wide unit
    convention; they are converted to samples internally using the
    chromatogram's :attr:`~pda_peak_finder.models.Chromatogram.sampling_interval`.
    """

    #: Minimum apex height (AU) passed to ``find_peaks``. ``None`` disables
    #: the height filter (prominence/distance still apply).
    min_height: float | None = None
    #: Minimum peak prominence (AU) passed to ``find_peaks``.
    min_prominence: float = 0.01
    #: Minimum separation between apexes, in minutes.
    min_distance_min: float = 0.05
    #: Savitzky-Golay smoothing window, in samples. Must be odd. ``None``
    #: (default) disables smoothing entirely.
    smoothing_window: int | None = None
    #: Savitzky-Golay polynomial order, used only when ``smoothing_window``
    #: is set.
    smoothing_polyorder: int = 3
    #: Relative height (fraction of prominence, 0-1) at which FWHM is
    #: measured. 0.5 = standard full-width-at-half-maximum.
    fwhm_rel_height: float = 0.5
    #: Relative height (fraction of prominence, 0-1) at which the
    #: integration start/end bounds are measured. Higher = wider window,
    #: closer to the peak base.
    bounds_rel_height: float = 0.9


def _index_to_time(times: np.ndarray, frac_index: float) -> float:
    """Linearly interpolate a fractional sample index (as returned by
    ``scipy.signal.peak_widths``) into a time value.
    """
    n = len(times)
    lo = int(np.floor(frac_index))
    hi = int(np.ceil(frac_index))
    lo = min(max(lo, 0), n - 1)
    hi = min(max(hi, 0), n - 1)
    if lo == hi:
        return float(times[lo])
    frac = frac_index - lo
    return float(times[lo] + frac * (times[hi] - times[lo]))


def _segment_area(times: np.ndarray, values: np.ndarray, start_time: float, end_time: float) -> float:
    """Trapezoidal area of ``values`` vs ``times`` between ``start_time`` and
    ``end_time`` (inclusive), interpolating the signal at both endpoints so
    the result does not depend on where samples happen to fall.
    """
    start_val = float(np.interp(start_time, times, values))
    end_val = float(np.interp(end_time, times, values))
    mask = (times > start_time) & (times < end_time)
    seg_times = np.concatenate(([start_time], times[mask], [end_time]))
    seg_values = np.concatenate(([start_val], values[mask], [end_val]))
    return float(np.trapezoid(seg_values, seg_times))


def detect_peaks(chromatogram: Chromatogram, config: PeakDetectionConfig | None = None) -> PeakTable:
    """Detect peaks in ``chromatogram`` and compute their properties.

    Returns a :class:`~pda_peak_finder.models.PeakTable` with peaks in
    elution order (increasing ``apex_time``), ``peak_id`` set to
    ``"P001"``, ``"P002"``, ... in that order, and ``injection_id`` copied
    from the chromatogram. ``lambda_max`` / ``spectrum`` are left as
    ``None`` — filling those is a downstream (spectral) module's job.
    """
    if config is None:
        config = PeakDetectionConfig()

    times = chromatogram.times
    raw_values = chromatogram.values

    values = raw_values
    if config.smoothing_window is not None:
        values = savgol_filter(
            raw_values,
            window_length=config.smoothing_window,
            polyorder=config.smoothing_polyorder,
        )

    sampling_interval = chromatogram.sampling_interval
    distance_samples = max(1, int(round(config.min_distance_min / sampling_interval)))

    find_peaks_kwargs: dict[str, float | int] = {
        "prominence": config.min_prominence,
        "distance": distance_samples,
    }
    if config.min_height is not None:
        find_peaks_kwargs["height"] = config.min_height

    peak_indices, _ = find_peaks(values, **find_peaks_kwargs)

    peaks: list[Peak] = []
    if len(peak_indices) > 0:
        _, _, fwhm_left_ips, fwhm_right_ips = peak_widths(
            values, peak_indices, rel_height=config.fwhm_rel_height
        )
        _, _, bounds_left_ips, bounds_right_ips = peak_widths(
            values, peak_indices, rel_height=config.bounds_rel_height
        )

        for i, idx in enumerate(peak_indices):
            start_time = _index_to_time(times, bounds_left_ips[i])
            end_time = _index_to_time(times, bounds_right_ips[i])
            fwhm_start = _index_to_time(times, fwhm_left_ips[i])
            fwhm_end = _index_to_time(times, fwhm_right_ips[i])

            peaks.append(
                Peak(
                    apex_time=float(times[idx]),
                    apex_index=int(idx),
                    height=float(raw_values[idx]),
                    start_time=start_time,
                    end_time=end_time,
                    fwhm=fwhm_end - fwhm_start,
                    area=_segment_area(times, raw_values, start_time, end_time),
                    injection_id=chromatogram.injection_id,
                )
            )

    peaks.sort(key=lambda p: p.apex_time)
    for i, peak in enumerate(peaks):
        peak.peak_id = f"P{i + 1:03d}"

    return PeakTable(
        peaks=peaks,
        injection_id=chromatogram.injection_id,
        source_label=chromatogram.label,
    )
