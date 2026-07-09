"""Chromatographic peak detection.

Public entry point: :func:`detect_peaks`, configured via
:class:`PeakDetectionConfig`. Consumes a
:class:`~pda_peak_finder.models.Chromatogram` (e.g. ``PDAData.maxplot()``)
and produces a :class:`~pda_peak_finder.models.PeakTable`.
"""

from __future__ import annotations

from .deconvolution import DeconvolutionConfig, detect_peaks_deconvolved
from .detect import PeakDetectionConfig, detect_peaks
from .filters import filter_peaks_by_height

__all__ = [
    "PeakDetectionConfig",
    "detect_peaks",
    "filter_peaks_by_height",
    "DeconvolutionConfig",
    "detect_peaks_deconvolved",
]
