"""Chromatographic peak detection.

Public entry point: :func:`detect_peaks`, configured via
:class:`PeakDetectionConfig`. Consumes a
:class:`~pda_peak_finder.models.Chromatogram` (e.g. ``PDAData.maxplot()``)
and produces a :class:`~pda_peak_finder.models.PeakTable`.
"""

from __future__ import annotations

from .detect import PeakDetectionConfig, detect_peaks

__all__ = [
    "PeakDetectionConfig",
    "detect_peaks",
]
