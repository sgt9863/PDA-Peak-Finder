"""Cross-injection peak tracking (alignment).

Matches peaks that represent the same compound across multiple injections,
primarily by retention time and optionally constrained by lambda-max,
producing a :class:`~pda_peak_finder.models.TrackingResult` of
:class:`~pda_peak_finder.models.PeakGroup` objects.
"""

from __future__ import annotations

from .config import TrackingConfig
from .matcher import track_peaks

__all__ = ["TrackingConfig", "track_peaks"]
