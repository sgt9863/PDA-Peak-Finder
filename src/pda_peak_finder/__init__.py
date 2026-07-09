"""PDA-Peak-Finder: peak detection and UV spectral analysis for PDA/UV 3D data.

Public surface is intentionally small: work through :mod:`pda_peak_finder.models`
data types and the high-level :mod:`pda_peak_finder.pipeline` /
:mod:`pda_peak_finder.reader` entry points.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .errors import (
    DataValidationError,
    PdaPeakFinderError,
    ReaderError,
)
from .models import (
    Chromatogram,
    InjectionMetadata,
    PDAData,
    Peak,
    PeakGroup,
    PeakTable,
    TrackingResult,
    UVSpectrum,
)

__all__ = [
    "__version__",
    "PdaPeakFinderError",
    "ReaderError",
    "DataValidationError",
    "PDAData",
    "InjectionMetadata",
    "Chromatogram",
    "UVSpectrum",
    "Peak",
    "PeakTable",
    "PeakGroup",
    "TrackingResult",
]
