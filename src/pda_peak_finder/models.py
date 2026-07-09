"""Core data models.

Every module in this package communicates through the types defined here.
File readers produce :class:`PDAData`; all downstream processing (peak
detection, spectrum extraction, tracking, export, plotting) consumes these
models and never touches raw files. This is what keeps the pipeline
independent of the ARW file format.

Unit conventions (fixed across the whole package):

* time        -> minutes
* wavelength  -> nm
* absorbance  -> AU
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .errors import DataValidationError


def _as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise DataValidationError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class InjectionMetadata:
    """Identity of one injection (= one acquired 3D dataset).

    ``injection_id`` is the key used everywhere downstream (peak tables,
    tracking, CSV output). Readers must fill it; when the source file has no
    natural identifier, the file stem is a good default.
    """

    injection_id: str
    sample_name: str = ""
    acquired_at: datetime | None = None
    instrument: str = ""
    channel: str = ""
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chromatogram:
    """A 1-D signal over time (MaxPlot, single-wavelength trace, ...)."""

    times: np.ndarray   # (T,) minutes, strictly increasing
    values: np.ndarray  # (T,) AU
    label: str = ""
    injection_id: str = ""

    def __post_init__(self) -> None:
        self.times = _as_1d_float("times", self.times)
        self.values = _as_1d_float("values", self.values)
        if self.times.shape != self.values.shape:
            raise DataValidationError(
                f"times {self.times.shape} and values {self.values.shape} differ"
            )
        if len(self.times) > 1 and not np.all(np.diff(self.times) > 0):
            raise DataValidationError("times must be strictly increasing")

    @property
    def sampling_interval(self) -> float:
        """Median sampling interval in minutes."""
        return float(np.median(np.diff(self.times)))


@dataclass
class UVSpectrum:
    """Absorbance vs wavelength at (or averaged around) one retention time."""

    wavelengths: np.ndarray  # (W,) nm, strictly increasing
    values: np.ndarray       # (W,) AU
    time: float | None = None  # RT the spectrum was taken at (minutes)
    label: str = ""
    injection_id: str = ""

    def __post_init__(self) -> None:
        self.wavelengths = _as_1d_float("wavelengths", self.wavelengths)
        self.values = _as_1d_float("values", self.values)
        if self.wavelengths.shape != self.values.shape:
            raise DataValidationError(
                f"wavelengths {self.wavelengths.shape} and values "
                f"{self.values.shape} differ"
            )
        if len(self.wavelengths) > 1 and not np.all(np.diff(self.wavelengths) > 0):
            raise DataValidationError("wavelengths must be strictly increasing")


@dataclass
class PDAData:
    """One injection's full 3D dataset: absorbance over time x wavelength.

    This is the contract every reader must fulfil:

    * ``times`` (T,) strictly increasing, minutes
    * ``wavelengths`` (W,) strictly increasing, nm
    * ``absorbance`` (T, W), AU — row i is the spectrum at ``times[i]``
    * ``metadata.injection_id`` non-empty
    """

    times: np.ndarray
    wavelengths: np.ndarray
    absorbance: np.ndarray
    metadata: InjectionMetadata

    def __post_init__(self) -> None:
        self.times = _as_1d_float("times", self.times)
        self.wavelengths = _as_1d_float("wavelengths", self.wavelengths)
        self.absorbance = np.asarray(self.absorbance, dtype=float)
        if self.absorbance.shape != (len(self.times), len(self.wavelengths)):
            raise DataValidationError(
                f"absorbance shape {self.absorbance.shape} does not match "
                f"(times, wavelengths) = ({len(self.times)}, {len(self.wavelengths)})"
            )
        for name, axis in (("times", self.times), ("wavelengths", self.wavelengths)):
            if len(axis) > 1 and not np.all(np.diff(axis) > 0):
                raise DataValidationError(f"{name} must be strictly increasing")
        if not self.metadata.injection_id:
            raise DataValidationError("metadata.injection_id must be non-empty")

    # -- basic projections -------------------------------------------------
    # Small numpy projections live on the model itself; anything smarter
    # (baseline correction, apex averaging, lambda-max) lives in `spectra`.

    def maxplot(self, wavelength_range: tuple[float, float] | None = None) -> Chromatogram:
        """Max absorbance across wavelengths at each time point (Empower MaxPlot)."""
        sl = self._wavelength_slice(wavelength_range)
        return Chromatogram(
            times=self.times,
            values=self.absorbance[:, sl].max(axis=1),
            label="MaxPlot",
            injection_id=self.metadata.injection_id,
        )

    def chromatogram_at(self, wavelength: float, bandwidth: float = 0.0) -> Chromatogram:
        """Trace at one wavelength, averaged over ``wavelength ± bandwidth/2``."""
        if bandwidth > 0:
            lo, hi = wavelength - bandwidth / 2, wavelength + bandwidth / 2
            mask = (self.wavelengths >= lo) & (self.wavelengths <= hi)
            if not mask.any():
                raise DataValidationError(
                    f"no wavelength points in {lo:g}-{hi:g} nm"
                )
            values = self.absorbance[:, mask].mean(axis=1)
        else:
            idx = int(np.argmin(np.abs(self.wavelengths - wavelength)))
            values = self.absorbance[:, idx]
            wavelength = float(self.wavelengths[idx])
        return Chromatogram(
            times=self.times,
            values=values,
            label=f"{wavelength:g} nm",
            injection_id=self.metadata.injection_id,
        )

    def spectrum_at(self, time: float) -> UVSpectrum:
        """Spectrum at the scan closest to ``time`` (minutes)."""
        idx = int(np.argmin(np.abs(self.times - time)))
        return UVSpectrum(
            wavelengths=self.wavelengths,
            values=self.absorbance[idx, :],
            time=float(self.times[idx]),
            label=f"RT {self.times[idx]:.3f} min",
            injection_id=self.metadata.injection_id,
        )

    def _wavelength_slice(self, wavelength_range: tuple[float, float] | None) -> Any:
        if wavelength_range is None:
            return slice(None)
        lo, hi = wavelength_range
        mask = (self.wavelengths >= lo) & (self.wavelengths <= hi)
        if not mask.any():
            raise DataValidationError(f"no wavelength points in {lo:g}-{hi:g} nm")
        return mask


@dataclass
class Peak:
    """One detected peak with its computed properties.

    Fields that a later pipeline stage has not filled yet are ``None``
    (e.g. ``lambda_max`` before spectrum extraction has run).
    """

    apex_time: float                 # RT, minutes
    apex_index: int                  # index into the source chromatogram
    height: float                    # AU (baseline-corrected if available)
    start_time: float | None = None  # integration start, minutes
    end_time: float | None = None    # integration end, minutes
    fwhm: float | None = None        # minutes
    area: float | None = None        # AU*min
    lambda_max: float | None = None  # nm
    spectrum: UVSpectrum | None = None
    injection_id: str = ""
    peak_id: str = ""                # unique within one injection, e.g. "P001"

    EXPORT_COLUMNS = (
        "injection_id",
        "peak_id",
        "apex_time",
        "height",
        "area",
        "fwhm",
        "lambda_max",
        "start_time",
        "end_time",
    )

    def as_record(self) -> dict[str, Any]:
        return {c: getattr(self, c) for c in self.EXPORT_COLUMNS}


@dataclass
class PeakTable:
    """All peaks of one injection, in elution order."""

    peaks: list[Peak]
    injection_id: str = ""
    source_label: str = ""  # which chromatogram detection ran on, e.g. "MaxPlot"

    def __iter__(self) -> Iterator[Peak]:
        return iter(self.peaks)

    def __len__(self) -> int:
        return len(self.peaks)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [p.as_record() for p in self.peaks], columns=list(Peak.EXPORT_COLUMNS)
        )


@dataclass
class PeakGroup:
    """One compound tracked across injections: injection_id -> matched Peak."""

    group_id: int
    members: dict[str, Peak]

    @property
    def mean_rt(self) -> float:
        return float(np.mean([p.apex_time for p in self.members.values()]))

    @property
    def rt_std(self) -> float:
        return float(np.std([p.apex_time for p in self.members.values()]))


@dataclass
class TrackingResult:
    """Peak groups aligned across a set of injections."""

    groups: list[PeakGroup]
    injection_ids: list[str]  # column order for matrix-style output

    def to_dataframe(self, value: str = "apex_time") -> pd.DataFrame:
        """Wide matrix: one row per group, one column per injection.

        ``value`` is any Peak attribute (``apex_time``, ``lambda_max``,
        ``area``, ...); missing matches become NaN.
        """
        rows = []
        for g in sorted(self.groups, key=lambda g: g.mean_rt):
            row: dict[str, Any] = {"group_id": g.group_id, "mean_rt": g.mean_rt}
            for inj in self.injection_ids:
                peak = g.members.get(inj)
                row[inj] = getattr(peak, value) if peak is not None else np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=["group_id", "mean_rt", *self.injection_ids])
