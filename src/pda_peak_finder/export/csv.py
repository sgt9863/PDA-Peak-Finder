"""CSV export of peak tables, tracking matrices, and peak spectra.

All writers accept ``str | Path`` for ``path``, create any missing parent
directories, write with ``pandas.DataFrame.to_csv(index=False)``, and return
the :class:`pathlib.Path` written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models import Peak, PeakTable, TrackingResult

SPECTRA_COLUMNS = ("injection_id", "peak_id", "wavelength", "absorbance")


def _resolve_path(path: str | Path) -> Path:
    """Coerce ``path`` to a :class:`Path` and ensure its parent dir exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_peak_table(table: PeakTable, path: str | Path) -> Path:
    """Write one injection's peak table to ``path``."""
    out = _resolve_path(path)
    table.to_dataframe().to_csv(out, index=False)
    return out


def write_peak_tables(tables: list[PeakTable], path: str | Path) -> Path:
    """Write all injections' peaks stacked into one long-form CSV."""
    out = _resolve_path(path)
    if tables:
        df = pd.concat([t.to_dataframe() for t in tables], ignore_index=True)
    else:
        df = pd.DataFrame(columns=list(Peak.EXPORT_COLUMNS))
    df.to_csv(out, index=False)
    return out


def write_tracking_matrix(
    result: TrackingResult, path: str | Path, value: str = "apex_time"
) -> Path:
    """Write the wide tracking matrix (one row per group) to ``path``."""
    out = _resolve_path(path)
    result.to_dataframe(value).to_csv(out, index=False)
    return out


def write_peak_spectra(table: PeakTable, path: str | Path) -> Path:
    """Write long-form UV spectra for every peak in ``table`` that has one.

    Columns: ``injection_id``, ``peak_id``, ``wavelength``, ``absorbance``.
    Peaks with ``spectrum is None`` are skipped. If no peak has a spectrum,
    an empty CSV with just the header row is written.
    """
    out = _resolve_path(path)
    rows = []
    for peak in table:
        spectrum = peak.spectrum
        if spectrum is None:
            continue
        for wavelength, absorbance in zip(spectrum.wavelengths, spectrum.values):
            rows.append(
                {
                    "injection_id": peak.injection_id,
                    "peak_id": peak.peak_id,
                    "wavelength": float(wavelength),
                    "absorbance": float(absorbance),
                }
            )
    df = pd.DataFrame(rows, columns=list(SPECTRA_COLUMNS))
    df.to_csv(out, index=False)
    return out
