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


REGRESSION_COLUMNS = (
    "group_id", "mean_rt", "n_injections", "injection_id",
    "apex_time", "fwhm", "area", "height", "lambda_max",
)


def regression_table(result: TrackingResult) -> pd.DataFrame:
    """Tidy (long-form) table for multiple-regression fitting.

    One row per (tracked compound, injection) with that peak's retention time,
    FWHM, area, height and lambda_max — the per-condition observations of each
    peak, aligned across conditions by :func:`track_peaks`. ``group_id``
    identifies the compound; ``mean_rt`` and ``n_injections`` describe the
    group. Missing matches simply produce no row for that (group, injection).
    """
    rows = []
    for g in sorted(result.groups, key=lambda g: g.mean_rt):
        n = len(g.members)
        for inj in result.injection_ids:
            peak = g.members.get(inj)
            if peak is None:
                continue
            rows.append({
                "group_id": g.group_id,
                "mean_rt": round(g.mean_rt, 5),
                "n_injections": n,
                "injection_id": inj,
                "apex_time": peak.apex_time,
                "fwhm": peak.fwhm,
                "area": peak.area,
                "height": peak.height,
                "lambda_max": peak.lambda_max,
            })
    return pd.DataFrame(rows, columns=list(REGRESSION_COLUMNS))


def write_regression_table(result: TrackingResult, path: str | Path) -> Path:
    """Write the tidy per-(compound, condition) regression table to ``path``."""
    out = _resolve_path(path)
    regression_table(result).to_csv(out, index=False)
    return out


def peak_matrix_table(
    result: TrackingResult, name_prefix: str = "P"
) -> pd.DataFrame:
    """Wide per-run table of each tracked peak's retention time and width.

    One row per injection (run); for every tracked peak (a group, ordered by
    mean retention time and named ``<prefix>01``, ``<prefix>02``, ...) two
    columns ``tR_<name>`` and ``Wh_<name>``. This mirrors the layout of an
    Empower-style transcription (``Run, ..., tR_TP, tR_IP1, ..., Wh_TP, ...``);
    join your condition metadata (T, F, ...) on the ``injection`` column.
    Missing detections are NaN.
    """
    groups = sorted(result.groups, key=lambda g: g.mean_rt)
    names = [f"{name_prefix}{i:02d}" for i in range(1, len(groups) + 1)]
    columns = ["injection"]
    for nm in names:
        columns += [f"tR_{nm}", f"Wh_{nm}"]
    rows = []
    for inj in result.injection_ids:
        row: dict = {"injection": inj}
        for nm, g in zip(names, groups):
            peak = g.members.get(inj)
            row[f"tR_{nm}"] = peak.apex_time if peak is not None else None
            row[f"Wh_{nm}"] = peak.fwhm if peak is not None else None
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def write_peak_matrix(
    result: TrackingResult, path: str | Path, name_prefix: str = "P"
) -> Path:
    """Write the wide per-run tR/Wh peak matrix to ``path``."""
    out = _resolve_path(path)
    peak_matrix_table(result, name_prefix).to_csv(out, index=False)
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
