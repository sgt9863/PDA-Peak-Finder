"""Waters Empower ARW reader.

ARW files are ASCII exports produced by Empower's export methods. The
sample files this reader was built against have the following layout
(one injection per file):

    "サンプル名"                       <- metadata label   (quoted)
    "Sample_run19"                     <- metadata value   (quoted)
    波長 <TAB> 199.9129 <TAB> ... <TAB> 399.8336   <- wavelength axis (nm)
    時間                               <- label marking the data block
    0        <TAB> 0        <TAB> ...   <- data: time(min) then absorbance(AU)/wl
    0.000833 <TAB> 1.6e-06  <TAB> ...
    ...

Concrete properties handled:

* encoding: Shift-JIS (cp932); UTF-8 and latin-1 are also accepted.
* line terminator: CR (``\\r``); ``\\r\\n`` and ``\\n`` are tolerated.
* delimiter: TAB. Numeric fields may carry leading spaces.
* metadata: label/value pairs on their own lines before the wavelength row
  (only the sample name in these exports); kept in ``metadata.extra``.
* the numeric block is time-major: one row per scan, first column = time
  in minutes, remaining columns = absorbance (AU) at each wavelength (nm).
* a truncated final row (fewer columns than the wavelength axis, e.g. when
  an export/transfer was cut off) is dropped rather than raising.

Units are already minutes / nm / AU, so no conversion is applied.

Parsing is intentionally structural (it locates the wavelength axis and the
numeric block by shape, not by the Japanese label text) so that
English-locale exports of the same layout also parse. If a future export
method differs materially, add a new branch or a sibling reader rather than
overloading this one.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ..errors import ReaderError
from ..models import InjectionMetadata, PDAData
from .base import SpectralDataReader

_ENCODINGS = ("utf-8", "cp932", "latin-1")
#: Header labels (case/locale variants) that identify the sample name value.
_SAMPLE_NAME_LABELS = frozenset(
    {"サンプル名", "sample name", "samplename", "sample", "sampleid", "sample id"}
)


class ArwReader(SpectralDataReader):
    """Reader for Waters Empower ARW 3D exports."""

    format_name: ClassVar[str] = "arw"
    file_patterns: ClassVar[tuple[str, ...]] = ("*.arw",)

    def read(self, path: Path) -> PDAData:
        path = Path(path)
        records = _decode_records(path)
        wl_idx = _find_wavelength_row(records)
        if wl_idx is None:
            raise ReaderError(
                f"{path.name}: could not locate a wavelength header row "
                "(a line of tab-separated numeric values)"
            )

        wavelengths = _parse_wavelength_row(records[wl_idx], path)
        n_wl = len(wavelengths)
        times, absorbance = _parse_data_block(
            records[wl_idx + 1:], n_wl, path
        )
        wavelengths, absorbance = _ensure_increasing_wavelengths(
            wavelengths, absorbance, path
        )
        metadata = _parse_metadata(records[:wl_idx], path)

        try:
            return PDAData(
                times=times,
                wavelengths=wavelengths,
                absorbance=absorbance,
                metadata=metadata,
            )
        except Exception as exc:  # surface model validation as a ReaderError
            raise ReaderError(f"{path.name}: {exc}") from exc


# -- helpers ---------------------------------------------------------------

def _decode_records(path: Path) -> list[str]:
    """Read the file and split into CR/LF-delimited records."""
    raw = path.read_bytes()
    text = None
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 never raises
        raise ReaderError(f"{path.name}: could not decode with {_ENCODINGS}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _find_wavelength_row(records: list[str], min_points: int = 3) -> int | None:
    """Index of the first row that is a numeric axis (the wavelength header).

    A wavelength row has many tab-separated fields where every field except
    possibly the first (a label such as "波長") parses as a float.
    """
    for i, rec in enumerate(records):
        fields = rec.split("\t")
        if len(fields) < min_points:
            continue
        numeric = fields if _is_float(fields[0]) else fields[1:]
        if len(numeric) >= min_points and all(_is_float(f) for f in numeric):
            return i
    return None


def _parse_wavelength_row(rec: str, path: Path) -> np.ndarray:
    fields = rec.split("\t")
    numeric = fields if _is_float(fields[0]) else fields[1:]
    try:
        return np.array([float(f) for f in numeric], dtype=float)
    except ValueError as exc:  # pragma: no cover - guarded by _find
        raise ReaderError(f"{path.name}: bad wavelength axis: {exc}") from exc


def _parse_data_block(
    records: list[str], n_wl: int, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Parse time-major data rows into (times, absorbance).

    Each valid row has ``n_wl + 1`` numeric fields (time + one per
    wavelength). Rows with a different field count — label lines like
    "時間", blank lines, or a truncated final row — are skipped.
    """
    expected = n_wl + 1
    # Keep only rows with the expected field count. This drops label lines
    # like "時間", blank lines, and a truncated final row.
    kept = [rec for rec in records if rec.count("\t") + 1 == expected]
    if not kept:
        raise ReaderError(f"{path.name}: no data rows with {expected} columns found")

    frame = pd.read_csv(
        io.StringIO("\n".join(kept)),
        sep="\t",
        header=None,
        dtype="float64",
    )
    matrix = frame.to_numpy(dtype=float)
    return matrix[:, 0], matrix[:, 1:]


def _ensure_increasing_wavelengths(
    wavelengths: np.ndarray, absorbance: np.ndarray, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    if wavelengths.size > 1 and not np.all(np.diff(wavelengths) > 0):
        order = np.argsort(wavelengths)
        wavelengths = wavelengths[order]
        absorbance = absorbance[:, order]
        if not np.all(np.diff(wavelengths) > 0):
            raise ReaderError(
                f"{path.name}: wavelength axis has duplicate values after sorting"
            )
    return wavelengths, absorbance


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        token = token[1:-1]
    return token.strip()


def _parse_metadata(header_records: list[str], path: Path) -> InjectionMetadata:
    """Turn the pre-wavelength header lines into InjectionMetadata.

    The lines are label/value pairs on separate rows. Unrecognised pairs are
    preserved in ``extra``. ``injection_id`` falls back to the file stem.
    """
    tokens = [_unquote(r) for r in header_records if r.strip() != ""]
    extra: dict[str, str] = {}
    sample_name = ""
    # Pair consecutive (label, value) lines.
    for i in range(0, len(tokens) - 1, 2):
        label, value = tokens[i], tokens[i + 1]
        extra[label] = value
        if label.strip().lower() in _SAMPLE_NAME_LABELS or label == "サンプル名":
            sample_name = value
    if len(tokens) % 2 == 1:  # dangling label with no value
        extra[f"header_{len(tokens) - 1}"] = tokens[-1]

    injection_id = sample_name or path.stem
    return InjectionMetadata(
        injection_id=injection_id,
        sample_name=sample_name,
        source_path=path,
        extra=extra,
    )
