"""Tests for the Waters Empower ARW reader.

The reader is exercised with small in-memory ARW files written in the real
export layout (Shift-JIS, CR line endings, metadata/wavelength/time blocks),
so no large sample file is required. A separate test reads the real sample
files under ``data/`` only when they are present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pda_peak_finder.errors import ReaderError
from pda_peak_finder.reader import load
from pda_peak_finder.reader.arw import ArwReader

CR = "\r"


def _write_arw(
    path: Path,
    wavelengths,
    times,
    absorbance,
    *,
    sample_label="サンプル名",
    sample_name="TestSample",
    wl_label="波長",
    time_label="時間",
    encoding="cp932",
    truncate_last=False,
    quote_meta=True,
) -> Path:
    """Write a minimal ARW file in the real Empower export layout."""
    lines: list[str] = []
    if sample_label is not None:
        q = '"' if quote_meta else ""
        lines.append(f"{q}{sample_label}{q}")
        lines.append(f"{q}{sample_name}{q}")
    lines.append(wl_label + "\t" + "\t".join(f"{w:g}" for w in wavelengths))
    lines.append(time_label)
    for i, t in enumerate(times):
        row = f"{t:g}\t" + "\t".join(f"{a:g}" for a in absorbance[i])
        lines.append(row)
    text = CR.join(lines)
    if truncate_last:
        # Chop the final row mid-way, mimicking a cut-off export/transfer.
        text = text[: -len(lines[-1]) // 2]
    path.write_bytes(text.encode(encoding))
    return path


def _demo_arrays():
    wavelengths = np.linspace(200.0, 400.0, 11)          # 11 wl
    times = np.linspace(0.0, 1.0, 6)                      # 6 scans
    # one gaussian band per scan, drifting height, peak near 280 nm
    absorbance = np.zeros((len(times), len(wavelengths)))
    for i, t in enumerate(times):
        absorbance[i] = (0.1 + t) * np.exp(-0.5 * ((wavelengths - 280.0) / 40.0) ** 2)
    return wavelengths, times, absorbance


def test_reads_basic_layout(tmp_path):
    wl, t, a = _demo_arrays()
    p = _write_arw(tmp_path / "run.arw", wl, t, a)
    data = load(p)

    assert data.metadata.injection_id == "TestSample"
    assert data.metadata.sample_name == "TestSample"
    assert data.absorbance.shape == (len(t), len(wl))
    np.testing.assert_allclose(data.wavelengths, wl, rtol=1e-6)
    np.testing.assert_allclose(data.times, t, atol=1e-6)
    np.testing.assert_allclose(data.absorbance, a, rtol=1e-5, atol=1e-9)


def test_truncated_final_row_is_dropped(tmp_path):
    wl, t, a = _demo_arrays()
    p = _write_arw(tmp_path / "trunc.arw", wl, t, a, truncate_last=True)
    data = load(p)
    # The last (partial) scan is dropped; all kept rows are full width.
    assert data.absorbance.shape == (len(t) - 1, len(wl))
    np.testing.assert_allclose(data.times, t[:-1], atol=1e-6)


def test_english_labels_structural_parsing(tmp_path):
    wl, t, a = _demo_arrays()
    p = _write_arw(
        tmp_path / "en.arw", wl, t, a,
        sample_label="Sample Name", sample_name="EN_Sample",
        wl_label="Wavelength", time_label="Time", encoding="utf-8",
    )
    data = load(p)
    assert data.metadata.injection_id == "EN_Sample"
    assert data.absorbance.shape == (len(t), len(wl))


def test_injection_id_falls_back_to_stem(tmp_path):
    wl, t, a = _demo_arrays()
    # No metadata header at all -> id comes from the file name.
    p = _write_arw(tmp_path / "nometa.arw", wl, t, a, sample_label=None)
    data = load(p)
    assert data.metadata.injection_id == "nometa"
    assert data.metadata.sample_name == ""


def test_descending_wavelengths_are_sorted(tmp_path):
    wl, t, a = _demo_arrays()
    wl_desc = wl[::-1]
    a_desc = a[:, ::-1]
    p = _write_arw(tmp_path / "desc.arw", wl_desc, t, a_desc)
    data = load(p)
    assert np.all(np.diff(data.wavelengths) > 0)
    # columns were reordered to match the sorted axis
    np.testing.assert_allclose(data.absorbance, a, rtol=1e-5, atol=1e-9)


def test_crlf_line_endings(tmp_path):
    wl, t, a = _demo_arrays()
    p = tmp_path / "crlf.arw"
    _write_arw(p, wl, t, a)
    # rewrite with CRLF terminators
    raw = p.read_bytes().replace(b"\r", b"\r\n")
    p.write_bytes(raw)
    data = load(p)
    assert data.absorbance.shape == (len(t), len(wl))


def test_missing_numeric_block_raises(tmp_path):
    p = tmp_path / "bad.arw"
    p.write_bytes(("サンプル名\r\"X\"\r時間\r").encode("cp932"))
    with pytest.raises(ReaderError):
        load(p)


def test_reader_is_registered_for_arw():
    from pda_peak_finder.reader import get_reader

    assert isinstance(get_reader(Path("whatever.arw")), ArwReader)


# -- real sample files (only when provided under data/) --------------------

_REAL = [
    Path(__file__).resolve().parents[1] / "data" / "sample1.arw",
    Path(__file__).resolve().parents[1] / "data" / "sample2.arw",
]


@pytest.mark.parametrize("path", _REAL, ids=lambda p: p.name)
def test_real_sample_files_if_present(path):
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    data = load(path)
    assert data.metadata.injection_id
    assert data.absorbance.shape == (len(data.times), len(data.wavelengths))
    assert len(data.wavelengths) == 326
    assert np.all(np.diff(data.times) > 0)
    assert np.all(np.diff(data.wavelengths) > 0)
    # sane physical ranges
    assert 195 < data.wavelengths[0] < 210
    assert 395 < data.wavelengths[-1] < 405
