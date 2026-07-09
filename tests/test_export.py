"""Tests for pda_peak_finder.export (CSV writers)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pda_peak_finder.export import (
    write_peak_spectra,
    write_peak_table,
    write_peak_tables,
    write_tracking_matrix,
)
from pda_peak_finder.models import Peak, PeakGroup, PeakTable, TrackingResult, UVSpectrum


def _make_peak(
    peak_id: str,
    injection_id: str,
    apex_time: float,
    height: float,
    spectrum: UVSpectrum | None = None,
) -> Peak:
    return Peak(
        apex_time=apex_time,
        apex_index=0,
        height=height,
        start_time=apex_time - 0.1,
        end_time=apex_time + 0.1,
        fwhm=0.05,
        area=height * 0.2,
        lambda_max=254.0,
        spectrum=spectrum,
        injection_id=injection_id,
        peak_id=peak_id,
    )


def _make_spectrum(injection_id: str, time: float) -> UVSpectrum:
    wavelengths = np.array([200.0, 210.0, 220.0])
    values = np.array([0.1, 0.5, 0.2])
    return UVSpectrum(
        wavelengths=wavelengths,
        values=values,
        time=time,
        label="apex",
        injection_id=injection_id,
    )


def test_write_peak_table(tmp_path):
    peaks = [
        _make_peak("P001", "INJ1", 4.0, 0.8),
        _make_peak("P002", "INJ1", 8.5, 1.2),
    ]
    table = PeakTable(peaks=peaks, injection_id="INJ1", source_label="MaxPlot")

    out_path = tmp_path / "sub" / "peaks.csv"
    result = write_peak_table(table, out_path)

    assert result == out_path
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert list(df.columns) == list(Peak.EXPORT_COLUMNS)
    assert len(df) == 2
    assert df.loc[0, "peak_id"] == "P001"
    assert df.loc[0, "apex_time"] == 4.0
    assert df.loc[1, "height"] == 1.2


def test_write_peak_tables_stacks_injections(tmp_path):
    table1 = PeakTable(
        peaks=[_make_peak("P001", "INJ1", 4.0, 0.8)],
        injection_id="INJ1",
        source_label="MaxPlot",
    )
    table2 = PeakTable(
        peaks=[
            _make_peak("P001", "INJ2", 4.1, 0.9),
            _make_peak("P002", "INJ2", 9.0, 1.5),
        ],
        injection_id="INJ2",
        source_label="MaxPlot",
    )

    out_path = tmp_path / "sub" / "all_peaks.csv"
    result = write_peak_tables([table1, table2], out_path)

    assert result == out_path
    df = pd.read_csv(out_path)
    assert list(df.columns) == list(Peak.EXPORT_COLUMNS)
    assert len(df) == 3
    assert set(df["injection_id"]) == {"INJ1", "INJ2"}
    assert df.loc[df["peak_id"].eq("P002") & df["injection_id"].eq("INJ2"), "apex_time"].iloc[0] == 9.0


def test_write_peak_tables_empty_list(tmp_path):
    out_path = tmp_path / "sub" / "empty.csv"
    result = write_peak_tables([], out_path)

    assert result == out_path
    df = pd.read_csv(out_path)
    assert list(df.columns) == list(Peak.EXPORT_COLUMNS)
    assert len(df) == 0


def test_write_tracking_matrix(tmp_path):
    peak_a1 = _make_peak("P001", "INJ1", 4.0, 0.8)
    peak_a2 = _make_peak("P001", "INJ2", 4.1, 0.9)
    peak_b1 = _make_peak("P002", "INJ1", 8.5, 1.2)

    group_a = PeakGroup(group_id=1, members={"INJ1": peak_a1, "INJ2": peak_a2})
    group_b = PeakGroup(group_id=2, members={"INJ1": peak_b1})

    result = TrackingResult(groups=[group_a, group_b], injection_ids=["INJ1", "INJ2"])

    out_path = tmp_path / "sub" / "tracking.csv"
    written = write_tracking_matrix(result, out_path)

    assert written == out_path
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["group_id", "mean_rt", "INJ1", "INJ2"]
    assert len(df) == 2

    row_a = df.loc[df["group_id"] == 1].iloc[0]
    assert row_a["INJ1"] == 4.0
    assert row_a["INJ2"] == 4.1

    row_b = df.loc[df["group_id"] == 2].iloc[0]
    assert row_b["INJ1"] == 8.5
    assert np.isnan(row_b["INJ2"])


def test_write_tracking_matrix_custom_value(tmp_path):
    peak_a1 = _make_peak("P001", "INJ1", 4.0, 0.8)
    group_a = PeakGroup(group_id=1, members={"INJ1": peak_a1})
    result = TrackingResult(groups=[group_a], injection_ids=["INJ1"])

    out_path = tmp_path / "tracking_height.csv"
    write_tracking_matrix(result, out_path, value="height")

    df = pd.read_csv(out_path)
    assert df.loc[0, "INJ1"] == 0.8


def test_write_peak_spectra_only_includes_peaks_with_spectrum(tmp_path):
    spectrum = _make_spectrum("INJ1", 4.0)
    peaks = [
        _make_peak("P001", "INJ1", 4.0, 0.8, spectrum=spectrum),
        _make_peak("P002", "INJ1", 8.5, 1.2, spectrum=None),
    ]
    table = PeakTable(peaks=peaks, injection_id="INJ1", source_label="MaxPlot")

    out_path = tmp_path / "sub" / "spectra.csv"
    result = write_peak_spectra(table, out_path)

    assert result == out_path
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["injection_id", "peak_id", "wavelength", "absorbance"]
    # Only P001 has a spectrum, with 3 wavelength points.
    assert len(df) == 3
    assert set(df["peak_id"]) == {"P001"}
    assert set(df["injection_id"]) == {"INJ1"}
    np.testing.assert_allclose(sorted(df["wavelength"]), [200.0, 210.0, 220.0])
    row = df.loc[df["wavelength"] == 210.0].iloc[0]
    assert row["absorbance"] == 0.5


def test_write_peak_spectra_empty_when_no_spectra(tmp_path):
    peaks = [_make_peak("P001", "INJ1", 4.0, 0.8, spectrum=None)]
    table = PeakTable(peaks=peaks, injection_id="INJ1", source_label="MaxPlot")

    out_path = tmp_path / "sub" / "no_spectra.csv"
    result = write_peak_spectra(table, out_path)

    assert result == out_path
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["injection_id", "peak_id", "wavelength", "absorbance"]
    assert len(df) == 0


def test_parent_directories_auto_created(tmp_path):
    table = PeakTable(peaks=[_make_peak("P001", "INJ1", 4.0, 0.8)], injection_id="INJ1")
    out_path = tmp_path / "a" / "b" / "c" / "out.csv"
    assert not out_path.parent.exists()

    write_peak_table(table, out_path)

    assert out_path.parent.exists()
    assert out_path.exists()
