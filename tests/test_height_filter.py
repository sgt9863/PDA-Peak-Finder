"""Tests for the peak-height range filter."""

from __future__ import annotations

from pda_peak_finder.models import Peak, PeakTable
from pda_peak_finder.peak_detection import filter_peaks_by_height


def _table(heights):
    peaks = [
        Peak(apex_time=float(i), apex_index=i, height=h, peak_id=f"P{i+1:03d}")
        for i, h in enumerate(heights)
    ]
    return PeakTable(peaks=peaks, injection_id="A")


def test_min_and_max_bounds():
    table = _table([0.05, 0.2, 1.5, 12.0])
    out = filter_peaks_by_height(table, min_height=0.1, max_height=10.0)
    assert [p.height for p in out] == [0.2, 1.5]
    assert [p.peak_id for p in out] == ["P001", "P002"]  # renumbered


def test_only_min():
    out = filter_peaks_by_height(_table([0.05, 0.2, 1.5]), min_height=0.1)
    assert [p.height for p in out] == [0.2, 1.5]


def test_only_max():
    out = filter_peaks_by_height(_table([0.05, 0.2, 1.5]), max_height=1.0)
    assert [p.height for p in out] == [0.05, 0.2]


def test_no_bounds_keeps_all():
    out = filter_peaks_by_height(_table([0.05, 0.2, 1.5]))
    assert len(out) == 3


def test_does_not_mutate_input():
    table = _table([0.05, 0.2, 1.5])
    out = filter_peaks_by_height(table, min_height=0.1)
    assert len(table) == 3
    assert len(out) == 2
    assert out.injection_id == "A"


def test_uv_range_via_scale():
    # 100-10000 µV with 1 AU = 1000 µV  ->  0.1-10 AU
    table = _table([0.05, 0.1, 3.086, 11.0])  # µV: 50, 100, 3086, 11000
    out = filter_peaks_by_height(table, min_height=100 / 1000, max_height=10000 / 1000)
    assert [p.height for p in out] == [0.1, 3.086]
