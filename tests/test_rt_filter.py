"""Tests for the retention-time range filter."""

from __future__ import annotations

from pda_peak_finder.models import Peak, PeakTable
from pda_peak_finder.peak_detection import filter_peaks_by_retention_time


def _table(rts):
    peaks = [
        Peak(apex_time=rt, apex_index=i, height=1.0, peak_id=f"P{i+1:03d}")
        for i, rt in enumerate(rts)
    ]
    return PeakTable(peaks=peaks, injection_id="A")


def test_min_and_max_bounds():
    table = _table([0.3, 1.0, 5.0, 12.0])
    out = filter_peaks_by_retention_time(table, rt_min=0.5, rt_max=10.0)
    assert [p.apex_time for p in out] == [1.0, 5.0]
    assert [p.peak_id for p in out] == ["P001", "P002"]  # renumbered


def test_only_min_drops_solvent_front():
    out = filter_peaks_by_retention_time(_table([0.2, 1.0, 5.0]), rt_min=0.5)
    assert [p.apex_time for p in out] == [1.0, 5.0]


def test_only_max():
    out = filter_peaks_by_retention_time(_table([0.2, 1.0, 5.0]), rt_max=2.0)
    assert [p.apex_time for p in out] == [0.2, 1.0]


def test_bounds_are_inclusive():
    out = filter_peaks_by_retention_time(_table([1.0, 2.0, 3.0]),
                                         rt_min=1.0, rt_max=3.0)
    assert [p.apex_time for p in out] == [1.0, 2.0, 3.0]


def test_no_bounds_keeps_all():
    out = filter_peaks_by_retention_time(_table([0.2, 1.0, 5.0]))
    assert len(out) == 3


def test_does_not_mutate_input():
    table = _table([0.2, 1.0, 5.0])
    out = filter_peaks_by_retention_time(table, rt_min=0.5)
    assert len(table) == 3
    assert len(out) == 2
    assert out.injection_id == "A"
