"""Tests for pda_peak_finder.tracking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pda_peak_finder.models import Peak, PeakTable
from pda_peak_finder.tracking import TrackingConfig, track_peaks

# Three compounds, RTs jittered by +/-0.05 min across injections A/B/C.
BASE_RTS = [4.0, 8.5, 13.0]
OFFSETS = {"A": -0.05, "B": 0.0, "C": 0.05}


def _make_peak(injection_id: str, rt: float, index: int) -> Peak:
    return Peak(
        apex_time=rt,
        apex_index=index,
        height=1.0,
        injection_id=injection_id,
        peak_id=f"P{index:03d}",
    )


def _make_table(injection_id: str, rts: list[float]) -> PeakTable:
    peaks = [_make_peak(injection_id, rt, i) for i, rt in enumerate(rts, start=1)]
    return PeakTable(peaks=peaks, injection_id=injection_id, source_label="MaxPlot")


def _default_tables() -> list[PeakTable]:
    return [
        _make_table(inj, [rt + OFFSETS[inj] for rt in BASE_RTS])
        for inj in ("A", "B", "C")
    ]


def test_track_peaks_groups_matching_compounds_across_injections():
    tables = _default_tables()
    result = track_peaks(tables)

    assert len(result.groups) == 3
    for group in result.groups:
        assert len(group.members) == 3
        assert set(group.members) == {"A", "B", "C"}

    assert result.injection_ids == ["A", "B", "C"]


def test_track_peaks_missing_peak_yields_two_member_group_and_nan_column():
    tables = _default_tables()
    # Drop the middle compound (RT ~8.5) from injection "C".
    c_rts = [rt + OFFSETS["C"] for rt in BASE_RTS if rt != 8.5]
    tables[2] = _make_table("C", c_rts)

    result = track_peaks(tables)

    assert len(result.groups) == 3
    group_by_mean_rt = sorted(result.groups, key=lambda g: g.mean_rt)
    middle_group = group_by_mean_rt[1]

    assert len(middle_group.members) == 2
    assert set(middle_group.members) == {"A", "B"}
    assert "C" not in middle_group.members

    df = result.to_dataframe()
    row = df.loc[df["group_id"] == middle_group.group_id].iloc[0]
    assert math.isnan(row["C"])
    # Other groups still have all three injections populated.
    for group in group_by_mean_rt:
        if group.group_id == middle_group.group_id:
            continue
        row = df.loc[df["group_id"] == group.group_id].iloc[0]
        assert not any(math.isnan(row[inj]) for inj in ("A", "B", "C"))


def test_small_rt_tolerance_splits_compounds_into_separate_groups():
    tables = _default_tables()
    # +/-0.05 min jitter exceeds a 0.01 min tolerance, so nothing should merge.
    config = TrackingConfig(rt_tolerance=0.01)
    result = track_peaks(tables, config)

    assert len(result.groups) == 9
    assert all(len(group.members) == 1 for group in result.groups)
