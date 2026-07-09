"""Tests for pda_peak_finder.peak_detection against synthetic ground truth."""

from __future__ import annotations

import math

import pytest

from pda_peak_finder.peak_detection import PeakDetectionConfig, detect_peaks
from pda_peak_finder.testing import SyntheticConfig, SyntheticPeakSpec, synthetic_pdadata

# Ground truth baked into pda_peak_finder.testing.synthetic_pdadata()'s default config.
EXPECTED_RT = [4.0, 8.5, 13.0]
EXPECTED_SIGMA = [0.10, 0.15, 0.12]
EXPECTED_HEIGHT = [0.8, 1.2, 0.5]
GAUSSIAN_FWHM_FACTOR = 2.3548  # FWHM = 2*sqrt(2*ln2) * sigma


def _default_maxplot():
    data = synthetic_pdadata()
    return data.maxplot()


def test_detects_exactly_three_peaks():
    table = detect_peaks(_default_maxplot())
    assert len(table) == 3


def test_apex_times_match_ground_truth():
    table = detect_peaks(_default_maxplot())
    peaks = list(table)
    for peak, expected_rt in zip(peaks, EXPECTED_RT):
        assert abs(peak.apex_time - expected_rt) < 0.05


def test_fwhm_matches_ground_truth():
    table = detect_peaks(_default_maxplot())
    peaks = list(table)
    for peak, sigma in zip(peaks, EXPECTED_SIGMA):
        expected_fwhm = GAUSSIAN_FWHM_FACTOR * sigma
        assert peak.fwhm is not None
        assert abs(peak.fwhm - expected_fwhm) / expected_fwhm < 0.15


def test_elution_order_and_peak_ids():
    chrom = _default_maxplot()
    table = detect_peaks(chrom)
    peaks = list(table)

    # strictly increasing apex_time (elution order)
    apex_times = [p.apex_time for p in peaks]
    assert apex_times == sorted(apex_times)

    assert [p.peak_id for p in peaks] == ["P001", "P002", "P003"]
    assert all(p.injection_id == chrom.injection_id for p in peaks)
    assert table.injection_id == chrom.injection_id
    assert table.source_label == chrom.label


def test_peak_geometry_is_sane():
    table = detect_peaks(_default_maxplot())
    for peak, expected_height in zip(table, EXPECTED_HEIGHT):
        assert peak.start_time < peak.apex_time < peak.end_time
        assert peak.area is not None and peak.area > 0.0
        assert math.isclose(peak.height, expected_height, rel_tol=0.05)


def test_min_prominence_filters_tiny_peak():
    peak_specs = [
        SyntheticPeakSpec(rt=4.0, width=0.10, height=0.8, lambda_max=254.0),
        SyntheticPeakSpec(rt=6.0, width=0.05, height=0.005, lambda_max=250.0),
        SyntheticPeakSpec(rt=8.5, width=0.15, height=1.2, lambda_max=280.0),
        SyntheticPeakSpec(rt=13.0, width=0.12, height=0.5, lambda_max=230.0),
    ]
    data = synthetic_pdadata(SyntheticConfig(peaks=peak_specs))
    chrom = data.maxplot()

    # Default min_prominence (0.01) should reject the 0.005 AU tiny peak.
    default_table = detect_peaks(chrom)
    assert len(default_table) == 3
    assert all(abs(p.apex_time - 6.0) > 0.5 for p in default_table)

    # Lowering min_prominence below the tiny peak's height should recover it.
    sensitive_table = detect_peaks(chrom, PeakDetectionConfig(min_prominence=0.001))
    assert len(sensitive_table) == 4
    assert any(abs(p.apex_time - 6.0) < 0.05 for p in sensitive_table)


def test_min_height_filter():
    chrom = _default_maxplot()
    table = detect_peaks(chrom, PeakDetectionConfig(min_height=1.0))
    # Only the RT=8.5 peak (height 1.2 AU) clears a 1.0 AU height floor.
    assert len(table) == 1
    assert abs(list(table)[0].apex_time - 8.5) < 0.05


def test_smoothing_does_not_break_detection():
    chrom = _default_maxplot()
    table = detect_peaks(chrom, PeakDetectionConfig(smoothing_window=11, smoothing_polyorder=3))
    assert len(table) == 3
    for peak, expected_rt in zip(table, EXPECTED_RT):
        assert abs(peak.apex_time - expected_rt) < 0.05
