"""Tests for pda_peak_finder.spectra against synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from pda_peak_finder.models import Peak, PeakTable
from pda_peak_finder.spectra import (
    SpectrumConfig,
    annotate_peaks,
    compute_lambda_max,
    extract_peak_spectrum,
)
from pda_peak_finder.testing import synthetic_pdadata

# Ground truth from testing.synthetic_pdadata()'s default SyntheticConfig.
GROUND_TRUTH = [
    (4.0, 254.0),
    (8.5, 280.0),
    (13.0, 230.0),
]
LAMBDA_TOL = 2.0  # nm


def _make_peak(data, rt: float, peak_id: str = "P001") -> Peak:
    """Build a Peak directly, with apex_index matching apex_time in data.times."""
    apex_index = int(np.argmin(np.abs(data.times - rt)))
    apex_time = float(data.times[apex_index])
    return Peak(
        apex_time=apex_time,
        apex_index=apex_index,
        height=float(data.absorbance[apex_index].max()),
        injection_id=data.metadata.injection_id,
        peak_id=peak_id,
    )


def _make_table(data) -> PeakTable:
    peaks = [
        _make_peak(data, rt, peak_id=f"P{i:03d}")
        for i, (rt, _lmax) in enumerate(GROUND_TRUTH, start=1)
    ]
    return PeakTable(peaks=peaks, injection_id=data.metadata.injection_id, source_label="MaxPlot")


@pytest.fixture
def data():
    return synthetic_pdadata()


def test_compute_lambda_max_matches_ground_truth(data):
    for rt, lmax in GROUND_TRUTH:
        peak = _make_peak(data, rt)
        spectrum = extract_peak_spectrum(data, peak)
        found = compute_lambda_max(spectrum)
        assert found == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_extract_peak_spectrum_sets_time_and_injection_id(data):
    rt, _lmax = GROUND_TRUTH[0]
    peak = _make_peak(data, rt)
    spectrum = extract_peak_spectrum(data, peak)
    assert spectrum.time == pytest.approx(peak.apex_time, abs=1e-6)
    assert spectrum.injection_id == data.metadata.injection_id
    assert spectrum.wavelengths.shape == data.wavelengths.shape


def test_extract_peak_spectrum_uses_apex_index_when_valid(data):
    """A deliberately mismatched apex_time is ignored when apex_index is valid."""
    rt, lmax = GROUND_TRUTH[1]
    apex_index = int(np.argmin(np.abs(data.times - rt)))
    peak = Peak(
        apex_time=999.0,  # bogus, out of range of data.times
        apex_index=apex_index,
        height=1.0,
        injection_id=data.metadata.injection_id,
    )
    spectrum = extract_peak_spectrum(data, peak)
    assert spectrum.time == pytest.approx(float(data.times[apex_index]), abs=1e-6)
    assert compute_lambda_max(spectrum) == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_extract_peak_spectrum_falls_back_to_nearest_scan_when_index_invalid(data):
    rt, lmax = GROUND_TRUTH[2]
    peak = Peak(
        apex_time=rt,
        apex_index=-1,  # invalid -> must fall back to nearest-scan-to-apex_time
        height=1.0,
        injection_id=data.metadata.injection_id,
    )
    spectrum = extract_peak_spectrum(data, peak)
    expected_idx = int(np.argmin(np.abs(data.times - rt)))
    assert spectrum.time == pytest.approx(float(data.times[expected_idx]), abs=1e-6)
    assert compute_lambda_max(spectrum) == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_compute_lambda_max_search_range_restricts_search(data):
    # Restricting the search window away from the true lambda_max of peak 2
    # (280 nm) should pick the max within the restricted window instead.
    rt, _lmax = GROUND_TRUTH[1]
    peak = _make_peak(data, rt)
    spectrum = extract_peak_spectrum(data, peak)
    restricted = compute_lambda_max(spectrum, search_range=(200.0, 250.0))
    assert 200.0 <= restricted <= 250.0
    assert restricted != pytest.approx(280.0, abs=LAMBDA_TOL)


def test_config_lambda_search_range_used_by_annotate_peaks(data):
    table = _make_table(data)
    config = SpectrumConfig(lambda_search_range=(200.0, 400.0))
    annotate_peaks(data, table, config)
    for peak, (_rt, lmax) in zip(table.peaks, GROUND_TRUTH):
        assert peak.lambda_max == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_annotate_peaks_fills_spectrum_and_lambda_max_for_all_peaks(data):
    table = _make_table(data)
    result = annotate_peaks(data, table)

    assert result is table  # same table returned
    assert len(result) == len(GROUND_TRUTH)
    for peak, (_rt, lmax) in zip(result.peaks, GROUND_TRUTH):
        assert peak.spectrum is not None
        assert peak.lambda_max is not None
        assert peak.lambda_max == pytest.approx(lmax, abs=LAMBDA_TOL)
        assert peak.spectrum.values.shape == data.wavelengths.shape


def test_apex_average_scans_still_matches_ground_truth(data):
    config = SpectrumConfig(apex_average_scans=5)
    for rt, lmax in GROUND_TRUTH:
        peak = _make_peak(data, rt)
        spectrum = extract_peak_spectrum(data, peak, config)
        assert compute_lambda_max(spectrum) == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_apex_average_scans_averages_neighboring_scans(data):
    rt, _lmax = GROUND_TRUTH[0]
    peak = _make_peak(data, rt)
    idx = peak.apex_index
    n = 3
    config = SpectrumConfig(apex_average_scans=n)
    spectrum = extract_peak_spectrum(data, peak, config)

    expected = data.absorbance[idx - n : idx + n + 1, :].mean(axis=0)
    np.testing.assert_allclose(spectrum.values, expected)


def test_baseline_subtract_zeroes_minimum(data):
    rt, _lmax = GROUND_TRUTH[0]
    peak = _make_peak(data, rt)
    config = SpectrumConfig(baseline_subtract=True)
    spectrum = extract_peak_spectrum(data, peak, config)
    assert spectrum.values.min() == pytest.approx(0.0, abs=1e-9)


def test_smoothing_window_preserves_lambda_max(data):
    rt, lmax = GROUND_TRUTH[1]
    peak = _make_peak(data, rt)
    config = SpectrumConfig(smoothing_window=11, smoothing_polyorder=3)
    spectrum = extract_peak_spectrum(data, peak, config)
    assert spectrum.values.shape == data.wavelengths.shape
    assert compute_lambda_max(spectrum) == pytest.approx(lmax, abs=LAMBDA_TOL)


def test_spectrum_config_defaults():
    config = SpectrumConfig()
    assert config.apex_average_scans == 0
    assert config.smoothing_window is None
    assert config.smoothing_polyorder == 3
    assert config.lambda_search_range is None
    assert config.baseline_subtract is False
