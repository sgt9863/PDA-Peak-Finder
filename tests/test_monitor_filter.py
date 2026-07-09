"""Tests for the monitoring-wavelength peak filter (e.g. drop non-230nm peaks)."""

from __future__ import annotations

import numpy as np

from pda_peak_finder.models import Peak, PeakTable, UVSpectrum
from pda_peak_finder.spectra import absorbance_at, filter_peaks_by_absorbance


def _peak(peak_id, a230, amax=1.0, rt=1.0):
    """A peak whose apex spectrum has absorbance ``a230`` at 230 nm."""
    wl = np.array([200.0, 230.0, 280.0])
    values = np.array([amax, a230, amax * 0.5])
    spec = UVSpectrum(wavelengths=wl, values=values, time=rt)
    return Peak(apex_time=rt, apex_index=0, height=amax, lambda_max=200.0,
                spectrum=spec, peak_id=peak_id)


def test_absorbance_at_picks_nearest_wavelength():
    spec = UVSpectrum(wavelengths=np.array([200.0, 230.0, 280.0]),
                      values=np.array([1.0, 0.3, 0.5]))
    assert absorbance_at(spec, 230.0) == 0.3
    assert absorbance_at(spec, 231.0) == 0.3  # nearest


def test_absolute_threshold_drops_weak_peaks():
    table = PeakTable(peaks=[
        _peak("P001", a230=0.50, rt=1.0),
        _peak("P002", a230=0.005, rt=2.0),   # below 0.01 -> dropped
        _peak("P003", a230=0.02, rt=3.0),
    ])
    out = filter_peaks_by_absorbance(table, 230.0, min_absorbance=0.01)
    rts = [p.apex_time for p in out]
    assert rts == [1.0, 3.0]
    # peak_ids are renumbered contiguously
    assert [p.peak_id for p in out] == ["P001", "P002"]


def test_relative_threshold():
    table = PeakTable(peaks=[
        _peak("P001", a230=0.05, amax=1.0),   # 5% of max
        _peak("P002", a230=0.30, amax=1.0),   # 30% of max
    ])
    out = filter_peaks_by_absorbance(table, 230.0, min_fraction=0.10)
    assert [p.apex_time for p in out] == [table.peaks[1].apex_time]


def test_peaks_without_spectrum_are_kept():
    p = Peak(apex_time=1.0, apex_index=0, height=1.0, peak_id="P001")
    out = filter_peaks_by_absorbance(PeakTable(peaks=[p]), 230.0, min_absorbance=1.0)
    assert len(out) == 1


def test_does_not_mutate_input_table():
    table = PeakTable(peaks=[_peak("P001", 0.005), _peak("P002", 0.5)], injection_id="A")
    out = filter_peaks_by_absorbance(table, 230.0, min_absorbance=0.01)
    assert len(table) == 2          # original untouched
    assert len(out) == 1
    assert out.injection_id == "A"  # metadata carried over
