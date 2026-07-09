"""Tests for spectral-similarity tracking across conditions + regression export."""

from __future__ import annotations

import numpy as np

from pda_peak_finder.export import regression_table
from pda_peak_finder.models import Peak, PeakTable, UVSpectrum
from pda_peak_finder.tracking import TrackingConfig, track_peaks

WL = np.linspace(200.0, 400.0, 201)


def _spectrum(lmax):
    return UVSpectrum(wavelengths=WL, values=np.exp(-0.5 * ((WL - lmax) / 20.0) ** 2))


def _peak(rt, lmax, inj, fwhm=0.1, area=1.0, pid="P001"):
    return Peak(apex_time=rt, apex_index=0, height=1.0, fwhm=fwhm, area=area,
                lambda_max=lmax, spectrum=_spectrum(lmax), injection_id=inj, peak_id=pid)


def test_spectral_matches_across_large_rt_shift():
    # same two compounds, but retention times shift a lot between conditions
    a = PeakTable(injection_id="A", peaks=[_peak(3.0, 254, "A"), _peak(6.0, 280, "A")])
    b = PeakTable(injection_id="B", peaks=[_peak(3.8, 254, "B"), _peak(6.5, 280, "B")])

    # RT-only (tol 0.2) cannot bridge the shift -> 4 separate groups
    rt_only = track_peaks([a, b], TrackingConfig(rt_tolerance=0.2))
    assert len(rt_only.groups) == 4

    # spectral tracking pairs them by UV spectrum -> 2 groups, both matched
    spec = track_peaks([a, b], TrackingConfig(use_spectral=True,
                                              min_spectral_similarity=0.95))
    assert len(spec.groups) == 2
    assert all(len(g.members) == 2 for g in spec.groups)


def test_spectral_does_not_merge_different_spectra_at_close_rt():
    a = PeakTable(injection_id="A", peaks=[_peak(3.0, 254, "A")])
    # close RT but a different compound (different spectrum) must NOT match
    b = PeakTable(injection_id="B", peaks=[_peak(3.05, 330, "B")])
    spec = track_peaks([a, b], TrackingConfig(use_spectral=True,
                                              min_spectral_similarity=0.95))
    assert len(spec.groups) == 2
    assert all(len(g.members) == 1 for g in spec.groups)


def test_regression_table_is_tidy_per_compound_condition():
    a = PeakTable(injection_id="A", peaks=[_peak(3.0, 254, "A", fwhm=0.10)])
    b = PeakTable(injection_id="B", peaks=[_peak(3.7, 254, "B", fwhm=0.12)])
    res = track_peaks([a, b], TrackingConfig(use_spectral=True))
    df = regression_table(res)
    assert list(df.columns)[:5] == ["group_id", "mean_rt", "n_injections",
                                    "injection_id", "apex_time"]
    # one compound seen in both conditions -> two rows, same group_id
    assert len(df) == 2
    assert df["group_id"].nunique() == 1
    assert set(df["injection_id"]) == {"A", "B"}
    assert sorted(df["fwhm"].tolist()) == [0.10, 0.12]
