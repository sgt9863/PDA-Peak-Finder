"""Tests for overlap-robust peak detection by deconvolution.

The core guarantee: overlapping / co-eluting peaks are separated into
individual components with accurate retention time and FWHM, and no peak is
dropped (a failed fit falls back to the seed estimate).
"""

from __future__ import annotations

import numpy as np

from pda_peak_finder.models import Chromatogram
from pda_peak_finder.peak_detection import (
    DeconvolutionConfig,
    detect_peaks_deconvolved,
)

FWHM_K = 2.3548  # FWHM = FWHM_K * sigma for a Gaussian


def _gauss(t, a, mu, s):
    return a * np.exp(-0.5 * ((t - mu) / s) ** 2)


def _chrom(t, y, inj="X"):
    return Chromatogram(times=t, values=y, label="230nm", injection_id=inj)


def test_two_overlapping_peaks_separated():
    t = np.linspace(4.5, 5.6, 1100)
    ch = _chrom(t, _gauss(t, 1.0, 5.00, 0.04) + _gauss(t, 0.7, 5.10, 0.05) + 0.001)
    peaks = list(detect_peaks_deconvolved(ch, DeconvolutionConfig(min_prominence=0.02, baseline_method="opening")))
    assert len(peaks) == 2
    rts = sorted(p.apex_time for p in peaks)
    assert abs(rts[0] - 5.00) < 0.01
    assert abs(rts[1] - 5.10) < 0.01
    # FWHM recovered per component (not the merged composite width)
    by_rt = {round(p.apex_time, 2): p for p in peaks}
    assert abs(by_rt[5.0].fwhm - FWHM_K * 0.04) < 0.01
    assert abs(by_rt[5.1].fwhm - FWHM_K * 0.05) < 0.01


def test_three_overlapping_with_shoulder():
    t = np.linspace(4.6, 5.7, 1100)
    truth = [(5.00, 0.05, 1.0), (5.12, 0.05, 0.85), (5.22, 0.06, 0.55)]
    ch = _chrom(t, sum(_gauss(t, a, mu, s) for mu, s, a in truth))
    peaks = list(detect_peaks_deconvolved(ch, DeconvolutionConfig(min_prominence=0.02, baseline_method="opening")))
    # exactly the three components, no spurious extras
    assert len(peaks) == 3
    for (mu, s, _), p in zip(truth, sorted(peaks, key=lambda p: p.apex_time)):
        assert abs(p.apex_time - mu) < 0.02
        assert abs(p.fwhm - FWHM_K * s) < 0.02


def test_gauss_and_emg_models_both_work():
    t = np.linspace(4.5, 5.6, 1100)
    ch = _chrom(t, _gauss(t, 1.0, 5.00, 0.04) + _gauss(t, 0.7, 5.10, 0.05) + 0.001)
    for model in ("emg", "gauss"):
        peaks = list(detect_peaks_deconvolved(
            ch, DeconvolutionConfig(model=model, min_prominence=0.02, baseline_method="opening")))
        assert len(peaks) == 2


def test_resolved_peaks_not_merged():
    t = np.linspace(0, 20, 4000)
    ch = _chrom(t, _gauss(t, 1.0, 4.0, 0.06) + _gauss(t, 1.0, 12.0, 0.06) + 0.0005)
    peaks = list(detect_peaks_deconvolved(ch, DeconvolutionConfig(min_prominence=0.05, baseline_method="opening")))
    rts = sorted(p.apex_time for p in peaks)
    assert len(peaks) == 2
    assert abs(rts[0] - 4.0) < 0.02 and abs(rts[1] - 12.0) < 0.02


def test_every_peak_has_rt_and_fwhm():
    t = np.linspace(4.6, 5.7, 1100)
    ch = _chrom(t, _gauss(t, 1.0, 5.0, 0.05) + _gauss(t, 0.6, 5.15, 0.05))
    peaks = list(detect_peaks_deconvolved(ch, DeconvolutionConfig(min_prominence=0.02, baseline_method="opening")))
    assert peaks
    for p in peaks:
        assert p.fwhm is not None and p.fwhm > 0
        assert p.apex_time is not None


def test_return_components_shape():
    t = np.linspace(4.6, 5.7, 1100)
    ch = _chrom(t, _gauss(t, 1.0, 5.0, 0.05) + _gauss(t, 0.6, 5.15, 0.05))
    table, components = detect_peaks_deconvolved(
        ch, DeconvolutionConfig(min_prominence=0.02, baseline_method="opening"), return_components=True)
    assert len(components) <= len(table)
    for tt, yy in components:
        assert len(tt) == len(yy)


def test_als_baseline_recovers_sharp_peaks_on_a_hump():
    # Sharp peaks riding on a broad baseline hump (as in a real late cluster):
    # the default ALS baseline must follow the hump and leave the sharp peaks,
    # not swallow them into one wide component.
    t = np.linspace(10.0, 14.0, 5000)
    hump = 0.008 * np.exp(-0.5 * ((t - 12.0) / 0.6) ** 2)          # broad hump
    peaks = (_gauss(t, 0.010, 11.6, 0.02) + _gauss(t, 0.012, 12.0, 0.02)
             + _gauss(t, 0.009, 12.4, 0.02))                       # sharp peaks
    ch = _chrom(t, hump + peaks + 0.001)
    found = list(detect_peaks_deconvolved(
        ch, DeconvolutionConfig(min_prominence=0.001, min_distance_min=0.05)))
    rts = sorted(p.apex_time for p in found)
    # the three sharp peaks are recovered (hump removed, not fit as one peak)
    for target in (11.6, 12.0, 12.4):
        assert any(abs(r - target) < 0.05 for r in rts), (target, rts)
    # no absurdly wide component absorbing the hump
    assert all(p.fwhm < 0.2 for p in found)
