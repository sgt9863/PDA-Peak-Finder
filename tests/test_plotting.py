"""Tests for pda_peak_finder.plotting.

Uses the Agg backend (headless) and synthetic PDAData so no ARW files or
other pipeline modules (peak detection, spectra) are needed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pda_peak_finder.models import Peak, PeakGroup, PeakTable, TrackingResult
from pda_peak_finder.plotting import (
    plot_chromatogram,
    plot_contour,
    plot_tracking,
    plot_uv_spectra,
    save_figure,
)
from pda_peak_finder.testing import synthetic_pdadata

# Ground-truth peaks baked into the default synthetic_pdadata().
_SYNTH_PEAKS = [
    (4.0, 0.8, 254.0),
    (8.5, 1.2, 280.0),
    (13.0, 0.5, 230.0),
]


def _make_peak_table(data, injection_id=None):
    injection_id = injection_id or data.metadata.injection_id
    peaks = []
    for i, (rt, height, lam) in enumerate(_SYNTH_PEAKS, start=1):
        apex_index = int(np.argmin(np.abs(data.times - rt)))
        peaks.append(
            Peak(
                apex_time=rt,
                apex_index=apex_index,
                height=height,
                start_time=rt - 0.3,
                end_time=rt + 0.3,
                fwhm=0.2,
                area=height * 0.3,
                lambda_max=lam,
                injection_id=injection_id,
                peak_id=f"P{i:03d}",
            )
        )
    return PeakTable(peaks=peaks, injection_id=injection_id, source_label="MaxPlot")


@pytest.fixture
def data():
    return synthetic_pdadata()


@pytest.fixture
def table(data):
    return _make_peak_table(data)


def test_plot_chromatogram_without_table(data):
    chrom = data.maxplot()
    fig = plot_chromatogram(chrom)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_chromatogram_with_table(data, table):
    chrom = data.maxplot()
    fig = plot_chromatogram(chrom, table=table)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_chromatogram_with_existing_ax(data, table):
    fig_in, ax = plt.subplots()
    chrom = data.maxplot()
    fig_out = plot_chromatogram(chrom, table=table, ax=ax)
    assert fig_out is fig_in
    plt.close(fig_out)


def test_plot_contour(data):
    fig = plot_contour(data)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_contour_custom_levels(data):
    fig = plot_contour(data, levels=10)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_uv_spectra_with_peak_table(data, table):
    for peak in table:
        peak.spectrum = data.spectrum_at(peak.apex_time)
    fig = plot_uv_spectra(table)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_uv_spectra_with_peak_list(data, table):
    peaks = list(table)
    for peak in peaks:
        peak.spectrum = data.spectrum_at(peak.apex_time)
    fig = plot_uv_spectra(peaks)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_uv_spectra_skips_missing_spectrum(data, table):
    peaks = list(table)
    # Leave spectrum as None for all but one peak.
    peaks[0].spectrum = data.spectrum_at(peaks[0].apex_time)
    fig = plot_uv_spectra(peaks)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_tracking():
    injection_ids = ["INJ001", "INJ002", "INJ003"]
    rng_rts = {
        1: [4.00, 4.02, 3.98],
        2: [8.50, 8.55, 8.49],
    }
    groups = []
    for group_id, rts in rng_rts.items():
        members = {}
        for inj, rt in zip(injection_ids, rts):
            members[inj] = Peak(
                apex_time=rt,
                apex_index=0,
                height=1.0,
                injection_id=inj,
                peak_id=f"G{group_id}",
            )
        groups.append(PeakGroup(group_id=group_id, members=members))

    result = TrackingResult(groups=groups, injection_ids=injection_ids)
    fig = plot_tracking(result)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_tracking_with_missing_member():
    injection_ids = ["INJ001", "INJ002"]
    members = {
        "INJ001": Peak(apex_time=4.0, apex_index=0, height=1.0, injection_id="INJ001", peak_id="G1"),
        # INJ002 has no match for this group.
    }
    groups = [PeakGroup(group_id=1, members=members)]
    result = TrackingResult(groups=groups, injection_ids=injection_ids)
    fig = plot_tracking(result)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_save_figure_writes_png(tmp_path, data):
    chrom = data.maxplot()
    fig = plot_chromatogram(chrom)
    target = tmp_path / "sub" / "fig.png"
    result_path = save_figure(fig, target)
    plt.close(fig)

    assert result_path == target
    assert target.exists()
    assert target.stat().st_size > 0
