"""Tests for the QDa/SIR-style labeled chromatogram and font/palette helpers."""

from __future__ import annotations

import matplotlib
from matplotlib.figure import Figure

from pda_peak_finder.peak_detection import detect_peaks
from pda_peak_finder.plotting import (
    configure_japanese_font,
    peak_palette,
    plot_labeled_chromatogram,
)
from pda_peak_finder.spectra import annotate_peaks
from pda_peak_finder.testing import synthetic_pdadata


def test_peak_palette_length():
    assert len(peak_palette(5)) == 5
    assert peak_palette(0) == []
    # each entry is an RGB triple
    assert all(len(c) == 3 for c in peak_palette(3))


def test_configure_japanese_font_returns_name_or_none():
    # Should not raise; returns a family name string or None.
    result = configure_japanese_font()
    assert result is None or isinstance(result, str)


def _table():
    data = synthetic_pdadata()
    mp = data.maxplot()
    table = detect_peaks(mp)
    annotate_peaks(data, table)
    return mp, table


def test_returns_figure_normalized():
    mp, table = _table()
    fig = plot_labeled_chromatogram(mp, table, normalize=True)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # normalized: y top axis limit leaves room above 1.0 for labels
    assert ax.get_ylim()[1] >= 1.3
    matplotlib.pyplot.close(fig)


def test_label_attr_variants_do_not_crash():
    mp, table = _table()
    for attr in ("lambda_max", "peak_id", "apex_time"):
        fig = plot_labeled_chromatogram(mp, table, label_attr=attr)
        assert isinstance(fig, Figure)
        matplotlib.pyplot.close(fig)


def test_non_normalized_mode():
    mp, table = _table()
    fig = plot_labeled_chromatogram(mp, table, normalize=False)
    assert isinstance(fig, Figure)
    matplotlib.pyplot.close(fig)
