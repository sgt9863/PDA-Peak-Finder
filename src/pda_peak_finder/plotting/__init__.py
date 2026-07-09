"""Matplotlib visualizations for pda_peak_finder (headless / Agg backend)."""

from __future__ import annotations

from .core import (
    configure_japanese_font,
    peak_palette,
    plot_chromatogram,
    plot_contour,
    plot_labeled_chromatogram,
    plot_tracking,
    plot_uv_spectra,
    save_figure,
)

__all__ = [
    "plot_chromatogram",
    "plot_labeled_chromatogram",
    "plot_contour",
    "plot_uv_spectra",
    "plot_tracking",
    "save_figure",
    "peak_palette",
    "configure_japanese_font",
]
