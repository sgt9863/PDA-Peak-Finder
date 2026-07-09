"""Matplotlib visualizations for pda_peak_finder (headless / Agg backend)."""

from __future__ import annotations

from .core import (
    plot_chromatogram,
    plot_contour,
    plot_tracking,
    plot_uv_spectra,
    save_figure,
)

__all__ = [
    "plot_chromatogram",
    "plot_contour",
    "plot_uv_spectra",
    "plot_tracking",
    "save_figure",
]
