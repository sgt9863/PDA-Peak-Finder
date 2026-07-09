"""Matplotlib visualizations for PDA/UV 3D data and peak-finding results.

This module owns all plotting for the package. It never mutates the data
models it is handed (:class:`~pda_peak_finder.models.Chromatogram`,
:class:`~pda_peak_finder.models.PDAData`, :class:`~pda_peak_finder.models.Peak`,
:class:`~pda_peak_finder.models.PeakTable`,
:class:`~pda_peak_finder.models.TrackingResult`) — it only reads them.

Units follow the rest of the package: time in minutes, wavelength in nm,
absorbance in AU.

Every ``plot_*`` function accepts an optional ``ax``; when omitted a new
``Figure``/``Axes`` pair is created. Every function returns the
``matplotlib.figure.Figure`` the axes belongs to, so callers can further
customize or save it (see :func:`save_figure`).
"""

from __future__ import annotations

# Headless backend: must be selected before pyplot is imported anywhere.
import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..models import Chromatogram, PDAData, Peak, PeakTable, TrackingResult

__all__ = [
    "plot_chromatogram",
    "plot_contour",
    "plot_uv_spectra",
    "plot_tracking",
    "save_figure",
]


def _resolve_ax(ax):
    """Return (figure, axes), creating a new Figure/Axes pair if needed."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def plot_chromatogram(
    chromatogram: Chromatogram,
    table: PeakTable | None = None,
    ax=None,
) -> Figure:
    """Plot a chromatogram trace, optionally annotated with detected peaks.

    If ``table`` is given, each peak's apex is marked (a vertical line up to
    ``height`` at ``apex_time``), the integration window is lightly shaded
    when both ``start_time`` and ``end_time`` are set, and each peak is
    labeled with its ``peak_id`` near the apex.
    """
    fig, ax = _resolve_ax(ax)

    ax.plot(chromatogram.times, chromatogram.values, color="tab:blue", linewidth=1.0,
             label=chromatogram.label or None, zorder=2)

    if table is not None:
        for peak in table:
            if peak.start_time is not None and peak.end_time is not None:
                ax.axvspan(peak.start_time, peak.end_time, color="tab:orange",
                           alpha=0.15, zorder=1)
            ax.vlines(peak.apex_time, ymin=0.0, ymax=peak.height,
                      color="tab:red", linewidth=1.0, linestyle="--", zorder=3)
            ax.plot(peak.apex_time, peak.height, marker="v", color="tab:red",
                    markersize=6, zorder=4)
            label = peak.peak_id or f"{peak.apex_time:.2f}"
            ax.annotate(
                label,
                xy=(peak.apex_time, peak.height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="tab:red",
            )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Absorbance (AU)")
    title = chromatogram.label or "Chromatogram"
    if chromatogram.injection_id:
        title = f"{title} ({chromatogram.injection_id})"
    ax.set_title(title)
    return fig


def plot_contour(data: PDAData, ax=None, levels: int = 30) -> Figure:
    """2D contour plot of absorbance with time on x and wavelength on y."""
    fig, ax = _resolve_ax(ax)

    contour = ax.contourf(
        data.times, data.wavelengths, data.absorbance.T, levels=levels, cmap="viridis"
    )
    fig.colorbar(contour, ax=ax, label="Absorbance (AU)")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Wavelength (nm)")
    title = "PDA Contour"
    if data.metadata.injection_id:
        title = f"{title} ({data.metadata.injection_id})"
    ax.set_title(title)
    return fig


def plot_uv_spectra(peaks: PeakTable | Iterable[Peak], ax=None) -> Figure:
    """Overlay UV spectra for peaks that carry a ``.spectrum``.

    Peaks without a spectrum (``peak.spectrum is None``) are silently
    skipped. Each plotted spectrum is labeled by ``peak_id`` and, when
    ``lambda_max`` is set, that wavelength is marked on the curve.
    """
    fig, ax = _resolve_ax(ax)

    for peak in peaks:
        spectrum = peak.spectrum
        if spectrum is None:
            continue
        label = peak.peak_id or (spectrum.label or None)
        line, = ax.plot(spectrum.wavelengths, spectrum.values, label=label)

        if peak.lambda_max is not None:
            idx = int(np.argmin(np.abs(spectrum.wavelengths - peak.lambda_max)))
            ax.plot(
                spectrum.wavelengths[idx],
                spectrum.values[idx],
                marker="o",
                color=line.get_color(),
                markersize=5,
            )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (AU)")
    ax.set_title("UV Spectra")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    return fig


def plot_tracking(result: TrackingResult, ax=None) -> Figure:
    """Scatter apex_time vs injection for each tracked peak group.

    Injections are placed on the x-axis in ``result.injection_ids`` order;
    points belonging to the same :class:`~pda_peak_finder.models.PeakGroup`
    share a color and are connected with a line so retention-time drift
    across injections is visible.
    """
    fig, ax = _resolve_ax(ax)

    injection_ids = result.injection_ids
    x_positions = {inj: i for i, inj in enumerate(injection_ids)}

    for group in result.groups:
        xs = []
        ys = []
        for inj in injection_ids:
            peak = group.members.get(inj)
            if peak is None:
                continue
            xs.append(x_positions[inj])
            ys.append(peak.apex_time)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linestyle="-", label=f"Group {group.group_id}")

    ax.set_xticks(range(len(injection_ids)))
    ax.set_xticklabels(injection_ids, rotation=45, ha="right")
    ax.set_xlabel("Injection")
    ax.set_ylabel("Apex time (min)")
    ax.set_title("Peak Tracking")
    if result.groups:
        ax.legend(fontsize=8)
    return fig


def save_figure(fig: Figure, path) -> Path:
    """Save ``fig`` to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
