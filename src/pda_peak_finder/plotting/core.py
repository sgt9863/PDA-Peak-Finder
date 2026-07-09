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
    "plot_labeled_chromatogram",
    "plot_contour",
    "plot_uv_spectra",
    "plot_tracking",
    "save_figure",
    "peak_palette",
    "configure_japanese_font",
]

#: Candidate CJK-capable font files, in preference order.
_JP_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
)


def configure_japanese_font(path: str | None = None) -> str | None:
    """Register a CJK font with matplotlib so Japanese labels render.

    Returns the font family name that was configured, or ``None`` if no CJK
    font could be found (in which case Japanese text falls back to tofu and
    callers may prefer ASCII labels). Safe to call repeatedly.
    """
    import os
    from matplotlib import font_manager

    candidates = [path] if path else list(_JP_FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            font_manager.fontManager.addfont(candidate)
            name = font_manager.FontProperties(fname=candidate).get_name()
            # Fall back to DejaVu Sans for glyphs the CJK font lacks (e.g. µ).
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def _resolve_ax(ax, figsize=None):
    """Return (figure, axes), creating a new Figure/Axes pair if needed."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def peak_palette(n: int) -> list:
    """A soft, evenly-spaced hue sequence for labeling many peaks.

    Identity is carried by each peak's text label and x-position, so color
    here only separates neighbours (as in a Waters QDa/SIR overlay). Hues are
    spread around the wheel at a gentle saturation so the result reads as a
    refined spectrum rather than a harsh rainbow.
    """
    import colorsys

    if n <= 0:
        return []
    return [colorsys.hls_to_rgb((i / max(n, 1)) % 1.0, 0.45, 0.55) for i in range(n)]


def _peak_label(peak, label_attr: str) -> str:
    if label_attr == "lambda_max" and peak.lambda_max is not None:
        return f"{peak.lambda_max:.0f}nm"
    if label_attr == "apex_time":
        return f"{peak.apex_time:.2f}"
    return peak.peak_id or f"{peak.apex_time:.2f}"


def plot_labeled_chromatogram(
    chromatogram: Chromatogram,
    table: PeakTable,
    ax=None,
    *,
    label_attr: str = "lambda_max",
    normalize: bool = True,
    fill: bool = True,
    colors=None,
    y_scale: float = 1.0,
    y_unit: str = "AU",
) -> Figure:
    """Waters QDa/SIR-style labeled chromatogram.

    Each detected peak is drawn as an isolated coloured trace segment
    (its neighbourhood of the chromatogram, roughly ``+/-3*FWHM`` around the
    apex) with a vertical label above it. With ``normalize=True`` each peak is
    scaled to unit apex height ("Y-axis normalized"), so the display reads as
    a row of labelled peaks regardless of their true absorbance — mirroring
    the amino-acid SIR overlay style. ``label_attr`` selects the label text:
    ``"lambda_max"`` (default), ``"peak_id"`` or ``"apex_time"``.
    """
    fig, ax = _resolve_ax(ax, figsize=(12, 4.8))
    peaks = list(table)
    colors = colors or peak_palette(len(peaks))

    scale = 1.0 if normalize else y_scale
    times = chromatogram.times
    apex_top = 1.0 if normalize else max((p.height * scale for p in peaks), default=1.0)
    drawn = []  # (apex_x, apex_y, color, text) for the labeling pass
    for i, peak in enumerate(peaks):
        color = colors[i % len(colors)]
        # window: the peak's own integration bounds keep each bump isolated
        # from its neighbours (so no stray sub-peaks bleed in); a small FWHM
        # pad lets the bump settle to baseline. Fall back to a FWHM window or
        # a few sampling intervals when bounds are missing.
        pad = 0.4 * peak.fwhm if peak.fwhm else 0.0
        if peak.start_time is not None and peak.end_time is not None:
            lo, hi = peak.start_time - pad, peak.end_time + pad
        elif peak.fwhm:
            lo, hi = peak.apex_time - 2.0 * peak.fwhm, peak.apex_time + 2.0 * peak.fwhm
        else:
            dt = chromatogram.sampling_interval
            lo, hi = peak.apex_time - 20 * dt, peak.apex_time + 20 * dt
        mask = (times >= lo) & (times <= hi)
        if not mask.any():
            continue
        seg_t = times[mask]
        seg_v = chromatogram.values[mask].astype(float)
        base = float(seg_v.min())
        denom = (peak.height - base) if normalize and peak.height > base else 1.0
        seg_v = (seg_v - base) / denom if normalize else seg_v * scale
        apex_y = 1.0 if normalize else peak.height * scale

        ax.plot(seg_t, seg_v, color=color, linewidth=1.1, zorder=3)
        if fill:
            ax.fill_between(seg_t, seg_v, color=color, alpha=0.10, zorder=2)
        drawn.append((peak.apex_time, apex_y, color, _peak_label(peak, label_attr)))

    # Label pass: stagger labels into tiers so dense clusters stay legible,
    # with a thin leader line from each apex up to its (raised) label.
    span = float(times[-1] - times[0]) or 1.0
    min_dx = 0.022 * span
    step = 0.16 * apex_top
    tier_last_x: list[float] = []
    max_tier = 0
    for apex_x, apex_y, color, text in drawn:
        tier = next(
            (t for t, lx in enumerate(tier_last_x) if apex_x - lx >= min_dx),
            len(tier_last_x),
        )
        if tier == len(tier_last_x):
            tier_last_x.append(apex_x)
        else:
            tier_last_x[tier] = apex_x
        max_tier = max(max_tier, tier)
        label_y = apex_top * 1.03 + tier * step
        ax.plot([apex_x, apex_x], [apex_y, label_y], color=color,
                linewidth=0.6, alpha=0.5, zorder=2)
        ax.annotate(
            text, xy=(apex_x, label_y), xytext=(0, 2),
            textcoords="offset points", rotation=90,
            ha="center", va="bottom", fontsize=7.5, color=color,
        )

    ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=1)
    ax.set_xlabel("保持時間 / Retention time (min)")
    ax.set_ylabel("正規化強度" if normalize else f"シグナル ({y_unit})")
    label_ceiling = apex_top * 1.03 + max_tier * step + 0.28 * apex_top
    if normalize:
        ax.set_ylim(0, max(1.32, label_ceiling))
        ax.set_yticks([0, 0.5, 1.0])
    else:
        ax.set_ylim(0, max(apex_top * 1.1, label_ceiling))
    ax.margins(x=0.01)
    title = (chromatogram.label or "Chromatogram")
    if normalize:
        title += " — Y軸ノーマライズ"
    if chromatogram.injection_id:
        title = f"{title}  ({chromatogram.injection_id})"
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig


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


def plot_deconvolution(
    chromatogram: Chromatogram,
    components,
    table: PeakTable | None = None,
    ax=None,
    *,
    xlim: tuple[float, float] | None = None,
    colors=None,
) -> Figure:
    """Show the raw trace with each deconvolved component drawn separately.

    ``components`` is the ``(t, y)`` list returned by
    :func:`~pda_peak_finder.peak_detection.detect_peaks_deconvolved` with
    ``return_components=True``. Each fitted component is filled in its own
    colour so overlapping/co-eluting peaks are visibly separated; the raw
    chromatogram is overlaid in black and peaks are labelled by RT.
    """
    fig, ax = _resolve_ax(ax, figsize=(12, 4.8))
    comps = list(components)
    colors = colors or peak_palette(len(comps))

    ax.plot(chromatogram.times, chromatogram.values, color="black",
            linewidth=1.0, zorder=5, label="測定トレース")
    for i, (t, y) in enumerate(comps):
        c = colors[i % len(colors)]
        ax.fill_between(t, y, color=c, alpha=0.35, zorder=2)
        ax.plot(t, y, color=c, linewidth=0.9, zorder=3)

    if table is not None:
        lo, hi = xlim if xlim else (chromatogram.times[0], chromatogram.times[-1])
        for pk in table:
            if lo <= pk.apex_time <= hi:
                ax.annotate(f"{pk.apex_time:.3f}", xy=(pk.apex_time, pk.height),
                            xytext=(0, 3), textcoords="offset points", rotation=90,
                            ha="center", va="bottom", fontsize=7, color="0.25")

    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("保持時間 / Retention time (min)")
    ax.set_ylabel("Absorbance (AU)")
    ax.set_title(f"デコンボリューション {chromatogram.label} ({chromatogram.injection_id})")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
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
    # A legend with one entry per group is unreadable past a dozen groups;
    # the connected points already show drift, so only label small sets.
    if 0 < len(result.groups) <= 12:
        ax.legend(fontsize=8, ncol=1)
    return fig


def save_figure(fig: Figure, path) -> Path:
    """Save ``fig`` to ``path``, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
