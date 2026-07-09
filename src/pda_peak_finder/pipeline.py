"""End-to-end analysis pipeline.

Ties the modules together into the workflow described in the project goal:

    load -> MaxPlot -> detect peaks -> peak properties -> UV spectra
         -> tracking (across injections) -> CSV export -> plots

Each stage lives in its own module; this file only orchestrates them and
holds the run-wide configuration. Callers use :func:`analyze_injection`
for one dataset or :func:`run_pipeline` for a whole batch with output
files written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import export
from .models import PDAData, PeakTable, TrackingResult
from .peak_detection import PeakDetectionConfig, detect_peaks
from .plotting import (
    plot_chromatogram,
    plot_contour,
    plot_tracking,
    plot_uv_spectra,
    save_figure,
)
from .reader import load, load_many
from .spectra import SpectrumConfig, annotate_peaks
from .tracking import TrackingConfig, track_peaks


@dataclass
class AnalysisConfig:
    """Run-wide configuration for the whole pipeline."""

    detection: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    #: Restrict the MaxPlot to this wavelength window (nm), or None for full range.
    wavelength_range: tuple[float, float] | None = None


@dataclass
class AnalysisResult:
    """Everything produced by a pipeline run."""

    datasets: list[PDAData]
    tables: list[PeakTable]
    tracking: TrackingResult | None = None


def analyze_injection(
    data: PDAData, config: AnalysisConfig | None = None
) -> PeakTable:
    """Run steps 2-5 for a single injection: MaxPlot -> detect -> spectra.

    Returns a PeakTable whose peaks carry RT, FWHM, area, lambda_max and the
    extracted UV spectrum.
    """
    config = config or AnalysisConfig()
    maxplot = data.maxplot(wavelength_range=config.wavelength_range)
    table = detect_peaks(maxplot, config.detection)
    table.source_label = maxplot.label
    annotate_peaks(data, table, config.spectrum)
    return table


def analyze_datasets(
    datasets: Sequence[PDAData], config: AnalysisConfig | None = None
) -> list[PeakTable]:
    """Run :func:`analyze_injection` over several already-loaded datasets."""
    config = config or AnalysisConfig()
    return [analyze_injection(d, config) for d in datasets]


def run_pipeline(
    paths: Iterable[Path | str],
    config: AnalysisConfig | None = None,
    *,
    format: str | None = None,
    output_dir: Path | str | None = None,
    do_tracking: bool = True,
    do_plots: bool = True,
) -> AnalysisResult:
    """Full workflow over a batch of files, optionally writing CSV + plots.

    ``paths`` is one file per injection. When ``output_dir`` is given, per-
    injection peak tables, a combined long table, tracking matrix, and plots
    are written there.
    """
    config = config or AnalysisConfig()
    datasets = load_many(paths, format=format)
    tables = analyze_datasets(datasets, config)

    tracking: TrackingResult | None = None
    if do_tracking and len(tables) > 1:
        tracking = track_peaks(tables, config.tracking)

    if output_dir is not None:
        _write_outputs(datasets, tables, tracking, Path(output_dir), do_plots)

    return AnalysisResult(datasets=datasets, tables=tables, tracking=tracking)


def _write_outputs(
    datasets: list[PDAData],
    tables: list[PeakTable],
    tracking: TrackingResult | None,
    output_dir: Path,
    do_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    export.write_peak_tables(tables, output_dir / "peaks_all.csv")
    for table in tables:
        inj = table.injection_id or "injection"
        export.write_peak_table(table, output_dir / f"peaks_{inj}.csv")
        export.write_peak_spectra(table, output_dir / f"spectra_{inj}.csv")
    if tracking is not None:
        export.write_tracking_matrix(
            tracking, output_dir / "tracking_rt.csv", value="apex_time"
        )
        export.write_tracking_matrix(
            tracking, output_dir / "tracking_lambda_max.csv", value="lambda_max"
        )

    if not do_plots:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for data, table in zip(datasets, tables):
        inj = table.injection_id or "injection"
        maxplot = data.maxplot()
        save_figure(
            plot_chromatogram(maxplot, table), plot_dir / f"chromatogram_{inj}.png"
        )
        save_figure(plot_contour(data), plot_dir / f"contour_{inj}.png")
        save_figure(plot_uv_spectra(table), plot_dir / f"spectra_{inj}.png")
    if tracking is not None:
        save_figure(plot_tracking(tracking), plot_dir / "tracking.png")
