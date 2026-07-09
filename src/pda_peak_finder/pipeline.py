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
from .peak_detection import (
    DeconvolutionConfig,
    PeakDetectionConfig,
    detect_peaks,
    detect_peaks_deconvolved,
    filter_peaks_by_height,
)
from .plotting import (
    plot_chromatogram,
    plot_contour,
    plot_tracking,
    plot_uv_spectra,
    save_figure,
)
from .reader import load, load_many
from .spectra import SpectrumConfig, annotate_peaks, filter_peaks_by_absorbance
from .tracking import TrackingConfig, track_peaks


@dataclass
class AnalysisConfig:
    """Run-wide configuration for the whole pipeline."""

    detection: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    #: Separate overlapping/co-eluting peaks by curve fitting so each peak's
    #: RT and FWHM survive overlap (for the regression data). Uses
    #: ``deconvolution`` instead of ``detection`` when True.
    deconvolve: bool = False
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    #: Restrict the MaxPlot to this wavelength window (nm), or None for full range.
    wavelength_range: tuple[float, float] | None = None
    #: Detect on the single-wavelength chromatogram at this wavelength (nm),
    #: e.g. 230 to mirror Empower's 230 nm trace. None = detect on the MaxPlot.
    base_wavelength: float | None = None
    #: Averaging bandwidth (nm) around ``base_wavelength`` (0 = nearest point).
    base_bandwidth: float = 0.0
    #: Drop peaks with little/no absorbance at this monitoring wavelength (nm).
    #: None disables the filter.
    monitor_wavelength: float | None = None
    #: Minimum absolute absorbance (AU) required at ``monitor_wavelength``.
    monitor_min_absorbance: float = 0.0
    #: Minimum absorbance at ``monitor_wavelength`` as a fraction of the peak's
    #: own spectral maximum (0 disables the relative test).
    monitor_min_fraction: float = 0.0
    #: Keep only peaks whose height is within [height_min, height_max] (AU).
    #: None disables that bound. (Convert detector-unit ranges, e.g. µV, to AU.)
    height_min: float | None = None
    height_max: float | None = None


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
    if config.base_wavelength is not None:
        base = data.chromatogram_at(config.base_wavelength, config.base_bandwidth)
    else:
        base = data.maxplot(wavelength_range=config.wavelength_range)
    if config.deconvolve:
        table = detect_peaks_deconvolved(base, config.deconvolution)
    else:
        table = detect_peaks(base, config.detection)
    table.source_label = base.label
    annotate_peaks(data, table, config.spectrum)
    if config.monitor_wavelength is not None:
        table = filter_peaks_by_absorbance(
            table,
            config.monitor_wavelength,
            min_absorbance=config.monitor_min_absorbance,
            min_fraction=config.monitor_min_fraction,
        )
    if config.height_min is not None or config.height_max is not None:
        table = filter_peaks_by_height(
            table, min_height=config.height_min, max_height=config.height_max
        )
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
