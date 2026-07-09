"""Command-line interface.

Two subcommands:

* ``analyze`` — run the full pipeline over one or more data files.
* ``demo``    — run the pipeline on built-in synthetic data (no files
  needed), so the tool can be exercised before the ARW reader exists.

Entry point: ``pda-peaks`` (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .peak_detection import PeakDetectionConfig
from .pipeline import AnalysisConfig, AnalysisResult, analyze_datasets, run_pipeline
from .spectra import SpectrumConfig
from .tracking import TrackingConfig


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    wl_range = tuple(args.wavelength_range) if args.wavelength_range else None
    return AnalysisConfig(
        detection=PeakDetectionConfig(
            min_height=args.min_height,
            min_prominence=args.min_prominence,
            min_distance_min=args.min_distance,
        ),
        spectrum=SpectrumConfig(apex_average_scans=args.apex_average_scans),
        tracking=TrackingConfig(rt_tolerance=args.rt_tolerance),
        wavelength_range=wl_range,
        monitor_wavelength=args.monitor_wavelength,
        monitor_min_absorbance=args.monitor_min_abs,
        monitor_min_fraction=args.monitor_min_fraction,
        # detection already enforces --min-height; the filter adds the upper bound
        height_max=args.max_height,
        rt_min=args.rt_min,
        rt_max=args.rt_max,
    )


def _print_summary(result: AnalysisResult) -> None:
    for table in result.tables:
        print(f"[{table.injection_id}] {len(table)} peaks ({table.source_label})")
        for peak in table:
            fwhm = "-" if peak.fwhm is None else f"{peak.fwhm:.3f}"
            lam = "-" if peak.lambda_max is None else f"{peak.lambda_max:.1f}"
            print(
                f"  {peak.peak_id}: RT={peak.apex_time:.3f} min  "
                f"FWHM={fwhm} min  lambda_max={lam} nm  height={peak.height:.4g}"
            )
    if result.tracking is not None:
        print(f"tracking: {len(result.tracking.groups)} peak groups across "
              f"{len(result.tracking.injection_ids)} injections")


def _cmd_analyze(args: argparse.Namespace) -> int:
    config = _build_config(args)
    result = run_pipeline(
        args.files,
        config,
        format=args.format,
        output_dir=args.output,
        do_tracking=not args.no_tracking,
        do_plots=not args.no_plots,
    )
    _print_summary(result)
    if args.output:
        print(f"outputs written to {args.output}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .testing import synthetic_pdadata

    config = _build_config(args)
    # Three injections with slightly different RTs, so tracking has something to do.
    datasets = [
        synthetic_pdadata(injection_id=f"DEMO{i+1}") for i in range(3)
    ]
    tables = analyze_datasets(datasets, config)
    result = AnalysisResult(datasets=datasets, tables=tables)
    if not args.no_tracking and len(tables) > 1:
        from .tracking import track_peaks

        result.tracking = track_peaks(tables, config.tracking)
    if args.output:
        from .pipeline import _write_outputs

        _write_outputs(datasets, tables, result.tracking, Path(args.output),
                       do_plots=not args.no_plots)
    _print_summary(result)
    if args.output:
        print(f"outputs written to {args.output}")
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="directory for CSV + plot outputs")
    p.add_argument("--min-height", type=float, default=None,
                   help="minimum peak height (AU)")
    p.add_argument("--max-height", type=float, default=None,
                   help="drop peaks taller than this height (AU)")
    p.add_argument("--min-prominence", type=float, default=0.01,
                   help="minimum peak prominence (AU)")
    p.add_argument("--min-distance", type=float, default=0.05,
                   help="minimum peak separation (minutes)")
    p.add_argument("--apex-average-scans", type=int, default=0,
                   help="average spectra over +/- N scans around each apex")
    p.add_argument("--monitor-wavelength", type=float, default=None,
                   help="drop peaks with little/no absorbance at this wavelength (nm)")
    p.add_argument("--monitor-min-abs", type=float, default=0.0,
                   help="min absolute absorbance (AU) required at --monitor-wavelength")
    p.add_argument("--monitor-min-fraction", type=float, default=0.0,
                   help="min absorbance at --monitor-wavelength as fraction of peak max")
    p.add_argument("--rt-min", type=float, default=None,
                   help="drop peaks eluting before this retention time (minutes)")
    p.add_argument("--rt-max", type=float, default=None,
                   help="drop peaks eluting after this retention time (minutes)")
    p.add_argument("--rt-tolerance", type=float, default=0.2,
                   help="RT tolerance for peak tracking (minutes)")
    p.add_argument("--wavelength-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="restrict MaxPlot to LO..HI nm")
    p.add_argument("--no-tracking", action="store_true",
                   help="skip cross-injection peak tracking")
    p.add_argument("--no-plots", action="store_true",
                   help="skip plot generation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pda-peaks",
        description="Detect and analyze peaks in PDA/UV 3D spectral data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze one or more data files")
    analyze.add_argument("files", nargs="+", type=Path, help="input data files")
    analyze.add_argument("--format", default=None,
                         help="force a reader format (default: auto-detect)")
    _add_common_args(analyze)
    analyze.set_defaults(func=_cmd_analyze)

    demo = sub.add_parser("demo", help="run on built-in synthetic data")
    _add_common_args(demo)
    demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
