# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PDA-Peak-Finder analyzes Waters Empower PDA/UV **3D spectral data** (time × wavelength × absorbance):
auto-detect UV-absorbing peaks, compute each peak's Retention Time / FWHM / λmax, track peaks
across injections, export CSV, and visualize. Python-only (numpy, pandas, scipy, matplotlib);
CLI-first, no GUI.

## Commands

```bash
python -m pip install -e ".[dev]"          # install package + dev deps (pytest)
python -m pytest                            # run all tests
python -m pytest tests/test_spectra.py -q   # single module
python -m pytest tests/test_spectra.py::test_name   # single test
pda-peaks demo -o results/                  # run full pipeline on built-in synthetic data
pda-peaks analyze data/*.arw -o results/    # analyze real Waters Empower ARW exports
```

`pda-peaks demo` and `pda_peak_finder.testing.synthetic_pdadata()` exercise the whole pipeline
without any files. The ARW reader is implemented; `data/*.arw` sample files (if present) are
gitignored and picked up by `tests/test_arw.py` only when they exist.

## Architecture

The design is **file-format-agnostic**: only the `reader` module knows about file formats;
everything else operates on the shared data model in `src/pda_peak_finder/models.py`, which is the
contract binding all modules together. Change the model deliberately — it ripples everywhere.

**Data flow:** `reader.load` → `PDAData` → `maxplot()` → `peak_detection.detect_peaks` → `PeakTable`
→ `spectra.annotate_peaks` (adds λmax + spectrum) → `tracking.track_peaks` → `TrackingResult`
→ `export.write_*` / `plotting.plot_*`. `pipeline.run_pipeline()` orchestrates all of it.

**Module responsibilities** (each owns its own directory; they communicate only through `models.py` types):
- `reader/` — file → validated `PDAData`. `base.py` has the `SpectralDataReader` ABC + registry
  (`register_reader`/`load`); `arw.py` parses Waters Empower ARW exports (Shift-JIS/cp932, CR line
  endings, TAB-separated; `波長` wavelength axis + `時間` time-major data block; structural, not
  label-dependent; drops a truncated final row).
- `peak_detection/` — detect peaks in a `Chromatogram`, compute RT/FWHM/area (scipy.signal).
- `spectra/` — extract UV spectrum at each apex, compute λmax, annotate the `PeakTable`.
- `tracking/` — match peaks across injections (greedy, RT-based, optional λmax).
- `export/` — write `PeakTable`/`TrackingResult`/spectra to CSV (pandas).
- `plotting/` — matplotlib figures (Agg backend). Returns `Figure`; `save_figure` writes files.
- `pipeline.py` / `cli.py` — orchestration and the `pda-peaks` CLI (`analyze` / `demo`).
- `testing.py` — `synthetic_pdadata()` builds a `PDAData` with known RT/FWHM/λmax for ground-truth tests.

Units are fixed everywhere: **time = minutes, wavelength = nm, absorbance = AU**. Readers are
responsible for converting to these. `PDAData.__post_init__` enforces strictly-increasing axes,
`(T, W)` shape, and non-empty `injection_id`, so malformed data can't reach downstream stages.

See `docs/ARCHITECTURE.md` for the data model table and reader-interface contract, and
`docs/ROADMAP.md` for planned work (ARW parsing is the next milestone).

## Conventions

- Adding a file format = one `SpectralDataReader` subclass + `register_reader()` in `reader/__init__.py`;
  no other module changes.
- Each module exposes a small public API from its `__init__.py` and a `*Config` dataclass for tunables.
- Tests assert against synthetic ground truth; keep every stage runnable on `synthetic_pdadata()`.
- `data/` and `results/` are gitignored (measurement data and outputs are not committed).
