"""Post-detection peak filters based on peak height and retention time."""

from __future__ import annotations

from ..models import Peak, PeakTable


def filter_peaks_by_retention_time(
    table: PeakTable,
    *,
    rt_min: float | None = None,
    rt_max: float | None = None,
) -> PeakTable:
    """Keep only peaks whose retention time lies within ``[rt_min, rt_max]``.

    Bounds are in minutes (the fixed time unit); ``None`` disables that side.
    Useful for dropping the solvent front / injection dip at the start of a
    run, or trailing junk past the region of interest. Peaks outside the
    window are dropped and the survivors are renumbered contiguously
    (P001, P002, ...). Returns a NEW PeakTable; the input is not modified.
    """
    kept: list[Peak] = []
    for peak in table.peaks:
        if rt_min is not None and peak.apex_time < rt_min:
            continue
        if rt_max is not None and peak.apex_time > rt_max:
            continue
        kept.append(peak)
    for i, peak in enumerate(kept, start=1):
        peak.peak_id = f"P{i:03d}"
    return PeakTable(
        peaks=kept,
        injection_id=table.injection_id,
        source_label=table.source_label,
    )


def filter_peaks_by_height(
    table: PeakTable,
    *,
    min_height: float | None = None,
    max_height: float | None = None,
) -> PeakTable:
    """Keep only peaks whose height lies within ``[min_height, max_height]``.

    Bounds are in the chromatogram's absorbance units (AU); ``None`` disables
    that side. Peaks outside the range are dropped and the survivors are
    renumbered contiguously (P001, P002, ...). Returns a NEW PeakTable; the
    input is not modified.

    Callers working in detector units (e.g. µV) convert their range to AU
    before calling — the models are always in AU.
    """
    kept: list[Peak] = []
    for peak in table.peaks:
        if min_height is not None and peak.height < min_height:
            continue
        if max_height is not None and peak.height > max_height:
            continue
        kept.append(peak)
    for i, peak in enumerate(kept, start=1):
        peak.peak_id = f"P{i:03d}"
    return PeakTable(
        peaks=kept,
        injection_id=table.injection_id,
        source_label=table.source_label,
    )
