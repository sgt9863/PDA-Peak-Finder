"""Post-detection peak filters based on peak height."""

from __future__ import annotations

from ..models import Peak, PeakTable


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
