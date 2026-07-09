"""Greedy cross-injection peak matching (tracking / alignment).

Peaks that represent the same compound across multiple injections are
grouped primarily by retention time proximity, with an optional lambda-max
constraint layered on top. The algorithm is a deterministic greedy matcher:

1. Injections (``PeakTable``s) are processed in input order.
2. Within one injection, every (peak, existing group) pair that satisfies
   the tolerances is a match candidate, ranked by cost — closer RT (plus
   optional lambda-max soft cost) wins, ties broken by peak/group input
   order.
3. Candidates are consumed cheapest-first; a peak or a group is used at
   most once per injection, so a peak that loses a group to a closer peer
   is free to match a different compatible group instead.
4. Any peak still unmatched after that starts a brand-new group.

Group RT (and lambda-max, when enabled) is simply the running mean of the
peaks already assigned to it — computed on demand from ``PeakGroup``, so no
separate running-average bookkeeping is needed.
"""

from __future__ import annotations

import numpy as np

from ..models import Peak, PeakGroup, PeakTable, TrackingResult
from .config import TrackingConfig


def track_peaks(
    tables: list[PeakTable], config: TrackingConfig | None = None
) -> TrackingResult:
    """Match peaks representing the same compound across injections.

    ``tables`` is processed in input order. ``TrackingResult.injection_ids``
    preserves that order (one entry per input table's ``injection_id``).
    """
    if config is None:
        config = TrackingConfig()

    groups: list[PeakGroup] = []
    next_group_id = 0

    for table in tables:
        inj = table.injection_id
        peaks = list(table.peaks)

        # Every compatible (peak, group) pair for this injection, with cost.
        candidates: list[tuple[float, int, int]] = []  # (cost, peak_idx, group_idx)
        for pi, peak in enumerate(peaks):
            for gi, group in enumerate(groups):
                if inj in group.members:
                    continue  # this group already has a peak from this injection
                cost = _match_cost(peak, group, config)
                if cost is not None:
                    candidates.append((cost, pi, gi))
        candidates.sort()  # ascending cost; ties -> lower peak_idx, then group_idx

        matched_group_for_peak: dict[int, int] = {}
        used_groups: set[int] = set()
        for cost, pi, gi in candidates:
            if pi in matched_group_for_peak or gi in used_groups:
                continue
            matched_group_for_peak[pi] = gi
            used_groups.add(gi)

        for pi, peak in enumerate(peaks):
            gi = matched_group_for_peak.get(pi)
            if gi is not None:
                groups[gi].members[inj] = peak
            else:
                groups.append(PeakGroup(group_id=next_group_id, members={inj: peak}))
                next_group_id += 1

    injection_ids = [table.injection_id for table in tables]
    return TrackingResult(groups=groups, injection_ids=injection_ids)


def _match_cost(peak: Peak, group: PeakGroup, config: TrackingConfig) -> float | None:
    """Cost of matching ``peak`` into ``group``, or None if incompatible."""
    rt_diff = abs(peak.apex_time - group.mean_rt)

    # Spectral-similarity matching: gate on UV-spectrum cosine similarity so a
    # compound is tracked across conditions despite large RT shifts; RT only
    # breaks ties among spectrally-compatible candidates.
    if config.use_spectral:
        if config.rt_max_shift is not None and rt_diff > config.rt_max_shift:
            return None
        sim = _spectral_similarity(peak, group)
        if sim is None or sim < config.min_spectral_similarity:
            return None
        return (1.0 - sim) + config.rt_soft_weight * rt_diff

    if rt_diff > config.rt_tolerance:
        return None

    cost = rt_diff
    if config.use_lambda_max:
        group_lambda = _mean_lambda_max(group)
        if peak.lambda_max is not None and group_lambda is not None:
            lambda_diff = abs(peak.lambda_max - group_lambda)
            if lambda_diff > config.lambda_tolerance:
                return None
            cost += config.lambda_weight * lambda_diff
        # If either side lacks a lambda_max, fall back to RT-only matching
        # for this pair rather than blocking it outright.
    return cost


def _mean_lambda_max(group: PeakGroup) -> float | None:
    values = [p.lambda_max for p in group.members.values() if p.lambda_max is not None]
    if not values:
        return None
    return float(np.mean(values))


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity of two baseline-subtracted spectra."""
    a = a - a.min()
    b = b - b.min()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _spectral_similarity(peak: Peak, group: PeakGroup) -> float | None:
    """Best cosine similarity between ``peak``'s spectrum and the group's."""
    if peak.spectrum is None:
        return None
    sims = []
    for member in group.members.values():
        if member.spectrum is None:
            continue
        a, b = peak.spectrum.values, member.spectrum.values
        if a.shape != b.shape:
            b = np.interp(peak.spectrum.wavelengths,
                          member.spectrum.wavelengths, member.spectrum.values)
        s = _cosine(np.asarray(a, float), np.asarray(b, float))
        if s is not None:
            sims.append(s)
    return max(sims) if sims else None
