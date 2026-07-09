"""Overlap-robust peak detection by curve deconvolution.

The plain :func:`~pda_peak_finder.peak_detection.detect_peaks` (scipy
``find_peaks`` + ``peak_widths``) misses shoulders and co-eluting peaks, and
its half-height width is distorted where peaks overlap. For building
multiple-regression data we must not lose any peak that could interfere with a
target, and we need each peak's retention time and FWHM even under overlap.

This module fits each cluster of overlapping peaks with a sum of peak models
(Exponentially Modified Gaussian by default — the standard tailing
chromatographic shape — or plain Gaussian) and reads each component's
retention time (apex), FWHM, height and area off its *own* fitted curve, so
overlapping peaks are separated rather than merged.

Pipeline: seed apexes (find_peaks) + shoulders (negative 2nd derivative) →
group seeds into clusters → fit each cluster (local linear baseline) → extract
per-component RT/FWHM/area. Fitting failures fall back to the seed estimate so
a peak is never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import exponnorm

from ..models import Chromatogram, Peak, PeakTable


@dataclass
class DeconvolutionConfig:
    """Tunables for :func:`detect_peaks_deconvolved`."""

    model: str = "emg"                 # "emg" (tailing) or "gauss"
    min_prominence: float = 0.0006     # apex seeding prominence (AU)
    min_distance_min: float = 0.02     # minimum apex separation (minutes)
    smoothing_window: int | None = 11  # Savitzky-Golay window (samples) for seeding
    smoothing_polyorder: int = 3
    shoulder_rel_prominence: float = 0.05  # 2nd-deriv shoulder seed sensitivity
    #: Baseline method: "als" (asymmetric least squares — follows broad humps
    #: under dense clusters) or "opening" (fast morphological, flat offsets only).
    baseline_method: str = "als"
    als_lambda: float = 1e6            # ALS smoothness (larger = stiffer baseline)
    als_p: float = 0.005              # ALS asymmetry (small = baseline hugs valleys)
    als_niter: int = 10
    baseline_window_min: float = 1.0   # rolling-opening baseline window (minutes)
    overlap_valley_ratio: float = 0.5  # group peaks whose valley stays above this
    max_peaks_per_cluster: int = 5     # split clusters larger than this (speed)
    merge_rt_ratio: float = 0.25       # merge components closer than this·FWHM (RT)
    fit_maxfev: int = 8000


# -- peak models (area-parameterised, numerically stable) ------------------

def _gauss(x, area, mu, sigma):
    sigma = max(sigma, 1e-9)
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _emg(x, area, mu, sigma, tau):
    """Exponentially Modified Gaussian via scipy's exponnorm (area-scaled)."""
    sigma = max(sigma, 1e-9)
    tau = max(tau, 1e-9)
    return area * exponnorm.pdf(x, tau / sigma, loc=mu, scale=sigma)


def _n_params(model: str) -> int:
    return 4 if model == "emg" else 3


def _model_sum(model: str):
    p = _n_params(model)
    fn = _emg if model == "emg" else _gauss

    def total(x, *params):
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), p):
            y = y + fn(x, *params[i:i + p])
        return y

    return total, fn


# -- metrics off a single fitted component ---------------------------------

def _curve_metrics(t_fine, y):
    """Return (rt, fwhm, height, start, end) read off a component curve."""
    imax = int(np.argmax(y))
    height = float(y[imax])
    rt = float(t_fine[imax])
    if height <= 0:
        return rt, None, height, None, None
    half = height / 2.0

    def _cross(idx_range):
        for j in idx_range:
            if y[j] <= half:
                # linear interpolation between j and its neighbour toward apex
                k = j + 1 if j < imax else j - 1
                if 0 <= k < len(y) and y[k] != y[j]:
                    frac = (half - y[j]) / (y[k] - y[j])
                    return t_fine[j] + frac * (t_fine[k] - t_fine[j])
                return t_fine[j]
        return None

    left = _cross(range(imax, -1, -1))
    right = _cross(range(imax, len(y)))
    fwhm = float(right - left) if (left is not None and right is not None) else None
    return rt, fwhm, height, left, right


# -- seeding ---------------------------------------------------------------

def _seed_apexes(times, values, config):
    dt = float(np.median(np.diff(times)))
    distance = max(1, int(round(config.min_distance_min / dt)))
    y = values
    if config.smoothing_window and config.smoothing_window < len(values):
        win = config.smoothing_window | 1  # force odd
        y = savgol_filter(values, win, min(config.smoothing_polyorder, win - 1))

    apex_idx, _ = find_peaks(y, prominence=config.min_prominence, distance=distance)
    seeds = set(int(i) for i in apex_idx)

    # shoulders / unresolved peaks: maxima of the negative second derivative
    # (each peak centre — resolved or buried — is a concavity). Threshold is
    # adaptive to the curvature signal's own range.
    if config.smoothing_window and config.smoothing_window < len(values):
        win = config.smoothing_window | 1
        d2 = savgol_filter(values, win, min(config.smoothing_polyorder, win - 1),
                           deriv=2, delta=dt)
        neg = -d2
        prom = config.shoulder_rel_prominence * float(neg.max() - neg.min())
        sh_idx, _ = find_peaks(neg, distance=distance, prominence=max(prom, 1e-15))
        # a shoulder seed must sit on a real signal (above the seeding floor)
        floor = values.min() + config.min_prominence
        for i in sh_idx:
            if values[int(i)] < floor:
                continue
            if all(abs(int(i) - s) > distance // 2 for s in seeds):
                seeds.add(int(i))
    return sorted(seeds)


def _estimate_sigma(times, values, idx):
    """Rough half-width-at-half-max in minutes around a seed index."""
    h = values[idx]
    half = h / 2.0
    n = len(values)
    li = idx
    while li > 0 and values[li] > half:
        li -= 1
    ri = idx
    while ri < n - 1 and values[ri] > half:
        ri += 1
    fwhm = max(times[ri] - times[li], 2 * np.median(np.diff(times)))
    return fwhm / 2.3548


# -- baseline & clustering -------------------------------------------------

def _baseline(values, dt, window_min):
    """Coarse baseline via morphological opening (removes peaks < window)."""
    w = max(3, int(round(window_min / dt)) | 1)
    opened = maximum_filter1d(minimum_filter1d(values, w), w)
    return uniform_filter1d(opened, w)


def _als_baseline(y, lam=1e6, p=0.005, niter=10):
    """Asymmetric Least Squares baseline (Eilers & Boelens).

    Follows broad drift and humps beneath dense peak clusters while ignoring
    the peaks themselves, so co-eluting sharp peaks sitting on a hump are not
    swallowed by a spurious wide component. ``lam`` sets stiffness, ``p`` the
    asymmetry (small p pulls the baseline down toward the valleys).
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), format="csc")
    DTD = lam * (D @ D.T)
    w = np.ones(L)
    z = y
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L, format="csc")
        z = spsolve((W + DTD).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def compute_baseline(values, dt, config: "DeconvolutionConfig | None" = None):
    """Baseline for a signal per ``config.baseline_method`` (ALS or opening)."""
    config = config or DeconvolutionConfig()
    if config.baseline_method == "opening":
        return _baseline(values, dt, config.baseline_window_min)
    return _als_baseline(np.asarray(values, float), config.als_lambda,
                         config.als_p, config.als_niter)


def _split_large(group, values, max_size):
    """Split a large cluster at its deepest internal valleys until each chunk
    has at most ``max_size`` seeds. Splitting at the most-resolved boundaries
    keeps big joint fits fast and stable while barely touching real overlaps.
    """
    if len(group) <= max_size:
        return [group]
    best_k, best_v = 1, np.inf
    for k in range(1, len(group)):
        v = float(values[group[k - 1]:group[k] + 1].min())
        if v < best_v:
            best_v, best_k = v, k
    left, right = group[:best_k], group[best_k:]
    return _split_large(left, values, max_size) + _split_large(right, values, max_size)


def _group_by_valley(values, seeds, config):
    """Group adjacent seeds that are NOT baseline-resolved (shallow valley).

    Operates on baseline-subtracted values: two neighbours belong to the same
    cluster when the signal between them stays above ``overlap_valley_ratio``
    of the smaller apex — i.e. they overlap and must be deconvolved together.
    """
    if not seeds:
        return []
    groups = [[seeds[0]]]
    for s in seeds[1:]:
        prev = groups[-1][-1]
        valley = float(values[prev:s + 1].min())
        h = min(float(values[prev]), float(values[s]))
        if h > 0 and valley > config.overlap_valley_ratio * h:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def _merge_duplicates(items, merge_rt_ratio):
    """Collapse near-coincident components (a single peak split by the fit).

    ``items`` is a list of ``(peak, component)`` sorted by RT; the taller of
    two near-coincident peaks (and its component) is kept.
    """
    if not items:
        return items
    merged = [items[0]]
    for pk, comp in items[1:]:
        prev_pk, _ = merged[-1]
        fwhms = [f for f in (pk.fwhm, prev_pk.fwhm) if f]
        tol = merge_rt_ratio * min(fwhms) if fwhms else 0.0
        if abs(pk.apex_time - prev_pk.apex_time) <= tol:
            if pk.height > prev_pk.height:
                merged[-1] = (pk, comp)
        else:
            merged.append((pk, comp))
    return merged


# -- main ------------------------------------------------------------------

def detect_peaks_deconvolved(
    chromatogram: Chromatogram,
    config: DeconvolutionConfig | None = None,
    *,
    return_components: bool = False,
):
    """Detect peaks by deconvolving overlapping clusters.

    Returns a :class:`PeakTable` whose peaks carry overlap-separated
    ``apex_time`` (RT), ``fwhm``, ``area``, ``height`` and integration bounds.
    With ``return_components=True`` also returns a list of ``(t, y)`` arrays,
    one per fitted component over the full time axis, for plotting.
    """
    config = config or DeconvolutionConfig()
    times = chromatogram.times
    raw = chromatogram.values.astype(float)
    dt = float(np.median(np.diff(times)))
    model = config.model
    p = _n_params(model)
    total_fn, comp_fn = _model_sum(model)

    baseline = compute_baseline(raw, dt, config)
    values = raw - baseline  # baseline-subtracted; fits/seeds work on this

    seeds = _seed_apexes(times, values, config)
    groups = _group_by_valley(values, seeds, config)
    # keep joint fits small & fast: split big clusters at their deepest valleys
    groups = [sg for g in groups
              for sg in _split_large(g, values, config.max_peaks_per_cluster)]
    items: list[tuple[Peak, tuple | None]] = []  # (peak, component-curve or None)

    for group in groups:
        sigs = [_estimate_sigma(times, values, s) for s in group]
        # tight window: 3 sigma beyond the outer seeds
        pad = 3 * max(sigs)
        lo_t = times[group[0]] - pad
        hi_t = times[group[-1]] + pad
        mask = (times >= lo_t) & (times <= hi_t)
        t = times[mask]
        y = values[mask]
        base_seg = baseline[mask]

        # split overly large groups into chunks to keep the fit stable
        fitted = None
        if len(group) <= config.max_peaks_per_cluster:
            p0, lo_b, hi_b = [], [], []
            for s, sig in zip(group, sigs):
                amp = max(float(values[s]), 1e-6)
                area0 = amp * sig * 2.5
                p0 += [area0, times[s], sig] + ([sig * 0.5] if model == "emg" else [])
                lo_b += [0.0, t[0], 1e-4] + ([1e-4] if model == "emg" else [])
                hi_b += [area0 * 1e3 + 1e-3, t[-1], (t[-1] - t[0])] + \
                        ([(t[-1] - t[0])] if model == "emg" else [])
            try:
                fitted, _ = curve_fit(total_fn, t, y, p0=p0, bounds=(lo_b, hi_b),
                                      maxfev=config.fit_maxfev)
            except Exception:
                fitted = None

        t_fine = np.linspace(t[0], t[-1], max(300, 6 * len(t)))
        for i, (s, sig) in enumerate(zip(group, sigs)):
            comp = None
            if fitted is not None:
                params = fitted[i * p:(i + 1) * p]
                y_comp = comp_fn(t_fine, *params)
                rt, fwhm, height, start, end = _curve_metrics(t_fine, y_comp)
                area = float(params[0])
                apex_index = int(np.argmin(np.abs(times - rt)))
                comp = (t_fine, y_comp)  # baseline-subtracted component bump
            else:  # fallback: seed estimate, never drop the peak
                rt = float(times[s])
                fwhm = float(2.3548 * sig)
                height = float(values[s])
                start = end = None
                area = None
                apex_index = int(s)
            peak = Peak(
                apex_time=rt, apex_index=apex_index, height=height,
                start_time=start, end_time=end, fwhm=fwhm, area=area,
                injection_id=chromatogram.injection_id,
            )
            items.append((peak, comp))

    items.sort(key=lambda it: it[0].apex_time)
    items = _merge_duplicates(items, config.merge_rt_ratio)
    # drop components with no measurable FWHM (boundary/degenerate artifacts):
    # a peak without a width is useless for the regression data and is almost
    # always spurious. Seed-fallback peaks always carry a FWHM and are kept.
    items = [it for it in items if it[0].fwhm and it[0].fwhm > 0]
    peaks = []
    components = []
    for i, (peak, comp) in enumerate(items, start=1):
        peak.peak_id = f"P{i:03d}"
        peaks.append(peak)
        if comp is not None:
            components.append(comp)
    table = PeakTable(peaks=peaks, injection_id=chromatogram.injection_id,
                      source_label=f"{chromatogram.label} (deconvolved)")
    if return_components:
        return table, components
    return table
