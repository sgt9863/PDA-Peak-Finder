"""Synthetic data generation for tests and demos.

This lets the whole pipeline be developed and tested WITHOUT any real ARW
files: it builds a :class:`~pda_peak_finder.models.PDAData` whose peaks have
known retention times, widths, and lambda-max values, so tests can assert
against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .models import InjectionMetadata, PDAData


@dataclass
class SyntheticPeakSpec:
    """Ground-truth description of one synthetic peak."""

    rt: float          # retention time, minutes
    width: float       # gaussian sigma in time, minutes
    height: float      # peak absorbance, AU
    lambda_max: float  # nm
    band_width: float = 30.0  # gaussian sigma of the UV band, nm


@dataclass
class SyntheticConfig:
    """Parameters for one synthetic injection."""

    peaks: list[SyntheticPeakSpec]
    t_start: float = 0.0
    t_end: float = 20.0
    n_times: int = 2000            # ~0.6 s/scan over 20 min
    wl_start: float = 200.0
    wl_end: float = 400.0
    n_wavelengths: int = 201       # 1 nm resolution
    noise: float = 0.0             # gaussian noise sigma, AU
    baseline: float = 0.0          # constant baseline offset, AU
    seed: int = 0


def synthetic_pdadata(
    config: SyntheticConfig | None = None,
    *,
    injection_id: str = "SYN001",
    **meta_kwargs,
) -> PDAData:
    """Build a synthetic PDAData from ground-truth peak specs.

    Each peak is separable: absorbance(t, w) = sum_k height_k
    * gaussian_t(t; rt_k, width_k) * gaussian_w(w; lambda_max_k, band_k),
    so the MaxPlot apex sits at ``rt`` and the apex spectrum peaks at
    ``lambda_max``.
    """
    if config is None:
        config = SyntheticConfig(
            peaks=[
                SyntheticPeakSpec(rt=4.0, width=0.10, height=0.8, lambda_max=254.0),
                SyntheticPeakSpec(rt=8.5, width=0.15, height=1.2, lambda_max=280.0),
                SyntheticPeakSpec(rt=13.0, width=0.12, height=0.5, lambda_max=230.0),
            ]
        )

    times = np.linspace(config.t_start, config.t_end, config.n_times)
    wavelengths = np.linspace(config.wl_start, config.wl_end, config.n_wavelengths)
    absorbance = np.full((config.n_times, config.n_wavelengths), config.baseline)

    for spec in config.peaks:
        time_profile = np.exp(-0.5 * ((times - spec.rt) / spec.width) ** 2)
        wl_profile = np.exp(-0.5 * ((wavelengths - spec.lambda_max) / spec.band_width) ** 2)
        absorbance += spec.height * np.outer(time_profile, wl_profile)

    if config.noise > 0:
        rng = np.random.default_rng(config.seed)
        absorbance = absorbance + rng.normal(0.0, config.noise, absorbance.shape)

    metadata = InjectionMetadata(
        injection_id=injection_id,
        sample_name=meta_kwargs.pop("sample_name", "synthetic"),
        acquired_at=meta_kwargs.pop("acquired_at", None),
        instrument=meta_kwargs.pop("instrument", "synthetic"),
        channel=meta_kwargs.pop("channel", "PDA"),
        extra=meta_kwargs.pop("extra", {}),
    )
    return PDAData(
        times=times,
        wavelengths=wavelengths,
        absorbance=absorbance,
        metadata=metadata,
    )
