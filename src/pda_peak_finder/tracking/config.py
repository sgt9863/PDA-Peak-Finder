"""Configuration for cross-injection peak tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackingConfig:
    """Parameters controlling how peaks are matched across injections.

    Matching is primarily by retention time (``rt_tolerance``); lambda-max
    can be layered on as an additional hard constraint (``use_lambda_max``
    + ``lambda_tolerance``) and, independently, as a soft tie-breaking cost
    (``lambda_weight``) among otherwise-compatible candidates.
    """

    #: Max |apex_time - group RT| (minutes) to join an existing group.
    rt_tolerance: float = 0.2
    #: Also require lambda_max to match within lambda_tolerance.
    use_lambda_max: bool = False
    #: Max |lambda_max - group lambda_max| (nm), only checked when use_lambda_max.
    lambda_tolerance: float = 5.0
    #: Soft cost weight added per nm of lambda_max difference when ranking
    #: candidates (0 = ignore for ranking, only used as a hard gate above).
    lambda_weight: float = 0.0
