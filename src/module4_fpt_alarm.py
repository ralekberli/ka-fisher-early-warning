"""
module4_fpt_alarm.py

First-passage-time (FPT) alarm: fires at the first decoding step t*
at which a smoothed lambda_max trajectory crosses a calibrated
threshold tau.

This is the mechanism behind the paper's central empirical result
(E6): a median 27-token lead time before hallucination onset, at an
85.6% detection rate.

The threshold tau is NOT a fixed constant — per the paper's
Calibration Necessity Protocol (Section 6), it must be empirically
calibrated per hardware/quantization configuration. The 244x gap
between a naive analytic threshold and the measured threshold on
Apple M5 / Q4_K_M is documented in data/calibration_244x.csv; do not
reuse that specific tau value on different hardware without
recalibration.
"""

from __future__ import annotations
import numpy as np


def smooth_trajectory(lambda_max_trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing to reduce per-token noise before
    threshold crossing is evaluated."""
    if len(lambda_max_trajectory) < window:
        return lambda_max_trajectory
    kernel = np.ones(window) / window
    return np.convolve(lambda_max_trajectory, kernel, mode="same")


def fpt_alarm(lambda_max_trajectory: np.ndarray, tau: float, *,
              smoothing_window: int = 5) -> int | None:
    """
    Returns the first decoding step index at which the smoothed
    lambda_max trajectory crosses tau, or None if it never does.

    Parameters
    ----------
    lambda_max_trajectory : np.ndarray
        Per-step lambda_max estimates from module2.
    tau : float
        Calibrated alarm threshold. Must be empirically determined for
        the target hardware/quantization configuration — see
        docs/known_limitations.md, item 7, and Section 6 of the paper.
    smoothing_window : int
        Moving-average window applied before threshold comparison.

    Returns
    -------
    int | None
        Index of the first alarm-triggering step, or None.
    """
    smoothed = smooth_trajectory(lambda_max_trajectory, window=smoothing_window)
    crossings = np.where(smoothed >= tau)[0]
    return int(crossings[0]) if len(crossings) > 0 else None


def lead_time(alarm_index: int | None, onset_index: int) -> int | None:
    """
    Computes lead time in tokens between alarm and labeled onset.
    Positive values mean the alarm fired before onset (a true early
    warning); non-positive or None values indicate a missed or late
    detection.
    """
    if alarm_index is None:
        return None
    return onset_index - alarm_index
