"""
module2_fisher_estimation.py

Implements the closed-form, top-K-constrained approximation to the
spectral sensitivity of the per-token empirical Fisher Information
Matrix, lambda_max(F_hat_t), as defined in Eq. 1 of the paper:

    lambda_max(F_hat_t) ~= p_max * (1 - p_max) * (1 + 0.5 * H_norm)

where:
    p_max   = top-1 token probability
    H_norm  = Shannon entropy of the top-K distribution, normalized to [0, 1]

This is reported in the paper as a validated engineering approximation,
not a derived theorem — it has not been validated against full-vocabulary
ground truth at scale. See docs/known_limitations.md, item 1.
"""

from __future__ import annotations
import math
import numpy as np


def shannon_entropy_normalized(probs: list[float]) -> float:
    """
    Shannon entropy of a (possibly truncated, top-K) probability
    distribution, normalized to [0, 1] by dividing by log(K).
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log(probs))
    h_max = math.log(len(probs))
    return float(h / h_max) if h_max > 0 else 0.0


def lambda_max_approx(top_k_probs: list[float]) -> float:
    """
    Computes the Eq. 1 approximation to lambda_max(F_hat_t) from a
    top-K probability distribution (already converted from logprobs
    via softmax/exp upstream).

    Parameters
    ----------
    top_k_probs : list[float]
        Top-K token probabilities at a single decoding step, sorted
        descending. p_max is taken as top_k_probs[0].

    Returns
    -------
    float
        Approximate lambda_max(F_hat_t) for this decoding step.
    """
    if not top_k_probs:
        return 0.0

    p_max = top_k_probs[0]
    h_norm = shannon_entropy_normalized(top_k_probs)

    return p_max * (1 - p_max) * (1 + 0.5 * h_norm)


def trajectory_lambda_max(per_step_top_k_probs: list[list[float]]) -> np.ndarray:
    """
    Applies lambda_max_approx across an entire generation trajectory.

    Parameters
    ----------
    per_step_top_k_probs : list of list of float
        One top-K probability list per decoding step.

    Returns
    -------
    np.ndarray
        Array of lambda_max estimates, one per decoding step.
    """
    return np.array([lambda_max_approx(step) for step in per_step_top_k_probs])
