"""
module1_logprob_parser.py

Converts the raw log-probability output of module0 into normalized
top-K probability distributions ready for module2's Fisher sensitivity
estimation.
"""

from __future__ import annotations
import numpy as np
from module0_ollama_client import TokenLogprobs


def logprobs_to_probs(token_logprobs: TokenLogprobs) -> list[float]:
    """
    Converts a TokenLogprobs object's raw top-K log-probabilities into
    a normalized probability distribution over those K candidates.

    Note: this renormalizes over only the top-K candidates exposed by
    the runtime, not the full vocabulary — consistent with the paper's
    explicit top-K-constrained scope (see docs/known_limitations.md,
    item 1).
    """
    raw = np.asarray(token_logprobs.top_logprobs, dtype=np.float64)
    probs = np.exp(raw - raw.max())  # numerically stable softmax
    probs = probs / probs.sum()
    return sorted(probs.tolist(), reverse=True)


def parse_trajectory(trajectory: list[TokenLogprobs]) -> list[list[float]]:
    """Applies logprobs_to_probs across an entire generation trajectory."""
    return [logprobs_to_probs(step) for step in trajectory]
