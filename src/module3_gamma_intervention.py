"""
module3_gamma_intervention.py

The Gamma_kappa truncation operator: once the FPT alarm (module4)
fires, truncate generation a fixed number of tokens later, with the
cutoff parameterized by an aggressiveness percentile kappa (e.g.
kappa = p75 for the paper's "aggressive" setting used in the
headline E3 result).

E3 finding: triggering Gamma_kappa at kappa=p75 reduces post-onset
token generation by 66% (Cohen's d = 1.95) relative to an
unintervened baseline, with mean output entropy statistically
unchanged — i.e. quality-preserving, not quality-destructive.

Note: this operator only truncates; it does not correct hallucinated
content into a verified-accurate answer. A separate, negative result
(resampling-based correction did not reduce hallucination rate) is
documented in docs/known_limitations.md, item 6, and is NOT
implemented here.
"""

from __future__ import annotations
import numpy as np


def truncation_offset(post_alarm_lengths: np.ndarray, kappa: float) -> int:
    """
    Computes the truncation offset (in tokens, after the alarm fires)
    corresponding to aggressiveness percentile kappa, based on a
    reference distribution of post-alarm generation lengths.

    Parameters
    ----------
    post_alarm_lengths : np.ndarray
        Reference distribution of token counts generated after the
        alarm fires, from a calibration run.
    kappa : float
        Aggressiveness percentile in [0, 1]. kappa=0.75 ("p75") is the
        aggressive setting used for the paper's headline E3 result;
        lower kappa values were swept for the dose-response analysis.

    Returns
    -------
    int
        Number of tokens to allow after the alarm fires before cutting
        generation off.
    """
    return int(np.percentile(post_alarm_lengths, kappa * 100))


def apply_intervention(generated_tokens: list[str], alarm_index: int | None,
                        offset: int) -> list[str]:
    """
    Applies the Gamma_kappa truncation to a generated token sequence.

    If no alarm fired (alarm_index is None), the sequence is returned
    unmodified — there is nothing to intervene on.
    """
    if alarm_index is None:
        return generated_tokens
    cutoff = alarm_index + offset
    return generated_tokens[:cutoff]
