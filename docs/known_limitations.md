# Known Limitations

This document mirrors the Limitations section of the accompanying paper. It is kept here as a standalone reference for anyone using this code/data independently of the paper text.

## 1. Top-K logprob constraint

All λ_max estimates use the closed-form approximation

```
λ_max(F̂_t) ≈ p_max(1 - p_max)(1 + 0.5 * H_norm)
```

derived from top-K=20 log-probabilities rather than full-vocabulary access (most edge inference runtimes, including Ollama, do not expose full-vocabulary logits). This approximation has not been validated against a full-vocabulary ground truth at the scale reported in the paper.

## 2. Single hardware platform

All measurements, including the calibration-drift findings reported in `data/calibration_244x.csv` and `data/calibration_244x_recovered_summary.json`, were obtained on a single device class (Apple M5, 32GB unified memory, Q4_K_M 4-bit quantization). No claim is made that these factors generalize to other chips, memory configurations, or quantization schemes (e.g., 8-bit or 2-bit weights) without independent recalibration.

## 3. Thinking-mode exclusion

Models run in extended "thinking" mode were excluded from the correlation (E1) and lead-time (E6) analyses because their trajectory structure differs substantially from non-thinking decoding — extended, structured generations produced near-zero post-onset trajectory length, making lead-time measurement unreliable for that subset. This is a documented scope restriction, not an oversight.

## 4. Statistical coverage across models

Some models in the twelve-model pool contributed comparatively few hallucination-onset events, limiting the per-model statistical power of E1 and E6 for those models. The qwen3:32b E1 result in particular is the strongest single-model result in the pool, not a representative average — see `data/e1_correlation_results.csv` for the full per-model breakdown, including the cross-model median (r = 0.479).

## 5. Oracle circularity in E1 (important — read before citing the r = 0.962 figure)

The hallucination-onset label used in E1, `H_mean` (mean output token entropy over a response), is **not independent** of the λ_max signal itself: both are derived from the top-K output distribution, and λ_max's own approximation (see Section 1 above) includes an entropy term, `H_norm`. This shared functional dependence is expected to **inflate** the reported correlation relative to what an independent, non-entropy-based hallucination-correctness oracle would yield.

**Practical implication:** treat `r = 0.962` (qwen3:32b) as an *upper bound under the current oracle*, not as a general estimate of how well λ_max predicts hallucination against ground truth. The cross-model median, `r = 0.479`, is the more conservative and more defensible summary statistic for E1 as currently measured.

This is identified as a priority item for follow-up work: re-running E1 against an independent label (e.g., human-annotated factual correctness, or a separate model-graded correctness judge that does not share λ_max's input features) to obtain an unbiased estimate of the signal's predictive strength.

## 6. Negative result: resampling-based correction

In separate testing (not part of the four pre-registered experiments E1/E3/E6/E7), resampling-based correction triggered by the same early-warning alarm produced a **0% measured reduction** in hallucination rate. We interpret this as evidence that converting an early-warning signal into an actual quality improvement — rather than the truncation studied in E3 — likely requires a structured fallback path (retrieval, summarization, or explicit abstention) rather than naive retry. This remains open work and is not implemented in this repository.

## 7. Generalizability of the threshold drift

**Update:** the original single-model (llama3.2, n=20) calibration measurement reported in early drafts of this work (KL p95 ≈ 15.9, drift ≈ 244×) had its raw per-sample log lost before this repository was assembled, and could not be independently re-verified from that exact run. We have since recovered a substantially larger set of raw, fully-traceable calibration measurements from the same May 2026 validation campaign: **397 individual KL-divergence measurements across 8 models** (see `data/calibration_kl_raw_recovered.csv` and `data/calibration_244x_recovered_summary.json`).

Two models (`deepseek-r1:latest`, `gemma4:latest`) were excluded from this pooled analysis: all 20 measurements for each returned exactly 0.0, which we interpret as a response-parsing artifact in the original measurement script rather than a genuine zero-divergence finding, and is documented transparently here rather than silently dropped.

Across the remaining 8 models, the measured drift factor (p95 KL divergence relative to the τ_theory = 0.065 baseline) ranges from **64.3× to 287.4×**, with a pooled (n=397) estimate of **129.9×**. The `llama3.2:latest` model specifically (n=120 across repeated runs) shows a drift factor of 236.8×, closely matching the originally reported single-run figure of ~244× and lending continuity to that earlier observation.

We read this multi-model evidence as **strengthening, not weakening**, the qualitative claim: the finding that quantization-induced calibration drift can be very large is now supported by a larger, multi-model, fully reproducible sample, rather than a single lost measurement. The drift factor remains substantially model-dependent (a roughly 4.5× spread between the lowest- and highest-drift models observed), reinforcing the paper's core recommendation that any specific numeric threshold be empirically recalibrated per deployment rather than transferred from any single reported figure, including the pooled 129.9× value reported here.

---

We report these limitations explicitly, in the same spirit as the paper itself: we would rather this release under-claim than over-claim.
