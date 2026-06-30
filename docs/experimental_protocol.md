# Experimental Protocol

This document expands on Table 1 and Remark 1 of the accompanying paper, for anyone reproducing the four pre-registered experiments independently.

## Hardware and software configuration

| Item | Value |
|---|---|
| Hardware | Apple M5 (MacBook Pro) |
| Unified memory | 32 GB |
| Operating system | macOS (Apple Silicon, native) |
| Inference runtimes | Ollama v0.30.7; MLX |
| Quantization | Q4_K_M (4-bit) |
| Models evaluated | 12, spanning four families (Llama, Qwen, Gemma, Phi/DeepSeek-derived) |
| Vocabulary size range | 100K–262K tokens |
| Logprob exposure | Top-K = 20 |
| Decoding mode | `think:false` for E1/E6 lead-time analysis |
| Stability gate (E7) | CV < 0.20, repeated runs per model |
| Intervention setting (E3) | Γ_κ, aggressive κ = p75 |

## On sampling configuration (why no single temperature/seed table is given)

Exact per-run sampling temperature, context window, random seeds, and the full prompt set are **deliberately varied** across the four experiments rather than held to a single fixed configuration, because each experiment answers a different question:

- **E7 (Stability)** requires *repeated runs under each model's default sampling settings* — the question being asked is "how much does λ_max vary run-to-run under realistic, non-deterministic decoding," so fixing temperature/seed would defeat the purpose.
- **E1 (Correlation) / E6 (Early Warning)** use a *fixed, larger prompt set* drawn from general factual and reasoning queries, applied consistently across all eligible models.
- **E3 (Intervention)** reuses the E6 alarm trajectories and applies the Γ_κ truncation operator at a fixed aggressiveness setting (κ = p75) for the headline result, with additional κ settings swept for the dose–response analysis mentioned in the paper.

Rather than publish placeholder or misleading single values for these parameters, the exact per-experiment configuration is tracked in the validation logs referenced by each CSV file in `data/` (see each file's accompanying notes). If you are attempting to reproduce a specific experiment, consult the corresponding data file first.

## Pre-registered experiments

| Code | Name | What it measures |
|---|---|---|
| E7 | Stability | Cross-run coefficient of variation (CV) of λ_max trajectories, gated at CV < 0.20, across all twelve models |
| E1 | Correlation | Pearson correlation between λ_max and a hallucination-onset label (H_mean — see `docs/known_limitations.md` for an important caveat), computed per model |
| E6 | Early Warning | Lead time, in tokens, between the first-passage-time (FPT) alarm and the labeled onset of hallucination, and the resulting detection rate |
| E3 | Intervention | Effect of the Γ_κ truncation operator on post-onset token count and on output quality (mean entropy, as a quality-preservation proxy) |

## Pipeline architecture

Five modules, executed in order:

1. **Logprob extraction** — queries the local Ollama server, extracts top-K=20 log-probabilities per decoding step.
2. **Fisher sensitivity estimation** — applies the closed-form λ_max approximation (paper Eq. 1) to each step's top-K distribution.
3. **FPT alarm computation** — smooths the λ_max trajectory and detects the first step at which it crosses a calibrated threshold τ.
4. **Γ_κ intervention** — truncates generation a fixed number of tokens after the alarm fires, parameterized by aggressiveness percentile κ.
5. **Orchestration** — runs the above across all twelve models for each of the four pre-registered experiments and aggregates results into the `data/` CSVs.

See `src/` for the corresponding module implementations.
