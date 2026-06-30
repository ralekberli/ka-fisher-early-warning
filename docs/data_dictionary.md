# Data Dictionary

All CSV files live in `data/` and contain real, measured data from the validation runs described in the paper — none of these are placeholder or synthetic values. Each file is described below.

## `e7_stability_results.csv`

One row per model (12 rows), summarizing the cross-run stability gate (E7).

| Column | Type | Description |
|---|---|---|
| `model_name` | string | Ollama model identifier, e.g. `qwen3:32b`, `llama3.2:latest` |
| `model_family` | string | Coarse family grouping (Llama / Qwen / Gemma / Hunyuan / GPT-OSS-derived / Phi-DeepSeek-derived) |
| `vocab_size` | int | Left blank in this release; not recorded per-run in the source logs |
| `n_runs` | int | Left blank in this release; not recorded per-run in the source logs |
| `cv_lambda_max` | float | Coefficient of variation of lambda_max across repeated runs |
| `passes_stability_gate` | bool | `cv_lambda_max < 0.20` |

## `e1_correlation_results.csv`

One row per model (10 rows: 9 non-thinking models plus 1 thinking-mode model excluded with a null correlation), reporting the E1 correlation.

| Column | Type | Description |
|---|---|---|
| `model_name` | string | |
| `decoding_mode` | string | `think:false`, or `think:true (excluded)` for the one excluded thinking-mode model |
| `pearson_r` | float | Correlation between lambda_max and the entropy-based hallucination-onset label H_mean; blank for the excluded model |
| `p_value`, `n_samples` | float, int | Left blank in this release; not retained per-model in the source summary file |
| `oracle_circularity_flag` | bool | Always `true` -- see `docs/known_limitations.md` Section 5 before treating these as unconditional predictive-strength estimates |
| `note` | string | Free-text notes carried over from the source file (e.g. the qwen3:32b rerun note, or the thinking-mode exclusion reason) |

## `e6_leadtime_events.csv`

One row per run (500 rows: 12 models x ~40+ prompts each), reporting the early-warning alarm outcome for that run.

| Column | Type | Description |
|---|---|---|
| `model_name` | string | |
| `event_id` | string | Source `run_id`, e.g. `qwen2.5-coder:7b_p000` |
| `alarm_token_index` | int | Decoding step at which the FPT alarm fired, computed as `onset_token_index - lead_time_tokens`; blank if not detected. Can be negative when lead time exceeds the onset index -- this is a known artifact of the lead-time computation on short generations and is preserved as-is rather than clipped, for transparency. |
| `onset_token_index` | int | Decoding step labeled as hallucination onset; blank if not detected (`onset_t = -1` in source) |
| `lead_time_tokens` | int | Reported lead time for this run; `0` for non-detected runs |
| `detected` | bool | `true` if `onset_token_index != -1` in the source data |

The paper's headline 27-token median lead time is computed only over the subset of rows where `lead_time_tokens > 0` (i.e., excluding both non-detections and zero-margin detections) -- see Section 5.3 of the paper and `docs/known_limitations.md` for this filtering convention.

## `e3_intervention_results.csv`

One row per trial (1800 rows: 12 models x 4 conditions x ~37+ prompts), comparing intervened vs. non-intervened generation.

| Column | Type | Description |
|---|---|---|
| `model_name` | string | |
| `kappa_setting` | float | Aggressiveness percentile threshold: `0.75` (aggressive/p75), `0.90` (conservative/p90), `0.85` (replication/p85), or blank for `control` (no gating) |
| `condition` | string | One of `control`, `aggressive`, `conservative`, `replication` |
| `intervened` | bool | `condition != "control"` |
| `post_onset_token_count` | int | Tokens generated after onset under this condition |
| `mean_output_entropy` | float | Quality-preservation proxy (H_mean for this trial) |

The paper's headline 66% reduction and Cohen's d=1.95 figures compare the `aggressive` condition against `control`, restricted to trials with a detected onset; see Section 5.4 of the paper.

## `calibration_244x.csv`

Single-row summary of the pooled, multi-model calibration-drift finding (see Section 6 of the paper and `calibration_244x_recovered_summary.json` below for the full per-model breakdown).

| Column | Type | Description |
|---|---|---|
| `hardware` | string | `Apple M5 32GB` |
| `quantization` | string | `Q4_K_M` |
| `kl_threshold_theoretical` | float | tau_theory ~= 0.065 |
| `kl_threshold_measured_p95` | float | Pooled p95 across 8 models, ~= 8.44 |
| `drift_factor` | float | ~= 130x |
| `n_models` | int | `8` (2 additional models excluded as measurement artifacts) |
| `n_measurements` | int | `397` |

## `calibration_244x_recovered_summary.json`

Full per-model breakdown supporting `calibration_244x.csv`: pooled and per-model kl_p50/kl_p95/kl_p99, drift factors, the two excluded models with their exclusion reason, and the original (pre-recovery) single-model measurement that motivated this investigation. This is the authoritative source for Table 3 in the paper.

## `calibration_kl_raw_recovered.csv`

The raw, per-sample KL-divergence measurements underlying the calibration finding above: 437 rows (one per individual measurement), spanning 10 models including the 2 excluded as artifacts.

| Column | Type | Description |
|---|---|---|
| `source_file` | string | Originating validation-log filename, for full traceability back to the raw measurement session |
| `model_name` | string | |
| `kl_divergence` | float | A single bigram KL-divergence measurement |
| `excluded_artifact` | bool | `true` for the 2 models (deepseek-r1, gemma4) excluded from the pooled statistic in the paper |
| `timestamp` | string | ISO timestamp of the measurement batch this value came from |
