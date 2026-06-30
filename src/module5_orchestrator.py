"""
module5_orchestrator.py

Orchestrates modules 0-4 across the four pre-registered experiments
(E7 stability, E1 correlation, E6 early-warning, E3 intervention) and
writes results into the data/ CSV schema documented in
docs/data_dictionary.md.

This is a skeleton orchestrator illustrating the pipeline's control
flow. Populate the model list, prompt sets, and tau calibration value
from your own validation run configuration (see
docs/experimental_protocol.md) before running at scale.
"""

from __future__ import annotations
import csv
import numpy as np

from module0_ollama_client import stream_generate
from module1_logprob_parser import parse_trajectory
from module2_fisher_estimation import trajectory_lambda_max, shannon_entropy_normalized
from module3_gamma_intervention import truncation_offset, apply_intervention
from module4_fpt_alarm import fpt_alarm, lead_time


# The twelve models evaluated in the paper, spanning four families
# (Llama, Qwen, Gemma, Hunyuan/GPT-OSS-derived). Sourced directly from
# data/e7_stability_results.csv -- the canonical record of which
# models were actually evaluated.
MODELS: list[str] = [
    "qwen2.5-coder:7b",
    "qwen:latest",
    "gemma4:latest",
    "huihui_ai/hunyuan-mt-abliterated:latest",
    "qwen3:32b",
    "gemma3:27b",
    "llama3.1:latest",
    "deepseek-r1:latest",
    "phi4-mini:latest",
    "gemma3:latest",
    "llama3.2:latest",
    "coney_/gpt-oss_claude-sonnet4.6:latest",
]

# Calibrated per Section 6 of the paper. DO NOT reuse this value on
# different hardware/quantization without recalibration — see
# docs/known_limitations.md, item 7.
TAU_CALIBRATED = 15.9  # KL-divergence p95 threshold, Apple M5 / Q4_K_M


def run_stability_experiment(models: list[str], n_runs: int = 5,
                              out_path: str = "data/e7_stability_results.csv") -> None:
    """E7: cross-run coefficient of variation of lambda_max trajectories."""
    rows = []
    for model in models:
        # Run n_runs repeated generations under default sampling settings
        # (see docs/experimental_protocol.md — E7 intentionally does NOT
        # fix temperature/seed, since the question is run-to-run variance
        # under realistic, non-deterministic decoding).
        run_means = []
        for _ in range(n_runs):
            trajectory = list(stream_generate(model, prompt="<calibration prompt set>"))
            probs = parse_trajectory(trajectory)
            lam = trajectory_lambda_max(probs)
            run_means.append(float(np.mean(lam)))

        cv = float(np.std(run_means) / np.mean(run_means)) if np.mean(run_means) else float("nan")
        rows.append({
            "model_name": model,
            "model_family": "",  # fill from model registry
            "vocab_size": "",
            "n_runs": n_runs,
            "cv_lambda_max": cv,
            "passes_stability_gate": cv < 0.20,
        })

    _write_csv(out_path, rows)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if not MODELS:
        raise SystemExit(
            "MODELS list is empty — populate it from your validation run "
            "configuration before running the orchestrator. See "
            "docs/experimental_protocol.md."
        )
    run_stability_experiment(MODELS)
