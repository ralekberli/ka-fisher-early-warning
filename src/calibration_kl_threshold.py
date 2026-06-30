"""
calibration_kl_threshold.py

Re-run of the KL-divergence calibration experiment behind the paper's
"Calibration Necessity Protocol" (Section 6). The original run
(May 2026) produced KL p50=1.370, KL p95=15.904 against a theoretical
baseline of 0.065 (a ~244x gap) — but its raw per-prompt logs were not
retained. This script reproduces the same methodology with full
logging this time, so the result is independently verifiable.

METHODOLOGY
-----------
For each of N "stable factual" prompts (short, single-correct-answer
questions where the model's output distribution should be sharply
peaked and consistent across resamples), generate K independent
completions. For each completion, compute the bigram KL divergence —
KL(P_t || P_{t+1}) — between the top-K output distribution at each
pair of consecutive decoding steps. This measures how much the
model's predictive distribution shifts from one token to the next;
large shifts on "stable factual" prompts (where the model should be
confident and consistent) are evidence of instability that an
inference-time safety threshold needs to account for.

The resulting per-step KL values are pooled across all prompts and
samples, and the 50th/95th/99th percentiles are reported as
calibration thresholds — directly comparable to the original
tau_theory / tau_measured framing in the paper.

ENERGY MEASUREMENT (optional, macOS-only)
-------------------------------------------
If --measure-energy is passed, this script also samples battery
percentage via `pmset -g batt` before and after each prompt, matching
the original battery-drain methodology. This is low-precision
(integer percentage steps) — for higher precision, consider running
`sudo powermetrics --samplers battery -n 1` separately and merging
manually; this is left as a documented option rather than built in,
since `powermetrics` requires sudo and is more invasive to automate
safely.

OUTPUT
------
Writes two files to --out-dir (default: data/):
  - calibration_kl_raw.csv       one row per (prompt, sample, step pair)
  - calibration_kl_summary.json  aggregate p50/p95/p99 and drift_factor

Run this on the same hardware/quantization configuration reported in
the paper (Apple M5, 32GB, Q4_K_M) to produce a directly comparable,
fully-traceable replacement for the original (lost) calibration run.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import requests


OLLAMA_HOST = "http://localhost:11434"
TOP_K = 20

# Theoretical baseline being tested against. The original 0.065 value
# was set by visual inspection of a small sample (3 models) and is
# itself flagged in the paper as warranting scrutiny — see
# docs/known_limitations.md. Kept here as the comparison point for
# continuity with the original framing; treat the resulting "drift
# factor" as relative to THIS specific baseline, not as an absolute
# truth.
TAU_THEORY = 0.065

# 20 "stable factual" prompts: short, single-correct-answer questions
# where a well-calibrated model's output distribution should be
# sharply peaked and highly reproducible across resamples. This is a
# freshly constructed list (the original prompt set was not
# preserved) — documented here in full so it is reproducible going
# forward.
STABLE_FACTUAL_PROMPTS: list[str] = [
    "The capital of Germany is",
    "Water boils at a temperature of",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "The author of Romeo and Juliet is",
    "The capital of Japan is",
    "The freezing point of water in Celsius is",
    "The number of continents on Earth is",
    "The chemical formula for table salt is",
    "The capital of France is",
    "The currency used in the United States is",
    "The longest river in the world is",
    "The smallest prime number is",
    "The capital of Italy is",
    "The number of days in a leap year is",
    "The chemical symbol for oxygen is",
    "The capital of Egypt is",
    "The first president of the United States was",
    "The square root of 144 is",
]


@dataclass
class StepRecord:
    prompt_idx: int
    prompt: str
    sample_idx: int
    step_t: int
    kl_divergence: float


def get_battery_pct() -> int | None:
    """macOS-only: reads current battery percentage via `pmset -g batt`."""
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        for token in out.split():
            if token.endswith("%;") or token.endswith("%"):
                return int(token.strip("%;"))
    except Exception:
        return None
    return None


def _extract_logprob_value(entry) -> float:
    """
    Normalizes a single top-K logprob entry to a float log-probability.

    Different Ollama versions/configurations have been observed to
    return either a flat float, or a dict shaped like
    {"token": "...", "logprob": -0.123, ...}. This handles both so the
    script does not silently break on a version change.
    """
    if isinstance(entry, dict):
        for key in ("logprob", "logProb", "log_prob"):
            if key in entry:
                return float(entry[key])
        raise ValueError(f"Unrecognized logprob entry shape: {entry!r}")
    return float(entry)


def stream_top_k_distributions(model: str, prompt: str, *, temperature: float = 0.7,
                                seed: int | None = None, debug: bool = False) -> list[list[float]]:
    """
    Streams a generation from a local Ollama model and returns the
    top-K probability distribution (softmax-normalized) at each
    decoding step.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            **({"seed": seed} if seed is not None else {}),
        },
        "logprobs": True,
        "top_logprobs": TOP_K,
    }

    distributions: list[list[float]] = []
    debug_printed = False
    with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)

            if "error" in chunk:
                raise RuntimeError(
                    f"Ollama returned an error for prompt {prompt!r}: {chunk['error']}. "
                    "This is commonly a GPU/Metal out-of-memory error on large models — "
                    "run `ollama ps` and stop any other loaded models, or try a smaller "
                    "model, before retrying."
                )

            logprobs = chunk.get("logprobs", [])
            if debug and not debug_printed:
                print("DEBUG raw chunk keys:", list(chunk.keys()))
                print("DEBUG len(logprobs):", len(logprobs))
                print("DEBUG raw logprobs[0:3]:", logprobs[:3])
                if logprobs and isinstance(logprobs[0], dict) and "top_logprobs" in logprobs[0]:
                    print("DEBUG len(top_logprobs nested):", len(logprobs[0]["top_logprobs"]))
                debug_printed = True
            if logprobs:
                # Ollama nests the actual top-K alternatives one level
                # deeper: logprobs is a list with ONE entry per decoding
                # step, and THAT entry's "top_logprobs" key holds the
                # real top-K candidate list. Fall back to treating
                # logprobs itself as the candidate list if "top_logprobs"
                # is absent, for robustness against future API changes.
                step_entry = logprobs[0]
                if isinstance(step_entry, dict) and "top_logprobs" in step_entry:
                    candidates = step_entry["top_logprobs"]
                else:
                    candidates = logprobs
                values = [_extract_logprob_value(e) for e in candidates[:TOP_K]]
                raw = np.asarray(values, dtype=np.float64)
                probs = np.exp(raw - raw.max())
                probs = probs / probs.sum()
                distributions.append(sorted(probs.tolist(), reverse=True))
            if chunk.get("done"):
                break

    if not distributions:
        raise RuntimeError(
            f"No token distributions were returned for prompt {prompt!r} "
            "(empty response from Ollama, but no explicit error field either). "
            "Check `ollama serve` logs for the actual failure."
        )

    return distributions


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-12) -> float:
    """
    KL(P || Q) over two top-K distributions. If P and Q have different
    lengths (different candidate sets at each step), they are padded
    with eps to the longer length — an approximation noted explicitly
    here for transparency.
    """
    n = max(len(p), len(q))
    p_arr = np.array(p + [eps] * (n - len(p)))
    q_arr = np.array(q + [eps] * (n - len(q)))
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * np.log((p_arr + eps) / (q_arr + eps))))


def run_calibration(model: str, n_samples_per_prompt: int, out_dir: str,
                     measure_energy: bool, debug: bool = False) -> None:
    raw_rows: list[dict] = []
    energy_rows: list[dict] = []

    for prompt_idx, prompt in enumerate(STABLE_FACTUAL_PROMPTS):
        for sample_idx in range(n_samples_per_prompt):
            batt_before = get_battery_pct() if measure_energy else None
            t_start = time.time()

            distributions = stream_top_k_distributions(
                model, prompt, seed=sample_idx, debug=(debug and prompt_idx == 0 and sample_idx == 0)
            )

            t_end = time.time()
            batt_after = get_battery_pct() if measure_energy else None

            for t in range(len(distributions) - 1):
                kl = kl_divergence(distributions[t], distributions[t + 1])
                raw_rows.append(asdict(StepRecord(
                    prompt_idx=prompt_idx,
                    prompt=prompt,
                    sample_idx=sample_idx,
                    step_t=t,
                    kl_divergence=kl,
                )))

            if measure_energy:
                energy_rows.append({
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "battery_pct_before": batt_before,
                    "battery_pct_after": batt_after,
                    "duration_sec": t_end - t_start,
                })

    # Write raw per-step KL records
    raw_path = f"{out_dir}/calibration_kl_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    # Aggregate statistics
    kl_values = np.array([r["kl_divergence"] for r in raw_rows])
    kl_p50 = float(np.percentile(kl_values, 50))
    kl_p95 = float(np.percentile(kl_values, 95))
    kl_p99 = float(np.percentile(kl_values, 99))

    if kl_p99 == 0.0:
        print(
            "WARNING: all computed KL-divergence values are exactly 0. This "
            "almost always means each decoding step only returned a single "
            "candidate distribution (top-K alternatives were not returned by "
            "Ollama), making every softmax trivially [1.0] and every KL(P||P)=0. "
            "Re-run with --debug and confirm 'DEBUG len(logprobs)' is > 1 before "
            "trusting these results."
        )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "hardware": "Apple M5 (fill in if different)",
        "quantization": "Q4_K_M (fill in if different)",
        "n_prompts": len(STABLE_FACTUAL_PROMPTS),
        "n_samples_per_prompt": n_samples_per_prompt,
        "n_step_pairs_total": len(raw_rows),
        "kl_p50": kl_p50,
        "kl_p95": kl_p95,
        "kl_p99": kl_p99,
        "tau_theory": TAU_THEORY,
        "drift_factor_p95": kl_p95 / TAU_THEORY if TAU_THEORY else None,
        "note": (
            "tau_theory=0.065 is the ORIGINAL baseline value from the lost "
            "May 2026 run; it was set by visual inspection of a small "
            "model sample and is flagged in docs/known_limitations.md as "
            "warranting independent scrutiny. Treat drift_factor_p95 as "
            "relative to this specific baseline, not as an absolute, "
            "independently re-derived theoretical value."
        ),
    }

    summary_path = f"{out_dir}/calibration_kl_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if measure_energy and energy_rows:
        energy_path = f"{out_dir}/calibration_energy_raw.csv"
        with open(energy_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(energy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(energy_rows)
        print(f"Wrote energy log to {energy_path}")
        print(
            "NOTE: pmset-based battery sampling is integer-percentage "
            "precision only. For finer-grained energy measurement, run "
            "`sudo powermetrics --samplers battery -i 1000 -n <N>` "
            "separately during the same session and merge manually."
        )

    print(f"Wrote {len(raw_rows)} raw KL records to {raw_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"KL p50={kl_p50:.4f}  p95={kl_p95:.4f}  p99={kl_p99:.4f}")
    print(f"Drift factor (p95 / tau_theory={TAU_THEORY}): {kl_p95 / TAU_THEORY:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:32b", help="Ollama model name")
    parser.add_argument("--n-samples", type=int, default=3,
                         help="Samples per prompt (3 in the original methodology)")
    parser.add_argument("--out-dir", default="data", help="Output directory for CSV/JSON")
    parser.add_argument("--measure-energy", action="store_true",
                         help="Also log battery percentage before/after each prompt (macOS only)")
    parser.add_argument("--debug", action="store_true",
                         help="Print raw Ollama response shape for the first chunk (diagnostic)")
    args = parser.parse_args()

    run_calibration(args.model, args.n_samples, args.out_dir, args.measure_energy, args.debug)
