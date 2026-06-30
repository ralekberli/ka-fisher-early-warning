"""
module0_ollama_client.py

Thin client for extracting top-K log-probabilities from a local Ollama
server during streaming generation. This is the entry point of the
five-module pipeline: all downstream Fisher-sensitivity estimation
depends on the per-token logprob payload produced here.

Tested against Ollama v0.30.7. Ollama's streaming API returns logprobs
as a flat list per token (not nested per-candidate dicts) as of this
version — the parsing below assumes that flat-list format. If you are
running a different Ollama version, verify the response shape before
trusting downstream results.
"""

from __future__ import annotations
import json
import requests
from dataclasses import dataclass
from typing import Iterator


OLLAMA_HOST = "http://localhost:11434"
TOP_K = 20  # matches the top-K constraint used throughout the paper


@dataclass
class TokenLogprobs:
    token: str
    top_logprobs: list[float]  # length <= TOP_K, sorted descending


def stream_generate(model: str, prompt: str, *, temperature: float = 0.7,
                     seed: int | None = None) -> Iterator[TokenLogprobs]:
    """
    Streams a generation from a local Ollama model and yields per-token
    top-K log-probabilities.

    NOTE: exact sampling configuration (temperature, seed) should match
    the experiment you are reproducing — see docs/experimental_protocol.md.
    This function exposes them as parameters rather than hardcoding a
    single value, since E7 vs. E1/E6 intentionally use different
    sampling regimes.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            **({"seed": seed} if seed is not None else {}),
            "top_k": TOP_K,
        },
        "logprobs": True,
    }

    with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            # Flat-list logprob format (Ollama v0.30.7+). Adjust here if
            # your Ollama version nests candidates differently.
            logprobs = chunk.get("logprobs", [])
            if logprobs:
                yield TokenLogprobs(
                    token=chunk.get("response", ""),
                    top_logprobs=[lp for lp in logprobs[:TOP_K]],
                )
            if chunk.get("done"):
                break
