# K–A Fisher Early-Warning: Validation Code & Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)
[![arXiv](https://img.shields.io/badge/arXiv-2606.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2606.XXXXX)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-lightgrey.svg)](LICENSE-DATA)

Code, validation logs, and per-model trajectory data supporting:

> R. Z. Alekberli and H. Karimov, "Runtime Fisher Spectral Sensitivity for Early Hallucination Detection in Edge-Deployed Language Models," arXiv:2606.XXXXX, 2026.

This repository is the companion data/code release referenced in the paper's Reproducibility section. It implements a model-agnostic, top-K-constrained approximation to the spectral sensitivity of the per-token empirical Fisher Information Matrix, $\lambda_{\max}(\hat F_t)$, used as a runtime early-warning signal for hallucination during autoregressive decoding on consumer edge hardware (tested on Apple M5 Silicon via [Ollama](https://ollama.com) and [MLX](https://github.com/ml-explore/mlx)).

## What is in this repository

| Directory | Contents |
|---|---|
| `src/` | Five-module pipeline: logprob extraction → Fisher sensitivity estimation → FPT alarm → intervention → orchestration |
| `data/` | CSV outputs for the four pre-registered experiments (stability, correlation, early-warning, intervention) and the calibration-drift measurement |
| `docs/` | Experimental protocol, data dictionary, and known limitations (including the E1 oracle-circularity caveat — see below) |

## Headline results

| Experiment | Finding | Statistical support |
|---|---|---|
| Stability | 10/12 models pass CV < 0.20 stability gate | CV < 0.20 |
| Correlation | Strongest single-model correlation, qwen3:32b (non-thinking) | r = 0.962 (upper bound — see [Known Limitations](docs/known_limitations.md)) |
| Early Warning | 27-token median lead time before hallucination onset, 85.6% detection | t = 33.2, p = 2.2 × 10⁻¹¹⁴ |
| Intervention | 66% reduction in post-onset tokens, quality-preserving | Cohen's d = 1.95 |
| Calibration drift | Pooled 130× gap (range 64×–287× across 8 models, n=397 measurements) between theoretical and measured KL-divergence threshold under 4-bit quantization | see `data/calibration_244x.csv` and `data/calibration_244x_recovered_summary.json` |

## Quickstart

```bash
git clone https://github.com/ralekberli/ka-fisher-early-warning.git
cd ka-fisher-early-warning
pip install -r src/requirements.txt

# Compute the lambda_max approximation (Eq. 1) for a sample top-K distribution
python3 -c "
from src.module2_fisher_estimation import lambda_max_approx
print(lambda_max_approx([0.9, 0.05, 0.03, 0.02]))
"

# Inspect the real, measured experimental data directly
python3 -c "
import csv
with open('data/e6_leadtime_events.csv') as f:
    rows = list(csv.DictReader(f))
n_detected = sum(1 for r in rows if r['detected'] == 'True')
print(f'{len(rows)} runs loaded; {n_detected} detections')
"
```

To reproduce the full pipeline end-to-end against live models, see `src/module5_orchestrator.py` (requires a local Ollama installation, v0.30.7 used in the paper) and `src/calibration_kl_threshold.py` for the calibration-drift measurement specifically. Exact per-experiment sampling configuration (temperature, seeds, prompt sets) is documented in `docs/experimental_protocol.md` rather than hardcoded, since some parameters were deliberately varied across experiments -- see that file for the exact configuration of the experiment you intend to reproduce.

## A note on what this is and is not

This release accompanies an empirical paper that is intentionally transparent about its limitations. In particular:

- The E1 correlation figures use an entropy-based hallucination-onset label that shares functional dependence with the signal itself. We report this explicitly as an **oracle-circularity caveat** — see [`docs/known_limitations.md`](docs/known_limitations.md) — and treat the headline r = 0.962 as an upper bound, not a general estimate of predictive strength.
- A separate, negative result (resampling-based correction did not reduce hallucination rate) is included for completeness, not omitted.
- All measurements come from a single hardware platform (Apple M5, 32GB, Q4_K_M quantization). The pooled 130× calibration gap (range 64×–287× across 8 models) is reported as a demonstration that such drift can be large and is itself model-dependent, not as a universal constant. Two models (deepseek-r1, gemma4) were excluded from this analysis due to a measurement artifact — see `docs/known_limitations.md`.

We would rather this repository under-claim than over-claim. Issues and replications on other hardware are very welcome.

## Citation

If you use this code or data, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{alekberli2026runtime,
  title         = {Runtime Fisher Spectral Sensitivity for Early Hallucination Detection in Edge-Deployed Language Models},
  author        = {Alekberli, Rahid Zahid and Karimov, Hikmat},
  year          = {2026},
  eprint        = {2606.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

This repository itself is also archived with its own DOI via Zenodo (see badge above); please cite the Zenodo DOI if referring specifically to the code/data release rather than the paper.

## License

Code in `src/` is released under the [MIT License](LICENSE). Data in `data/` and documentation in `docs/` are released under [CC BY 4.0](LICENSE-DATA).

## Contact

Rahid Zahid Alekberli and Hikmat Karimov — Institute of Defense Technologies and Cybersecurity Research, Azerbaijan Technical University, Baku.
