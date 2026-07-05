# K–A Fisher Early-Warning: Validation Code & Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21134196.svg)](https://doi.org/10.5281/zenodo.21134196)
[![Preprint](https://img.shields.io/badge/preprint-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.21133067)
[![arXiv](https://img.shields.io/badge/arXiv-2606.28432-b31b1b.svg)](https://arxiv.org/abs/2606.28432)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC--BY--4.0-lightgrey.svg)](LICENSE-DATA)

Code, validation logs, and per-model trajectory data supporting:

> R. Z. Alekberli and H. Karimov, "Runtime Fisher Spectral Sensitivity for Early Hallucination Detection in Edge-Deployed Language Models," preprint, Zenodo, 2026. https://doi.org/10.5281/zenodo.21133067

A companion theoretical paper by the same authors is available on arXiv: R. Z. Alekberli and H. Karimov, "Spectral Perturbation of the Empirical Fisher Information Matrix under Weight Quantization," [arXiv:2606.28432](https://arxiv.org/abs/2606.28432), 2026.

This repository is the companion data/code release referenced in the empirical paper's Reproducibility section. It implements a model-agnostic, top-K-constrained approximation to the spectral sensitivity of the per-token empirical Fisher Information Matrix, $\lambda_{\max}(\hat F_t)$, used as a runtime early-warning signal for hallucination during autoregressive decoding on consumer edge hardware (tested on Apple M5 Silicon via [Ollama](https://ollama.com) and [MLX](https://github.com/ml-explore/mlx)).

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
| Calibration drift | 244× gap between theoretical and measured KL-divergence threshold under 4-bit quantization (llama3.2, corroborated by a 237× estimate across 120 repeated measurements on the same model) | see `data/calibration_244x.csv`; multi-model robustness check across 8 models in `data/calibration_244x_recovered_summary.json` |

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
- All measurements come from a single hardware platform (Apple M5, 32GB, Q4_K_M quantization). The reported 244× calibration gap (llama3.2) is corroborated by 120 repeated measurements on the same model (237×) and falls within a 64×–287× range observed across a broader 8-model robustness check — see `docs/known_limitations.md` for the full breakdown, including two models excluded due to a measurement artifact.
- The empirical paper is currently available as a preprint (Zenodo DOI above); peer-reviewed submission is planned but has not yet occurred.

We would rather this repository under-claim than over-claim. Issues and replications on other hardware are very welcome.

## Citation

If you use this code or data, please cite both the paper and this repository:

```bibtex
@misc{alekberli2026runtime_paper,
  title     = {Runtime Fisher Spectral Sensitivity for Early Hallucination Detection in Edge-Deployed Language Models},
  author    = {Alekberli, Rahid Zahid and Karimov, Hikmat},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21133067},
  note      = {Preprint}
}

@misc{alekberli2026runtime_code,
  title     = {K–A Fisher Early-Warning: Validation Code \& Data},
  author    = {Alekberli, Rahid Zahid and Karimov, Hikmat},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21134196},
  note      = {Code and data release, v1.0.3}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

Code in `src/` is released under the [MIT License](LICENSE). Data in `data/` and documentation in `docs/` are released under [CC BY 4.0](LICENSE-DATA).

## Contact

Rahid Zahid Alekberli and Hikmat Karimov — Institute of Defense Technologies and Cybersecurity Research, Azerbaijan Technical University, Baku.
