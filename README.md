# How Far Can You Get Without a GPU? Lightweight Hallucination Detection Across QA, Dialogue, and Summarisation

[![arXiv](https://img.shields.io/badge/arXiv-2606.29809-b31b1b.svg)](https://arxiv.org/abs/2606.29809)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and results for a systematic benchmark of **five CPU-feasible hallucination
detection methods** across **all three tasks** of the
[HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) benchmark.

> **[How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight
> Hallucination Detection Across Question Answering, Dialogue, and
> Summarisation](https://arxiv.org/abs/2606.29809)**
> Kriti Faujdar, Smit Kadvani. arXiv:2606.29809, 2026.

## Overview

The most accurate hallucination detectors rely on GPUs, proprietary APIs, or
white-box access to the generating model. This project asks how far detection can
get using only **lightweight, CPU-feasible methods built on public models**, and
benchmarks five of them across question answering (QA), dialogue, and
summarisation.

**Headline finding: method ranking is task-dependent.** A similarity–NLI ensemble
wins on QA, NLI leads on dialogue, and *every* method collapses to near-random on
summarisation — mapping the practical frontier of GPU-free detection.

### Results (test n = 2,000 per task; F1 / AUC-ROC)

| Method | QA | Dialogue | Summarisation |
|---|---|---|---|
| ROUGE-L | 0.726 / 0.817 | 0.607 / 0.612 | 0.568 / 0.563 |
| Semantic Similarity | 0.675 / 0.736 | 0.544 / 0.656 | 0.627 / 0.528 |
| BERTScore | 0.733 / 0.821 | 0.638 / 0.637 | 0.658 / 0.469 |
| NLI | 0.725 / 0.795 | 0.679 / **0.713** | 0.609 / 0.567 |
| **Ensemble (Sim+NLI)** | **0.792 / 0.873** | **0.694 / 0.749** | 0.518 / 0.574 |

Full numbers: [`results/metrics/all_tasks.csv`](results/metrics/all_tasks.csv).

## Methods

1. **ROUGE-L** — lexical longest-common-subsequence overlap with the source.
2. **Semantic Similarity** — cosine similarity of [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings (~22M params).
3. **BERTScore** — token-level contextual F-measure with a DistilBERT backbone.
4. **NLI** — entailment scoring with [`DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (~184M params); score = 1 − P(entailment).
5. **Ensemble (Sim+NLI)** — `α·NLI + (1−α)·Sim`, with α grid-searched on validation.

All thresholds (and the ensemble weight α) are calibrated on a held-out
validation split and applied unchanged to the test split.

## Repository Structure

```
.
├── main.py                  # Entry point: runs all 5 methods × 3 tasks
├── src/
│   ├── data_loader.py       # Loads HaluEval (QA/dialogue/summarisation), val/test split
│   ├── scorers.py           # All five detection methods + unified runner
│   ├── evaluate.py          # Metrics, results table, error examples
│   └── visualize.py         # Heatmaps, ROC curves, confusion matrices, score dists
├── results/
│   ├── metrics/all_tasks.csv   # Per-task, per-method metrics
│   └── figures/                # Heatmaps, ROC, confusion matrices, score distributions
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

No GPU is required — all inference runs on CPU.

> **macOS note:** XGBoost-style native libs are not needed here, but if you add
> them, install OpenMP via `brew install libomp`.

## Usage

```bash
python main.py
```

This will, for each of the three HaluEval tasks:

1. Download the task from HuggingFace (first run only).
2. Draw 3,000 samples → 1,000 validation / 2,000 test (stratified, seed 42).
3. Run all five detection methods, calibrating thresholds on validation.
4. Save metrics to `results/metrics/all_tasks.csv` and figures to `results/figures/`.
5. Print per-task classification reports and NLI error examples.

## Key Findings

- **Task-dependence:** No single method is best across all tasks. The ensemble
  leads on QA, NLI on dialogue.
- **Lexical beats semantic in-task:** TF-IDF-style overlap (ROUGE-L, BERTScore)
  is competitive because HaluEval hallucinations carry a consistent lexical
  signature.
- **Summarisation is a systematic failure mode:** all methods fall to near-random
  (AUC-ROC ≤ 0.574) because subtle factual edits hide inside long, mostly faithful
  summaries that overlap metrics and a truncated NLI premise cannot localise.

## Citation

If you use this code or findings, please cite:

```bibtex
@article{faujdar2026howfar,
  title   = {How Far Can You Get Without a {GPU}? A Systematic Benchmark of
             Lightweight Hallucination Detection Across Question Answering,
             Dialogue, and Summarisation},
  author  = {Faujdar, Kriti and Kadvani, Smit},
  journal = {arXiv preprint arXiv:2606.29809},
  year    = {2026}
}
```

## Authors

- Kriti Faujdar — <kritifaujdar@gmail.com>
- Smit Kadvani — <smit.kadvani@gmail.com>

## License

Released under the [MIT License](LICENSE).
