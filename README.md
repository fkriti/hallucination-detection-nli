# How Far Can You Get Without a GPU? Lightweight Hallucination Detection Across QA, Dialogue, and Summarisation

[![arXiv](https://img.shields.io/badge/arXiv-2606.29809-b31b1b.svg)](https://arxiv.org/abs/2606.29809)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and results for a systematic benchmark of **four CPU-feasible hallucination
detectors and a score-level ensemble** across **all three tasks** of the
[HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) benchmark.

> **[How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight
> Hallucination Detection Across Question Answering, Dialogue, and
> Summarisation](https://arxiv.org/abs/2606.29809)**
> Kriti Faujdar, Smit Kadvani. arXiv:2606.29809, 2026.

## Overview

The most accurate hallucination detectors rely on GPUs, proprietary APIs, or
white-box access to the generating model. This project asks how far detection can
get using only **lightweight, CPU-feasible methods built on public models**, and
benchmarks four detectors plus a score-level ensemble across question answering
(QA), dialogue, and summarisation.

**Headline findings.** The similarity–NLI ensemble is the most consistent method: it
ranks first on QA and on dialogue, where NLI is the strongest standalone detector.
On summarisation every method performs near chance in the single-pass setting, but
that is largely an artifact of source truncation: SummaC-style chunk aggregation
lifts summarisation AUC-ROC from 0.567 to 0.683, still on CPU with the same model,
at roughly 20× the NLI calls.

### Results (test n = 2,000 per task; F1 / AUC-ROC)

| Method | QA | Dialogue | Summarisation |
|---|---|---|---|
| ROUGE-L | 0.726 / 0.817 | 0.607 / 0.612 | 0.568 / 0.563 |
| Semantic Similarity | 0.675 / 0.736 | 0.544 / 0.656 | 0.627 / 0.528 |
| BERTScore | 0.733 / 0.821 | 0.638 / 0.637 | 0.658 / 0.469 |
| NLI | 0.725 / 0.795 | 0.679 / 0.713 | 0.609 / 0.567 |
| **Ensemble (Sim+NLI)** | **0.792 / 0.873** | **0.694 / 0.749** | 0.518 / 0.574 |

Full numbers: [`results/metrics/all_tasks.csv`](results/metrics/all_tasks.csv).

## Methods

1. **ROUGE-L** — lexical longest-common-subsequence overlap with the source.
2. **Semantic Similarity** — cosine similarity of [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings (~22M params).
3. **BERTScore** — token-level contextual F-measure with a DistilBERT backbone.
4. **NLI** — entailment scoring with [`DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) (~184M params); score = 1 − P(entailment).
5. **Ensemble (Sim+NLI)** — `α·NLI + (1−α)·Sim`, with α grid-searched on validation.

**Score orientation.** Higher scores always mean more likely hallucinated. For the
three overlap detectors (1–3) the raw score is used, *not* one minus it: HaluEval's
hallucinated candidates are generated to be topically plausible, so they overlap
more with the source. Orientation is fixed in advance; the threshold only sets the
operating point and cannot flip a score's direction.

**Input length limits.** The NLI premise is capped at 800 source characters,
BERTScore references at 512 characters, and `all-MiniLM-L6-v2` truncates at its
default 256 tokens; only ROUGE-L sees the full source.

All thresholds (and the ensemble weight α) are calibrated on a held-out
validation split and applied unchanged to the test split.

## Repository Structure

```
.
├── main.py                            # Entry point: 4 detectors + ensemble × 3 tasks
├── src/
│   ├── data_loader.py                 # Loads HaluEval (QA/dialogue/summarisation), val/test split
│   ├── scorers.py                     # Four detectors, the ensemble, and a unified runner
│   ├── evaluate.py                    # Metrics, results table, error examples, raw-score dump
│   └── visualize.py                   # Heatmaps, ROC curves, confusion matrices, score dists
├── experiment_truncation_ablation.py  # NLI premise budget sweep on summarisation
├── experiment_chunk_aggregation.py    # SummaC-style chunk aggregation vs single-pass
├── experiment_bootstrap_ci.py         # Bootstrap CIs + paired significance tests
├── measure_throughput.py              # Measured CPU throughput per method
├── results/
│   ├── metrics/                       # all_tasks.csv, ablation/aggregation/CI results, raw scores
│   └── figures/                       # Heatmaps, ROC, confusion matrices, score distributions
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
3. Run the four detectors and the ensemble, calibrating thresholds on validation.
4. Save metrics to `results/metrics/all_tasks.csv` and figures to `results/figures/`.
5. Print per-task classification reports and NLI error examples.
6. Write per-instance scores to `results/metrics/raw_scores.json`.

### Follow-up experiments

Each script is standalone and writes to `results/metrics/`:

```bash
python experiment_truncation_ablation.py   # premise budget sweep (summarisation)
python experiment_chunk_aggregation.py     # chunk aggregation vs single-pass
python experiment_bootstrap_ci.py          # bootstrap CIs (needs raw_scores.json)
python measure_throughput.py               # measured CPU throughput
```

`experiment_chunk_aggregation.py` runs on a 500-instance subset by default; set
`CHUNK_AGG_FULL=1` to use the complete splits (several hours on CPU).
`experiment_bootstrap_ci.py` reads the saved per-instance scores, so it needs no
model inference and finishes in seconds.

## Key Findings

- **The ensemble is the most consistent method.** It ranks first on QA
  (AUC-ROC 0.873) and on dialogue (0.749); NLI is the strongest *standalone*
  method on dialogue (0.713). Paired bootstrap tests on AUC-ROC confirm both leads;
  the ranking on summarisation is **not** statistically resolved.
- **Contextual overlap beats sentence-embedding similarity on QA.** BERTScore
  (AUC-ROC 0.821) edges ROUGE-L (0.817), and both clearly beat semantic similarity
  (0.736).
- **The summarisation failure is largely an artifact of truncation.** Every method
  performs near chance in the single-pass setting (AUC-ROC ≤ 0.574; BERTScore's
  0.469 reflects the fixed score orientation, and reversing it gives only 0.532).
  The NLI premise sees ~23% of a median 3,458-character document, and inputs beyond
  the checkpoint's configured 512 positions degrade its judgments. Raising the
  budget to 1,600 characters gives 0.629, and SummaC-style chunk aggregation
  reaches **0.683** on CPU with the same model, at roughly 20× the NLI calls.
  Summarisation remains the hardest task, but the results do not support treating
  lightweight detection as intrinsically unsuited to it.

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
