"""Does sentence-level NLI aggregation rescue summarisation?

The single-pass NLI detector truncates the source to a fixed character budget, so
on summarisation it never sees most of the document. This experiment tests a
SummaC-ZS-style alternative that removes truncation entirely: the source is split
into chunks, the candidate into sentences, and every candidate sentence is scored
against every source chunk.

We deliberately reuse the same DeBERTa NLI model as the main experiments, so the
only thing that changes is the aggregation strategy, not the underlying model.

Aggregation (following SummaC-ZS): entailment is maxed over source chunks for each
candidate sentence, then combined across candidate sentences. We report both the
mean combiner (SummaC-ZS default) and the min combiner, which is more sensitive to
a single unsupported sentence.

To keep this CPU-feasible we run on a subset, and we re-run the single-pass
baseline on the *same* subset so the comparison is like-for-like.
"""
import json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.data_loader import load_task
from src import scorers

import os
# Per-split cap, to keep runtime practical on CPU. Set CHUNK_AGG_FULL=1 to run on
# the complete splits so the numbers are directly comparable to the main results.
N_SUBSET = 500
FULL = os.environ.get("CHUNK_AGG_FULL", "0") == "1"
CHUNK_CHARS = 800     # same per-call budget as the single-pass detector
MAX_CHUNKS = 10
MAX_CAND_SENTS = 6
BATCH = 32

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text, cap):
    sents = [s.strip() for s in SENT_SPLIT.split(str(text)) if s.strip()]
    return sents[:cap] if sents else [str(text)[:CHUNK_CHARS]]


def chunk_source(text, size=CHUNK_CHARS, cap=MAX_CHUNKS):
    text = str(text)
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return chunks[:cap] if chunks else [text]


def entailment_probs(pairs, nli):
    """Return P(entailment) for each (premise, hypothesis) pair."""
    from tqdm import tqdm
    out = []
    payload = [{"text": p, "text_pair": h} for p, h in pairs]
    for i in tqdm(range(0, len(payload), BATCH), desc="NLI-agg", leave=False):
        for res in nli(payload[i:i + BATCH]):
            d = {x["label"].lower(): x["score"] for x in res}
            out.append(d.get("entailment", 0.0))
    return out


def chunk_agg_scores(df, nli):
    """SummaC-ZS style. Returns (mean_combiner_scores, min_combiner_scores)."""
    pairs, index = [], []
    for row_i, (src, cand) in enumerate(zip(df["source"], df["candidate"])):
        chunks = chunk_source(src)
        sents = split_sentences(cand, MAX_CAND_SENTS)
        for s_i, sent in enumerate(sents):
            for ch in chunks:
                pairs.append((f"Document: {ch}", sent))
                index.append((row_i, s_i))

    probs = entailment_probs(pairs, nli)

    # max over source chunks, per (row, candidate sentence)
    per_sent = {}
    for (row_i, s_i), p in zip(index, probs):
        key = (row_i, s_i)
        per_sent[key] = max(per_sent.get(key, 0.0), p)

    by_row = {}
    for (row_i, s_i), p in per_sent.items():
        by_row.setdefault(row_i, []).append(p)

    mean_scores, min_scores = [], []
    for row_i in range(len(df)):
        vals = by_row.get(row_i, [0.0])
        mean_scores.append(1.0 - float(np.mean(vals)))   # hallucination score
        min_scores.append(1.0 - float(np.min(vals)))
    return mean_scores, min_scores


def evaluate(name, val_scores, test_scores, val_y, test_y, out):
    thr = scorers._best_threshold(val_y, val_scores)
    preds = scorers._predict(test_scores, thr)
    auc = roc_auc_score(test_y, test_scores)
    f1 = f1_score(test_y, preds)
    acc = accuracy_score(test_y, preds)
    out[name] = {"auc_roc": round(auc, 4), "f1": round(f1, 4),
                 "accuracy": round(acc, 4), "threshold": round(thr, 4)}
    print(f"  {name:<34} AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}", flush=True)


def main():
    val_df, test_df = load_task("summarisation")
    if not FULL:
        val_df = val_df.head(N_SUBSET).reset_index(drop=True)
        test_df = test_df.head(N_SUBSET).reset_index(drop=True)
    val_y, test_y = val_df["label"].tolist(), test_df["label"].tolist()

    print(f"Summarisation {'FULL splits' if FULL else 'subset'}: "
          f"val={len(val_df)} test={len(test_df)}")
    n_chunks = [len(chunk_source(s)) for s in test_df["source"]]
    n_sents = [len(split_sentences(c, MAX_CAND_SENTS)) for c in test_df["candidate"]]
    print(f"Median source chunks/sample: {int(np.median(n_chunks))}, "
          f"median candidate sentences: {int(np.median(n_sents))}\n")

    out = {"val_n": len(val_df), "test_n": len(test_df), "full_splits": FULL,
           "median_chunks": int(np.median(n_chunks)),
           "median_cand_sents": int(np.median(n_sents))}

    nli = scorers._get_nli()

    print("Single-pass NLI (800-char premise) on the same subset:", flush=True)
    evaluate("single_pass_nli_800",
             scorers.nli_scores(val_df, max_src=800),
             scorers.nli_scores(test_df, max_src=800),
             val_y, test_y, out)

    print("\nChunk aggregation (no truncation):", flush=True)
    v_mean, v_min = chunk_agg_scores(val_df, nli)
    t_mean, t_min = chunk_agg_scores(test_df, nli)
    evaluate("chunk_agg_mean", v_mean, t_mean, val_y, test_y, out)
    evaluate("chunk_agg_min", v_min, t_min, val_y, test_y, out)

    suffix = "_full" if FULL else ""
    path = f"results/metrics/chunk_aggregation{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
