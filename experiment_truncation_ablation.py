"""Truncation ablation: does the summarisation failure come from the premise budget?

Runs the NLI detector on the summarisation task at several source-truncation
lengths. Thresholds are fitted on validation and applied to test, matching the
main experimental protocol.

Also reports how many tokens each setting actually produces, since the NLI model
has a fixed context window that bounds what any truncation length can deliver.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.data_loader import load_task
from src import scorers

TRUNC_LENGTHS = [400, 800, 1600, 3200]
TASK = "summarisation"

val_df, test_df = load_task(TASK)
val_y = val_df["label"].tolist()
test_y = test_df["label"].tolist()

# How long are the source documents, and what does the model actually see?
tok = scorers._get_nli().tokenizer
model_max = tok.model_max_length
src_chars = [len(s) for s in test_df["source"]]
print(f"Task: {TASK}")
print(f"Source length (chars): median={int(np.median(src_chars))} "
      f"mean={int(np.mean(src_chars))} p90={int(np.percentile(src_chars,90))}")
print(f"NLI model max tokens: {model_max}\n")

results = {"model_max_tokens": int(model_max),
           "source_chars": {"median": int(np.median(src_chars)),
                            "mean": int(np.mean(src_chars)),
                            "p90": int(np.percentile(src_chars, 90))},
           "settings": {}}

for L in TRUNC_LENGTHS:
    print(f"--- max_src = {L} chars ---", flush=True)

    # how many tokens does this truncation actually yield (before model cutoff)?
    sample_prem = [f"Document: {s[:L]}" for s in test_df["source"].head(200)]
    tok_lens = [len(tok(p)["input_ids"]) for p in sample_prem]
    truncated_frac = float(np.mean([t > model_max for t in tok_lens]))

    val_scores = scorers.nli_scores(val_df, max_src=L)
    test_scores = scorers.nli_scores(test_df, max_src=L)

    thr = scorers._best_threshold(val_y, val_scores)
    preds = scorers._predict(test_scores, thr)

    auc = roc_auc_score(test_y, test_scores)
    f1 = f1_score(test_y, preds)
    acc = accuracy_score(test_y, preds)

    results["settings"][str(L)] = {
        "auc_roc": round(auc, 4), "f1": round(f1, 4), "accuracy": round(acc, 4),
        "threshold": round(thr, 4),
        "median_premise_tokens": int(np.median(tok_lens)),
        "frac_hitting_model_limit": round(truncated_frac, 3),
    }
    print(f"  AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  "
          f"median_tokens={int(np.median(tok_lens))}  "
          f"hit_model_limit={truncated_frac:.1%}\n", flush=True)

with open("results/metrics/truncation_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved -> results/metrics/truncation_ablation.json")
