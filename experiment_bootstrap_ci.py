"""Bootstrap confidence intervals and pairwise significance tests.

Aggregate point estimates cannot show whether one method genuinely outperforms
another. We resample the test set with replacement and recompute each metric, which
gives confidence intervals without re-running any model.

We also bootstrap the paired difference between methods. If the interval for a
difference contains zero, the ranking between those two methods is not resolved by
our test set.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

N_BOOT = 2000
SEED = 42
RAW = "results/metrics/raw_scores.json"

rng = np.random.default_rng(SEED)


def boot_indices(n, b):
    return rng.integers(0, n, size=(b, n))


def metric_ci(labels, scores, preds, idx_matrix):
    labels = np.asarray(labels); scores = np.asarray(scores); preds = np.asarray(preds)
    aucs, f1s = [], []
    for idx in idx_matrix:
        y = labels[idx]
        if y.min() == y.max():        # degenerate resample, skip
            continue
        aucs.append(roc_auc_score(y, scores[idx]))
        f1s.append(f1_score(y, preds[idx], zero_division=0))
    return np.array(aucs), np.array(f1s)


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    with open(RAW) as f:
        data = json.load(f)

    rows, boot_store = [], {}

    for task, blob in data.items():
        labels = blob["labels"]
        idx_matrix = boot_indices(len(labels), N_BOOT)
        boot_store[task] = {}

        for method, m in blob["methods"].items():
            aucs, f1s = metric_ci(labels, m["scores"], m["preds"], idx_matrix)
            boot_store[task][method] = {"auc": aucs, "f1": f1s}
            a_lo, a_hi = ci(aucs)
            f_lo, f_hi = ci(f1s)
            rows.append({
                "Task": task, "Method": method,
                "AUC-ROC": round(float(np.mean(aucs)), 4),
                "AUC 95% CI": f"[{a_lo:.3f}, {a_hi:.3f}]",
                "F1": round(float(np.mean(f1s)), 4),
                "F1 95% CI": f"[{f_lo:.3f}, {f_hi:.3f}]",
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/metrics/bootstrap_ci.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved -> results/metrics/bootstrap_ci.csv\n")

    # Pairwise: is the top method significantly better than the runner-up?
    print("Paired bootstrap: best vs runner-up by AUC-ROC (per task)")
    sig_rows = []
    for task, methods in boot_store.items():
        means = {m: float(np.mean(v["auc"])) for m, v in methods.items()}
        ranked = sorted(means, key=means.get, reverse=True)
        best, second = ranked[0], ranked[1]
        diff = methods[best]["auc"] - methods[second]["auc"]
        d_lo, d_hi = ci(diff)
        resolved = "yes" if d_lo > 0 else "no (CI spans 0)"
        sig_rows.append({
            "Task": task, "Best": best, "Runner-up": second,
            "ΔAUC": round(float(np.mean(diff)), 4),
            "95% CI": f"[{d_lo:.3f}, {d_hi:.3f}]",
            "Separated": resolved,
        })
        print(f"  {task:<14} {best} > {second}: "
              f"Δ={np.mean(diff):.4f} CI=[{d_lo:.3f}, {d_hi:.3f}] -> {resolved}")

    pd.DataFrame(sig_rows).to_csv("results/metrics/bootstrap_pairwise.csv", index=False)
    print("\nSaved -> results/metrics/bootstrap_pairwise.csv")


if __name__ == "__main__":
    main()
