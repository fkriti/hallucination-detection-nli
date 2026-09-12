import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)


def compute_metrics(labels, preds, scores, method_name, task_name):
    return {
        "Task":      task_name,
        "Method":    method_name,
        "Accuracy":  round(accuracy_score(labels, preds),                        4),
        "Precision": round(precision_score(labels, preds,  zero_division=0),     4),
        "Recall":    round(recall_score(labels, preds,     zero_division=0),     4),
        "F1":        round(f1_score(labels, preds,         zero_division=0),     4),
        "AUC-ROC":   round(roc_auc_score(labels, scores),                        4),
    }


def build_results_df(task_results):
    """task_results: {task: {method: scorer_output}}"""
    rows = []
    for task, method_results in task_results.items():
        df_test = method_results["_test_df"]
        labels  = df_test["label"].tolist()
        for method, r in method_results.items():
            if method.startswith("_"):
                continue
            rows.append(compute_metrics(
                labels, r["preds"], r["scores"], method, task
            ))
    return pd.DataFrame(rows)


def save_metrics(df, out_path):
    df.to_csv(out_path, index=False)
    print(f"Metrics saved → {out_path}")
    print(df.to_string(index=False))


def save_raw_scores(task_results, out_path):
    """Persist per-instance test scores, predictions and labels.

    Aggregate metrics alone cannot support confidence intervals, so we keep the
    raw per-instance output to allow bootstrap resampling after the fact without
    re-running any model.
    """
    import json
    payload = {}
    for task, method_results in task_results.items():
        labels = method_results["_test_df"]["label"].tolist()
        methods = {}
        for method, r in method_results.items():
            if method.startswith("_"):
                continue
            methods[method] = {
                "scores": [float(s) for s in r["scores"]],
                "preds":  [int(p) for p in r["preds"]],
            }
        payload[task] = {"labels": [int(l) for l in labels], "methods": methods}

    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Raw per-instance scores saved → {out_path}")


def print_classification_reports(task_results):
    for task, method_results in task_results.items():
        df_test = method_results["_test_df"]
        labels  = df_test["label"].tolist()
        print(f"\n{'='*60}\nTask: {task.upper()}\n{'='*60}")
        for method, r in method_results.items():
            if method.startswith("_"):
                continue
            print(f"\n  [{method}]")
            print(classification_report(
                labels, r["preds"],
                target_names=["Faithful", "Hallucinated"],
                zero_division=0,
            ))


def build_error_examples(task_results, n=3):
    """Return false-positive and false-negative examples per task."""
    examples = {}
    for task, method_results in task_results.items():
        df_test = method_results["_test_df"]
        # use NLI predictions for error analysis
        if "NLI" not in method_results:
            continue
        preds  = method_results["NLI"]["preds"]
        labels = df_test["label"].tolist()

        fp_idx = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == 1 and l == 0][:n]
        fn_idx = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == 0 and l == 1][:n]

        examples[task] = {
            "false_positives": df_test.iloc[fp_idx][["source", "candidate"]].to_dict("records"),
            "false_negatives": df_test.iloc[fn_idx][["source", "candidate"]].to_dict("records"),
        }
    return examples
