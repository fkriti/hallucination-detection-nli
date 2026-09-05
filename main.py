import os, sys, warnings
warnings.filterwarnings("ignore")

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

from src.data_loader import TASK_DISPLAY, load_task
from src.scorers import run_all_scorers
from src.evaluate import (
    build_results_df, save_metrics, print_classification_reports,
    build_error_examples, save_raw_scores,
)
from src.visualize import (
    plot_metrics_heatmap, plot_roc_curves,
    plot_confusion_matrices, plot_score_distributions,
    plot_metrics_bar,
)

TASKS = list(TASK_DISPLAY.keys())


def main():
    # ── 1. Run all scorers on all tasks ──────────────────────────────────────
    task_results = {}

    for task in TASKS:
        print(f"\n{'='*55}")
        print(f"  Task: {TASK_DISPLAY[task]}  (val=1000  test=2000)")
        print(f"{'='*55}")

        val_df, test_df = load_task(task)
        print(f"  Label balance — val: {val_df['label'].value_counts().to_dict()}"
              f"  test: {test_df['label'].value_counts().to_dict()}")

        scorer_results = run_all_scorers(val_df, test_df, verbose=True)
        scorer_results["_test_df"] = test_df
        task_results[task] = scorer_results

    # ── 2. Metrics ───────────────────────────────────────────────────────────
    print("\nBuilding results table...")
    results_df = build_results_df(task_results)
    save_metrics(results_df, "results/metrics/all_tasks.csv")
    save_raw_scores(task_results, "results/metrics/raw_scores.json")
    print_classification_reports(task_results)

    # ── 3. Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_metrics_heatmap(results_df, "F1",      "results/figures")
    plot_metrics_heatmap(results_df, "AUC-ROC", "results/figures")
    plot_metrics_bar(results_df, "results/figures")
    plot_roc_curves(task_results, "results/figures")
    plot_confusion_matrices(task_results, "results/figures")
    plot_score_distributions(task_results, "results/figures")

    # ── 4. Error analysis ────────────────────────────────────────────────────
    print("\nError analysis (NLI false positives / negatives):")
    examples = build_error_examples(task_results, n=3)
    for task, ex in examples.items():
        print(f"\n  [{TASK_DISPLAY[task]}]")
        print("  False Positives (faithful flagged as hallucinated):")
        for e in ex["false_positives"]:
            cand = str(e["candidate"])[:120]
            print(f"    candidate: {cand}...")
        print("  False Negatives (hallucinated missed):")
        for e in ex["false_negatives"]:
            cand = str(e["candidate"])[:120]
            print(f"    candidate: {cand}...")

    print("\nDone. Results in results/")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
