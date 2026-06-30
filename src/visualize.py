import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix
from src.data_loader import TASK_DISPLAY
from src.scorers import SCORER_NAMES

TASK_NAMES = list(TASK_DISPLAY.keys())
COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(SCORER_NAMES)))


def plot_metrics_heatmap(results_df, metric, out_dir):
    pivot = results_df.pivot(index="Method", columns="Task", values=metric)
    pivot = pivot.reindex(index=SCORER_NAMES, columns=TASK_NAMES)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn",
        vmin=0.5, vmax=1.0, ax=ax,
        linewidths=0.5, cbar_kws={"label": metric},
        xticklabels=[TASK_DISPLAY[t] for t in TASK_NAMES],
    )
    ax.set_title(f"{metric} — All Methods × All Tasks")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    plt.tight_layout()
    safe = metric.replace("-", "_").replace(" ", "_")
    plt.savefig(f"{out_dir}/heatmap_{safe}.png", dpi=150)
    plt.close()
    print(f"Saved: heatmap_{safe}.png")


def plot_roc_curves(task_results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, TASK_NAMES):
        method_results = task_results[task]
        labels = method_results["_test_df"]["label"].tolist()
        for i, method in enumerate(SCORER_NAMES):
            if method not in method_results:
                continue
            scores = method_results[method]["scores"]
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{method} ({auc_val:.3f})", color=COLORS[i], lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(TASK_DISPLAY[task], fontsize=10)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=6, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.suptitle("ROC Curves (AUC-ROC) by Task", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=150)
    plt.close()
    print("Saved: roc_curves.png")


def plot_confusion_matrices(task_results, out_dir):
    n_methods = len(SCORER_NAMES)
    n_tasks   = len(TASK_NAMES)
    fig, axes = plt.subplots(n_methods, n_tasks, figsize=(3.5 * n_tasks, 3 * n_methods))

    for i, method in enumerate(SCORER_NAMES):
        for j, task in enumerate(TASK_NAMES):
            ax = axes[i][j]
            method_results = task_results[task]
            labels = method_results["_test_df"]["label"].tolist()
            if method not in method_results:
                ax.axis("off"); continue
            preds = method_results[method]["preds"]
            cm    = confusion_matrix(labels, preds, normalize="true")
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        xticklabels=["Faith.", "Hall."],
                        yticklabels=["Faith.", "Hall."],
                        cbar=False, annot_kws={"size": 7})
            if i == 0: ax.set_title(TASK_DISPLAY[task], fontsize=9)
            if j == 0: ax.set_ylabel(method[:16], fontsize=7)
            ax.tick_params(labelsize=7)

    plt.suptitle("Normalised Confusion Matrices", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_score_distributions(task_results, out_dir):
    methods_to_plot = ["Sem. Similarity", "NLI"]
    fig, axes = plt.subplots(len(methods_to_plot), 3,
                             figsize=(12, 4 * len(methods_to_plot)))

    for i, method in enumerate(methods_to_plot):
        for j, task in enumerate(TASK_NAMES):
            ax = axes[i][j]
            method_results = task_results[task]
            df  = method_results["_test_df"]
            if method not in method_results:
                ax.axis("off"); continue
            scores = method_results[method]["scores"]
            data = [
                [s for s, l in zip(scores, df["label"]) if l == 0],
                [s for s, l in zip(scores, df["label"]) if l == 1],
            ]
            ax.violinplot(data, positions=[0, 1], showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Faithful", "Hallucinated"], fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            if i == 0: ax.set_title(TASK_DISPLAY[task], fontsize=9)
            if j == 0: ax.text(-0.35, 0.5, method, transform=ax.transAxes,
                               va="center", ha="right", fontsize=8, rotation=90)

    plt.suptitle("Score Distributions by True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_distributions.png", dpi=150)
    plt.close()
    print("Saved: score_distributions.png")


def plot_metrics_bar(results_df, out_dir):
    for metric in ["F1", "AUC-ROC"]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, task in zip(axes, TASK_NAMES):
            sub = results_df[results_df["Task"] == task]
            sub = sub.set_index("Method").reindex(SCORER_NAMES)
            bars = ax.bar(range(len(SCORER_NAMES)), sub[metric],
                          color=COLORS, width=0.6)
            ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)
            ax.set_xticks(range(len(SCORER_NAMES)))
            ax.set_xticklabels(SCORER_NAMES, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0.5, 1.05)
            ax.set_title(TASK_DISPLAY[task], fontsize=10)
            if ax == axes[0]: ax.set_ylabel(metric)
        plt.suptitle(f"{metric} by Method and Task", fontsize=11)
        plt.tight_layout()
        safe = metric.replace("-", "_")
        plt.savefig(f"{out_dir}/bar_{safe}.png", dpi=150)
        plt.close()
        print(f"Saved: bar_{safe}.png")
