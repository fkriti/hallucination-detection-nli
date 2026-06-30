from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Task specifications ───────────────────────────────────────────────────────
TASK_SPECS = {
    "qa": {
        "config":        "qa_samples",
        "source_col":    "knowledge",
        "context_col":   "question",
        "candidate_col": "answer",
    },
    "dialogue": {
        "config":        "dialogue_samples",
        "source_col":    "knowledge",
        "context_col":   "dialogue_history",
        "candidate_col": "response",
    },
    "summarisation": {
        "config":        "summarization_samples",
        "source_col":    "document",
        "context_col":   None,
        "candidate_col": "summary",
    },
}

TASK_DISPLAY = {
    "qa":            "QA",
    "dialogue":      "Dialogue",
    "summarisation": "Summarisation",
}

VAL_SIZE  = 1000   # for threshold / ensemble-weight selection
TEST_SIZE = 2000   # for final reported metrics


def load_task(task: str, seed: int = 42):
    """
    Load one HaluEval task and return (val_df, test_df).
    Each df has columns: source, context (may be None column), candidate, label.
    """
    spec = TASK_SPECS[task]
    ds   = load_dataset("pminervini/HaluEval", spec["config"])
    df   = ds["data"].to_pandas()

    df["source"]    = df[spec["source_col"]]
    df["context"]   = df[spec["context_col"]] if spec["context_col"] else ""
    df["candidate"] = df[spec["candidate_col"]]
    df["label"]     = (df["hallucination"] == "yes").astype(int)

    df = df[["source", "context", "candidate", "label"]].copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    total = VAL_SIZE + TEST_SIZE
    df = df.iloc[:total]

    val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed,
        stratify=df["label"]
    )
    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)
