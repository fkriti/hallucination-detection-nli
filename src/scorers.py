"""
All five hallucination detection scorers in one module.
Each scorer returns (scores, predictions) after fitting threshold on val_df.
"""
import numpy as np
from sklearn.metrics import roc_curve

# ── shared threshold helper ───────────────────────────────────────────────────

def _best_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


def _predict(scores, threshold):
    return [1 if s >= threshold else 0 for s in scores]


# ── 1. ROUGE-L ────────────────────────────────────────────────────────────────

def rouge_l_scores(df):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for src, cand in zip(df["source"], df["candidate"]):
        score = scorer.score(src, cand)["rougeL"].fmeasure
        scores.append(float(score))
    return scores


# ── 2. Semantic Similarity ────────────────────────────────────────────────────

_sbert_model = None

def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def similarity_scores(df):
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    model = _get_sbert()
    src_emb  = model.encode(df["source"].tolist(),    batch_size=32, show_progress_bar=False)
    cand_emb = model.encode(df["candidate"].tolist(), batch_size=32, show_progress_bar=False)
    scores = [float(cos_sim([s], [c])[0][0]) for s, c in zip(src_emb, cand_emb)]
    return scores


# ── 3. BERTScore ─────────────────────────────────────────────────────────────

def bertscore_scores(df, model_type="distilbert-base-uncased", batch_size=32):
    from bert_score import score as bs_score
    MAX_CHARS = 512
    refs  = [s[:MAX_CHARS] for s in df["source"].tolist()]
    cands = df["candidate"].tolist()
    _, _, F = bs_score(
        cands, refs,
        model_type=model_type,
        batch_size=batch_size,
        device="cpu",
        verbose=False,
    )
    # In HaluEval, hallucinated answers are topically plausible, so they
    # exhibit HIGHER semantic overlap with the source — same pattern as
    # semantic similarity. Use BERTScore-F directly as the hallucination score.
    return F.numpy().tolist()


# ── 4. NLI ───────────────────────────────────────────────────────────────────

_nli_pipeline = None

def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=-1, top_k=None,
        )
    return _nli_pipeline


def nli_scores(df, batch_size=16, max_src=800):
    """Score with the NLI detector.

    max_src caps how many source characters enter the premise. The default of 800
    is the setting used for the main results; the truncation ablation varies it.
    """
    from tqdm import tqdm
    nli = _get_nli()
    MAX_SRC  = max_src
    MAX_CTX  = 200

    premises = []
    for src, ctx in zip(df["source"], df["context"]):
        if ctx and str(ctx).strip():
            premises.append(f"Document: {src[:MAX_SRC]} Context: {str(ctx)[-MAX_CTX:]}")
        else:
            premises.append(f"Document: {src[:MAX_SRC]}")

    hypotheses = df["candidate"].tolist()
    pairs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]

    raw = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="NLI", leave=False):
        raw.extend(nli(pairs[i: i + batch_size]))

    scores = []
    for output in raw:
        label_scores = {item["label"].lower(): item["score"] for item in output}
        entailment   = label_scores.get("entailment", 0.0)
        scores.append(1.0 - entailment)
    return scores


# ── 5. Ensemble (Sim + NLI) ───────────────────────────────────────────────────

def ensemble_scores(sim_scores, nli_scores_list, alpha=0.5):
    """alpha * nli + (1-alpha) * sim"""
    return [alpha * n + (1.0 - alpha) * s
            for s, n in zip(sim_scores, nli_scores_list)]


def find_best_alpha(val_labels, val_sim, val_nli):
    """Grid-search alpha ∈ {0.1, 0.2, …, 0.9} to maximise val AUC-ROC."""
    from sklearn.metrics import roc_auc_score
    best_alpha, best_auc = 0.5, -1
    for alpha in np.arange(0.1, 1.0, 0.1):
        scores = ensemble_scores(val_sim, val_nli, alpha)
        auc = roc_auc_score(val_labels, scores)
        if auc > best_auc:
            best_auc, best_alpha = auc, alpha
    return round(best_alpha, 1)


# ── Unified runner ────────────────────────────────────────────────────────────

SCORER_NAMES = [
    "ROUGE-L",
    "Sem. Similarity",
    "BERTScore",
    "NLI",
    "Ensemble (Sim+NLI)",
]


def run_all_scorers(val_df, test_df, verbose=True):
    """
    Returns dict: {scorer_name: {"scores": [...], "preds": [...], "threshold": float}}
    Threshold is selected on val_df and applied to test_df.
    """
    results = {}
    val_labels  = val_df["label"].tolist()

    def _fit_and_score(name, val_scores, test_scores):
        threshold  = _best_threshold(val_labels, val_scores)
        test_preds = _predict(test_scores, threshold)
        results[name] = {
            "scores":    test_scores,
            "preds":     test_preds,
            "threshold": threshold,
        }
        if verbose:
            from sklearn.metrics import f1_score, roc_auc_score
            f1  = f1_score(test_df["label"], test_preds)
            auc = roc_auc_score(test_df["label"], test_scores)
            print(f"    {name:<22} F1={f1:.4f}  AUC={auc:.4f}  τ={threshold:.4f}")

    # 1. ROUGE-L
    if verbose: print("  → ROUGE-L...", flush=True)
    _fit_and_score("ROUGE-L", rouge_l_scores(val_df), rouge_l_scores(test_df))

    # 2. Semantic Similarity
    if verbose: print("  → Semantic Similarity...", flush=True)
    val_sim  = similarity_scores(val_df)
    test_sim = similarity_scores(test_df)
    _fit_and_score("Sem. Similarity", val_sim, test_sim)

    # 3. BERTScore
    if verbose: print("  → BERTScore...", flush=True)
    _fit_and_score("BERTScore", bertscore_scores(val_df), bertscore_scores(test_df))

    # 4. NLI
    if verbose: print("  → NLI...", flush=True)
    val_nli  = nli_scores(val_df)
    test_nli = nli_scores(test_df)
    _fit_and_score("NLI", val_nli, test_nli)

    # 5. Ensemble
    if verbose: print("  → Ensemble...", flush=True)
    alpha    = find_best_alpha(val_labels, val_sim, val_nli)
    val_ens  = ensemble_scores(val_sim, val_nli, alpha)
    test_ens = ensemble_scores(test_sim, test_nli, alpha)
    _fit_and_score(f"Ensemble (Sim+NLI)", val_ens, test_ens)
    results["Ensemble (Sim+NLI)"]["alpha"] = alpha
    if verbose:
        print(f"      → best α={alpha}")

    return results
