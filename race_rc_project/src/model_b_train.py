"""
model_b_train.py -- Model B: Distractor & Hint Generator Training
=================================================================
ALL training runs on GPU (NVIDIA RTX 5070 Ti):
  - Sentence-BERT (all-MiniLM-L6-v2) loaded on CUDA for hint generation
  - XGBoost GPU for distractor ranking
  - PyTorch cosine similarity on CUDA tensors

Implements:
  - Distractor candidate extraction (TF-IDF + cosine similarity)
  - ML-based distractor ranking (XGBoost GPU)
  - Graduated hint generation (SBERT on GPU with TF-IDF fallback)
"""

import os
import sys

# Ensure dotasim environment is active
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from check_pkgs import verify_environment
    verify_environment()
except ImportError:
    print("[WARN] Environment check skipped (check_pkgs.py not found).")

import time
import re
import string
import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_race_data, clean_text, load_artifact, save_artifact

# ---- GPU Setup (Strict) ----
if not torch.cuda.is_available():
    print("FATAL ERROR: CUDA is NOT available! This project is strictly optimized for GPU execution on RTX 5070 Ti.")
    print("Please check your drivers and CUDA installation. Exiting to prevent CPU fallback.")
    sys.exit(1)

DEVICE = torch.device("cuda")
print(f"[GPU] ACTIVE: {torch.cuda.get_device_name(0)}")
print("[GPU] Enforcement: SBERT and XGBoost will run exclusively on CUDA device 0.")

# ---- Optional: Sentence-Transformers (GPU) ----
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("[WARN] sentence-transformers not installed. Using TF-IDF only for hints.")

# ---- Optional: XGBoost (GPU) ----
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed. Using fallback for distractor ranking.")


# =============================================================
#  1. Sentence Splitting
# =============================================================

def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# =============================================================
#  2. Distractor Candidate Extraction
# =============================================================

def extract_distractor_candidates(
    article: str,
    question: str,
    correct_answer: str,
    all_options: list[str],
    vectorizer: TfidfVectorizer,
) -> list[dict]:
    """
    Extract candidate distractor phrases from the article.
    Scored by TF-IDF cosine similarity to the correct answer.
    """
    sentences = split_sentences(article)
    correct_clean = clean_text(correct_answer)

    texts = [correct_clean] + [clean_text(s) for s in sentences]
    try:
        tfidf_matrix = vectorizer.transform(texts)
    except Exception:
        return []

    answer_vec = tfidf_matrix[0:1]
    sent_vecs = tfidf_matrix[1:]
    sims = sklearn_cosine(answer_vec, sent_vecs).flatten()

    candidates = []
    existing_options_clean = {clean_text(o) for o in all_options}

    for i, sent in enumerate(sentences):
        sent_clean = clean_text(sent)
        if sent_clean == correct_clean or sent_clean in existing_options_clean:
            continue
        words = sent_clean.split()
        if len(words) < 2:
            continue
        candidate_text = " ".join(words[:15]) if len(words) > 15 else sent_clean
        candidates.append({
            "text": candidate_text,
            "sim_to_answer": float(sims[i]),
            "sentence_idx": i,
        })

    # Sort by medium similarity (plausible but wrong)
    candidates.sort(key=lambda x: abs(x["sim_to_answer"] - 0.3))
    return candidates


# =============================================================
#  3. Distractor Ranking -- XGBoost GPU
# =============================================================

def build_distractor_training_data(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    max_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build labeled data for distractor ranking.
    Positive = actual distractors, Negative = correct answer.
    """
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    option_cols = ["A", "B", "C", "D"]
    all_features = []
    all_labels = []
    n = min(len(df), max_samples)

    for idx in range(n):
        row = df.iloc[idx]
        correct_idx = label_map[row["answer"]]
        correct_text = clean_text(row[option_cols[correct_idx]])
        article_clean = clean_text(row["article"])
        question_clean = clean_text(row["question"])

        texts = [article_clean, question_clean, correct_text]
        for opt_col in option_cols:
            texts.append(clean_text(row[opt_col]))

        try:
            tfidf_mat = vectorizer.transform(texts)
        except Exception:
            continue

        art_vec = tfidf_mat[0:1]
        q_vec = tfidf_mat[1:2]
        correct_vec = tfidf_mat[2:3]

        for opt_i, opt_col in enumerate(option_cols):
            opt_vec = tfidf_mat[3 + opt_i: 4 + opt_i]
            sim_to_answer = float(sklearn_cosine(opt_vec, correct_vec).flatten()[0])
            sim_to_question = float(sklearn_cosine(opt_vec, q_vec).flatten()[0])
            sim_to_article = float(sklearn_cosine(opt_vec, art_vec).flatten()[0])

            opt_text = clean_text(row[opt_col])
            opt_words = set(opt_text.split())
            ans_words = set(correct_text.split())
            word_overlap = len(opt_words & ans_words) / max(len(opt_words), 1)
            opt_len = len(opt_text.split())

            features = [sim_to_answer, sim_to_question, sim_to_article, word_overlap, opt_len]
            all_features.append(features)
            is_distractor = 1 if opt_i != correct_idx else 0
            all_labels.append(is_distractor)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"[INFO] Distractor data: X={X.shape} y={y.shape} distractor_rate={y.mean():.4f}")
    return X, y


def train_distractor_ranker_gpu(X_train, y_train, X_val, y_val):
    """Train distractor ranker using XGBoost on GPU."""
    print("\n" + "=" * 60)
    print("  [GPU] Training: Distractor Ranker (XGBoost CUDA)")
    print("=" * 60)
    start = time.time()

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    else:
        # Fallback: LogisticRegression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)

    elapsed = time.time() - start
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1}


# =============================================================
#  4. Hint Generation -- SBERT on GPU
# =============================================================

def generate_hints_tfidf(article, question, correct_answer, vectorizer, n_hints=3):
    """TF-IDF fallback for hint generation."""
    sentences = split_sentences(article)
    if not sentences:
        return []

    target = clean_text(question + " " + correct_answer)
    texts = [target] + [clean_text(s) for s in sentences]
    try:
        tfidf_mat = vectorizer.transform(texts)
    except Exception:
        return []

    target_vec = tfidf_mat[0:1]
    sent_vecs = tfidf_mat[1:]
    sims = sklearn_cosine(target_vec, sent_vecs).flatten()

    ranked = sorted(enumerate(sentences), key=lambda x: sims[x[0]], reverse=True)
    hints = []
    for rank, (sent_idx, sent_text) in enumerate(ranked[:n_hints]):
        hint_level = n_hints - rank
        hints.append({"level": hint_level, "text": sent_text, "similarity": float(sims[sent_idx])})
    hints.sort(key=lambda x: x["level"])
    return hints


def generate_hints_sbert(article, question, correct_answer, sbert_model, n_hints=3):
    """GPU-accelerated hint generation using Sentence-BERT."""
    sentences = split_sentences(article)
    if not sentences:
        return []

    target = question + " " + correct_answer
    all_texts = [target] + sentences

    # SBERT encode on GPU
    embeddings = sbert_model.encode(all_texts, convert_to_tensor=True, show_progress_bar=False)
    target_emb = embeddings[0:1]
    sent_embs = embeddings[1:]

    sims = torch.nn.functional.cosine_similarity(target_emb, sent_embs).cpu().numpy()

    ranked = sorted(enumerate(sentences), key=lambda x: sims[x[0]], reverse=True)
    hints = []
    for rank, (sent_idx, sent_text) in enumerate(ranked[:n_hints]):
        hint_level = n_hints - rank
        hints.append({"level": hint_level, "text": sent_text, "similarity": float(sims[sent_idx])})
    hints.sort(key=lambda x: x["level"])
    return hints


# =============================================================
#  5. Evaluate Distractor Quality
# =============================================================

def evaluate_distractors(df, vectorizer, ranker_model, n_samples=100):
    """Evaluate distractor generation quality."""
    print("\n" + "=" * 60)
    print(f"  Evaluating Distractor Quality on {n_samples} samples")
    print("=" * 60)

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    option_cols = ["A", "B", "C", "D"]
    correct_not_selected = 0
    total = 0

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        correct_idx = label_map[row["answer"]]
        correct_text = row[option_cols[correct_idx]]

        candidates = extract_distractor_candidates(
            row["article"], row["question"], correct_text,
            [row[c] for c in option_cols], vectorizer
        )
        if len(candidates) >= 3:
            for c in candidates[:3]:
                if clean_text(c["text"]) != clean_text(correct_text):
                    correct_not_selected += 1
                total += 1

    accuracy = correct_not_selected / max(total, 1)
    print(f"  Distractor accuracy (not selecting correct answer): {accuracy:.4f}")
    return accuracy


# =============================================================
#  6. Main Training Pipeline
# =============================================================

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "model_b")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data and vectorizer
    train_df, val_df, test_df = load_race_data(DATA_DIR)
    vectorizer = load_artifact(os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"))

    # ---- Distractor Ranker (XGBoost GPU) ----
    print("\n[PHASE] Building distractor training data...")
    X_train_d, y_train_d = build_distractor_training_data(train_df, vectorizer, max_samples=15000)
    X_val_d, y_val_d = build_distractor_training_data(val_df, vectorizer, max_samples=3000)

    ranker, ranker_res = train_distractor_ranker_gpu(X_train_d, y_train_d, X_val_d, y_val_d)
    save_artifact(ranker, os.path.join(MODEL_DIR, "distractor_ranker.pkl"))

    # ---- SBERT Hint Generation (GPU) ----
    if HAS_SBERT:
        print("\n[PHASE] Loading Sentence-BERT model on GPU...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        save_artifact(sbert, os.path.join(MODEL_DIR, "sbert_model.pkl"))

        row = val_df.iloc[0]
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_idx = label_map[row["answer"]]
        option_cols = ["A", "B", "C", "D"]
        hints = generate_hints_sbert(
            row["article"], row["question"],
            row[option_cols[correct_idx]], sbert
        )
        print("\n[DEMO] SBERT Hints for sample 0:")
        for h in hints:
            print(f"  Hint {h['level']} (sim={h['similarity']:.3f}): {h['text'][:100]}...")
    else:
        print("\n[INFO] SBERT not available; using TF-IDF fallback for hints.")
        row = val_df.iloc[0]
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_idx = label_map[row["answer"]]
        option_cols = ["A", "B", "C", "D"]
        hints = generate_hints_tfidf(
            row["article"], row["question"],
            row[option_cols[correct_idx]], vectorizer
        )
        print("\n[DEMO] TF-IDF Hints for sample 0:")
        for h in hints:
            print(f"  Hint {h['level']} (sim={h['similarity']:.3f}): {h['text'][:100]}...")

    # ---- Evaluate Distractors ----
    evaluate_distractors(val_df, vectorizer, ranker, n_samples=200)

    # ---- Save Results ----
    pd.DataFrame([{
        "Model": "Distractor_Ranker_GPU",
        "Accuracy": ranker_res.get("accuracy", 0),
        "Macro_F1": ranker_res.get("f1_macro", 0),
        "Precision": ranker_res.get("precision", 0),
        "Recall": ranker_res.get("recall", 0),
    }]).to_csv(os.path.join(MODEL_DIR, "model_b_results.csv"), index=False)
    print(f"\nDONE: Model B training complete. Models saved to: {MODEL_DIR}")
