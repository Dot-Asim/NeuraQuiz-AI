"""
preprocessing.py — Dataset Loading & Feature Engineering
=========================================================
Handles RACE dataset loading, cleaning, feature extraction (TF-IDF + One-Hot),
and train/val/test split management.

GPU-Accelerated where possible using PyTorch tensors for similarity computations.
"""

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import pickle
import string
import re
import os
import sys

# Ensure dotasim environment is active
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..")))
try:
    from check_pkgs import verify_environment

    verify_environment()
except ImportError:
    print("[WARN] Environment check skipped (check_pkgs.py not found).")


# ─────────────────────────────────────────────────────────────
# 0. GPU Initialization
# ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("WARNING: CUDA not found! Falling back to CPU.")


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[GPU] Preprocessing active on: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[CPU] Preprocessing active")

# ─────────────────────────────────────────────────────────────
# 1. Dataset Loading & Auto-Splitting
# ─────────────────────────────────────────────────────────────


def load_race_data(
        data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load RACE train/dev/test CSVs and perform basic cleaning."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "dev.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Check if files are identical
    if len(train_df) == len(val_df) == len(test_df):
        print(
            "[WARN] Detected identical file sizes for train/val/test. Performing manual split (80/10/10)..."
        )
        full_df = train_df.sample(
            frac=1, random_state=42).reset_index(
            drop=True)
        n = len(full_df)
        train_df = full_df.iloc[: int(n * 0.8)]
        val_df = full_df.iloc[int(n * 0.8): int(n * 0.9)]
        test_df = full_df.iloc[int(n * 0.9):]

    for df in [train_df, val_df, test_df]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    option_cols = ["A", "B", "C", "D"]
    for df in [train_df, val_df, test_df]:
        df[option_cols] = df[option_cols].fillna("")

    print(
        f"[INFO] Dataset split -> Train: {
            len(train_df)} | Val: {
            len(val_df)} | Test: {
                len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# 2. Text Cleaning
# ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# 3. Feature Engineering — TF-IDF Vectorization
# ─────────────────────────────────────────────────────────────


def build_tfidf_vectorizer(
    corpus: list[str],
    max_features: int = 15000,
    ngram_range: tuple = (1, 2),
) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on the given corpus."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.9,
    )
    vectorizer.fit(corpus)
    print(f"[INFO] TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def build_onehot_vectorizer(
    corpus: list[str],
    max_features: int = 15000,
) -> CountVectorizer:
    """Fit a CountVectorizer (One-Hot Encoding) on the given corpus."""
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        binary=True,  # This enforces One-Hot Encoding
        min_df=5,
        max_df=0.9,
    )
    vectorizer.fit(corpus)
    print(f"[INFO] One-Hot vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def get_sims_gpu(vecs1_sparse, vecs2_sparse, batch_size=10000):
    """
    Compute row-wise cosine similarity using GPU with batching to manage VRAM.
    """
    n = vecs1_sparse.shape[0]
    all_sims = []

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        v1 = torch.tensor(
            vecs1_sparse[i:end].toarray(), dtype=torch.float32).to(DEVICE)
        v2 = torch.tensor(
            vecs2_sparse[i:end].toarray(), dtype=torch.float32).to(DEVICE)

        v1_norm = torch.nn.functional.normalize(v1, p=2, dim=1)
        v2_norm = torch.nn.functional.normalize(v2, p=2, dim=1)

        sims = torch.sum(v1_norm * v2_norm, dim=1).cpu().numpy()
        all_sims.append(sims)

    return np.concatenate(all_sims)


# ─────────────────────────────────────────────────────────────
# 4. SBERT Semantic Embeddings (GPU)
# ─────────────────────────────────────────────────────────────


def _load_sbert():
    """Load Sentence-BERT on GPU for semantic similarity features.
    DISABLED: Professor explicitly banned Neural Networks.
    """
    print("[WARN] Neural Networks banned by instructor. Skipping SBERT features.")
    return None


def _sbert_cosine_gpu(emb1, emb2):
    """Row-wise cosine similarity between two embedding arrays on GPU."""
    v1 = torch.tensor(emb1, dtype=torch.float32).to(DEVICE)
    v2 = torch.tensor(emb2, dtype=torch.float32).to(DEVICE)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=1)
    v2 = torch.nn.functional.normalize(v2, p=2, dim=1)
    sims = torch.sum(v1 * v2, dim=1).cpu().numpy()
    return sims


# ─────────────────────────────────────────────────────────────
# 5. Feature Engineering — Verification Features
# ─────────────────────────────────────────────────────────────


def build_verification_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    sbert_model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs feature matrix X and label vector y for Model A.

    Features (per option):
      1. tfidf_sim_article_option   — TF-IDF cosine sim (article vs option)
      2. tfidf_sim_question_option  — TF-IDF cosine sim (question vs option)
      3. tfidf_sim_article_question — TF-IDF cosine sim (article vs question)
      4. word_overlap               — fraction of option words in article
      5. option_length              — number of words in option
      6. question_length            — number of words in question
      7. sbert_sim_article_option   — SBERT semantic sim (article vs option) [NEW]
      8. sbert_sim_question_option  — SBERT semantic sim (question vs option) [NEW]
      9. sbert_sim_article_question — SBERT semantic sim (article vs question) [NEW]
    """
    option_cols = ["A", "B", "C", "D"]

    # --- TF-IDF features ---
    print("[INFO] Vectorizing articles, questions, and options (TF-IDF)...")
    articles_tfidf = vectorizer.transform(df["article"].apply(clean_text))
    questions_tfidf = vectorizer.transform(df["question"].apply(clean_text))

    opt_vecs = {}
    for col in option_cols:
        opt_vecs[col] = vectorizer.transform(df[col].apply(clean_text))

    print("[INFO] Computing TF-IDF similarity features on GPU...")
    sim_art_q_tfidf = get_sims_gpu(articles_tfidf, questions_tfidf)

    # --- SBERT features (if available) ---
    sbert_art_emb = None
    sbert_q_emb = None
    sbert_opt_embs = {}
    if sbert_model is not None:
        print("[GPU] Encoding articles with SBERT (this may take a few minutes)...")
        articles_text = df["article"].astype(str).tolist()
        questions_text = df["question"].astype(str).tolist()
        sbert_art_emb = sbert_model.encode(
            articles_text,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True)
        print("[GPU] Encoding questions with SBERT...")
        sbert_q_emb = sbert_model.encode(
            questions_text,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for col in option_cols:
            print(f"[GPU] Encoding option {col} with SBERT...")
            sbert_opt_embs[col] = sbert_model.encode(
                df[col].astype(str).tolist(),
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        sim_art_q_sbert = _sbert_cosine_gpu(sbert_art_emb, sbert_q_emb)
    else:
        sim_art_q_sbert = None

    # --- Build features per option ---
    all_X = []
    all_y = []

    for i, col in enumerate(option_cols):
        # TF-IDF sims
        sim_art_opt = get_sims_gpu(articles_tfidf, opt_vecs[col])
        sim_q_opt = get_sims_gpu(questions_tfidf, opt_vecs[col])

        # Word overlap
        print(f"[INFO]   Computing word overlap for option {col}...")
        art_words_list = df["article"].apply(
            lambda x: set(clean_text(str(x)).split()))
        opt_words_list = df[col].apply(
            lambda x: set(clean_text(str(x)).split()))
        overlap = np.array(
            [
                len(ow & aw) / max(len(ow), 1)
                for ow, aw in zip(opt_words_list, art_words_list)
            ],
            dtype=np.float32,
        )

        # Labels
        y_col = (df["answer"] == col).astype(int).values

        # Lengths
        opt_len = df[col].str.split().str.len().fillna(
            0).values.astype(np.float32)
        q_len = df["question"].str.split().str.len().fillna(
            0).values.astype(np.float32)

        # Build feature row
        features = [
            sim_art_opt,
            sim_q_opt,
            sim_art_q_tfidf,
            overlap,
            opt_len,
            q_len]

        # SBERT features
        if sbert_model is not None and col in sbert_opt_embs:
            sbert_sim_art_opt = _sbert_cosine_gpu(
                sbert_art_emb, sbert_opt_embs[col])
            sbert_sim_q_opt = _sbert_cosine_gpu(
                sbert_q_emb, sbert_opt_embs[col])
            features.extend(
                [sbert_sim_art_opt, sbert_sim_q_opt, sim_art_q_sbert])

        X_col = np.column_stack(features)
        all_X.append(X_col)
        all_y.append(y_col)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"[INFO] Features built: {X.shape} | Positive rate: {y.mean():.4f}")
    return X.astype(np.float32), y.astype(np.int32)


# ─────────────────────────────────────────────────────────────
# 5. Encode Answer Labels (for multi-class classification)
# ─────────────────────────────────────────────────────────────


def encode_answer_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode A/B/C/D labels into integer labels 0-3."""
    le = LabelEncoder()
    y = le.fit_transform(df["answer"])
    return y, le


# ─────────────────────────────────────────────────────────────
# 6. Save / Load Utilities
# ─────────────────────────────────────────────────────────────


def save_artifact(obj, filepath: str):
    """Pickle any Python object to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved -> {filepath}")


def load_artifact(filepath: str):
    """Load a pickled Python object from disk."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"[INFO] Loaded <- {filepath}")
    return obj


# ─────────────────────────────────────────────────────────────
# 7. CLI Entry Point — Run full preprocessing pipeline
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Load data
    train_df, val_df, test_df = load_race_data(DATA_DIR)

    # Step 2: Fit TF-IDF on training articles
    train_articles = train_df["article"].apply(clean_text).tolist()
    tfidf = build_tfidf_vectorizer(train_articles)
    save_artifact(tfidf, os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl"))

    # Step 2b: Fit One-Hot Encoding on training articles (Required Primary
    # baseline)
    onehot = build_onehot_vectorizer(train_articles)
    save_artifact(onehot, os.path.join(PROCESSED_DIR, "onehot_vectorizer.pkl"))

    # Step 3: Load SBERT for semantic features (GPU)
    sbert = _load_sbert()

    # Step 4: Build verification features with SBERT
    print(
        "\n[INFO] Building verification features for TRAIN (this may take a while)..."
    )
    X_train, y_train = build_verification_features(
        train_df, tfidf, sbert_model=sbert)
    save_artifact(
        (X_train, y_train),
        os.path.join(PROCESSED_DIR, "train_verification_features.pkl"),
    )

    print("\n[INFO] Building verification features for VAL...")
    X_val, y_val = build_verification_features(
        val_df, tfidf, sbert_model=sbert)
    save_artifact((X_val, y_val), os.path.join(
        PROCESSED_DIR, "val_verification_features.pkl"))

    print("\n[INFO] Building verification features for TEST...")
    X_test, y_test = build_verification_features(
        test_df, tfidf, sbert_model=sbert)
    save_artifact((X_test, y_test), os.path.join(
        PROCESSED_DIR, "test_verification_features.pkl"))

    # Step 5: Encode answer labels
    y_train_mc, le = encode_answer_labels(train_df)
    save_artifact(le, os.path.join(PROCESSED_DIR, "label_encoder.pkl"))

    print(
        f"\nDONE: Preprocessing complete! Features: {
            X_train.shape[1]} dims | Saved to: {PROCESSED_DIR}")
