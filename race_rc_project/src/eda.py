"""
eda.py -- Exploratory Data Analysis for the RACE Dataset
=========================================================
Generates publication-quality plots saved to results/eda/:
  1. Answer distribution (train/val/test)
  2. Article word-length distribution
  3. Question word-length distribution
  4. Option word-length distribution
  5. Question type analysis (wh-words)
  6. Top-20 TF-IDF terms
  7. Passage difficulty (middle vs. high)
  8. Cosine similarity distributions
  9. Summary statistics table
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from check_pkgs import verify_environment
    verify_environment()
except ImportError:
    print("[WARN] Environment check skipped.")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_race_data, clean_text, build_tfidf_vectorizer, load_artifact

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0e23",
    "axes.facecolor": "#13112b",
    "axes.edgecolor": "#2d2b55",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#a5b4fc",
    "ytick.color": "#a5b4fc",
    "grid.color": "#1e1b4b",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

PALETTE = ["#6366f1", "#a78bfa", "#c084fc", "#34d399", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]


def save_fig(fig, name, eda_dir):
    path = os.path.join(eda_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {path}")


def run_eda(project_root):
    DATA_DIR = os.path.join(project_root, "..", "dataset")
    EDA_DIR = os.path.join(project_root, "results", "eda")
    os.makedirs(EDA_DIR, exist_ok=True)

    train_df, val_df, test_df = load_race_data(DATA_DIR)

    # ── 1. Answer Distribution ─────────────────────────────────
    print("\n[EDA] 1. Answer distribution...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, df) in zip(axes, [("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        counts = df["answer"].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color=PALETTE[:4], edgecolor="none", width=0.6)
        ax.set_title(f"{name} ({len(df):,} rows)", fontweight="bold", fontsize=12)
        ax.set_xlabel("Answer Label")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--")
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + len(df)*0.005, f"{v:,}",
                    ha="center", va="bottom", fontsize=9, color="#a5b4fc")
    fig.suptitle("Answer Label Distribution (RACE Dataset)", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "answer_distribution", EDA_DIR)

    # ── 2. Article Word-Length Distribution ─────────────────────
    print("[EDA] 2. Article word-length distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        lengths = df["article"].str.split().str.len()
        ax.hist(lengths, bins=60, alpha=0.6, color=PALETTE[i], label=f"{name} (μ={lengths.mean():.0f})", edgecolor="none")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Article Word-Length Distribution", fontweight="bold", fontsize=13)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    save_fig(fig, "article_length_dist", EDA_DIR)

    # ── 3. Question Word-Length Distribution ────────────────────
    print("[EDA] 3. Question word-length distribution...")
    fig, ax = plt.subplots(figsize=(10, 5))
    lengths = train_df["question"].str.split().str.len()
    ax.hist(lengths, bins=40, color=PALETTE[1], edgecolor="none", alpha=0.8)
    ax.axvline(lengths.mean(), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean = {lengths.mean():.1f}")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Question Length Distribution (Training Set)", fontweight="bold", fontsize=13)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    save_fig(fig, "question_length_dist", EDA_DIR)

    # ── 4. Option Word-Length Distribution ──────────────────────
    print("[EDA] 4. Option word-length distribution...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col, color in zip(axes, ["A", "B", "C", "D"], PALETTE[:4]):
        lengths = train_df[col].str.split().str.len()
        ax.hist(lengths, bins=30, color=color, edgecolor="none", alpha=0.8)
        ax.set_title(f"Option {col} (μ={lengths.mean():.1f})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Words")
        ax.grid(axis="y", linestyle="--")
    fig.suptitle("Option Length Distribution (Training Set)", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "option_length_dist", EDA_DIR)

    # ── 5. Question Type Analysis ──────────────────────────────
    print("[EDA] 5. Question type analysis...")
    wh_words = ["what", "which", "who", "where", "when", "why", "how"]
    type_counts = {}
    for w in wh_words:
        type_counts[w.capitalize()] = train_df["question"].str.lower().str.startswith(w).sum()
    other = len(train_df) - sum(type_counts.values())
    type_counts["Other"] = other

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    bars = ax.barh(labels, values, color=PALETTE[:len(labels)], edgecolor="none", height=0.6)
    ax.set_xlabel("Count")
    ax.set_title("Question Type Distribution (by First Word)", fontweight="bold", fontsize=13)
    ax.grid(axis="x", linestyle="--")
    for bar, v in zip(bars, values):
        ax.text(v + len(train_df)*0.005, bar.get_y() + bar.get_height()/2,
                f"{v:,} ({v/len(train_df)*100:.1f}%)", va="center", fontsize=9, color="#a5b4fc")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "question_type_dist", EDA_DIR)

    # ── 6. Top-20 TF-IDF Terms ─────────────────────────────────
    print("[EDA] 6. Top TF-IDF terms...")
    processed_dir = os.path.join(project_root, "data", "processed")
    tfidf_path = os.path.join(processed_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        vectorizer = load_artifact(tfidf_path)
        feature_names = vectorizer.get_feature_names_out()
        # Get mean TF-IDF scores across a sample of training articles
        sample = train_df["article"].apply(clean_text).head(5000)
        tfidf_matrix = vectorizer.transform(sample)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-20:][::-1]
        top_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        terms = [t[0] for t in top_terms][::-1]
        scores = [t[1] for t in top_terms][::-1]
        ax.barh(terms, scores, color=PALETTE[0], edgecolor="none", height=0.6)
        ax.set_xlabel("Mean TF-IDF Score")
        ax.set_title("Top 20 TF-IDF Terms (Training Articles)", fontweight="bold", fontsize=13)
        ax.grid(axis="x", linestyle="--")
        fig.tight_layout()
        save_fig(fig, "top_tfidf_terms", EDA_DIR)
    else:
        print("  [SKIP] TF-IDF vectorizer not found. Run preprocessing.py first.")

    # ── 7. Correct Answer vs Option Position ───────────────────
    print("[EDA] 7. Correct answer position bias...")
    fig, ax = plt.subplots(figsize=(8, 5))
    pos_counts = train_df["answer"].value_counts().sort_index()
    total = pos_counts.sum()
    bars = ax.bar(pos_counts.index, pos_counts.values / total * 100, color=PALETTE[:4], edgecolor="none", width=0.5)
    ax.axhline(25, color="#ef4444", linestyle="--", linewidth=1.5, label="Expected (25%)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Answer Position")
    ax.set_title("Correct Answer Position Bias", fontweight="bold", fontsize=13)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", linestyle="--")
    for bar, v in zip(bars, pos_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v/total*100 + 0.3, f"{v/total*100:.1f}%",
                ha="center", fontsize=10, color="#a5b4fc", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "answer_position_bias", EDA_DIR)

    # ── 8. Summary Statistics Table ────────────────────────────
    print("[EDA] 8. Summary statistics...")
    stats = []
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        art_len = df["article"].str.split().str.len()
        q_len = df["question"].str.split().str.len()
        stats.append({
            "Split": name,
            "Samples": len(df),
            "Avg Article Length": f"{art_len.mean():.0f}",
            "Median Article Length": f"{art_len.median():.0f}",
            "Max Article Length": f"{art_len.max():.0f}",
            "Avg Question Length": f"{q_len.mean():.1f}",
            "Answer Distribution": dict(df["answer"].value_counts().sort_index()),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(EDA_DIR, "summary_statistics.csv"), index=False)
    print(stats_df.to_string(index=False))

    print(f"\n[DONE] EDA complete. All plots saved to: {EDA_DIR}")
    return EDA_DIR


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_eda(PROJECT_ROOT)
