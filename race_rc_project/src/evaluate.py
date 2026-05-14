"""
evaluate.py -- Metric Computation for Model A & Model B
========================================================
Computes Accuracy, F1, Precision, Recall, Confusion Matrix,
Exact Match on the TEST set for all trained models.

GPU: Uses CUDA for PyTorch MLP inference during evaluation.
"""

import json
from preprocessing import load_artifact
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    r2_score,
)
import torch
import pandas as pd
import numpy as np
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


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, X, y, model_name="Model"):
    """Full evaluation of a classification model."""
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec = recall_score(y, y_pred, average="macro", zero_division=0)
    em = np.mean(y == y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1:       {f1_mac:.4f}")
    print(f"  Weighted F1:    {f1_wt:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  Exact Match:    {em:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y, y_pred, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "precision_macro": prec,
        "recall_macro": rec,
        "exact_match": em,
    }


def evaluate_mlp_gpu(checkpoint_path, X, y, model_name="MLP_GPU"):
    """Evaluate PyTorch MLP on GPU."""
    from model_a_train import VerifierMLP

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = VerifierMLP(ckpt["input_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    scaler = ckpt["scaler"]

    model.eval()
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y, preds)
    f1_mac = f1_score(y, preds, average="macro", zero_division=0)
    prec = precision_score(y, preds, average="macro", zero_division=0)
    rec = recall_score(y, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y, preds)

    print(f"\n{'=' * 60}")
    print(f"  [GPU] Evaluation: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_mac:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y, preds, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "precision_macro": prec,
        "recall_macro": rec,
    }


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_A_DIR = os.path.join(PROJECT_ROOT, "models", "model_a")
    MODEL_B_DIR = os.path.join(PROJECT_ROOT, "models", "model_b")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test features
    X_test, y_test = load_artifact(
        os.path.join(PROCESSED_DIR, "test_verification_features.pkl")
    )

    # ---- Model A Evaluation ----
    model_files = [
        ("XGBoost_GPU", "xgboost_gpu.pkl"),
        ("LogisticRegression", "logistic_regression.pkl"),
        ("RandomForest", "random_forest.pkl"),
        ("NaiveBayes", "naive_bayes.pkl"),
    ]

    all_results = []
    all_confusion_matrices = {}
    for name, fname in model_files:
        path = os.path.join(MODEL_A_DIR, fname)
        if os.path.exists(path):
            model = load_artifact(path)
            res = evaluate_model(model, X_test, y_test, model_name=name)
            all_results.append(res)
            y_pred = model.predict(X_test)
            all_confusion_matrices[name] = confusion_matrix(
                y_test, y_pred).tolist()

    # SVM evaluation (needs scaler)
    svm_path = os.path.join(MODEL_A_DIR, "svm.pkl")
    if os.path.exists(svm_path):
        svm_model, svm_scaler = load_artifact(svm_path)
        X_test_svm = svm_scaler.transform(X_test)
        res = evaluate_model(svm_model, X_test_svm, y_test, model_name="SVM")
        all_results.append(res)
        y_pred = svm_model.predict(X_test_svm)
        all_confusion_matrices["SVM"] = confusion_matrix(
            y_test, y_pred).tolist()

    # MLP GPU evaluation
    mlp_path = os.path.join(MODEL_A_DIR, "mlp_gpu.pth")
    if os.path.exists(mlp_path):
        res = evaluate_mlp_gpu(mlp_path, X_test, y_test, model_name="MLP_GPU")
        all_results.append(res)

    # Summary table
    if all_results:
        summary = pd.DataFrame(
            [
                {
                    "Model": r["model"],
                    "Accuracy": r["accuracy"],
                    "Macro F1": r["f1_macro"],
                    "Precision": r["precision_macro"],
                    "Recall": r["recall_macro"],
                }
                for r in all_results
            ]
        )
        print("\n" + "=" * 60)
        print("  TEST SET -- MODEL A COMPARISON")
        print("=" * 60)
        print(summary.to_string(index=False))
        summary.to_csv(
            os.path.join(RESULTS_DIR, "model_a_test_results.csv"), index=False
        )

    # Save confusion matrices as JSON for dashboard visualization
    if all_confusion_matrices:
        with open(os.path.join(RESULTS_DIR, "confusion_matrices.json"), "w") as f:
            json.dump(all_confusion_matrices, f)
        print(
            f"  Confusion matrices saved for: {
                list(
                    all_confusion_matrices.keys())}")

    # ---- Model B Evaluation ----
    ranker_path = os.path.join(MODEL_B_DIR, "distractor_ranker.pkl")
    if os.path.exists(ranker_path):
        from model_b_train import build_distractor_training_data
        from preprocessing import load_race_data

        DATA_DIR = os.path.join(PROJECT_ROOT, "..", "dataset")
        _, _, test_df = load_race_data(DATA_DIR)
        vectorizer = load_artifact(
            os.path.join(
                PROCESSED_DIR,
                "tfidf_vectorizer.pkl"))
        X_test_d, y_test_d = build_distractor_training_data(
            test_df, vectorizer, max_samples=3000
        )

        ranker = load_artifact(ranker_path)
        res_b = evaluate_model(
            ranker, X_test_d, y_test_d, model_name="Distractor_Ranker_GPU"
        )

        # R² Score (required by lab PDF for Model B)
        if hasattr(ranker, "predict_proba"):
            y_proba = ranker.predict_proba(X_test_d)[:, 1]
            r2 = r2_score(y_test_d, y_proba)
        else:
            y_pred_d = ranker.predict(X_test_d)
            r2 = r2_score(y_test_d, y_pred_d)
        print(f"  Model B R² Score: {r2:.4f}")

        pd.DataFrame([{"Model": "Distractor_Ranker_GPU",
                       "Accuracy": res_b["accuracy"],
                       "Macro F1": res_b["f1_macro"],
                       "Precision": res_b["precision_macro"],
                       "Recall": res_b["recall_macro"],
                       "R2_Score": r2,
                       }]).to_csv(os.path.join(RESULTS_DIR,
                                               "model_b_test_results.csv"),
                                  index=False)

        # Save confusion matrix for Model B
        y_pred_b = ranker.predict(X_test_d)
        cm_b = confusion_matrix(y_test_d, y_pred_b).tolist()
        with open(os.path.join(RESULTS_DIR, "confusion_matrix_model_b.json"), "w") as f:
            json.dump({"Distractor_Ranker_GPU": cm_b}, f)

    print(f"\nDONE: Evaluation complete. Results saved to: {RESULTS_DIR}")
