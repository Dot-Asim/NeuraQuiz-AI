"""
model_a_train.py -- Model A: Question & Answer Verifier Training
================================================================
ALL training runs on GPU (NVIDIA RTX 5070 Ti):
  - XGBoost with device='cuda'
  - PyTorch Neural Network on CUDA
  - PyTorch-based K-Means on CUDA tensors
  - Scikit-learn lightweight models (LR, NB, RF) for ensemble diversity

Prints full metrics: Accuracy, Precision, Recall, F1, Confusion Matrix.
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

# XGBoost (GPU)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed.")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_artifact, save_artifact

# ---- GPU Setup (Strict) ----
if not torch.cuda.is_available():
    print("FATAL ERROR: CUDA is NOT available! This project is strictly optimized for GPU execution on RTX 5070 Ti.")
    print("Please check your drivers and CUDA installation. Exiting to prevent CPU fallback.")
    sys.exit(1)

DEVICE = torch.device("cuda")
print(f"[GPU] ACTIVE: {torch.cuda.get_device_name(0)}")
print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("[GPU] Enforcement: All compatible operations will run on CUDA device 0.")


# =============================================================
#  1. Load Pre-computed Features
# =============================================================

def load_features(processed_dir: str):
    """Load train/val/test verification features."""
    X_train, y_train = load_artifact(os.path.join(processed_dir, "train_verification_features.pkl"))
    X_val, y_val     = load_artifact(os.path.join(processed_dir, "val_verification_features.pkl"))
    X_test, y_test   = load_artifact(os.path.join(processed_dir, "test_verification_features.pkl"))
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================
#  2. XGBoost -- GPU Accelerated (tree_method=hist, device=cuda)
# =============================================================

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    """XGBoost on GPU -- the main powerhouse model."""
    if not HAS_XGB:
        print("[SKIP] XGBoost not available.")
        return None, {}

    print("\n" + "=" * 60)
    print("  [GPU] Training: XGBoost (CUDA)")
    print("=" * 60)
    start = time.time()

    # scale_pos_weight handles the 75/25 class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    spw = neg_count / max(pos_count, 1)
    print(f"  Class imbalance: {neg_count}:{pos_count} -> scale_pos_weight={spw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        early_stopping_rounds=40,
        scale_pos_weight=spw,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    elapsed = time.time() - start
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


# =============================================================
#  3. PyTorch Neural Network -- Full GPU Training
# =============================================================

class VerifierMLP(nn.Module):
    """Multi-layer perceptron for answer verification, runs on GPU."""
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_pytorch_mlp(X_train, y_train, X_val, y_val, epochs=50, batch_size=1024, lr=1e-3):
    """Train a PyTorch MLP entirely on GPU."""
    print("\n" + "=" * 60)
    print("  [GPU] Training: PyTorch MLP (CUDA)")
    print("=" * 60)
    start = time.time()

    # Normalize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    # Move to GPU tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_v_t = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    y_v_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Weighted loss for class imbalance (3:1 ratio)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    class_weights = torch.tensor([1.0, neg_count / max(pos_count, 1)], dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {class_weights.cpu().tolist()}")

    model = VerifierMLP(X_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_v_t)
                preds = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            avg_loss = total_loss / len(X_tr_t)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict().copy()

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - start
    model.eval()
    with torch.no_grad():
        preds = model(X_v_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)

    print(f"\n  Best MLP Results:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, preds)}")
    print(classification_report(y_val, preds, zero_division=0))
    return model, scaler, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


# =============================================================
#  4. GPU K-Means Clustering (PyTorch tensors on CUDA)
# =============================================================

def train_gpu_kmeans(X_train, y_train, n_clusters=2, max_iter=100):
    """K-Means entirely on GPU using PyTorch tensors."""
    print("\n" + "=" * 60)
    print(f"  [GPU] Training: K-Means Clustering (k={n_clusters}, CUDA)")
    print("=" * 60)
    start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    # Initialize centroids randomly
    indices = torch.randperm(X_t.shape[0])[:n_clusters]
    centroids = X_t[indices].clone()

    for it in range(max_iter):
        # Assign clusters: compute distances to all centroids
        dists = torch.cdist(X_t, centroids)
        labels = dists.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                new_centroids[c] = X_t[mask].mean(dim=0)
            else:
                new_centroids[c] = centroids[c]

        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < 1e-4:
            print(f"  Converged at iteration {it+1}")
            break

    labels_np = labels.cpu().numpy()
    elapsed = time.time() - start

    # Purity
    purity = 0
    for c in range(n_clusters):
        mask = labels_np == c
        if mask.sum() > 0:
            most_common = np.bincount(y_train[mask]).max()
            purity += most_common
    purity /= len(y_train)

    print(f"  Time: {elapsed:.1f}s | Purity: {purity:.4f}")
    return centroids.cpu().numpy(), scaler, {"purity": purity, "time": elapsed}


# =============================================================
#  5. Lightweight CPU Models (for ensemble diversity)
# =============================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: Logistic Regression (CPU - fast)")
    print("=" * 60)
    start = time.time()
    model = LogisticRegression(max_iter=3000, C=0.5, solver="lbfgs", class_weight="balanced")
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"  Time: {elapsed:.1f}s | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("  Training: Random Forest (CPU - fast)")
    print("=" * 60)
    start = time.time()
    model = RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"  Time: {elapsed:.1f}s | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


def train_svm(X_train, y_train, X_val, y_val):
    """Support Vector Machine classifier (required by lab PDF).
    Uses stratified subsample since SVM is O(n^2) on large datasets."""
    print("\n" + "=" * 60)
    print("  Training: SVM (CPU)")
    print("=" * 60)
    start = time.time()
    scaler = StandardScaler()

    # SVM is O(n^2); subsample for feasibility on 281K rows
    max_svm_samples = 30000
    if len(X_train) > max_svm_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_svm_samples, replace=False)
        X_tr_sub = X_train[idx]
        y_tr_sub = y_train[idx]
        print(f"  Using subsample: {max_svm_samples}/{len(X_train)} (SVM is O(n^2))")
    else:
        X_tr_sub = X_train
        y_tr_sub = y_train

    X_tr = scaler.fit_transform(X_tr_sub)
    X_v = scaler.transform(X_val)
    model = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr_sub)
    elapsed = time.time() - start
    y_pred = model.predict(X_v)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"  Time: {elapsed:.1f}s | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, scaler, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


def train_naive_bayes(X_train, y_train, X_val, y_val):
    """Gaussian Naive Bayes classifier (required by lab PDF)."""
    print("\n" + "=" * 60)
    print("  Training: Gaussian Naive Bayes (CPU)")
    print("=" * 60)
    start = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
    print(f"  Time: {elapsed:.1f}s | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


# =============================================================
#  5b. Label Propagation (Semi-Supervised)
# =============================================================

def train_label_propagation(X_train, y_train, X_val, y_val, unlabeled_fraction=0.5):
    """
    Semi-supervised Label Propagation.
    Masks a fraction of training labels as -1 (unlabeled) to simulate
    a semi-supervised setting, then propagates labels through the graph.
    """
    print("\n" + "=" * 60)
    print(f"  Training: Label Propagation (Semi-Supervised, {unlabeled_fraction*100:.0f}% unlabeled)")
    print("=" * 60)
    start = time.time()

    # Scale features for graph-based method
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    # Mask a fraction of labels as unlabeled (-1)
    rng = np.random.RandomState(42)
    y_semi = y_train.copy()
    mask = rng.rand(len(y_semi)) < unlabeled_fraction
    y_semi[mask] = -1
    n_labeled = (y_semi != -1).sum()
    n_unlabeled = (y_semi == -1).sum()
    print(f"  Labeled: {n_labeled} | Unlabeled: {n_unlabeled}")

    # Use a subset if dataset is very large (LP is O(n^2))
    max_samples = 20000
    if len(X_tr) > max_samples:
        idx = rng.choice(len(X_tr), max_samples, replace=False)
        X_tr_sub = X_tr[idx]
        y_semi_sub = y_semi[idx]
        print(f"  Using subset: {max_samples} samples (LP is memory-intensive)")
    else:
        X_tr_sub = X_tr
        y_semi_sub = y_semi

    model = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
    model.fit(X_tr_sub, y_semi_sub)
    elapsed = time.time() - start

    y_pred = model.predict(X_v)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"  Time: {elapsed:.1f}s | Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(classification_report(y_val, y_pred, zero_division=0))
    return model, scaler, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "time": elapsed}


# =============================================================
#  6. GPU Ensemble (Soft Voting)
# =============================================================

def build_gpu_ensemble(models_dict, X_val, y_val, mlp_model=None, mlp_scaler=None):
    """Soft voting ensemble -- MLP inference on GPU, others on CPU."""
    print("\n" + "=" * 60)
    print("  Building: GPU-Accelerated Soft Voting Ensemble")
    print("=" * 60)

    prob_list = []
    valid_names = []

    for name, model in models_dict.items():
        if model is None:
            continue
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            y_pred_temp = np.argmax(probs, axis=1)
            weight = f1_score(y_val, y_pred_temp, average="macro", zero_division=0)
            
            # Penalize highly biased models like Naive Bayes
            if weight < 0.45:
                weight *= 0.1
                
            prob_list.append(probs * weight)
            valid_names.append(f"{name} (w={weight:.2f})")
            print(f"    + {name}: shape {probs.shape}, weight {weight:.3f}")

    # Add MLP predictions (GPU)
    if mlp_model is not None and mlp_scaler is not None:
        mlp_model.eval()
        X_v_scaled = mlp_scaler.transform(X_val)
        X_v_t = torch.tensor(X_v_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = mlp_model(X_v_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        prob_list.append(probs)
        valid_names.append("MLP_GPU")
        print(f"    + MLP_GPU: shape {probs.shape}")

    if not prob_list:
        print("  [WARN] No models available for ensemble.")
        return None, {}

    avg_probs = np.mean(prob_list, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    print(f"  Ensemble ({', '.join(valid_names)})")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    return valid_names, {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1}


# =============================================================
#  7. Main Training Pipeline
# =============================================================

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "model_a")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load features
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(PROCESSED_DIR)

    results = {}

    # ---- GPU Models ----
    # 1. XGBoost GPU
    xgb_model, xgb_res = train_xgboost_gpu(X_train, y_train, X_val, y_val)
    results["XGBoost_GPU"] = xgb_res
    if xgb_model:
        save_artifact(xgb_model, os.path.join(MODEL_DIR, "xgboost_gpu.pkl"))

    # 2. PyTorch MLP GPU (DISABLED DUE TO INSTRUCTOR BAN ON NEURAL NETWORKS)
    mlp_model, mlp_scaler = None, None
    print("\n[SKIP] PyTorch MLP is disabled because Neural Networks are not allowed by the instructor.")

    # 3. GPU K-Means
    km_centroids, km_scaler, km_res = train_gpu_kmeans(X_train, y_train, n_clusters=2)
    results["KMeans_GPU"] = km_res
    save_artifact((km_centroids, km_scaler), os.path.join(MODEL_DIR, "kmeans_gpu.pkl"))

    # ---- CPU Models (fast, for ensemble diversity) ----
    lr_model, lr_res = train_logistic_regression(X_train, y_train, X_val, y_val)
    results["LogisticRegression"] = lr_res
    save_artifact(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    rf_model, rf_res = train_random_forest(X_train, y_train, X_val, y_val)
    results["RandomForest"] = rf_res
    save_artifact(rf_model, os.path.join(MODEL_DIR, "random_forest.pkl"))

    # 5. SVM (required by lab PDF)
    svm_model, svm_scaler, svm_res = train_svm(X_train, y_train, X_val, y_val)
    results["SVM"] = svm_res
    save_artifact((svm_model, svm_scaler), os.path.join(MODEL_DIR, "svm.pkl"))

    # 6. Naive Bayes (required by lab PDF)
    nb_model, nb_res = train_naive_bayes(X_train, y_train, X_val, y_val)
    results["NaiveBayes"] = nb_res
    save_artifact(nb_model, os.path.join(MODEL_DIR, "naive_bayes.pkl"))

    # 7. Label Propagation (semi-supervised, required by lab PDF)
    lp_model, lp_scaler, lp_res = train_label_propagation(X_train, y_train, X_val, y_val)
    results["LabelPropagation"] = lp_res
    save_artifact((lp_model, lp_scaler), os.path.join(MODEL_DIR, "label_propagation.pkl"))

    # ---- Ensemble ----
    ensemble_models = {
        "XGBoost_GPU": xgb_model,
        "LogisticRegression": lr_model,
        "RandomForest": rf_model,
        "SVM": svm_model,
        "NaiveBayes": nb_model,
    }
    ens_names, ens_res = build_gpu_ensemble(ensemble_models, X_val, y_val,
                                             mlp_model=mlp_model, mlp_scaler=mlp_scaler)
    results["Ensemble_GPU"] = ens_res

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("  MODEL A -- RESULTS SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    print(summary.to_string())
    summary.to_csv(os.path.join(MODEL_DIR, "model_a_results.csv"))

    print(f"\nDONE: Model A training complete. Models saved to: {MODEL_DIR}")
