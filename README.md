# NeuraQuiz AI 🧠⚡

**NeuraQuiz AI** is a state-of-the-art, high-performance Intelligent Reading Comprehension (RC) and Quiz Generation system. Built to push the boundaries of classical Machine Learning and modern Neural Networks, it transforms raw text into interactive, high-fidelity learning experiences.

Optimized for **NVIDIA RTX 5070 Ti** and **CUDA 13.2**, NeuraQuiz AI leverages hybrid AI architectures to deliver near-instantaneous inference and deep linguistic analysis.

---

## 🌌 The Neural Noir Experience

NeuraQuiz features a premium **Neural Noir** interface—a dark-themed, motion-rich UI designed for focus and aesthetic excellence. 

- **Frontend:** React 18 + Vite (TypeScript)
- **Backend:** FastAPI (Python 3.10+)
- **Design System:** Custom CSS-in-JS with high-fidelity animations and glassmorphic elements.

---

## 🛠️ Core AI Architecture

### Model A: Question & Answer Verifier
A robust verification layer that validates the quality and accuracy of question-answer pairs.
- **Supervised Learning:** XGBoost (GPU Accelerated), Random Forest, SVM, Naive Bayes.
- **Unsupervised Learning:** K-Means & GMM for structural pattern recognition.
- **Semi-Supervised:** Label Propagation for dataset augmentation.
- **Ensemble:** Soft-voting mechanism for maximum precision.

### Model B: Distractor & Hint Generator
The "intelligence" behind the quiz, creating plausible decoys and helpful nudges.
- **Distractors:** Hybrid TF-IDF Cosine Similarity coupled with a custom ML Ranker.
- **Hints:** Context-aware generation powered by **Sentence-BERT (SBERT)** with GPU fallback.

---

## 🏗️ Project Structure

```bash
NeuraQuiz-AI/
├── race_rc_project/
│   ├── api/             # FastAPI High-performance Backend
│   ├── frontend/        # React + Vite "Neural Noir" UI
│   ├── src/             # Core ML Logic & Training Pipeline
│   ├── models/          # Trained Model Weights (.pkl, .bin)
│   ├── data/            # Processed RACE Dataset
│   └── results/         # Evaluation Metrics & Reports
├── dataset/             # Raw RACE CSV Files (Train/Dev/Test)
├── APP_GUIDE.md         # Detailed Component Documentation
└── RUN_GUIDE.md         # Deployment & Execution Guide
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **NVIDIA GPU** (Optional, but recommended for CUDA acceleration)

### 2. Backend Setup
```bash
cd race_rc_project
pip install -r requirements.txt
python api/main.py
```

### 3. Frontend Setup
```bash
cd race_rc_project/frontend
npm install
npm run dev
```

---

## 📊 Performance & Hardware
This project is purpose-built for high-end hardware optimization:
- **GPU:** NVIDIA GeForce RTX 5070 Ti (12GB VRAM)
- **CUDA:** 13.2
- **Acceleration:** RAPIDS (XGBoost GPU), PyTorch (SBERT GPU)

---

## 📝 License
This project is part of the **AI LAB Project** series. All rights reserved.

---

*Designed and Developed with ❤️ by NeuraQuiz AI Team*
