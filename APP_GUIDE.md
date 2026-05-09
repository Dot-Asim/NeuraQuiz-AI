# 🧠 NeuraQuiz AI — Streamlit App Documentation

## What Is This App?
**NeuraQuiz AI** is a premium reading comprehension and quiz generation platform built with **Streamlit**. It uses 9 trained ML models and SBERT to turn any English passage into an interactive quiz with AI-powered answer verification and graduated hints.

---

## App Screens (4 Total)

### Screen 1: 🏠 Home — Article Input
This is the landing page with an animated particle hero section.

**What you can do:**
- **Paste a passage** — Type or paste any English reading passage into the text area
- **Random Sample** — Click "🔀 Random Sample" to auto-load a passage from the RACE dataset
- **Enter a question** — Type a comprehension question about the passage
- **Enter the correct answer** — Provide the ground-truth answer
- **Launch Quiz Engine** — Click "🚀 Launch Quiz Engine" to generate a full quiz with distractors

**Behind the scenes:** The inference engine uses Model B (XGBoost GPU Distractor Ranker + SBERT) to generate 3 plausible wrong answers and 3 graduated hints.

---

### Screen 2: 📋 Quiz Engine — Interactive Quiz
The generated quiz is displayed here with 4 multiple-choice options (1 correct + 3 AI-generated distractors).

**What you can do:**
- **Select an option** — Click on A, B, C, or D
- **Verify Answer** — Click "✅ Verify Answer" to check your answer using Model A (9-model ensemble)
- **Get Hints** — Navigate to the hint system if you're stuck
- **New Quiz** — Start over with a new passage

**After verification:**
- ✅ Green banner + balloons if correct
- ❌ Red banner showing the right answer if wrong
- **Model Confidence** — Shows how confident the AI was (%)
- **Verification Speed** — Shows inference latency in milliseconds
- Score tracker updates in the sidebar

---

### Screen 3: 💡 Hint System — Graduated SBERT Hints
A progressive hint system powered by Sentence-BERT semantic similarity.

**What you can do:**
- **Reveal hints one at a time** — Click "🔓 Reveal Hint" to see the next hint
- Each hint shows a **relevant sentence** from the passage with a **similarity score**
- Hints go from vague (low similarity) to obvious (high similarity)
- **Show Answer** — After all 3 hints are revealed, you can see the correct answer

**Behind the scenes:** SBERT (running on GPU) encodes the passage sentences and the correct answer, then ranks sentences by semantic similarity to give you increasingly helpful clues.

---

### Screen 4: 📊 Analytics Lab — Performance Dashboard
A full developer/professor dashboard showing model metrics and visualizations.

**What it displays:**
- **Metric Cards** — 9 models trained, distractor accuracy, GPU status, total queries
- **EDA Gallery** — All 7 exploratory data analysis plots displayed in tabs:
  - Answer distribution (train/val/test)
  - Article length distribution
  - Question length distribution
  - Option length distribution
  - Question type distribution
  - Top 20 TF-IDF terms
  - Answer position bias
- **Summary Statistics Table** — Expandable dataset statistics
- **Model A Performance** — Bar chart comparing all models (Accuracy, F1, Precision, Recall)
- **Confusion Matrix Heatmaps** — Plotly heatmaps for XGBoost, LR, RF, NB, SVM
- **Model B Results** — Distractor ranker metrics + R² Score + confusion matrix
- **Test Set Results** — Tables showing final evaluation on held-out test data
- **Inference Latency** — Time-series chart tracking query speed over the session
- **Export Log** — Download all inference logs as CSV

---

## Sidebar
Always visible on the left:
- **Navigation** — Radio buttons to switch between the 4 screens
- **GPU Status** — Live indicator showing your RTX 5070 Ti is active
- **Session Score** — Tracks correct/total answers during the session
- **Query Counter** — Number of inferences run

---

## Design Features
- **Dark theme** with indigo/purple gradient palette
- **Glassmorphism** — Frosted glass card effects with `backdrop-filter: blur()`
- **Inter font** — Premium Google Font loaded for all text
- **Micro-animations** — Hover effects, fade-ins, and bounce transitions on buttons
- **Three.js particles** — Animated particle background on the home screen
- **Custom scrollbar** — Styled gradient scrollbar matching the theme
- **Hidden Streamlit branding** — No default header/footer/menu

---

## How to Run
```powershell
conda activate dotasim
streamlit run race_rc_project/ui/app.py
```
Opens at `http://localhost:8501` in your browser.
