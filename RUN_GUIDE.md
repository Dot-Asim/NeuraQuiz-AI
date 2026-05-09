# 🚀 NeuraQuiz AI — Execution Guide
**Intelligent Reading Comprehension & Quiz Generation System**

This system is optimized for **NVIDIA RTX 5070 Ti** GPU acceleration. Follow these steps in order to process the dataset, train the models, and launch the dashboard.

---

## 1. Environment Setup
Always ensure your Conda environment is active before running any command:
```powershell
conda activate dotasim
```

## 2. GPU Verification
Run the diagnostic tool to ensure your 5070 Ti is ready:
```powershell
python check_cuda.py
```
*Note: All scripts will fail with a FATAL error if CUDA is not detected.*

---

## 3. Full Training Pipeline
You must run these scripts in order to build the feature artifacts and train the models.

| Step | Command | Description |
|:---|:---|:---|
| **1. Preprocess** | `python race_rc_project/src/preprocessing.py` | Generates GPU-accelerated TF-IDF & Overlap features. |
| **2. Train Model A** | `python race_rc_project/src/model_a_train.py` | Trains 9 verification models (XGBoost, MLP, SVM, etc.) on GPU. |
| **3. Train Model B** | `python race_rc_project/src/model_b_train.py` | Trains Distractor Ranker & loads SBERT for hints. |
| **4. Evaluate** | `python race_rc_project/src/evaluate.py` | Generates confusion matrices and final R² scores. |
| **5. EDA** | `python race_rc_project/src/eda.py` | Generates distribution plots for the analytics dashboard. |

---

## 4. Launching the App
Once training is complete, launch the premium Streamlit dashboard to generate quizzes and view analytics.

```powershell
streamlit run race_rc_project/ui/app.py
```

---

## 💡 Troubleshooting & Notes
- **GPU Enforcement**: All scripts are hard-coded to use `cuda:0`.
- **VRAM Management**: Preprocessing uses a batch size of 10,000 to stay within 12.8GB VRAM limits.
- **Results**: Check the `race_rc_project/results/` folder for CSV logs, confusion matrices, and EDA plots.
- **Model Storage**: Models are saved in `race_rc_project/models/` for persistent use by the UI.
