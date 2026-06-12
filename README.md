
# Multi-Scale Character-Level CNN and Hybrid Feature Fusion
## for Lightweight Phishing URL Detection

**Course:** CISC 867 Deep Learning — Queen's University  
**Team:** Ahmed Al-Shobaki (Student A), Zainab Zahran (Student B), Fatma Abu Al Wafa (Student C)  
**Repository:** https://github.com/zainbmaged/Deeplearning-URLCNN

---

## Project Overview

This project builds a complete phishing URL detection system evaluated across classical and deep learning approaches. The same Mendeley 2026 dataset underpins both directions: raw URL strings for character-level deep models and handcrafted lexical features for classical ML baselines.

**Dataset:** Mendeley 2026 Phishing URL Dataset  
**DOI:** `10.17632/3jddhy2f6s.1`  
**Size:** 129,776 cleaned URLs (74,972 legitimate / 54,804 phishing)  
**Reproducibility:** `RANDOM_STATE = 42` used throughout all experiments.

---

## Repository Structure

```
.
├── StudentA/          — Data preparation pipeline (Ahmed Al-Shobaki)
├── StudentB/          — Deep learning models (Zainab Zahran)
└── StudentC/          — Classical ML baselines (Fatma Abu Al Wafa)
```

---

## Student A — Data Preparation Pipeline

**Notebook:** `StudentA/notebooks/01_studentA_data_preparation - Final Submission.ipynb`

### What Was Done

- Audited and cleaned the raw dataset (149,726 → 129,776 rows after removing 19,950 duplicates).
- Created a stratified 70/15/15 split (train: 90,843 / val: 19,466 / test: 19,467).
- Engineered 23 handcrafted lexical and structural URL features.
- Fitted `StandardScaler` on training data only and applied it to validation and test sets to prevent leakage.
- Generated balanced (50/50, n = 16,442) and imbalanced (80/20, n = 14,057) evaluation test sets.
- Created a SMOTE-balanced training file (104,960 samples) for optional classical ML experiments.
- Produced EDA tables and figures for the report.

### Key Outputs

| Directory | Contents |
|---|---|
| `StudentA/data/splits/` | `train.csv`, `val.csv`, `test.csv`, `test_balanced.csv`, `test_imbalanced_80_20.csv` |
| `StudentA/data/features/` | Unscaled feature files (23 features + label) for all splits |
| `StudentA/data/features_scaled/` | Scaled feature files + SMOTE training file |
| `StudentA/models/` | `standard_scaler.joblib` |
| `StudentA/reports/` | EDA tables and figures |

### Documentation

- `StudentA/README_data.md` — full data documentation
- `StudentA/AI_USAGE.md` — AI usage disclosure
- `StudentA/LOG.md` — development log

---

## Student B — Deep Learning Models

**Notebook:** `StudentB/notebooks/StudentB_DeepLearning_Models.ipynb`

### What Was Done

Four deep learning architectures were implemented, trained, and rigorously compared on CPU.

| Model | Parameters | Validation F1 | Test F1 (Original) | F1/M-Params |
|---|---|---|---|---|
| Multi-Scale Char-CNN | 179,329 | 0.9908 | 0.9926 | 5.71 |
| CNN-BiLSTM | 360,400 | 0.9925 | 0.9920 | 2.80 |
| FastText-CNN | 5.34M | 0.9918 | **0.9938** | 1.59 |
| Hybrid Feature Fusion | 193,200 | 0.9924 | 0.9926 | 5.30 |

An ablation study was conducted on three Hybrid Fusion feature subsets (Top-25%, Top-50%, All-23). All models were benchmarked across three test distributions (original, balanced 50/50, imbalanced 80/20) and CPU inference time was measured per URL.

### Key Results

- **FastText-CNN** achieves the best overall test F1 (0.9938) and the fastest inference (0.78 ms/URL).
- **Char-CNN** achieves the best parameter efficiency (5.71 F1/M-params).
- **Hybrid Fusion** achieves the highest precision across all test sets.
- All deep models substantially outperform the best classical baseline (Stacking Ensemble F1 = 0.9519).

### Key Outputs

| Directory | Contents |
|---|---|
| `StudentB/notebooks/` | Final deep learning notebook |
| `StudentB/model_artifacts/` | Saved checkpoints for all models and ablation variants |

### Documentation

- `StudentB/README.md` — reproduction instructions and AI usage log
- `StudentB/LOG.md` — development log

---

## Student C — Classical ML Baselines

**Notebook:** `StudentC/notebooks/StudentC_Classical_ML_Baselines.ipynb`

### What Was Done

Five model families were evaluated under three class-imbalance handling strategies across the original, balanced, and imbalanced test distributions.

| Model | Test F1 | ROC-AUC | Inference (ms/URL) |
|---|---|---|---|
| Stacking Ensemble (best) | **0.9519** | **0.9918** | 0.337 |
| RF (no handling) | 0.9506 | 0.9919 | 0.363 |
| RF (weighted) | 0.9504 | 0.9918 | — |
| XGB (weighted) | 0.9483 | 0.9905 | 0.034 |
| LR (no handling) | 0.8067 | 0.9115 | 0.003 |

Additional analyses:
- RF Gini feature importance (top features: `path_length`, `hostname_length`, `shannon_entropy`, `path_depth`, `url_length`).
- Feature subset ablation: Top-5 achieves test F1 = 0.9154; Top-10 achieves 0.9414; All-23 achieves 0.9506.
- Threshold sensitivity analysis: peak F1 = 0.9539 at threshold = 0.40; FNR = 4.57%.
- ROC curve analysis and confusion matrix analysis across all three test distributions.

### Key Outputs

| Directory | Contents |
|---|---|
| `StudentC/notebooks/` | Final classical ML notebook |
| `StudentC/Best_models/` | Best trained model artifacts |
| `StudentC/Reports/` | Tables and figures |

### Documentation

- `StudentC/README.md` — reproduction instructions
- `StudentC/AI_USAGE_Person C.md` — AI usage disclosure
- `StudentC/LOG_Person C.md` — development log

---

## How to Reproduce Results

### Requirements

Each student folder contains a `requirements.txt`. Key dependencies:

**Student A / Student C:**
```
python==3.9
numpy, pandas, scikit-learn, imbalanced-learn
xgboost, matplotlib, seaborn
```

**Student B:**
```
torch==2.3.0+cpu
torchtext==0.18.0
--index-url https://download.pytorch.org/whl/cpu
tqdm, scikit-learn
```

### Steps

**Student A (Data Pipeline):**
1. Place the raw dataset at `StudentA/data/raw/mendeley_2026_commoncrawl_phishtank.csv`.
2. Run `StudentA/notebooks/01_studentA_data_preparation - Final Submission.ipynb` top to bottom.
3. All splits, feature files, and EDA outputs will be generated automatically.

**Student B (Deep Learning):**
1. Ensure Student A outputs are available in `StudentA/data/`.
2. To run without retraining: upload model artifacts from `StudentB/model_artifacts/`.
3. Run `StudentB/notebooks/StudentB_DeepLearning_Models.ipynb` top to bottom.

**Student C (Classical ML):**
1. Ensure Student A feature files are available in `StudentA/data/features/` and `StudentA/data/features_scaled/`.
2. Run `StudentC/notebooks/StudentC_Classical_ML_Baselines.ipynb` top to bottom.

---

## GenAI Disclosure

In accordance with the course GenAI policy: AI tools (Claude, ChatGPT, Gemini, Kimi) were used for brainstorming, debugging, and grammar/phrasing review only. No AI tool was used to fabricate results, generate figures or tables, or write report sections. Full per-student AI usage logs are provided in each student's directory.

---

## Report

The final report is available in the repository root:  
`Group9_Final_Report_submission.pdf`

All code, data splits, trained models, and reproduction scripts are available at:  
https://github.com/zainbmaged/Deeplearning-URLCNN
