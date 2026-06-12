# Student B Progress Log — Zainab Zahran

This log summarizes the main development steps for the Student B deep learning model work, derived from commit history and notebook development.

Final notebook:

`StudentB/notebooks/StudentB_DeepLearning_Models.ipynb`

---

## Entry 1 — Repository setup and initial commit

**Date:** May 13, 2026

Set up the repository structure and created the initial `.gitignore` files to exclude large data files and model artifacts from version control.

Commits:
- `Initial commit`
- `Create gitignore` (×3 — iterative refinement)

---

## Entry 2 — Char-CNN baseline upload

**Date:** May 13, 2026

Uploaded the first version of the deep learning notebook containing the Multi-Scale Char-CNN baseline model.

This includes:
- CharVocab class (vocabulary size 92, min_freq = 3, cap = 100),
- URLDataset with random token-masking augmentation (p = 0.10),
- WeightedRandomSampler for class imbalance handling,
- Multi-Scale Char-CNN architecture: embedding(64) → three parallel Conv1D branches (k = 3, 5, 7; 128 filters each) + BN + ReLU + GlobalMaxPool → FC(384→128) → FC(128→1),
- Training configuration: BCEWithLogitsLoss (pos_weight = 1.368), Adam (lr = 1e-3, weight_decay = 1e-4), ReduceLROnPlateau, gradient clipping, early stopping.

Commit:
- `Add files via upload`

---

## Entry 3 — CNN-BiLSTM and FastText-CNN models

**Date:** Jun 6, 2026

Implemented and trained two additional deep learning architectures:

**CNN-BiLSTM:** Extended the Char-CNN with a bidirectional LSTM layer to capture sequential character dependencies in both directions. Global max-pooling was removed to preserve the full 200-timestep sequence for the BiLSTM. Trained for 10 epochs (reduced from 15 due to CPU training time). Checkpoint saving by best validation F1 introduced to avoid restarting from scratch.

**FastText-CNN:** Replaced character-level embeddings with word-level FastText embeddings. Word-level vocabulary built by splitting URLs on common separators, yielding 15,711 words (frequency ≥ 3). Max sequence length set to 35 tokens. FastText embedding layer fine-tuned rather than frozen to adapt pretrained Common Crawl representations to the phishing domain.

Key bugs resolved during this phase:
- Frozen FastText embedding layer (`freeze=True` → `freeze=False`),
- Zero-initialized OOV embeddings replaced with random normal initialization,
- torch/torchtext version mismatch fixed (`torch==2.3.0+cpu`, `torchtext==0.18.0`),
- Flipped early stopping condition logic corrected.

Commit:
- `CNN + BiLSTM hybrid approach & FastText + CNN`

---

## Entry 4 — Model artifacts uploaded

**Date:** Jun 6, 2026

Uploaded saved model checkpoint files for all trained models to support reproducibility without retraining.

Commit:
- `Adding models artifact for reproducibility`

---

## Entry 5 — Hybrid Feature Fusion model and ablation study

**Date:** Jun 8, 2026

Implemented Architecture D: Hybrid Feature Fusion, combining the Char-CNN deep character-level features with the 23 handcrafted statistical features provided by Student A.

Architecture:
- Statistical features projected through FC(23→64) + ReLU + Dropout(0.3),
- Concatenated with 128-dimensional Char-CNN output,
- Final classification through FC(192→64) + ReLU, Dropout(0.3), FC(64→1).

Conducted ablation study across three feature subsets:
- Top-25% (5 features),
- Top-50% (9 features),
- All-23 features.

Uploaded corresponding model artifacts for all Hybrid Fusion variants.

Commits:
- `Hybrid Feature Fusion and ablation study`
- `Adding hybrid fusion models artifacts`

---

## Entry 6 — Test set inference across all models

**Date:** Jun 9, 2026

Ran final inference across all three test distributions (original, balanced, imbalanced 80/20) for all four model families and all Hybrid Fusion ablation variants.

Collected and organized:
- accuracy, precision, recall, F1 across all test sets,
- F1/M-params efficiency scores,
- ROC-AUC per test set,
- CPU inference time per URL benchmarked on 1,000 samples.

Commit:
- `Test set inference for all models comparing metrics`

---

## Entry 7 — Final documentation

**Date:** Jun 12, 2026

Added Student B README with reproduction instructions, AI usage log with full details of bugs encountered and fixes applied, and organized notebook structure for submission.

Commit:
- `Adding student B readme`

---

## Current Status

Student B deep learning work is complete.

All four model families are trained and evaluated:
- Multi-Scale Char-CNN (179,329 parameters) — best parameter efficiency (5.71 F1/M-params on validation),
- CNN-BiLSTM — highest raw validation F1 (0.9925) but longest inference time (6.27 ms/URL),
- FastText-CNN — best overall test F1 (0.9938) and fastest inference (0.78 ms/URL),
- Hybrid Feature Fusion (Full, Top-25, Top-50) — highest precision across all test sets.

All model artifacts, the final notebook, and supporting documentation are available in the `StudentB/` directory.
