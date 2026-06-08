# Student A Progress Log

This log summarizes the main development steps for the Student A data preparation work. Some work was first developed locally through notebook drafts, then organized into the final notebook before being added to the team repository.

Final notebook:

`StudentA/notebooks/01_studentA_data_preparation.ipynb`

This log can be merged later into the main team `LOG.md`.

---

## Entry 1 — Dataset selection and setup

We reviewed possible phishing URL datasets and decided to use a dataset with raw URL strings and binary labels.

This was important because our project needs the same data for two directions:
- raw URLs for the character-level CNN/RNN model,
- handcrafted URL features for classical machine learning baselines.

The dataset uses:

- `URL`
- `Label`

Label mapping:
- `0` = legitimate URL
- `1` = phishing URL

---

## Entry 2 — Raw dataset audit

We loaded the raw dataset and checked the basic structure before doing any cleaning.

We checked:
- dataset shape,
- column names,
- data types,
- missing values,
- empty URL strings,
- binary label values,
- initial class distribution.

The raw dataset had:

- 149,726 rows
- 2 columns: `URL`, `Label`

The labels were valid binary labels. We also found duplicate URLs, so we needed a cleaning step before splitting.

---

## Entry 3 — Cleaning and duplicate handling

We stripped whitespace from the URL strings first. This was done because the same URL with extra spaces should not be treated as a different URL.

After that, we checked:
- duplicate URLs,
- conflicting labels for the same URL.

We removed duplicate URLs before splitting to reduce data leakage risk.

Final cleaned dataset:

- 129,776 rows
- 74,972 legitimate URLs
- 54,804 phishing URLs

No conflicting labels were found.

Output:

`StudentA/data/processed/clean_urls.csv`

---

## Entry 4 — Train / validation / test split

We created a stratified 70/15/15 split.

We used stratification because the dataset is mildly imbalanced, and we wanted the class ratio to stay similar across all splits.

Split sizes:

- Train: 90,843 rows
- Validation: 19,466 rows
- Test: 19,467 rows

The validation and test sets differ by one row because the cleaned dataset size does not divide perfectly into 70/15/15.

Outputs:

- `StudentA/data/splits/train.csv`
- `StudentA/data/splits/val.csv`
- `StudentA/data/splits/test.csv`

---

## Entry 5 — Handcrafted feature extraction

We extracted 23 handcrafted URL features from the raw URL strings.

The features cover:
- URL and hostname lengths,
- path and query lengths,
- counts of dots, hyphens, slashes, digits, and other characters,
- digit ratio,
- HTTPS usage,
- IP address usage,
- subdomain depth,
- path depth,
- hostname hyphen flag,
- Shannon entropy,
- suspicious keyword count.

During review, we updated the URL parser to use `parsed.hostname` instead of `parsed.netloc`. This made hostname-based features cleaner because `netloc` can include ports.

Outputs:

- `StudentA/data/features/train_features.csv`
- `StudentA/data/features/val_features.csv`
- `StudentA/data/features/test_features.csv`

Each file contains 23 features plus the label column.

---

## Entry 6 — Feature scaling

We created scaled feature files using `StandardScaler`.

The scaler was fitted only on the training feature set, then applied to validation and test sets. This avoids using validation or test information during preprocessing.

Outputs:

- `StudentA/data/features_scaled/train_features_scaled.csv`
- `StudentA/data/features_scaled/val_features_scaled.csv`
- `StudentA/data/features_scaled/test_features_scaled.csv`
- `StudentA/models/standard_scaler.joblib`

The unscaled files are still kept because tree-based models can use them directly.

---

## Entry 7 — EDA for the report

We generated basic EDA tables and figures for the mid-report.

The EDA includes:
- class distribution,
- mean feature comparison by class,
- URL length by class,
- capped URL length plot,
- Shannon entropy by class,
- digit ratio by class,
- binary feature rates,
- top TLDs by class,
- feature correlation heatmap.

We noticed that URL length has large outliers. We did not remove these outliers from the data. We only used a capped plot to make the visualization easier to read.

Outputs:

- `StudentA/reports/tables/`
- `StudentA/reports/figures/`

---

## Entry 8 — Balanced and imbalanced test sets

We created two extra test sets for later evaluation.

Balanced test set:
- 16,442 rows
- 50% legitimate
- 50% phishing

Imbalanced test set:
- 14,057 rows
- 80% legitimate
- 20% phishing

These sets are only for evaluation, not training.

Outputs:

- `StudentA/data/splits/test_balanced.csv`
- `StudentA/data/splits/test_imbalanced_80_20.csv`

---

## Entry 9 — Feature files for evaluation test sets

After creating the balanced and imbalanced test sets, we extracted the same 23 handcrafted features for them.

We also created scaled versions using the same scaler fitted on the original training data.

Outputs:

- `StudentA/data/features/test_balanced_features.csv`
- `StudentA/data/features/test_imbalanced_80_20_features.csv`
- `StudentA/data/features_scaled/test_balanced_features_scaled.csv`
- `StudentA/data/features_scaled/test_imbalanced_80_20_features_scaled.csv`

This makes the evaluation sets ready for classical ML models.

---

## Entry 10 — SMOTE training file

We created one SMOTE-balanced training file for optional classical ML experiments.

SMOTE was applied only to the scaled training features.

Before SMOTE:
- Legitimate: 52,480
- Phishing: 38,363

After SMOTE:
- Legitimate: 52,480
- Phishing: 52,480

Output:

`StudentA/data/features_scaled/train_features_smote_scaled.csv`

Important note: this file should not be used for the character-level CNN/RNN model. SMOTE creates synthetic feature vectors, not real URL strings.

---

## Entry 11 — Final checks and cleanup

At the end, we added final checks to make sure the saved files match the notebook outputs.

We checked:
- cleaned dataset shape,
- split row totals,
- unscaled feature file shapes,
- scaled feature file shapes.

The saved files matched the current pipeline outputs.

We also added supporting documentation:

- `StudentA/README_data.md`
- `StudentA/AI_USAGE.md`
- `StudentA/requirements.txt`

---

## Entry 12 — Feature ablation files and fusion handoff

We created top-5 feature ablation files (path_length, hostname_length,
shannon_entropy, path_depth, url_length) for Student C's ablation study.
We added a utility function confirming the feature vector format for
Student B's Hybrid Feature Fusion model.

---

## Entry 13 — Error analysis setup and final report writing

We prepared the error analysis framework using Student C's FP/FN URL lists.
We wrote the Discussion, Limitations, and Conclusion sections for the final report.
Final pipeline checks confirmed all 37 documented output files.

---

## Current status

Student A data preparation is complete and ready for team integration.

Student B can use the raw URL split files for the character-level CNN/RNN model.

Student C can use:
- unscaled feature files for tree-based models,
- scaled feature files for Logistic Regression or SVM,
- the SMOTE-scaled training file as an optional classical ML experiment.

The EDA tables and figures are ready to use in the mid-report.