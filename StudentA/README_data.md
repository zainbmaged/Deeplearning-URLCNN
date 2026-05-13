# Data README — Student A

## Project
Phishing URL Detection using Machine Learning and Deep Learning

## Dataset Source

We use the Mendeley 2026 phishing URL dataset:

**Phishing URL Dataset**  
DOI: `10.17632/3jddhy2f6s.1`

The dataset contains raw URL strings with binary labels:

- `0` = legitimate URL
- `1` = phishing URL

This dataset is suitable for our project because it provides raw URLs. This allows the team to use the same dataset for both:

- character-level CNN/RNN models using raw URL strings,
- classical machine learning models using handcrafted URL features.

---

## Raw Dataset

The raw dataset contains:

- Rows: `149,726`
- Columns: `URL`, `Label`

Initial checks included:

- column names,
- data types,
- binary label values,
- missing values,
- empty URL strings,
- duplicate URLs,
- conflicting labels.

---

## Cleaning Process

The cleaning process included:

1. Stripping whitespace from URL strings.
2. Checking missing values.
3. Checking empty URL strings.
4. Checking that labels are binary.
5. Checking duplicate URLs.
6. Checking conflicting labels for the same URL.
7. Removing duplicate URLs before splitting.

Duplicate URLs were removed before train/validation/test splitting to reduce data leakage risk.

Final cleaned dataset:

| Item | Count |
|---|---:|
| Raw rows | 149,726 |
| Clean rows | 129,776 |
| Legitimate URLs | 74,972 |
| Phishing URLs | 54,804 |

Output file:

```text
data/processed/clean_urls.csv
```

---

## Train / Validation / Test Split

We used a stratified 70/15/15 split.

Stratification was used to keep the class ratio consistent across train, validation, and test sets.

| Split | Rows |
|---|---:|
| Train | 90,843 |
| Validation | 19,466 |
| Test | 19,467 |

Output files:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

---

## Handcrafted URL Features

We extracted 23 handcrafted lexical and structural URL features from each URL.

Feature groups:

| Group | Examples |
|---|---|
| Length-based features | `url_length`, `hostname_length`, `path_length`, `query_length` |
| Character-count features | `num_dots`, `num_hyphens`, `num_digits`, `num_slashes` |
| Ratio and complexity features | `digit_ratio`, `shannon_entropy` |
| URL structure features | `has_https`, `has_ip_address`, `subdomain_depth`, `path_depth`, `hostname_has_hyphen`, `tld_in_path` |
| Keyword feature | `suspicious_keyword_count` |

Each feature file contains:

- 23 handcrafted features,
- 1 target label column.

Unscaled feature files:

```text
data/features/train_features.csv
data/features/val_features.csv
data/features/test_features.csv
```

---

## Feature Scaling

We created scaled versions of the feature files using `StandardScaler`.

Important detail:

The scaler was fitted only on the training features. The same scaler was then applied to validation and test features to avoid data leakage.

Scaled feature files:

```text
data/features_scaled/train_features_scaled.csv
data/features_scaled/val_features_scaled.csv
data/features_scaled/test_features_scaled.csv
```

Saved scaler:

```text
models/standard_scaler.joblib
```

---

## Balanced and Imbalanced Test Sets

We created two additional test sets for later evaluation.

### Balanced test set

| Class | Count |
|---|---:|
| Legitimate | 8,221 |
| Phishing | 8,221 |

Total rows: `16,442`

Output:

```text
data/splits/test_balanced.csv
```

### Imbalanced test set

| Class | Count |
|---|---:|
| Legitimate | 11,246 |
| Phishing | 2,811 |

Total rows: `14,057`

Class ratio:

- 80% legitimate
- 20% phishing

Output:

```text
data/splits/test_imbalanced_80_20.csv
```

These additional test sets are for evaluation only, not training.

---

## Evaluation Feature Files

We extracted the same 23 handcrafted URL features for the balanced and imbalanced test sets.

Unscaled files:

```text
data/features/test_balanced_features.csv
data/features/test_imbalanced_80_20_features.csv
```

Scaled files:

```text
data/features_scaled/test_balanced_features_scaled.csv
data/features_scaled/test_imbalanced_80_20_features_scaled.csv
```

The scaled versions use the same scaler fitted on the original training feature set.

---

## SMOTE Training File

We created one SMOTE-balanced training file.

Important detail:

SMOTE was applied only to the scaled training features. It was not applied to validation or test data.

Before SMOTE:

| Class | Count |
|---|---:|
| Legitimate | 52,480 |
| Phishing | 38,363 |

After SMOTE:

| Class | Count |
|---|---:|
| Legitimate | 52,480 |
| Phishing | 52,480 |

Output:

```text
data/features_scaled/train_features_smote_scaled.csv
```

This file should be used only for optional classical machine learning experiments with handcrafted features.

It should not be used for the character-level CNN/RNN because SMOTE creates synthetic feature vectors, not real URL strings.

---

## Recommended Use by Team Members

### Student B — Deep Learning

Use raw URL split files:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/splits/test_balanced.csv
data/splits/test_imbalanced_80_20.csv
```

These files are suitable for character-level CNN/RNN models.

### Student C — Classical Machine Learning

Use unscaled feature files for tree-based models:

```text
data/features/
```

Use scaled feature files for scale-sensitive models such as Logistic Regression or SVM:

```text
data/features_scaled/
```

Use the SMOTE file only as an optional training-only classical ML experiment:

```text
data/features_scaled/train_features_smote_scaled.csv
```

---

## Report Outputs

EDA tables:

```text
reports/tables/
```

EDA figures:

```text
reports/figures/
```

These outputs are prepared for the mid-report and final report.
