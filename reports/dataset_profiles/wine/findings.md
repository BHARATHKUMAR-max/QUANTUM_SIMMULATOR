# Exploratory Data Profile: Wine Recognition Dataset

> **Generated:** 2026-11-19T06:23:26.238431+00:00  
> **Source:** Wine Recognition Dataset, Forina, M. et al., Institute of Pharmaceutical and Food Analysis and Technologies, Genoa (1991)  
> **Module:** `backend/data/dataset_profiler.py`

---

## 1. Dataset Overview

The Wine recognition dataset is a classical multiclass benchmark derived from chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.

* **Total Samples ($N$):** 178
* **Total Features ($D$):** 13 (all continuous, real-valued numerical constituents)
* **Target Classes:** 3 (class_0, class_1, class_2)
* **Memory Footprint:** 19.73 KB

---

## 2. Sample and Feature Counts

* **Sample Matrix Shape:** `(178, 13)`
* **Target Vector Shape:** `(178,)`
* **Feature Scope:** Encapsulates 13 chemical and physical measurements: alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, and proline.

---

## 3. Target Class Distribution

| Class Label | Encoded Index | Sample Count | Proportion |
|---|---|---|---|
| **class_0** | 0 | 59 | 33.15% |
| **class_1** | 1 | 71 | 39.89% |
| **class_2** | 2 | 48 | 26.97% |

* **Majority Class:** class_1 (71 samples, 39.89%)
* **Minority Class:** class_2 (48 samples, 26.97%)
* **Imbalance Ratio:** **1.479:1** (class_1 : class_2)
* **Class Balance Status:** Mild imbalance across all 3 classes. Stratified splitting is required to ensure consistent class representation across training and testing partitions.

---

## 4. Missing-Value Findings

* **Missing Values Detected:** 0
* **Columns with Missing Data:** 0
* **Assessment:** The scikit-learn Wine dataset is complete with zero missing values across all 13 features.

---

## 5. Duplicate-Row Findings

* **Duplicate Rows:** 0
* **Unique Instances:** 178
* **Identical Columns:** False
* **Assessment:** Every sample in the dataset is a unique observation. No duplicate feature columns exist.

---

## 6. Feature Range and Variance Observations

The 13 features exhibit **substantial disparities in numerical scale and variance**:

### Largest Feature Variances
| Feature Name | Sample Variance (\sigma^2) | Sample Std Dev (\sigma) |
|---|---|---|
| `proline` | 99,166.72 | 314.91 |
| `magnesium` | 203.99 | 14.28 |
| `alcalinity_of_ash` | 11.15 | 3.34 |
| `color_intensity` | 5.37 | 2.32 |
| `malic_acid` | 1.25 | 1.12 |

### Smallest Feature Variances
| Feature Name | Sample Variance (\sigma^2) | Sample Std Dev (\sigma) |
|---|---|---|
| `nonflavanoid_phenols` | 0.01548863 | 0.12445334 |
| `hue` | 0.05224496 | 0.22857157 |
| `ash` | 0.07526464 | 0.27434401 |
| `proanthocyanins` | 0.32759467 | 0.57235886 |
| `total_phenols` | 0.39168954 | 0.62585105 |

* **Scale Disparity:** Variances range from over **$99,166$** (`proline`, values up to 1680) down to **$0.015$** (`nonflavanoid_phenols`, values between 0.13 and 0.66).
* **Implication:** Z-score standardization (`StandardScaler`) is strictly required prior to model training to prevent features with large magnitudes from dominating classical gradients and quantum angle embeddings.

---

## 7. Highly Correlated Feature Pairs (|r| >= 0.95)

* **Pairs with $|r| \ge 0.95$:** **0** (none exceed 0.95).
* **Top Collinear Feature Pairs:**
| Feature 1 | Feature 2 | Pearson Correlation ($r$) |
|---|---|---|
| `total_phenols` | `flavanoids` | +0.8646 (Top Correlation) |
| `flavanoids` | `od280/od315_of_diluted_wines` | +0.7872 (Top Correlation) |
| `total_phenols` | `od280/od315_of_diluted_wines` | +0.6999 (Top Correlation) |
| `flavanoids` | `proanthocyanins` | +0.6527 (Top Correlation) |
| `alcohol` | `proline` | +0.6437 (Top Correlation) |

* **Observation:** The strongest linear correlation is between `total_phenols` and `flavanoids` ($r = +0.8646$), followed by `flavanoids` and `od280/od315_of_diluted_wines` ($r = +0.7872$). Unlike the Breast Cancer dataset, no pairs exhibit extreme collinearity ($r \ge 0.95$).

---

## 8. Important Observations for Future Preprocessing

1. **Mandatory Scaling:** Large scale differences between `proline` and `nonflavanoid_phenols` require standard scaling.
2. **Quantum Feature Dimension Ceiling:** With 13 features, the dataset exceeds the 10-qubit simulator ceiling. Dimensionality reduction (Mutual Information, ANOVA, or PCA) must reduce inputs from 13 down to $k \le 10$ features for VQC and QKSVM.
3. **Multiclass QML Requirement:** The Wine benchmark has 3 classes. VQC will implement a multi-wire or One-vs-Rest probability mapping, and classical evaluation will report Macro and Weighted multi-class metrics.
4. **Primary Fair Comparison:** Classical models in *Matched Reduced Mode* will evaluate on the exact same $k \le 10$ selected features.

---

## 9. Limitations of Exploratory Analysis

* **No Generalization Claims:** These findings describe the 178 chemical samples in the benchmark and cannot be assumed to generalize across arbitrary agricultural datasets.
* **Correlation != Predictive Power:** Moderate correlations do not indicate single-feature predictive importance.
* **Exploratory Scope:** Statistics here represent global dataset characteristics and must not be used as parameter inputs for model training.

---

## 10. Reminder: Training-Only Preprocessing

> **Strict Non-Leakage Rule:**  
> All scalers (`StandardScaler`), imputers (`SimpleImputer`), PCA transformations, and feature selection masks (`SelectKBest`) during model training must be **fitted exclusively on training partitions (`X_train`)**. The exploratory statistics documented here are for reference only and must never be applied directly to test instances.
