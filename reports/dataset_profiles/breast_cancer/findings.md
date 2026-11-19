# Exploratory Data Profile: Breast Cancer Wisconsin (Diagnostic)

> **Generated:** 2026-11-19T06:23:18.638154+00:00  
> **Source:** WDBC Dataset, Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)  
> **Module:** `backend/data/dataset_profiler.py`

---

## 1. Dataset Overview

The Breast Cancer Wisconsin (Diagnostic) dataset is a classical binary classification benchmark derived from digitized fine needle aspirate (FNA) images of breast masses. Features describe characteristics of cell nuclei present in the image.

* **Total Samples ($N$):** 569
* **Total Features ($D$):** 30 (all continuous, real-valued numerical)
* **Target Classes:** 2 (malignant, benign)
* **Memory Footprint:** 138.06 KB

---

## 2. Sample and Feature Counts

* **Sample Matrix Shape:** `(569, 30)`
* **Target Vector Shape:** `(569,)`
* **Feature Scope:** Encapsulates 10 fundamental nuclear characteristics evaluated across three statistical summaries: **Mean**, **Standard Error**, and **"Worst"** (mean of the three largest values).

---

## 3. Target Class Distribution

| Class Label | Encoded Value | Sample Count | Proportion | Imbalance Ratio |
|---|---|---|---|---|
| **benign** (Majority) | 1 | 357 | 62.74% | **1.684:1** |
| **malignant** (Minority) | 0 | 212 | 37.26% | 1.00:1 |

* **Class Balance Status:** Moderate imbalance (62.74% benign vs 37.26% malignant). While not severely imbalanced (< 20%), stratified partitioning is necessary during training/testing splits to preserve class parity across evaluation folds.

---

## 4. Missing-Value Findings

* **Missing Values Detected:** 0
* **Columns with Missing Data:** 0
* **Assessment:** The scikit-learn distribution of the dataset is completely intact with zero null or NaN values across all 569 instances. Imputation pipelines in production will serve as safeguards for custom uploaded datasets.

---

## 5. Duplicate-Row Findings

* **Duplicate Rows:** 0
* **Unique Instances:** 569
* **Identical Columns:** False
* **Assessment:** Every sample in the dataset represents a distinct observation. No redundant or duplicated feature columns exist.

---

## 6. Feature Range and Variance Observations

The features exhibit **drastic disparities in numerical scale and variance**:

### Largest Feature Variances
| Feature Name | Sample Variance (\sigma^2) | Sample Std Dev (\sigma) |
|---|---|---|
| `worst area` | 324,167.39 | 569.36 |
| `mean area` | 123,843.55 | 351.91 |
| `area error` | 2,069.43 | 45.49 |
| `worst perimeter` | 1,129.13 | 33.60 |
| `mean perimeter` | 590.44 | 24.30 |

### Smallest Feature Variances
| Feature Name | Sample Variance (\sigma^2) | Sample Std Dev (\sigma) |
|---|---|---|
| `fractal dimension error` | 0.00000700 | 0.00264607 |
| `smoothness error` | 0.00000902 | 0.00300252 |
| `concave points error` | 0.00003807 | 0.00617029 |
| `mean fractal dimension` | 0.00004985 | 0.00706036 |
| `symmetry error` | 0.00006833 | 0.00826637 |

* **Scale Disparity:** Variances range from over **$569,000$** (`worst area`) down to **$0.000003$** (`smoothness error`).
* **Implication:** Unscaled features would cause classical distance-based algorithms (SVM, MLP) and quantum parameterised angle rotations (\theta = x_i \pmod{2\pi}) to fail catastrophically. Standardization is mandatory.

---

## 7. Highly Correlated Feature Pairs (|r| >= 0.95)

Identified **15** feature pairs exceeding the collinearity threshold of |r| >= 0.95:

| Feature 1 | Feature 2 | Pearson Correlation ($r$) |
|---|---|---|
| `mean radius` | `mean perimeter` | +0.9979 |
| `worst radius` | `worst perimeter` | +0.9937 |
| `mean radius` | `mean area` | +0.9874 |
| `mean perimeter` | `mean area` | +0.9865 |
| `worst radius` | `worst area` | +0.9840 |
| `worst perimeter` | `worst area` | +0.9776 |
| `radius error` | `perimeter error` | +0.9728 |
| `mean perimeter` | `worst perimeter` | +0.9704 |
| `mean radius` | `worst radius` | +0.9695 |
| `mean perimeter` | `worst radius` | +0.9695 |

* **Collinearity Drivers:** Perimeter, area, and radius are geometrically coupled (A \approx \pi r^2, P = 2\pi r), producing near-perfect linear relationships (r > 0.99).
* **Note on Causation:** These correlation coefficients describe mutual linear association, not biological or physical causation. Feature reduction is required to eliminate redundancy.

---

## 8. Important Observations for Future Preprocessing

1. **Mandatory Scaling:** Features must undergo z-score standardization (`StandardScaler`) prior to model ingestion.
2. **Quantum Feature Dimension Ceiling:** The 30 features far exceed the 10-qubit simulator ceiling. Dimensionality reduction (Mutual Information, ANOVA, or PCA) to k <= 10 features is strictly required for VQC and Quantum Kernel SVM.
3. **Primary Fair Comparison:** Classical models will be evaluated in *Matched Reduced Mode* on the exact same k <= 10 selected features as quantum models.

---

## 9. Limitations of Exploratory Analysis

* **No Generalization Claims:** These findings apply strictly to the 569 instances in the Wisconsin Diagnostic benchmark.
* **Correlation != Predictive Power:** High collinearity between radius and perimeter does not dictate which feature is more informative for classification.
* **Exploratory Scope:** Visualizations and descriptive tables represent global dataset properties and must not be used as training parameters.

---

## 10. Reminder: Training-Only Preprocessing

> **Strict Non-Leakage Rule:**  
> All scalers (`StandardScaler`), imputers (`SimpleImputer`), PCA transformations, and feature selection masks (`SelectKBest`) during model training must be **fitted exclusively on training partitions (`X_train`)**. The exploratory statistics documented here are for reference only and must never be applied directly to test instances.
