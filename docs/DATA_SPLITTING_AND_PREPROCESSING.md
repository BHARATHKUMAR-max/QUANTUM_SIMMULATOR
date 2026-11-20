# Data Splitting and Preprocessing Specification — QML Benchmark Studio

> **Status:** Formal Specification & Engineering Guide — Day 8, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR  
> **Modules:** `backend/data/splitting.py` · `backend/data/preprocessing.py` · `backend/data/schemas.py`  
> **Scope Reference:** [docs/MVP_SCOPE.md](MVP_SCOPE.md) · [docs/ARCHITECTURE.md](ARCHITECTURE.md) · [docs/DATA_FLOW.md](DATA_FLOW.md)

---

## 1. Overview & Core Philosophy

In machine learning benchmarking—and especially when comparing Classical Machine Learning against Quantum Machine Learning (QML) models—data leakage represents a primary threat to scientific validity. If information from held-out evaluation instances permeates training pipelines (via global scalers, whole-dataset imputers, or un-partitioned encoders), reported test accuracies and barren plateau observations become spurious.

The **QML Benchmark Studio** data preparation subsystem guarantees strict mathematical containment:

```text
                           ┌────────────────────────┐
                           │     DatasetBundle      │
                           │   (Features X, y)      │
                           └───────────┬────────────┘
                                       │
                      create_train_test_split (stratified)
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼                                   ▼
          ┌────────────────────┐              ┌────────────────────┐
          │  Training (X_train)│              │  Held-Out (X_test) │
          └──────────┬─────────┘              └──────────┬─────────┘
                     │                                   │
            fit_preprocessor                             │
                     │                                   │
                     ▼                                   │
          ┌────────────────────┐                         │
          │ Fitted Preprocessor│                         │
          │ (Scaler, Imputer,  │                         │
          │      Encoder)      │                         │
          └──────────┬─────────┘                         │
                     │                                   │
        transform_with_preprocessor         transform_with_preprocessor
                     │ (Fit & Transform)                 │ (Transform ONLY)
                     ▼                                   ▼
          ┌────────────────────┐              ┌────────────────────┐
          │  X_train_processed │              │   X_test_processed │
          └────────────────────┘              └────────────────────┘
```

### Guiding Leakage Prevention Principles

1. **Strict Fit Quarantine:** Preprocessor objects (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are **fitted exclusively on training partitions** (`X_train` or `X_fold_train`).
2. **Transform-Only Test Evaluation:** Held-out test partitions (`X_test`) and validation fold slices (`X_fold_val`) are transformed purely through already-fitted estimators without recalculating statistics.
3. **Target Isolation:** The target column $y$ is isolated during initial bundle loading and never exists inside feature matrices $X$, $X_{\text{train}}$, or $X_{\text{test}}$.
4. **Fold Independence in Cross-Validation:** Preprocessing is never fitted once on the entire training set before $k$-fold cross-validation. Instead, each fold fits its own isolated preprocessor on `X_fold_train`.
5. **No Data Snooping:** Feature distributions, categorical levels, and scaling parameters reflect strictly the observations available at training time.

---

## 2. Train/Test Partitioning (`backend/data/splitting.py`)

### 2.1 Interface & Parameters

```python
from backend.data.splitting import create_train_test_split

split_result = create_train_test_split(
    dataset_bundle=bundle,
    test_size=0.2,       # Default: 80% train, 20% test
    random_state=42,     # Reproducible random seed
    stratify=True,       # Enforce class proportion preservation
    shuffle=True         # Shuffle rows before splitting
)
```

### 2.2 Stratification & Class Balance

- **Binary & Multiclass:** Stratification ensures that each target class representation in `X_train` and `X_test` accurately reflects the source dataset distribution:
  $$\frac{N_{\text{class}, \text{train}}}{N_{\text{train}}} \approx \frac{N_{\text{class}, \text{test}}}{N_{\text{test}}} \approx \frac{N_{\text{class}, \text{total}}}{N_{\text{total}}}$$
- **Fail-Safe Integrity:** If any class contains fewer than 2 instances, stratified splitting raises an explicit `ValueError` explaining the minority class shortfall. The system **never** silently falls back to unstratified splitting unless `stratify=False` is passed explicitly by the user.

### 2.3 Original Index Preservation

- Original DataFrame row indices (`bundle.X.index`) are retained verbatim in `X_train.index`, `X_test.index`, `train_indices`, and `test_indices`.
- This ensures full auditability back to original data files without index resets or offset confusion.

### 2.4 Random Seed Strategy

Reproducibility is paramount for research benchmarking. The platform supports deterministic single-run experiments (default seed `42`) and standardized multi-seed statistical repetitions:

$$\text{DEFAULT\_SPLIT\_SEEDS} = [42, 101, 202, 303, 404]$$

---

## 3. Preprocessing Transformations (`backend/data/preprocessing.py`)

The preprocessing engine utilizes scikit-learn's `ColumnTransformer` to route semantic feature types to dedicated transformers. All strategies are configured via `PreprocessingConfig` (Day 8).

### 3.1 Numerical Features

**Imputation (configurable via `numerical_imputation`):**

| Value | Strategy | When to Use |
|---|---|---|
| `"median"` **(default)** | `SimpleImputer(strategy="median")` | Skewed distributions, outlier-heavy features |
| `"mean"` | `SimpleImputer(strategy="mean")` | Approximately Gaussian features |

Imputation statistics are computed **strictly on X_train**.

**Scaling (configurable via `numerical_scaling`):**

| Value | Class | Transformation | Use Case |
|---|---|---|---|
| `"standard"` **(default)** | `StandardScaler` | $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ | SVM, Logistic Regression, QML |
| `"minmax"` | `MinMaxScaler(feature_range=(0,1))` | $z = \frac{x - x_{\min}^{\text{train}}}{x_{\max}^{\text{train}} - x_{\min}^{\text{train}}}$ | Neural nets, bounded ranges |
| `"none"` | `FunctionTransformer(func=None)` | $z = x$ (identity passthrough) | Pre-normalised data, ablation studies |

All scaling parameters ($\mu$, $\sigma$, $x_{\min}$, $x_{\max}$) are computed **exclusively from X_train**. Constant features (zero variance) are handled gracefully by both `StandardScaler` (zeroed) and `MinMaxScaler` (zeroed) without crashes.

### 3.2 Categorical Features

**Imputation (configurable via `categorical_imputation`):**
- `"most_frequent"` **(default and only current option):** Fills missing values with the training-set mode.

**Encoding (configurable via `categorical_encoding`):**

| Value | Class | Behaviour | Unseen category handling |
|---|---|---|---|
| `"onehot"` **(default)** | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | Expands each category to a binary indicator column | All-zeros row |
| `"ordinal"` | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` | Maps each category to an integer code | `-1.0` sentinel |

Category mappings are learned **strictly from X_train**. Unseen test categories **never** crash the pipeline or leak into learned encodings.

### 3.3 Boolean Features
- **Cast:** Boolean columns are first cast to `float64` via `FunctionTransformer(_cast_bool_to_float)` before imputation (required for `SimpleImputer` compatibility with dtype `bool`).
- **Imputation:** `SimpleImputer(strategy="most_frequent")` fills missing booleans with the training mode.
- **Mathematical Representation:** Produces clean binary $\{0.0, 1.0\}$ features directly consumable by classical estimators and quantum angle embedding gates ($\theta_i = x_i \cdot \pi$).

### 3.4 All-Missing Training Feature Policy (`preserve_empty_features`)

A training feature with 100% missing values cannot provide a meaningful imputation statistic. The `preserve_empty_features` parameter (default `True`) governs the platform's response:

| Setting | Behaviour |
|---|---|
| `True` (default) | Column is preserved. `SimpleImputer(keep_empty_features=True)` fills training values with `0.0`/`0`. Column dimensions are consistent across train and test. Affected columns are reported in `preprocessor.all_missing_training_features`. |
| `False` | Column is removed from `active_columns` before the transformer is built. Downstream shape reflects the reduced dimension. |

**Leakage guarantee:** In both policies, test-data values in an all-missing training column are **never used to infer imputation statistics**. The imputer statistic is computed solely from training data.

---

## 4. PreprocessingConfig — Configuration Schema (`backend/data/schemas.py`)

All preprocessing strategies are controlled through the `PreprocessingConfig` dataclass, validated at construction time.

```python
from backend.data.schemas import PreprocessingConfig

# Default configuration (backward-compatible with Day 7)
cfg = PreprocessingConfig()
# numerical_imputation    = "median"
# numerical_scaling       = "standard"
# categorical_imputation  = "most_frequent"
# categorical_encoding    = "onehot"
# preserve_empty_features = True

# Custom configuration
cfg_custom = PreprocessingConfig(
    numerical_imputation="mean",
    numerical_scaling="minmax",
    categorical_encoding="ordinal",
    preserve_empty_features=False,
)

# From dict (supports shorthand aliases)
cfg_dict = PreprocessingConfig.from_dict({
    "imputation": "mean",
    "scaling": "minmax",
    "encoding": "ordinal",
    "preserve_empty": False,
})
```

### Validation Rules

| Field | Valid Values | Default |
|---|---|---|
| `numerical_imputation` | `"median"`, `"mean"` | `"median"` |
| `numerical_scaling` | `"standard"`, `"minmax"`, `"none"` | `"standard"` |
| `categorical_imputation` | `"most_frequent"` | `"most_frequent"` |
| `categorical_encoding` | `"onehot"`, `"ordinal"` | `"onehot"` |
| `preserve_empty_features` | `True`, `False` | `True` |

Invalid values raise `ValueError` with a descriptive message at construction time.

### Dict Aliases (`from_dict`)

| Alias key | Maps to full field |
|---|---|
| `imputation` | `numerical_imputation` |
| `scaling` | `numerical_scaling` |
| `encoding` | `categorical_encoding` |
| `preserve_empty` | `preserve_empty_features` |
| `numerical_imputer_strategy` | `numerical_imputation` (Day 7 backward compat) |
| `scale_features` (bool) | `numerical_scaling` = `"standard"` / `"none"` (Day 7) |
| `categorical_imputer_strategy` | `categorical_imputation` (Day 7 backward compat) |

---

## 4. Cross-Validation Procedure (`backend/data/splitting.py`)

Cross-validation is conducted **strictly inside the training partition**. The held-out test partition ($X_{\text{test}}, y_{\text{test}}$) remains completely quarantined until final model evaluation.

### 4.1 Fold Generation

```python
from backend.data.splitting import create_stratified_cv_splits

# Generate 3 reproducible stratified folds from training data
folds = create_stratified_cv_splits(
    X_train=split_result.X_train,
    y_train=split_result.y_train,
    n_splits=3,
    random_state=42
)
```

### 4.2 Leakage-Free Fold Execution Helper

To prevent inter-fold contamination, `prepare_cv_fold_data()` fits a fresh preprocessor on each fold's training slice:

```python
from backend.data.preprocessing import prepare_cv_fold_data

for fold in folds:
    (
        fold_preprocessor,
        X_fold_train_proc,
        X_fold_val_proc,
        y_fold_train,
        y_fold_val
    ) = prepare_cv_fold_data(
        X_train=split_result.X_train,
        y_train=split_result.y_train,
        fold=fold,
        feature_types=bundle.feature_types,
        config=PreprocessingConfig(numerical_scaling="minmax", categorical_encoding="ordinal"),
    )
    
    # Train classical or quantum model on X_fold_train_proc
    # Evaluate on X_fold_val_proc
```

---

## 5. End-to-End Pipeline Usage Examples

### 5.1 One-Line Integrated Preprocessing

```python
from backend.data import load_dataset, prepare_split_and_preprocessed_data

# 1. Load benchmark dataset
bundle = load_dataset("breast_cancer")

# 2. Split and preprocess in a single leakage-safe invocation
processed = prepare_split_and_preprocessed_data(
    dataset_bundle=bundle,
    test_size=0.2,
    random_state=42,
    stratify=True
)

print(processed.X_train_processed.shape)  # (455, 30)
print(processed.X_test_processed.shape)   # (114, 30)
print(processed.transformed_feature_names[:3])
```

### 5.2 Step-by-Step Low-Level Control

```python
from backend.data.splitting import create_train_test_split
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor

# 1. Create split
split = create_train_test_split(bundle, test_size=0.2, random_state=42)

# 2. Fit preprocessor on X_train only
preprocessor = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)

# 3. Transform X_train
X_train_processed = transform_with_preprocessor(preprocessor, split.X_train)

# 4. Transform X_test (held-out test set)
X_test_processed = transform_with_preprocessor(preprocessor, split.X_test)
```

---

## 6. How Future Model Modules Consume Processed Data

Downstream classical algorithms (`SVM`, `RandomForest`, `LogisticRegression`) and quantum circuits (`Variational Quantum Classifier`, `Quantum Kernel SVM`) will consume the output of this pipeline without modifying partitioning or feature transformations:

```text
ProcessedDatasetResult
  ├── X_train_processed (pd.DataFrame / .to_numpy()) ────► Model Training (Fit)
  ├── y_train (pd.Series / .to_numpy())             ────► Model Training (Fit)
  ├── X_test_processed (pd.DataFrame / .to_numpy())  ────► Model Evaluation (Predict)
  └── y_test (pd.Series / .to_numpy())              ────► Model Evaluation (Score)
```

1. **NumPy Statevector Ingestion:** Quantum circuits can convert processed feature vectors to NumPy arrays via `X_train_processed.to_numpy()` and pass them to PennyLane angle or amplitude embedding templates.
2. **Dimensionality Reduction Stage (Day 8+):** When feature selection (e.g., SelectKBest, PCA) is introduced, selectors will be fitted on `X_train_processed` exclusively, keeping the initial preprocessing pipeline completely decoupled and immutable.
