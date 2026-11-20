# Correlation Filtering & PCA Specification — QML Benchmark Studio

> **Status:** Formal Specification & Engineering Guide — Day 10, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR  
> **Modules:** `backend/data/correlation_filter.py` · `backend/data/dimensionality_reduction.py` · `backend/data/schemas.py`  
> **Scope Reference:** [docs/MVP_SCOPE.md](MVP_SCOPE.md) · [docs/ARCHITECTURE.md](ARCHITECTURE.md) · [docs/DATA_FLOW.md](DATA_FLOW.md)

---

## 1. Overview & Purpose

In classical machine learning and especially in Quantum Machine Learning (QML), high-dimensional and collinear feature spaces impair model performance and drastically inflate quantum circuit complexity (e.g., number of required qubits in angle embedding).

Day 10 introduces two modular, leakage-safe unsupervised dimensionality reduction techniques:

1. **Correlation-Based Feature Filtering** (`backend/data/correlation_filter.py`):
   - Removes collinear features based on pairwise absolute Pearson correlation $|r| \ge \text{threshold}$ (default: $0.95$).
   - Reduces redundancy without altering feature representation (retains original features).
2. **Principal Component Analysis (PCA)** (`backend/data/dimensionality_reduction.py`):
   - Projects preprocessed data onto orthogonal principal components.
   - Supports fixed component count (`n_components`) or cumulative explained variance target (`variance_target`).
   - Produces new composite features named `PC1`, `PC2`, ..., explicitly distinct from original features.

---

## 2. Correlation Filtering (`backend/data/correlation_filter.py`)

### 2.1 Concept and Mathematical Definition

Pairwise Pearson correlation between candidate features $X_i$ and $X_j$:

$$r(X_i, X_j) = \frac{\sum_{k=1}^n (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)}{\sqrt{\sum_{k=1}^n (x_{ki} - \bar{x}_i)^2} \sqrt{\sum_{k=1}^n (x_{kj} - \bar{x}_j)^2}}$$

Two features are considered redundant if $|r(X_i, X_j)| \ge \text{threshold}$.

### 2.2 Deterministic Removal Policy

When a set of features are mutually collinear above the threshold:
1. Feature pairs are examined in the established column order ($i < j$).
2. The **earlier feature** ($X_i$) is **retained**, while the **later feature** ($X_j$) is **marked for removal**.
3. Once marked for removal, $X_j$ cannot trigger removal of subsequent features.
4. Retained columns maintain their original relative order.
5. Every removal records a detailed reason in `CorrelatedPairInfo`.

### 2.3 Configuration (`CorrelationFilterConfig`)

```python
from backend.data.schemas import CorrelationFilterConfig

# Default config: threshold = 0.95, method = "pearson"
cfg = CorrelationFilterConfig()

# Custom threshold
cfg_custom = CorrelationFilterConfig(threshold=0.90)
```

| Field | Type | Default | Validation Rules |
|---|---|---|---|
| `threshold` | `float` | `0.95` | Must satisfy $0.0 < \text{threshold} \le 1.0$ |
| `method` | `str` | `"pearson"` | Must be `"pearson"` |

### 2.4 API Signatures

```python
from backend.data.correlation_filter import fit_correlation_filter, transform_with_filter, prepare_cv_fold_filter

# 1. Fit on training data ONLY
artifact = fit_correlation_filter(X_train_processed, config=cfg)

# 2. Transform train and test partitions
X_train_filtered = transform_with_filter(artifact, X_train_processed)
X_test_filtered  = transform_with_filter(artifact, X_test_processed)

# 3. Cross-Validation fold helper
fold_artifact, X_fold_tr_filt, X_fold_val_filt = prepare_cv_fold_filter(
    X_train_processed, fold=cv_fold, config=cfg
)
```

### 2.5 Metadata Structure (`CorrelationFilterArtifact`)

- `retained_feature_names`: List of kept column names.
- `removed_feature_names`: List of discarded column names.
- `correlated_pairs`: List of `CorrelatedPairInfo(feature_a, feature_b, correlation, removed_feature, reason)`.
- `n_original`, `n_retained`, `n_removed`: Summary counts.
- `warnings`: Advisory notes (e.g. constant columns with undefined correlation).

---

## 3. PCA Dimensionality Reduction (`backend/data/dimensionality_reduction.py`)

### 3.1 Concept

PCA computes orthogonal linear combinations of centered input features that maximize variance:

$$\mathbf{z}_k = \mathbf{X} \mathbf{w}_k \quad \text{subject to} \quad \|\mathbf{w}_k\|_2 = 1, \quad \mathbf{w}_k \perp \mathbf{w}_j \, \forall j < k$$

Components are labeled `PC1`, `PC2`, ..., `PCn`.

> [!IMPORTANT]
> **PCA Components vs. Selected Features**:  
> PCA components are **not** selected original features; they are dense orthogonal linear projections. Output column headers are named `PC1, PC2, ...` to prevent ambiguity.

### 3.2 Configuration Modes (`PCAConfig`)

Supports two distinct configuration modes:

**Mode A — Explicit Component Count (`n_components`):**
```python
from backend.data.schemas import PCAConfig

cfg_mode_a = PCAConfig(n_components=10)
```

**Mode B — Cumulative Explained Variance Target (`variance_target`):**
```python
cfg_mode_b = PCAConfig(variance_target=0.95)
```

| Field | Type | Default | Rules |
|---|---|---|---|
| `n_components` | `int \| None` | `None` | Positive integer $\le \min(n_{\text{samples}}, n_{\text{features}})$ |
| `variance_target` | `float \| None` | `None` | $0.0 < \text{variance\_target} \le 1.0$ |
| `random_state` | `int` | `42` | Seed for deterministic solver |
| `whiten` | `bool` | `False` | Normalize component variance to 1.0 |

Exactly one of `n_components` or `variance_target` must be specified.

### 3.3 API Signatures

```python
from backend.data.dimensionality_reduction import fit_pca, transform_with_pca, prepare_cv_fold_pca

# 1. Fit on training data ONLY
pca_artifact = fit_pca(X_train_processed, config=cfg)

# 2. Transform train and test partitions
X_train_pca = transform_with_pca(pca_artifact, X_train_processed)
X_test_pca  = transform_with_pca(pca_artifact, X_test_processed)

# 3. Cross-Validation fold helper
fold_pca, X_fold_tr_pca, X_fold_val_pca = prepare_cv_fold_pca(
    X_train_processed, fold=cv_fold, config=cfg
)
```

### 3.4 Metadata Structure (`PCAArtifact`)

- `original_feature_names`: Input feature names prior to projection.
- `component_names`: Output component names (`["PC1", "PC2", ...]`).
- `n_components_actual`: Actual number of components selected.
- `explained_variance_ratio`: Individual component variance proportions.
- `cumulative_explained_variance`: Monotonically increasing cumulative variance.
- `total_variance_retained`: Total explained variance ratio retained.
- `warnings`: Advisory notes.

---

## 4. Pipeline Integration & Recommended Order

The modular pipeline components can be sequenced flexibly depending on benchmark needs:

```text
ProcessedDatasetResult
         │
         ▼
[Optional: Correlation Filtering] ──► Removes redundant |r| >= 0.95 features
         │
         ▼
[Optional: Supervised Feature Selection (MI / ANOVA)] ──► Keeps top k task-relevant features
         │
         ▼
[Optional: PCA Projection] ──► Compresses to orthogonal PC1..PCk
         │
         ▼
Model Training (Classical ML / PennyLane QML)
```

All steps follow strict training-only fitting and transform-only evaluation on test/validation data.
