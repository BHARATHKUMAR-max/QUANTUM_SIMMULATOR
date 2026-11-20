# Feature Selection Specification — QML Benchmark Studio

> **Status:** Formal Specification & Engineering Guide — Day 9, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR  
> **Module:** `backend/data/feature_selection.py` · `backend/data/schemas.py`  
> **Scope Reference:** [docs/MVP_SCOPE.md](MVP_SCOPE.md) · [docs/ARCHITECTURE.md](ARCHITECTURE.md) · [docs/DATA_FLOW.md](DATA_FLOW.md)

---

## 1. Overview & Purpose

Feature selection reduces input dimensionality by retaining only the most informative features with respect to the target variable. In QML Benchmark Studio, this serves two purposes:

1. **Scientific validity** — Fewer features → fewer required qubits in angle-embedding quantum circuits, making QML feasible on the `default.qubit` simulator.
2. **Benchmark fairness** — Classical and quantum models are compared on the same reduced feature set, isolating model architecture as the independent variable.

> [!IMPORTANT]
> Feature selection is a **fitted transformation**, like scaling and encoding. Scores must be computed **exclusively from training data**. Test data must never influence which features are selected.

---

## 2. Architecture Position

Feature selection sits between preprocessing and model training in the pipeline:

```text
DatasetBundle
    │
    ▼
create_train_test_split()
    │
    ├── X_train ──► fit_preprocessor()  ──► transform_with_preprocessor() ──► X_train_proc
    │                                                                              │
    │                                                                    fit_selector(X_train_proc, y_train)
    │                                                                              │
    │                                                                    FeatureSelectorArtifact (fitted)
    │                                                                              │
    │                                                           ┌──────────────────┴──────────────────┐
    │                                                           │                                     │
    │                                               transform_with_selector()         transform_with_selector()
    │                                                           │                                     │
    │                                               X_train_selected             X_test_selected  ◄── X_test_proc
    │
    └── X_test ──► transform_with_preprocessor() ──► X_test_proc ──► (see above)
```

**Key invariants:**
- `fit_selector()` only receives `X_train_proc` and `y_train`.
- `transform_with_selector()` on `X_test_proc` uses the already-fitted selector.
- No test targets or test features enter score computation.

---

## 3. Selection Methods

### 3.1 Mutual Information (`method="mutual_info"`)

Uses scikit-learn's `mutual_info_classif`:

$$MI(X_j; Y) = \sum_{x,y} p(x, y) \log\frac{p(x,y)}{p(x)\,p(y)}$$

| Property | Detail |
|---|---|
| Assumptions | None — model-free, captures non-linear relationships |
| Output | Score ≥ 0 for each feature; no p-values |
| Stochastic | Yes — nearest-neighbour estimation uses random sampling |
| Reproducible | Requires fixed `random_state` |
| Constant features | Score = 0.0 (correctly zero mutual information) |
| Multiclass | Supported natively |

### 3.2 ANOVA F-test (`method="anova"`)

Uses scikit-learn's `f_classif`:

$$F = \frac{\text{between-group variance}}{\text{within-group variance}}$$

| Property | Detail |
|---|---|
| Assumptions | Linear relationship, approximately normal within-group distribution |
| Output | F-score and p-value for each feature |
| Stochastic | No — deterministic |
| Constant features | F-score = NaN (replaced with 0.0, warning emitted) |
| Multiclass | Supported — computes between-group vs within-group variance across all classes |

**NaN/Inf handling:** Any NaN or infinite scores (from constant or near-constant features) are replaced with `0.0` before selection. ANOVA p-values that are NaN are replaced with `1.0` (not significant). Both replacements are recorded in `FeatureSelectorArtifact.warnings`.

---

## 4. Feature Count Configuration (`FeatureSelectionConfig`)

```python
from backend.data.schemas import FeatureSelectionConfig

# Default: mutual_info, auto, all features, seed=42
cfg = FeatureSelectionConfig()

# ANOVA, manual selection of exactly 10 features
cfg = FeatureSelectionConfig(
    method="anova",
    mode="manual",
    k_features=10,
)

# Mutual info, auto, select 50% of features
cfg = FeatureSelectionConfig(
    method="mutual_info",
    mode="auto",
    proportion=0.5,
    random_state=42,
)
```

### Configuration Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"mutual_info"` | `"mutual_info"` or `"anova"` |
| `mode` | `str` | `"auto"` | `"auto"` (data-driven count) or `"manual"` (explicit k) |
| `k_features` | `int \| None` | `None` | Features to select. Required for `"manual"`. Caps in `"auto"`. |
| `proportion` | `float \| None` | `None` | Fraction in `(0.0, 1.0]` for auto mode. Used when `k_features` is `None`. |
| `random_state` | `int` | `42` | Seed for `mutual_info_classif` reproducibility. |

### Selection Count Resolution (Auto Mode)

| Condition | Resolved k |
|---|---|
| `k_features` set | `min(k_features, n_candidate)` — warns if capped |
| `proportion` set, no `k_features` | `max(1, floor(proportion × n_candidate))` |
| Neither set | `n_candidate` (all features) |

### Validation Errors

| Condition | Exception |
|---|---|
| `method` not in `{"mutual_info", "anova"}` | `ValueError` |
| `mode` not in `{"auto", "manual"}` | `ValueError` |
| `mode="manual"` with no `k_features` | `ValueError` |
| `k_features ≤ 0` or non-integer | `ValueError` |
| `k_features > n_candidate` in `manual` mode | `ValueError` |
| `proportion` not in `(0.0, 1.0]` | `ValueError` |

---

## 5. Public API

### 5.1 `fit_selector(X_train, y_train, config) → FeatureSelectorArtifact`

```python
from backend.data.feature_selection import fit_selector

cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=10)
artifact = fit_selector(X_train_processed, y_train, cfg)
```

- Fitted **exclusively on training data**.
- Returns a `FeatureSelectorArtifact` containing the fitted `SelectKBest`, scores, rankings, and feature names.

### 5.2 `transform_with_selector(artifact, X) → pd.DataFrame`

```python
from backend.data.feature_selection import transform_with_selector

X_train_sel = transform_with_selector(artifact, X_train_processed)
X_test_sel  = transform_with_selector(artifact, X_test_processed)
```

- **Transform-only** — the fitted selector is never modified.
- Returns a DataFrame with `artifact.selected_feature_names` as columns.
- Raises `ValueError` if `X.columns` does not match `artifact.candidate_feature_names`.

### 5.3 `build_selection_preview(artifact) → FeatureSelectionPreview`

```python
from backend.data.feature_selection import build_selection_preview

preview = build_selection_preview(artifact)
print(preview.to_dict())  # JSON-serializable
```

Returns a structured `FeatureSelectionPreview` suitable for API responses and frontend display.

### 5.4 `prepare_cv_fold_selection(X_train, y_train, fold, config)`

```python
from backend.data.feature_selection import prepare_cv_fold_selection

fold_sel, X_fold_train_sel, X_fold_val_sel = prepare_cv_fold_selection(
    X_train_proc, y_train, fold, cfg
)
```

Fits an **independent selector** on each fold's training slice. The validation slice is transformed without refitting — same leakage-safe contract as `prepare_cv_fold_data()`.

---

## 6. FeatureSelectorArtifact — Schema

| Field | Type | Description |
|---|---|---|
| `selector` | `SelectKBest` | Fitted sklearn selector |
| `method` | `str` | `"mutual_info"` or `"anova"` |
| `mode` | `str` | `"auto"` or `"manual"` |
| `candidate_feature_names` | `List[str]` | All input feature names |
| `selected_feature_names` | `List[str]` | Selected feature names (original order preserved) |
| `scores` | `List[float]` | Score per candidate feature (NaN → 0.0) |
| `pvalues` | `List[float] \| None` | ANOVA p-values (`None` for MI) |
| `rankings` | `List[int]` | Rank per candidate (1 = highest score) |
| `k_requested` | `int` | Resolved number of features to select |
| `k_selected` | `int` | Actual selected feature count |
| `random_state` | `int` | Seed used |
| `warnings` | `List[str]` | Advisory messages (NaN scores, capping, etc.) |
| `config` | `Dict` | Original `FeatureSelectionConfig` as dict |
| `fitted_timestamp_utc` | `str` | ISO 8601 timestamp |

---

## 7. FeatureSelectionPreview — Frontend Metadata

```json
{
  "method": "anova",
  "mode": "manual",
  "k_requested": 10,
  "k_selected": 10,
  "selected_feature_names": ["mean radius", "mean perimeter", ...],
  "warnings": [],
  "features": [
    {
      "name": "mean radius",
      "score": 1312.47,
      "pvalue": 1.23e-150,
      "rank": 1,
      "selected": true
    },
    {
      "name": "mean texture",
      "score": 42.1,
      "pvalue": 3.5e-11,
      "rank": 15,
      "selected": false
    }
  ]
}
```

`FeatureSelectionPreview.to_dict()` returns a fully JSON-serializable dictionary with no numpy types.

---

## 8. Leakage Prevention Guarantees

| Guarantee | Mechanism |
|---|---|
| Scores computed from training data only | `selector.fit(X_train_arr, y_train_arr)` |
| Test data transform-only | `selector.transform(X_test_arr)` — no `fit_transform` |
| Test targets never used | Only `y_train` is passed to `fit_selector()` |
| CV fold isolation | Each fold calls independent `fit_selector(X_fold_train, y_fold_train, ...)` |
| Fitted state immutable after transform | Verified by adversarial test suite |

---

## 9. Feature Names After Preprocessing

When `fit_preprocessor()` with `categorical_encoding="onehot"` expands categorical columns, the transformed feature names follow the pattern `cat__<col>_<category>` (produced by `ColumnTransformer`). The feature selector operates on these expanded names:

```python
# Example: after one-hot encoding 'city' (London, Paris, Tokyo)
# Transformed names: ['num__age', 'cat__city_London', 'cat__city_Paris', 'cat__city_Tokyo', 'bool__subscribed']

artifact = fit_selector(X_train_proc, y_train, cfg)
# artifact.candidate_feature_names == ['num__age', 'cat__city_London', ...]
# artifact.selected_feature_names   == ['num__age', 'cat__city_Tokyo', ...]
```

No feature name fabrication occurs. Selected names are always a subset of `preprocessor.transformed_feature_names`.

---

## 10. Complete End-to-End Example

```python
from backend.data import load_dataset, create_train_test_split
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor
from backend.data.feature_selection import fit_selector, transform_with_selector, build_selection_preview
from backend.data.schemas import PreprocessingConfig, FeatureSelectionConfig

# 1. Load dataset
bundle = load_dataset("breast_cancer")

# 2. Split
split = create_train_test_split(bundle, test_size=0.2, random_state=42)

# 3. Preprocess (fit on X_train only)
prep_cfg = PreprocessingConfig(numerical_scaling="standard")
preprocessor = fit_preprocessor(split.X_train, config=prep_cfg)
X_train_proc = transform_with_preprocessor(preprocessor, split.X_train)
X_test_proc  = transform_with_preprocessor(preprocessor, split.X_test)

# 4. Feature selection (fit on X_train_proc only)
sel_cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=15)
selector = fit_selector(X_train_proc, split.y_train, sel_cfg)

# 5. Apply selection (transform-only on test)
X_train_sel = transform_with_selector(selector, X_train_proc)
X_test_sel  = transform_with_selector(selector, X_test_proc)

print(f"Original features : {X_train_proc.shape[1]}")  # 30
print(f"Selected features : {X_train_sel.shape[1]}")   # 15
print(f"Selected names    : {selector.selected_feature_names}")

# 6. Preview for API / frontend
preview = build_selection_preview(selector)
print(preview.to_dict())
```
