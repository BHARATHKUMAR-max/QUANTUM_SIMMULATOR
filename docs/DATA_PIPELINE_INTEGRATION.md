# Complete Data Pipeline Integration Guide — QML Benchmark Studio

> **Status:** Formal Specification & Integration Verification — Day 12, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR  
> **Modules:** `backend/data/` (`dataset_loader.py`, `csv_validator.py`, `dataset_profiler.py`, `splitting.py`, `preprocessing.py`, `feature_selection.py`, `correlation_filter.py`, `dimensionality_reduction.py`, `comparison_modes.py`, `schemas.py`)  
> **Test Suite:** `tests/test_data_pipeline_integration.py` (27 integration tests)  
> **Example Script:** `examples/reproducible_data_pipeline.py`  
> **Scope Reference:** [docs/MVP_SCOPE.md](MVP_SCOPE.md) · [docs/ARCHITECTURE.md](ARCHITECTURE.md) · [docs/DATA_FLOW.md](DATA_FLOW.md)

---

## 1. Overview & Pipeline Architecture

The **QML Benchmark Studio** data subsystem provides a unified, leak-proof, scientifically rigorous pipeline that transforms raw tabular inputs (built-in datasets or custom CSV uploads) into validated feature matrices ready for side-by-side classical and quantum machine learning model training.

```text
 1. DATA INGESTION & AUDIT
    ├── Built-in: load_dataset("breast_cancer") | load_dataset("wine")
    └── User CSV: validate_csv_dataset(file) ──► load_csv_dataset(file)
                               │
                               ▼
 2. DATASET PROFILING
    └── profile_dataset(bundle) ──► export_profile(profile, bundle)
                               │
                               ▼
 3. STRATIFIED PARTITIONING
    └── create_train_test_split(bundle, test_size=0.2, stratify=True, random_state=42)
            ├── X_train, y_train  (80%)
            └── X_test,  y_test   (20%, strictly quarantined)
                               │
                               ▼
 4. LEAKAGE-SAFE PREPROCESSING (Fit on X_train ONLY)
    ├── fit_preprocessor(X_train, config) ──► PreprocessorArtifact
    ├── transform_with_preprocessor(artifact, X_train) ──► X_train_proc
    └── transform_with_preprocessor(artifact, X_test)  ──► X_test_proc
                               │
                               ▼
 5. MODULAR FEATURE REDUCTION (Fit on X_train_proc ONLY)
    ├── Mutual Information (mutual_info_classif)
    ├── ANOVA F-test (f_classif)
    ├── Correlation Filtering (Pearson |r| >= threshold)
    └── PCA Dimensionality Reduction (n_components / variance_target)
                               │
                               ▼
 6. COMPARISON MODE ORCHESTRATION & VALIDATION
    ├── Mode A (Matched Reduced Features, PRIMARY) ──► is_matched_input = True
    ├── Mode B (Quantum-Limited Features)           ──► is_matched_input = True
    └── Mode C (Best Practical Configuration)       ──► is_matched_input = False
                               │
                               ▼
 7. HARDWARE COMPATIBILITY & METADATA EXPORT
    ├── validate_qubit_compatibility(n_features, n_qubits, encoding_strategy)
    ├── verify_matched_inputs(X_classical, X_quantum)
    └── ComparisonModeResult.to_dict()  (JSON-serializable)
```

---

## 2. Step-by-Step Pipeline Walkthrough

### Step 1: Ingestion and CSV Audit

#### Built-in Datasets
```python
from backend.data import load_dataset

bundle = load_dataset("breast_cancer")  # 569 samples x 30 features, binary
# or
bundle = load_dataset("wine")           # 178 samples x 13 features, 3-class
```

#### Custom CSV Upload
```python
from backend.data import validate_csv_dataset, load_csv_dataset

# 1. Structural and statistical validation audit
val_res = validate_csv_dataset("data/upload.csv", target_column="target", min_rows=15)
if not val_res.is_valid:
    for err in val_res.errors:
        print(f"[{err.code}] {err.message} -> Action: {err.suggested_action}")
    raise ValueError("CSV failed pre-flight audit.")

# 2. Ingest validated file
bundle = load_csv_dataset("data/upload.csv", target_column="target", dataset_name="custom_data")
```

---

### Step 2: Dataset Profiling

```python
from backend.data import profile_dataset, export_profile

profile = profile_dataset(bundle)
print("Samples:", profile.n_samples)
print("Features:", profile.n_features)
print("Class breakdown:", profile.class_distribution.counts)

# Export reports and 5 analytical charts
artifacts = export_profile(profile, bundle, output_dir="reports/dataset_profiles/custom_data")
```

---

### Step 3: Reproducible Stratified Partitioning

```python
from backend.data import create_train_test_split, create_stratified_cv_splits

# 80/20 train/test split with class-distribution preservation
split = create_train_test_split(
    dataset_bundle=bundle,
    test_size=0.2,
    random_state=42,
    stratify=True,
)

# K-Fold Cross-Validation splits strictly inside X_train
folds = create_stratified_cv_splits(split.X_train, split.y_train, n_splits=3, random_state=42)
```

---

### Step 4: Leakage-Safe Preprocessing

Preprocessors (`SimpleImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder`) are fitted **exclusively on `X_train`**:

```python
from backend.data import fit_preprocessor, transform_with_preprocessor
from backend.data.schemas import PreprocessingConfig

prep_cfg = PreprocessingConfig(
    numerical_imputation="median",    # or "mean"
    numerical_scaling="standard",     # or "minmax", "none"
    categorical_encoding="onehot",    # or "ordinal"
    preserve_empty_features=True,     # prevents silent drops of all-NaN columns
)

preprocessor = fit_preprocessor(split.X_train, feature_types=bundle.feature_types, config=prep_cfg)
X_train_proc = transform_with_preprocessor(preprocessor, split.X_train)
X_test_proc  = transform_with_preprocessor(preprocessor, split.X_test)
```

---

### Step 5: Feature Reduction & Selection

#### A. Supervised Feature Selection (Mutual Information / ANOVA)
```python
from backend.data import fit_selector, transform_with_selector
from backend.data.schemas import FeatureSelectionConfig

sel_cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=6)
selector = fit_selector(X_train_proc, split.y_train, config=sel_cfg)
X_train_sel = transform_with_selector(selector, X_train_proc)
X_test_sel  = transform_with_selector(selector, X_test_proc)
```

#### B. Correlation Filtering
```python
from backend.data import fit_correlation_filter, transform_with_filter
from backend.data.schemas import CorrelationFilterConfig

corr_cfg = CorrelationFilterConfig(threshold=0.95)
corr_art = fit_correlation_filter(X_train_proc, config=corr_cfg)
X_train_filt = transform_with_filter(corr_art, X_train_proc)
X_test_filt  = transform_with_filter(corr_art, X_test_proc)
```

#### C. PCA Dimensionality Reduction
```python
from backend.data import fit_pca, transform_with_pca
from backend.data.schemas import PCAConfig

pca_cfg = PCAConfig(n_components=4)  # or variance_target=0.95
pca_art = fit_pca(X_train_proc, config=pca_cfg)
X_train_pca = transform_with_pca(pca_art, X_train_proc)  # Columns: PC1..PC4
X_test_pca  = transform_with_pca(pca_art, X_test_proc)   # Columns: PC1..PC4
```

---

### Step 6: Comparison Mode Orchestration

```python
from backend.data import prepare_comparison_mode, verify_matched_inputs
from backend.data.schemas import ComparisonModeConfig

# Mode A: Matched Reduced Features (PRIMARY SCIENTIFIC COMPARISON)
cmp_cfg = ComparisonModeConfig(
    mode="matched_reduced",
    reduction_method="pca",
    pca_config=PCAConfig(n_components=4),
    qubit_count=4,
    encoding_strategy="angle",
)

result = prepare_comparison_mode(
    X_train_processed=X_train_proc,
    y_train=split.y_train,
    X_test_processed=X_test_proc,
    y_test=split.y_test,
    config=cmp_cfg,
)

# Output verification
assert result.is_valid is True
assert result.is_matched_input is True
assert result.qubit_compatibility.is_compatible is True

# Strict identity assertion
matched, errors = verify_matched_inputs(result.X_train_classical, result.X_train_quantum)
assert matched is True
```

---

## 3. Leakage Prevention Protocol

Data leakage compromises empirical benchmark validity. QML Benchmark Studio enforces strict architectural containment:

1. **Quarantined Estimator Fitting:**
   - Every statistical transformation (`fit_preprocessor`, `fit_selector`, `fit_correlation_filter`, `fit_pca`) is invoked exclusively on training partitions (`X_train` or `X_fold_train`).
2. **Transform-Only Test Projection:**
   - Test instances (`X_test`) and validation fold instances (`X_fold_val`) are projected purely using pre-fitted artifacts.
   - Slicing, modifying, or refitting estimators on test data is impossible.
3. **Cross-Validation Isolation:**
   - `prepare_cv_fold_comparison()` fits completely independent transformers inside each fold's training slice (`fold.train_indices`). Zero information crosses fold boundaries.

---

## 4. How to Execute Tests & Verification Scripts

### Running the End-to-End Integration Test Suite

From the project root:

```powershell
.venv\Scripts\python -m pytest tests/test_data_pipeline_integration.py -v
```

### Running the Complete Regression Suite (284 tests)

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

### Running the Reproducible Demonstration Example

```powershell
.venv\Scripts\python examples/reproducible_data_pipeline.py
```

---

## 5. Known Limitations & Scope Boundaries

1. **Simulator Qubit Limits**: PennyLane's `default.qubit` statevector simulator complexity scales as $O(2^N)$. Supported qubit range is strictly bounded to 2–10 qubits ($MIN\_QUBITS=2, MAX\_QUBITS=10$).
2. **Encoding Mapping**: The MVP strictly implements 1-feature-per-qubit mapping for angle encoding and ZZ feature map. Silent truncation or zero-padding is prohibited.
3. **Data Re-Uploading**: Compact multi-feature mapping via repeated rotation layers is documented for post-MVP exploration and is not implemented in the current MVP.
