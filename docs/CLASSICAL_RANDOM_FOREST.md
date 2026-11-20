# Classical Random Forest Classifier

**Project:** QML Benchmark Studio  
**Module:** `backend.classical.random_forest`  
**Interface:** `RandomForestClassifier` (wrapping `sklearn.ensemble.RandomForestClassifier`)

---

## 1. Overview and Role in Benchmark Suite

In the QML Benchmark Studio research platform, **Random Forest** serves as the canonical non-linear tree-ensemble classical baseline. While Linear/RBF Support Vector Machines evaluate maximum-margin hyperplanes and kernel transformations, and Multi-Layer Perceptrons evaluate smooth continuous neural representations, Random Forest evaluates:
1. Orthogonal axis-aligned decision boundaries constructed via recursive binary splitting.
2. Bootstrap aggregation (bagging) over decorrelated decision trees.
3. Inherent non-linear interactions across high-dimensional feature spaces without explicit kernel design.
4. Gini impurity reduction as an intrinsic measure of feature importance.

This model provides a strong, robust, and interpretable classical comparator against upcoming Quantum Classifiers (Variational Quantum Classifier and Quantum Kernel Support Vector Classifier).

---

## 2. Architecture & Implementation

The `RandomForestClassifier` class encapsulates `sklearn.ensemble.RandomForestClassifier` inside the established QML Benchmark Studio classical model lifecycle:

```
[Configuration Dataclass / Parameters]
                 │
                 ▼
     [RandomForestClassifier] ─── (Wraps sklearn.ensemble.RandomForestClassifier)
                 │
   ┌─────────────┴─────────────┐
   ▼                           ▼
[fit(X, y, feature_names)]   [predict(X) / predict_proba(X)]
   │                           │
   ├─ High-res CPU timing      ├─ High-res CPU timing
   ├─ Gini Feature Importance  └─ Validated predictions / probabilities
   └─ fitted_ = True
```

### Key Components

- **Hyperparameter Validation:** Validates `n_estimators >= 1`, `max_depth` (positive int or None), `min_samples_split >= 2`, `max_features` (`sqrt`, `log2`, float in (0, 1], int >= 1, or None), `class_weight` (`balanced`, `balanced_subsample`, dict, or None), `random_state`, and `n_jobs`.
- **Reproducibility:** Fixes `random_state` across both bootstrap sampling and feature subsampling, ensuring bit-exact deterministic training across identical datasets.
- **CPU Parallelization:** Multi-threaded bagging execution is configured via `n_jobs` (default `-1` uses all available logical CPU cores).
- **Device Reporting:** Explicitly metadata-tagged as `"execution_device": "CPU"` to preserve strict academic transparency, noting that tree ensembles execute via CPU threads rather than host GPU accelerators.

---

## 3. Feature Importance Extraction

Scikit-learn computes feature importance based on **Mean Decrease in Impurity (MDI)** (Gini importance):

$$\text{Importance}(f) = \sum_{t \in T, v(t) = f} \frac{N_t}{N} \Delta I(t)$$

where $N_t / N$ is the proportion of samples reaching node $t$ and $\Delta I(t)$ is the decrease in Gini impurity achieved by splitting on feature $f$.

### Implementation Details

1. **Sum to 1.0:** Total MDI scores are normalized across all trees such that $\sum_{i} \text{importance}_i = 1.0$.
2. **Feature Name Mapping:**
   - When trained on a DataFrame or when `feature_names` are passed to `fit(X, y, feature_names=...)`, names map directly to column identifiers (e.g. `num__worst perimeter`, `cat__color_red`).
   - When trained on raw NumPy 2D arrays without names, fallback identifiers `feature_0`, `feature_1`, ... are generated deterministically.
3. **Structured Properties:**
   - `.feature_importances_`: Ordered dictionary `{feature_name: score}` preserving input column sequence.
   - `.top_features_`: Ranked list `[{"rank": 1, "feature": name, "importance": float}, ...]` sorted descending by importance.

### Academic Limitations of MDI Feature Importance

> [!WARNING]
> While Mean Decrease in Impurity (MDI) provides quick interpretability, it has known academic limitations:
> 1. **High-Cardinality Bias:** MDI exhibits a systematic statistical bias toward numerical features with high variance or categorical features with many unique levels, as they provide more split opportunities.
> 2. **Correlation Smearing:** In the presence of strongly collinear features, importance is divided arbitrarily among correlated features rather than concentrating on the primary predictive variable.
> 3. **Training Data Bias:** MDI is computed on training splits, reflecting in-sample tree construction rather than out-of-sample generalization.
> For production interpretability, permutation importance or SHAP values are recommended when comparing against quantum feature kernels.

---

## 4. Preprocessing and Leakage Safety

Random Forest is integrated directly into the `backend.data.preprocessing` pipeline:
- **Imputation:** Handles missing data using median/mean imputers fitted strictly on training folds.
- **Scaling:** While tree splits are invariant to monotonic monotonic transformations (standardization or min-max scaling), all inputs pass through the standard preprocessor for consistency with other baselines.
- **Leakage Safety:** The model receives strictly preprocessed train folds. Test and validation folds are transformed using the fitted preprocessor artifact without refitting.

---

## 5. Benchmark Performance Summary

Benchmarking executed on real datasets via `examples/run_random_forest_baseline.py` yielded the following authentic results:

| Dataset | Model Config | Accuracy | Balanced Acc | Macro F1 | ROC-AUC | Fit Time | Inf Time | Top Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Breast Cancer** (Binary) | 100 Trees, Depth None | **0.9561** | **0.9504** | **0.9526** | **0.9939** | 0.1367s | 0.0377s | `num__worst area` (0.1400) |
| **Breast Cancer** (Binary) | 200 Trees, Depth 10 | **0.9561** | **0.9504** | **0.9526** | **0.9932** | 0.2968s | 0.0502s | `num__worst perimeter` (0.1331) |
| **Wine** (Multiclass 3-class)| 100 Trees, Depth None | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.1718s | 0.0294s | `num__color_intensity` (0.1983) |

Saved JSON evaluation artifacts:
- `reports/model_results/random_forest/breast_cancer_rf_100.json`
- `reports/model_results/random_forest/breast_cancer_rf_200_depth10.json`
- `reports/model_results/random_forest/wine_rf_100.json`
