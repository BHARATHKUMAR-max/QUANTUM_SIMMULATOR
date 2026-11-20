# Model Configuration Validation & Recommendation Engine

**Project:** QML Benchmark Studio  
**Modules:**
- `backend.classical.model_configs`
- `backend.classical.model_registry`
- `backend.classical.model_recommender`

---

## 1. Overview & Architectural Role

In the **QML Benchmark Studio** platform, comparing classical and quantum models requires rigorous, reproducible, and validated experiment configurations. Day 17 establishes:
1. **Centralized, Typed Configuration Schemas:** Enforcing hyperparameter bounds, valid options, type checking, and deterministic seeds across all four classical baselines (`DummyBaseline`, `SVMClassifier`, `MLPClassifier`, `RandomForestClassifier`).
2. **Unified Classical Model Registry:** A decoupled registry supporting model discovery, metadata inspection, configuration resolution, and lifecycle instantiation.
3. **Transparent, Rule-Based Recommendation Engine:** A deterministic advisory engine that inspects dataset scale, feature characteristics, and available compute budgets to suggest candidate models and initial hyperparameters.

> [!IMPORTANT]
> **Strict No-Winner-Prediction Guarantee:**
> The recommendation engine is an experiment design advisor, NOT a predictive model. It never predicts which model will achieve higher accuracy, AUC, or F1 score, and never declares an empirical winner prior to benchmark execution.

---

## 2. Model Configuration Schemas

All model configurations inherit from `BaseModelConfig` and implement strict type validation, parameter boundaries, and JSON export via `.to_dict()`.

### A. Dummy Classifier (`DummyClassifierConfig`)

Predicts the most frequent class observed in the training distribution, establishing the zero-knowledge performance floor.

| Parameter | Type | Default | Validation Rules & Constraints |
| :--- | :--- | :--- | :--- |
| `strategy` | `str` | `"most_frequent"` | Must be one of `("most_frequent", "prior", "stratified", "uniform", "constant")`. |
| `random_state` | `Optional[int]` | `42` | Must be an integer or `None`. |
| `constant` | `Optional[Union[int, str]]` | `None` | Required when `strategy='constant'`; otherwise ignored. |

### B. Support Vector Machine (`SVMConfig`)

Encapsulates classical maximum-margin hyperplane classification via `sklearn.svm.SVC`.

| Parameter | Type | Default | Validation Rules & Constraints |
| :--- | :--- | :--- | :--- |
| `kernel` | `str` | `"rbf"` | Must be `"linear"` or `"rbf"`. |
| `C` | `float` | `1.0` | Must be strictly positive ($C > 0$). |
| `gamma` | `Union[str, float]` | `"scale"` | Must be `"scale"`, `"auto"`, or a positive float ($\gamma > 0$). |
| `probability` | `bool` | `False` | Boolean flag. When `False`, exposes decision function scores; when `True`, enables Platt scaling. |
| `random_state` | `Optional[int]` | `42` | Integer seed or `None`. |
| `class_weight` | `Optional[Union[str, Dict]]` | `None` | Must be `None`, `"balanced"`, or a dictionary of positive class weights. |

### C. Multi-Layer Perceptron (`MLPConfig`)

Encapsulates feed-forward neural networks trained via backpropagation (`sklearn.neural_network.MLPClassifier`).

| Parameter | Type | Default | Validation Rules & Constraints |
| :--- | :--- | :--- | :--- |
| `hidden_layer_sizes` | `Tuple[int, ...]` | `(64, 32)` | Tuple/list of positive integers. E.g., `(32,)`, `(64, 32)`, `(128, 64)`. Must not be empty. |
| `activation` | `str` | `"relu"` | Must be one of `("relu", "tanh", "logistic", "identity")`. |
| `solver` | `str` | `"adam"` | Must be one of `("adam", "sgd", "lbfgs")`. |
| `alpha` | `float` | `0.0001` | Non-negative L2 regularization penalty ($\alpha \ge 0$). |
| `learning_rate_init` | `float` | `0.001` | Strictly positive initial learning rate ($\eta > 0$). |
| `max_iter` | `int` | `300` | Strictly positive maximum epochs ($N_{\text{iter}} > 0$). |
| `early_stopping` | `bool` | `False` | Boolean flag determining validation early stopping. |
| `random_state` | `Optional[int]` | `42` | Integer seed or `None`. |
| `validation_fraction`| `float` | `0.1` | Reserved validation fraction in $(0.0, 1.0)$. |
| `n_iter_no_change` | `int` | `10` | Epoch patience for early stopping ($\ge 1$). |

### D. Random Forest (`RandomForestConfig`)

Encapsulates bagged ensemble decision trees (`sklearn.ensemble.RandomForestClassifier`).

| Parameter | Type | Default | Validation Rules & Constraints |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `int` | `100` | Strictly positive tree count ($N_{\text{trees}} \ge 1$). |
| `max_depth` | `Optional[int]` | `None` | Tree depth constraint (int $\ge 1$) or `None` for unconstrained growth. |
| `min_samples_split` | `Union[int, float]` | `2` | Integer $\ge 2$ or float in $(0.0, 1.0]$. |
| `max_features` | `Union[str, int, float, None]` | `"sqrt"` | Must be `"sqrt"`, `"log2"`, float in $(0.0, 1.0]$, int $\ge 1$, or `None`. |
| `class_weight` | `Optional[Union[str, Dict]]` | `None` | Must be `None`, `"balanced"`, `"balanced_subsample"`, or a dict of positive weights. |
| `random_state` | `Optional[int]` | `42` | Integer seed or `None`. |
| `n_jobs` | `Optional[int]` | `-1` | CPU worker threads (`-1` uses all logical cores). |
| `compute_feature_importances` | `bool` | `True` | Whether to extract normalized Gini impurity importances. |

---

## 3. Using the Model Registry

The model registry provides a decoupled API for querying available models, resolving configuration overrides, and instantiating model wrappers.

```python
from backend.classical import (
    list_available_models,
    get_default_config,
    validate_model_config,
    create_model,
)

# 1. Discover available classical models
models = list_available_models()
# Output: ['dummy', 'svm', 'mlp', 'random_forest']

# 2. Inspect default configuration
svm_default = get_default_config("svm")
print(svm_default.to_dict())

# 3. Apply manual overrides
overrides = {
    "kernel": "linear",
    "C": 2.5,
    "class_weight": "balanced",
}
validated_cfg = validate_model_config("svm", overrides)

# 4. Instantiate model
model = create_model("svm", validated_cfg)
# Model is ready for fit(X_train, y_train) and predict(X_test)
```

### Defensive Override Protection
If an unexpected parameter is passed in an override dictionary, the registry rejects it immediately with a descriptive `ValueError`:
```python
validate_model_config("svm", {"use_gpu": True})
# ValueError: Unknown parameter(s) for SVMConfig: ['use_gpu']. Allowed parameters are: ['C', 'class_weight', 'gamma', 'kernel', 'probability', 'random_state']
```

---

## 4. Transparent Rule-Based Recommendations

The recommendation engine (`backend.classical.model_recommender`) inspects dataset characteristics and available compute budgets to generate advisory reports.

### Heuristic Decision Rules

1. **Dummy Baseline:**
   - Always recommended (`recommended=True`) for every classification dataset.
   - Serves as the minimal zero-knowledge benchmark floor to detect trivial majority-class classifiers.
2. **Linear SVM:**
   - Recommended for numerical, standardized datasets.
   - Preferred over RBF SVM when dataset scale exceeds 1,500 samples in `"low"` compute budgets, or 5,000 samples in `"medium"` budgets to avoid $O(N^2)$ kernel matrix memory overhead.
3. **RBF SVM:**
   - Recommended for evaluating smooth non-linear decision boundaries when $N \le 5,000$.
   - Crucial comparator for Quantum Kernel Support Vector Classifiers (QK-SVC).
   - If class imbalance ratio $> 3.0$, automatically suggests `class_weight="balanced"`.
4. **Multi-Layer Perceptron (MLP):**
   - Guardrail: If $N < 50$ samples, marked `recommended=False` due to severe risk of neural network overfitting or divergence on tiny samples.
   - Adapts architecture to compute budget:
     - Low budget: Single layer `(32,)`, `max_iter=150`.
     - Medium budget: Two layers `(64, 32)`, `max_iter=200`.
     - High budget: Two layers `(128, 64)`, `max_iter=300`.
   - Notes that scikit-learn MLPClassifier does not support native `class_weight` reweighting.
5. **Random Forest:**
   - Recommended as the primary tree-ensemble baseline. Robust to unscaled data and outliers.
   - Adapts ensemble size to compute budget:
     - Low budget: `n_estimators=50`, `max_depth=8`.
     - Medium budget: `n_estimators=100`, `max_depth=None`.
     - High budget: `n_estimators=200`, `max_depth=None`.
   - If class imbalance ratio $> 3.0$, suggests `class_weight="balanced"`.
6. **Execution Transparency:**
   - All classical baselines execute on CPU multi-threading. The recommender explicitly notes that host GPU accelerators (such as NVIDIA GeForce RTX 4050) are not utilized by scikit-learn classical models.

---

## 5. Compute Budget Tiers

| Tier | SVM Policy | MLP Architecture | Random Forest Budget | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`low`** | Linear preferred if $N > 1500$; RBF only for small $N$. | 1 layer `(32,)`, 150 max epochs | 50 trees, max depth 8 | Quick smoke testing, CPU-constrained environments. |
| **`medium`** | RBF up to 5,000 samples; Linear thereafter. | 2 layers `(64, 32)`, 200 max epochs | 100 trees, unconstrained depth | Standard balanced research benchmarking. |
| **`high`** | RBF with unconstrained scale | 2 layers `(128, 64)`, 300 max epochs | 200 trees, unconstrained depth | Thorough variance reduction and convergence. |

---

## 6. Example Usage & Dataset Profiles

### A. Breast Cancer Wisconsin (Binary, $N=569$, $D=30$)

```python
from backend.data.dataset_loader import load_breast_cancer_data
from backend.data.dataset_profiler import profile_dataset
from backend.classical import recommend_models

bundle = load_breast_cancer_data()
profile = profile_dataset(bundle)

report = recommend_models(profile, compute_budget="medium")
for rec in report.recommendations:
    print(f"Model: {rec.model_name:<15} Recommended: {rec.recommended}  Compute: {rec.compute_estimate}")
```

**Output Summary:**
- `dummy`: Recommended (`strategy="most_frequent"`, compute: `low`).
- `svm`: Recommended (`kernel="rbf"`, `C=1.0`, compute: `low` to `medium`).
- `mlp`: Recommended (`hidden_layer_sizes=[64, 32]`, `max_iter=200`, compute: `medium`).
- `random_forest`: Recommended (`n_estimators=100`, compute: `low`).

### B. Wine Dataset (Multiclass 3-Class, $N=178$, $D=13$)

**Output Summary:**
- All non-dummy models automatically detect multiclass structure and annotate reasons:
  - `svm`: Notes One-vs-One decision voting.
  - `mlp`: Notes Softmax cross-entropy distribution over 3 classes.
  - `random_forest`: Notes inherent multiclass classification via per-node class distribution statistics.
