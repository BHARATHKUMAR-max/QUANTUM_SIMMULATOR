# Classical Support Vector Classification (SVM)

**Project:** QML Benchmark Studio  
**Layer:** Classical ML Engine (`backend/classical/`)  
**Status:** Implemented & Verified (335/335 Tests Passing)  
**Date:** November 20, 2026

---

## 1. Overview & Architectural Role

Support Vector Machines (SVMs) represent one of the most mathematically rigorous and high-performing classical supervised learning paradigms. In **QML Benchmark Studio**, the `SVMClassifier` (`backend/classical/svm.py`) serves as a primary classical benchmark against which quantum models are evaluated:
- **Linear SVM** serves as a baseline for linearly separable feature geometries.
- **RBF (Radial Basis Function) SVM** acts as the direct classical counterpart to **Quantum Kernel SVM (QK-SVM)**, which maps classical inputs into a high-dimensional quantum Hilbert space using parameterized quantum circuits (e.g. ZZ Feature Maps).

---

## 2. Supported Configurations & Hyperparameters

The `SVMClassifier` wraps scikit-learn's `sklearn.svm.SVC` with input validation, profiling metrics, and JSON metadata extraction.

### 2.1 Hyperparameters

| Parameter | Type | Default | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `kernel` | `str` | `"rbf"` | Kernel type. Supported: `"linear"`, `"rbf"`. |
| `C` | `float` | `1.0` | Regularization parameter. Strictly positive ($C > 0$). Controls trade-off between margin width and margin boundary errors. |
| `gamma` | `str` or `float` | `"scale"` | Kernel coefficient for RBF. Supported: `"scale"` ($1 / (N \cdot \text{Var}(X))$), `"auto"` ($1 / N$), or float $> 0$. Ignored for `"linear"`. |
| `probability` | `bool` | `False` | Enables Platt scaling for probability calibration via internal 5-fold cross-validation. Disabled by default due to computational overhead. |
| `random_state` | `int` or `None` | `42` | Random seed controlling Platt scaling pseudo-randomness when `probability=True`. |
| `class_weight` | `str` or `dict` | `None` | Balanced class weighting (`"balanced"`) or custom label weight mapping. |

---

## 3. Preprocessing & Feature Scaling Invariants

Support Vector Classifiers maximize geometric margins between classes using distance metrics (Euclidean distance for linear and RBF kernels). Consequently:
- **Sensitivity to Scale:** Features with larger numerical magnitudes dominate the optimization objective if unscaled.
- **Pipeline Integration:** `SVMClassifier` must always receive features scaled with `StandardScaler` (or `MinMaxScaler` where bounded inputs are required) fitted **exclusively on training data** via `backend.data.preprocessing`.
- **Zero Leakage:** Transformers (`SimpleImputer`, `StandardScaler`) are never fitted on validation or held-out test partitions.

---

## 4. Probability Estimation Overhead vs. Decision Scores

In scikit-learn, `SVC` does not natively produce calibrated probabilities. Setting `probability=True` activates **Platt Scaling** (fitting a logistic regression model over the SVM's decision margins using internal 5-fold cross-validation).

### 4.1 Cost Comparison & Recommendations
- **Training Time Overhead:** Platt scaling increases training time by approximately $3\times$ to $5\times$.
- **When `probability=False` (Default):**
  - Continuous scores are obtained via `decision_function(X)`.
  - For **binary classification**, decision scores (signed Euclidean distance to the hyperplane) are rank-equivalent to probability scores and compute mathematically exact binary ROC-AUC.
  - For **multiclass classification**, unnormalized decision margins cannot be treated as probability distributions; multiclass ROC-AUC is reported as `None` with an informative warning.
- **When `probability=True`:**
  - Calibrated class probabilities are obtained via `predict_proba(X)`.
  - Computes full One-vs-Rest (OvR) multiclass ROC-AUC (`roc_auc_ovr_macro` and `roc_auc_ovr_weighted`).

---

## 5. Binary vs. Multiclass Execution

### 5.1 Binary Classification (e.g. Breast Cancer)
- **Predictions:** 1D array of class labels `(n_samples,)`.
- **Decision Scores:** 1D array `(n_samples,)` representing signed distance to the separating hyperplane. Positive scores indicate prediction of the positive class.
- **Probabilities:** 2D matrix `(n_samples, 2)` summing to 1.0 per sample.

### 5.2 Multiclass Classification (e.g. Wine)
- **Predictions:** 1D array of class labels `(n_samples,)`.
- **Decision Scores:** 2D matrix `(n_samples, n_classes)` representing One-vs-Rest margin distances for each class.
- **Probabilities:** 2D matrix `(n_samples, n_classes)` summing to 1.0 per row across all classes.

---

## 6. Execution Hardware: CPU vs. GPU Consideration

- **Execution Engine:** Standard `scikit-learn` `SVC` is built on LIBSVM, which runs as a multi-threaded C/C++ routine on **CPU hardware**.
- **Hardware Platform:** Although the host machine possesses an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM, CUDA-capable), standard scikit-learn does not invoke CUDA kernels.
- **Academic Transparency:** Model metadata exported via `to_dict()` explicitly records `"execution_device": "CPU"`. No GPU acceleration is claimed or simulated for classical SVM baselines.

---

## 7. Integration With Common Evaluation

`SVMClassifier` outputs integrate seamlessly with `evaluate_predictions()` from `backend/evaluation/metrics.py`:

```python
from backend.classical.svm import SVMClassifier
from backend.evaluation.metrics import evaluate_predictions

# 1. Train SVM with decision scores (fast, no Platt scaling)
clf = SVMClassifier(kernel="linear", C=1.0, probability=False)
clf.fit(X_train_proc, y_train)

y_pred = clf.predict(X_test_proc)
y_scores = clf.decision_function(X_test_proc)

# 2. Evaluate using shared evaluation engine
result = evaluate_predictions(
    y_true=y_test,
    y_pred=y_pred,
    y_score=y_scores,
    class_labels=[0, 1],
    positive_label=1,
)

print("Accuracy:", result.accuracy)
print("Binary ROC-AUC:", result.roc_auc)
```

---

## 8. Executable Example Runner

A comprehensive benchmark demonstration script is located at `examples/run_svm_baseline.py`:

```powershell
.venv\Scripts\python examples/run_svm_baseline.py
```

### Generated Artifacts
Results are persisted as JSON files under `reports/model_results/svm/`:
1. `reports/model_results/svm/breast_cancer_linear_svm.json`
2. `reports/model_results/svm/breast_cancer_rbf_svm.json`
3. `reports/model_results/svm/wine_rbf_svm.json`
