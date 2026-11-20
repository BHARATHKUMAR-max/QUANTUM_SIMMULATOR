# Classical Multi-Layer Perceptron (MLP) Classifier

**Project:** QML Benchmark Studio  
**Layer:** Classical ML Engine (`backend/classical/`)  
**Status:** Implemented & Verified (353/353 Tests Passing)  
**Date:** November 20, 2026

---

## 1. Overview & Architectural Role

The **Multi-Layer Perceptron (MLP)** is a fundamental feed-forward artificial neural network architecture. In **QML Benchmark Studio**, the `MLPClassifier` (`backend/classical/mlp.py`) serves as the primary classical deep learning benchmark.

### 1.1 Comparison with Variational Quantum Classifiers (VQC)
- **Conceptual Parallel:** Both MLP and VQC optimize parameterized continuous functions via gradient descent to learn non-linear decision boundaries.
- **Fundamental Distinction:**
  - Classical MLPs apply linear affine transformations followed by fixed non-linear activation functions ($\sigma(Wx + b)$) across Euclidean vector spaces.
  - VQCs apply unitary matrix rotations ($U(\theta)$) over quantum statevectors in exponentially high-dimensional Hilbert spaces ($\mathbb{C}^{2^n}$), with non-linearity arising from quantum state projection and measurement operators.
  - While MLP is an essential classical benchmark, it is **not mathematically equivalent** to a parameterized quantum circuit.

---

## 2. Supported Configurations & Hyperparameters

The `MLPClassifier` wraps scikit-learn's `sklearn.neural_network.MLPClassifier` with input validation, profiling metrics, and convergence diagnostics.

### 2.1 Hyperparameter Specifications

| Parameter | Type | Default | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `hidden_layer_sizes` | `tuple` or `list` | `(64, 32)` | Neurons per hidden layer. Supports arbitrary positive integer tuples (e.g. `(32,)`, `(64,)`, `(64, 32)`, `(128, 64)`). |
| `activation` | `str` | `"relu"` | Activation function for hidden layers: `"relu"`, `"tanh"`, `"logistic"`, `"identity"`. |
| `solver` | `str` | `"adam"` | Optimization algorithm: `"adam"` (stochastic gradient-based optimizer), `"sgd"`, `"lbfgs"`. |
| `alpha` | `float` | `0.0001` | L2 regularization penalty parameter ($\alpha \ge 0$). Penalizes large weight magnitudes. |
| `learning_rate_init` | `float` | `0.001` | Initial step size for parameter updates ($> 0$). |
| `max_iter` | `int` | `200` | Maximum number of training epochs ($> 0$). |
| `random_state` | `int` or `None` | `42` | Seed for deterministic weight initialization and batch shuffling. |
| `early_stopping` | `bool` | `False` | Whether to terminate training early when validation score stagnates. |
| `validation_fraction`| `float` | `0.1` | Fraction of training samples reserved for early stopping validation ($0 < f < 1$). |
| `n_iter_no_change` | `int` | `10` | Number of epochs without cost improvement before stopping ($> 0$). |

---

## 3. Preprocessing & Feature Scaling Invariants

Feed-forward neural networks optimize weights using backpropagation and gradient descent:
- **Gradient Sensitivity:** Unscaled features create distorted, elongated loss surfaces, leading to slow training, vanishing gradients, or premature divergence.
- **Pipeline Requirement:** `MLPClassifier` must always receive features scaled with `StandardScaler` (zero mean, unit variance) fitted **exclusively on training data** via `backend.data.preprocessing`.
- **Leakage Prevention:** Scalers and imputers are strictly quarantined from held-out test data. Test features are transformed solely using parameters derived from training partitions.

---

## 4. Convergence Diagnostics & Warning Capture

A known characteristic of iterative stochastic optimizers (such as Adam or SGD) is failure to reach global convergence within `max_iter`:

### 4.1 Diagnostic Principles
1. **No Silent Suppression:** `ConvergenceWarning` is captured via Python's `warnings` subsystem, logged in the model's metadata, and re-emitted for caller visibility.
2. **Honest Status Reporting:** Reaching `max_iter` without meeting convergence criteria does not imply success. The model explicitly records:
   - `converged_`: Boolean flag (`True` if converged, `False` if `ConvergenceWarning` occurred).
   - `convergence_warning_`: Full diagnostic message string.
   - `n_iter_`: Actual iterations completed.
   - `loss_`: Final loss value.
   - `best_loss_`: Best loss observed during training.
3. **Availability Despite Non-Convergence:** Even if the iteration ceiling is hit, the model still provides its best-effort weights for inference and evaluation without throwing unhandled exceptions.

---

## 5. Execution Hardware: CPU vs. GPU Transparency

- **Execution Engine:** Standard `scikit-learn` `MLPClassifier` is implemented in Cython/C++ and executes entirely on the **CPU**.
- **Hardware Context:** Although the development machine features an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM, CUDA-capable), standard scikit-learn does not invoke CUDA kernels.
- **Transparency Policy:** Model metadata exported via `to_dict()` explicitly records `"execution_device": "CPU"`. No GPU acceleration is claimed.

---

## 6. Integration With Common Evaluation

`MLPClassifier` supports native probability outputs via `predict_proba()`. These integrate directly with `evaluate_predictions()` from `backend/evaluation/metrics.py`:

```python
from backend.classical.mlp import MLPClassifier
from backend.evaluation.metrics import evaluate_predictions

# 1. Train MLP model
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
clf.fit(X_train_proc, y_train)

# 2. Generate predictions and probability distributions
y_pred = clf.predict(X_test_proc)
y_proba = clf.predict_proba(X_test_proc)

# 3. Evaluate using shared evaluation engine
result = evaluate_predictions(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    class_labels=[0, 1],
    positive_label=1,
)

print("Accuracy:", result.accuracy)
print("Binary ROC-AUC:", result.roc_auc)
```

---

## 7. Executable Benchmark Runner

A standalone benchmark script is available at `examples/run_mlp_baseline.py`:

```powershell
.venv\Scripts\python examples/run_mlp_baseline.py
```

### Persisted Benchmark Results
Outputs are saved in JSON format under `reports/model_results/mlp/`:
1. `reports/model_results/mlp/breast_cancer_mlp_32.json`
2. `reports/model_results/mlp/breast_cancer_mlp_64_32.json`
3. `reports/model_results/mlp/wine_mlp_64_32.json`

Each file records the model hyperparameters, convergence diagnostics, execution timings, confusion matrices, and standardized classification metrics.
