# Quantum VQC Classification Interface

## 1. Overview and Architecture

The Variational Quantum Classifier (VQC) prediction interface in `backend/quantum/vqc_classifier.py` provides a standardized, scikit-learn-compatible classification layer built atop PennyLane's quantum simulation engine (`ParameterizedVQC`).

The system bridges continuous quantum circuit expectation values $\langle Z_i \rangle \in [-1, 1]$ into discrete class predictions and normalized probability distributions $P(y=k \mid x) \in [0, 1]$.

```
                     ┌────────────────────────┐
                     │ Feature Vector x (R^d) │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   Angle Encoding       │
                     │  (RX / RY / RZ / ...)  │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   Variational Layers   │
                     │  (Rotations + CNOTs)   │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │ Pauli-Z Expectations   │
                     │ <Z_0>, ..., <Z_{n-1}>  │
                     └───────────┬────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                 │
[Binary Classification]                          [Multiclass (K >= 3)]
Readout Wire Wire_r                              All Wires <Z_0...Z_{n-1}>
        │                                                 │
        ▼                                                 ▼
Formula: P(y=1) = (1 - <Z_r>) / 2                Linear Head: s = W * <Z> + b
        P(y=0) = (1 + <Z_r>) / 2                          │
        │                                                 ▼
        ▼                                        Numerically Stable Softmax:
Argmax over probabilities                        exp(s_k - max(s)) / sum(...)
        │                                                 │
        └────────────────────────┬────────────────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │ Class Decoding via     │
                     │ LabelEncoder.inverse   │
                     └────────────────────────┘
```

---

## 2. Binary Classification Convention

For binary classification ($K = 2$), quantum measurement on a designated readout wire $r \in \{0, \dots, n-1\}$ provides a single expectation value $z = \langle Z_r \rangle \in [-1, 1]$.

### 2.1 Pauli-Z to Probability Mapping

Under standard quantum computational basis convention:
- Pure state $|0\rangle$: $\langle Z \rangle = +1$
- Pure state $|1\rangle$: $\langle Z \rangle = -1$

To align with standard statistical classification conventions where $y=1$ denotes the positive class:
$$P(y = 1 \mid x) = \frac{1 - z}{2}$$
$$P(y = 0 \mid x) = \frac{1 + z}{2}$$

This ensures:
- When $\langle Z \rangle = -1$ (state $|1\rangle$), $P(y=1) = 1.0$ and $P(y=0) = 0.0$.
- When $\langle Z \rangle = +1$ (state $|0\rangle$), $P(y=1) = 0.0$ and $P(y=0) = 1.0$.
- When $\langle Z \rangle = 0.0$ (equal superposition $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$), $P(y=1) = P(y=0) = 0.5$.

### 2.2 Output Shapes and Class Ordering
- For $N$ samples, `predict_proba(X)` returns shape `(N, 2)`.
- Column 0 corresponds strictly to `classes_[0]` (negative class).
- Column 1 corresponds strictly to `classes_[1]` (positive class).
- Class labels are determined via `LabelEncoder.classes_`, preserving original types (integers or strings).

---

## 3. Multiclass Classification Architecture ($K \ge 3$)

For multiclass problems with $K \ge 3$ classes, a single Pauli-$Z$ measurement is insufficient to represent $K$-way categorical distributions.

### 3.1 Architecture: Quantum Backbone + Trainable Linear Head

1. **Expectation Vector**: All $n$ qubit expectation values are evaluated across the circuit:
   $$h(x) = \left[ \langle Z_0 \rangle, \langle Z_1 \rangle, \dots, \langle Z_{n-1} \rangle \right]^T \in [-1, 1]^n$$
2. **Trainable Classical Head**: A linear transformation maps expectations to class logits:
   $$s(x) = W h(x) + b \in \mathbb{R}^K$$
   where $W \in \mathbb{R}^{K \times n}$ and $b \in \mathbb{R}^K$.
3. **Numerically Stable Softmax**:
   $$P(y = k \mid x) = \frac{\exp\left(s_k(x) - \max_j s_j(x)\right)}{\sum_{j=1}^K \exp\left(s_j(x) - \max_j s_j(x)\right)}$$

### 3.2 Output Shapes
- For $N$ samples and $K$ classes, `predict_proba(X)` returns shape `(N, K)`.
- Probabilities sum to $1.0$ along each row.
- `predict(X)` returns shape `(N,)` with class labels matching the original training labels.

---

## 4. Scikit-Learn Estimator Compatibility

`VQCClassifier` implements standard scikit-learn conventions:

```python
from backend.quantum import VQCClassifier, VQCClassifierConfig, VQCConfig

config = VQCClassifierConfig(
    vqc_config=VQCConfig(n_qubits=4, n_layers=2),
    readout_wire=0,
    random_state=42,
)
clf = VQCClassifier(config=config)

# Fit labels and initialize parameters
clf.fit(X_train, y_train)

# Inference
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Parameter access
weights = clf.get_vqc_weights()
clf.set_vqc_weights(weights)
```

### Parameter Management
- `get_vqc_weights()` / `set_vqc_weights(weights)`: Direct access to circuit rotation weights of shape `(n_layers, n_qubits, 3)`.
- `get_head_parameters()` / `set_head_parameters(weights, bias)`: Direct access to linear head parameters for multiclass classification.
- `get_timing()`: Execution wall-clock timing for fit, circuit evaluations, and predict operations.

---

## 5. Evaluation Framework Integration

`VQCClassifier` outputs are directly consumable by `backend.evaluation.metrics.evaluate_predictions`:

```python
from backend.evaluation.metrics import evaluate_predictions

# Binary or Multiclass evaluation
metrics = evaluate_predictions(
    y_true=y_test,
    y_pred=clf.predict(X_test),
    y_proba=clf.predict_proba(X_test),
    positive_label=clf.classes_[1] if len(clf.classes_) == 2 else None,
)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"Macro F1: {metrics.f1_macro:.4f}")
print(f"ROC-AUC:  {metrics.roc_auc:.4f}")
```

---

## 6. Distinguishing Circuit Expectations vs. Trained Predictions

- **Circuit Expectations ($\langle Z \rangle$)**: Raw physical observables in $[-1, 1]$. They depend purely on circuit parameters $\theta$, input features $x$, and ansatz geometry.
- **Untrained Predictions**: Before parameter optimization (implemented in Day 25), `VQCClassifier` outputs deterministic pseudo-random or zero-initialized baseline predictions.
- **Trained Predictions**: Formed when gradient-based optimizers (e.g., Adam, Nesterov, or COBYLA) minimize cross-entropy loss against ground truth target classes.
