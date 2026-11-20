# Comparison Modes & Compatibility Specification — QML Benchmark Studio

> **Status:** Formal Specification & Engineering Guide — Day 11, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR  
> **Modules:** `backend/data/comparison_modes.py` · `backend/data/schemas.py`  
> **Scope Reference:** [docs/MVP_SCOPE.md](MVP_SCOPE.md) · [docs/ARCHITECTURE.md](ARCHITECTURE.md) · [docs/DATA_FLOW.md](DATA_FLOW.md)

---

## 1. Overview & Core Philosophy

The primary objective of **QML Benchmark Studio** is benchmarking Quantum Machine Learning (QML) models against Classical Machine Learning models on equal, scientifically defensible algorithmic footing.

In empirical research, claiming an algorithmic advantage or parity is meaningless if:
- One model family is evaluated on 30 features while the other is evaluated on 4 features.
- Two model families are given representations with equal dimensionality ($k=4$) but different feature identities (e.g., PCA components vs. raw features).
- Excess features are silently truncated (e.g. slicing `X[:, :n_qubits]`) without explicit mathematical reduction.
- Test or validation samples leak into feature selection or dimensionality reduction.

To provide both rigorous scientific comparability and real-world practical insight, QML Benchmark Studio defines **three formal comparison modes**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 COMPARISON MODES                                       │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Mode A: Matched Reduced       │ Mode B: Quantum-Limited       │ Mode C: Best Practical │
│ (PRIMARY SCIENTIFIC MODE)     │ (HARDWARE-CONSTRAINED)        │ (UNCONSTRAINED)        │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ • Identical feature columns   │ • Qubit count determines      │ • Classical model uses │
│ • Identical column ordering   │   permitted feature count     │   full or best feature │
│ • Identical sample rows       │ • Models get identical        │   representation       │
│ • Identical transformations   │   quantum-compatible input    │ • Quantum model uses   │
│ • is_matched_input = True     │ • is_matched_input = True     │   quantum-compatible   │
│ • Validates pure model        │ • Tests classical models      │ • is_matched_input =   │
│   architecture differences    │   under quantum bottleneck    │   False (Practical)    │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

## 2. Mode A — Matched Reduced Features (PRIMARY SCIENTIFIC COMPARISON)

### 2.1 Principle

Mode A is the **primary baseline mode** for all formal scientific comparisons in this project. Both classical and quantum models receive the **exact same feature representation**:
- Same number of features.
- Same feature identities (exact matching column names).
- Same feature order.
- Same numerical values.

```text
ProcessedDatasetResult (e.g. 30 features)
               │
               ▼
   fit_reduction (X_train only)
   [PCA / Mutual Info / ANOVA / Correlation Filter]
               │
               ▼
       X_train_reduced (e.g. 6 features)
               │
       ┌───────┴───────┐
       ▼               ▼
Classical Model  Quantum Model
 (6 features)     (6 qubits)
```

### 2.2 Verification Contract (`verify_matched_inputs`)

A comparison under Mode A is strictly invalid if:
1. Shape mismatch exists ($X_{\text{classical}}.\text{shape} \ne X_{\text{quantum}}.\text{shape}$).
2. Column names differ (even if shapes match, e.g., `["PC1", "PC2"]` vs. `["feat_a", "feat_b"]`).
3. Column ordering differs (e.g., `["PC1", "PC2"]` vs. `["PC2", "PC1"]`).
4. Matrix values differ ($\|X_{\text{classical}} - X_{\text{quantum}}\|_\infty > 10^{-9}$).

---

## 3. Mode B — Quantum-Limited Features

### 3.1 Principle

Near-term quantum simulators and NISQ devices are constrained by qubit counts. Mode B investigates:
> *"How well do classical machine learning baselines perform when forced into the exact same hardware bottleneck as the quantum model?"*

In Mode B:
- `qubit_count` ($n_{\text{qubits}} \in [2, 10]$) is **mandatory** and dictates the allowed input dimensionality $k = n_{\text{qubits}}$.
- The candidate dataset must be reduced to exactly $k$ features via explicit reduction (PCA, Mutual Information, ANOVA).
- Both classical and quantum models receive this identical $k$-dimensional representation.
- `is_matched_input = True`.

---

## 4. Mode C — Best Practical Configuration

### 4.1 Principle

In industrial application, a data scientist would not arbitrarily compress tabular features from 30 down to 4 for a Random Forest or SVM just because an experimental quantum circuit requires 4 qubits.

Mode C enables **practical benchmarking**:
- The classical model uses its best practical configuration (e.g. all 30 features, or correlation-filtered features).
- The quantum model uses a quantum-compatible configuration (e.g. 4 PCA components on 4 qubits).
- The result is explicitly labeled:
  `is_matched_input = False`
- Results in Mode C reflect real-world deployment trade-offs rather than controlled theoretical fairness.

---

## 5. Feature-to-Qubit Compatibility Rules

Quantum circuits simulated on PennyLane's `default.qubit` adhere to strict physical boundaries:

| Parameter / Requirement | Specification | Enforcement |
|---|---|---|
| **Supported Qubit Range** | $2 \le n_{\text{qubits}} \le 10$ | Values outside $[2, 10]$ raise `ValueError` |
| **Angle Encoding** (`"angle"`) | $1\text{ feature} \leftrightarrow 1\text{ qubit}$ ($n_{\text{features}} == n_{\text{qubits}}$) | Exact dimensional match required |
| **ZZ Feature Map** (`"zz_feature_map"`) | $1\text{ feature} \leftrightarrow 1\text{ qubit}$ ($n_{\text{features}} == n_{\text{qubits}}$) | Exact dimensional match required |
| **Excess Features** | $n_{\text{features}} > n_{\text{qubits}}$ | **Never silently truncated.** Explicit reduction required. |
| **Fewer Features** | $n_{\text{features}} < n_{\text{qubits}}$ | **Never silently padded or duplicated.** Qubit count must match. |

### Feature-to-Qubit Mapping

When compatible, a deterministic dictionary mapping each physical qubit to its input feature is generated:
```json
{
  "q0": "PC1",
  "q1": "PC2",
  "q2": "PC3",
  "q3": "PC4"
}
```

---

## 6. Critical Project Invariant: No Silent Feature Loss

To preserve experimental integrity across all 12 backlog epics:
1. **No Implicit Truncation**: Writing `X[:, :n_qubits]` or silently dropping trailing features when $n_{\text{features}} > n_{\text{qubits}}$ is prohibited.
2. **No Implicit Padding**: Fabricating zero-columns, cloning features, or synthesizing dummy data when $n_{\text{features}} < n_{\text{qubits}}$ is prohibited.
3. If an input dimensionality mismatch occurs, the execution stops with a clear, structured `QubitCompatibilityResult` containing explicit diagnostic errors.

---

## 7. Leakage Prevention in Comparison Modes

All transformation pipelines (PCA, Mutual Information, ANOVA, Correlation Filter) adhere to the established non-leakage protocol:
- **Fit Exclusively on $X_{\text{train}}$**: Estimators learn statistics ($\mu, \sigma$, covariance, F-scores, mutual information) purely from training instances.
- **Transform-Only on $X_{\text{test}}$**: Held-out test partitions are projected using pre-fitted transformers without modification.
- **Cross-Validation Isolation**: `prepare_cv_fold_comparison` fits fresh transformers inside each fold's training slice (`fold.train_indices`). No information traverses fold boundaries.

---

## 8. Data Re-Uploading (Future Extension Note)

> [!NOTE]
> **Data Re-Uploading Strategy (Non-MVP Roadmap)**  
> In theoretical quantum machine learning, *data re-uploading* (Pérez-Salinas et al., 2020) interleaves parameterised rotation layers with repeated feature-encoding layers. This allows a circuit with $N$ qubits to process $M > N$ features across sequential rotational layers.  
> **Status in MVP:** Data re-uploading is **not implemented** in the current MVP scope. The studio enforces the primary 1-feature-per-qubit mapping for angle encoding and the standard ZZ feature map. Data re-uploading is designated for post-MVP exploration.

---

## 9. Usage Examples

### 9.1 Mode A (Matched Reduced Features with PCA)

```python
from backend.data.comparison_modes import prepare_comparison_mode
from backend.data.schemas import ComparisonModeConfig, PCAConfig

cfg = ComparisonModeConfig(
    mode="matched_reduced",
    reduction_method="pca",
    pca_config=PCAConfig(n_components=4),
    qubit_count=4,
)

result = prepare_comparison_mode(X_train_proc, y_train, X_test_proc, y_test, config=cfg)

assert result.is_matched_input is True
print("Classical features:", result.classical_metadata.feature_names)  # ['PC1', 'PC2', 'PC3', 'PC4']
print("Quantum features  :", result.quantum_metadata.feature_names)    # ['PC1', 'PC2', 'PC3', 'PC4']
```

### 9.2 Mode B (Quantum-Limited Features)

```python
cfg_b = ComparisonModeConfig(
    mode="quantum_limited",
    qubit_count=6,
    reduction_method="anova",  # Automatically selects top 6 features matching qubits
)

result_b = prepare_comparison_mode(X_train_proc, y_train, X_test_proc, y_test, config=cfg_b)
assert result_b.X_train_classical.shape[1] == 6
assert result_b.X_train_quantum.shape[1] == 6
```

### 9.3 Mode C (Best Practical Configuration)

```python
cfg_c = ComparisonModeConfig(
    mode="best_practical",
    classical_reduction_method="none",  # Full 30 features for classical
    quantum_reduction_method="pca",      # 4 PCA components for quantum
    qubit_count=4,
)

result_c = prepare_comparison_mode(X_train_proc, y_train, X_test_proc, y_test, config=cfg_c)
assert result_c.is_matched_input is False  # Explicitly unmatched practical comparison
assert result_c.X_train_classical.shape[1] == 30
assert result_c.X_train_quantum.shape[1] == 4
```
