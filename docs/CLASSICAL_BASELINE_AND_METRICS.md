# Classical Baseline & Common Evaluation Metrics

**Project:** QML Benchmark Studio  
**Layer:** Classical ML Engine (`backend/classical/`) & Evaluation Subsystem (`backend/evaluation/`)  
**Status:** Implemented & Verified (312/312 Tests Passing)  
**Date:** November 19, 2026

---

## 1. Overview & Architectural Role

In benchmarking quantum machine learning against classical paradigms, establish an empirical baseline before evaluating complex models. The **Dummy Baseline** establishes the zero-knowledge performance floor, representing a classifier that outputs the most frequent training label for all inputs.

Concurrently, the **Common Evaluation Module** (`backend/evaluation/`) provides an impartial, model-agnostic evaluation framework. Crucially:
- The evaluation subsystem operates strictly on `(y_true, y_pred)` and optional probabilities `(y_proba)`.
- It is decoupled from estimator implementations.
- Classical algorithms (`DummyBaseline`, `SVM`, `RandomForest`, `MLP`) and quantum models (`VQC`, `Quantum Kernel SVM`) emit identical evaluation schemas without requiring specialized metric routines.

---

## 2. Dummy Baseline Implementation

The `DummyBaseline` class (`backend/classical/dummy_baseline.py`) wraps `sklearn.dummy.DummyClassifier(strategy="most_frequent")`.

### 2.1 Interface & Lifecycle

```python
class DummyBaseline:
    def __init__(self, strategy: str = "most_frequent", random_state: Optional[int] = None):
        ...

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> "DummyBaseline":
        """Fits strictly on training features and targets. Learns the majority class."""
        ...

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns a 1D array predicting the training majority class for all rows."""
        ...

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Returns 1.0 probability for the majority class and 0.0 for others."""
        ...

    @property
    def classes_(self) -> np.ndarray:
        """Ordered class labels discovered during fit."""
        ...

    @property
    def most_frequent_class_(self) -> Any:
        """The specific class label identified as most frequent in training."""
        ...
```

### 2.2 Invariants & Leakage Safety
1. **Quarantine of Held-Out Data:** Fitting executes strictly on `(X_train, y_train)`. Test or validation features are never inspected during fitting.
2. **Type Preservation:** Preserves original label types (integers, strings, or booleans) without forced numerical coercion.
3. **Deterministic Probabilities:** Emits a 2D matrix of shape `(n_samples, n_classes)` where the column for `most_frequent_class_` is `1.0` and all other columns are `0.0`.
4. **Input Validation:** Rejects empty or mismatched datasets with descriptive `ValueError`s; raises `sklearn.exceptions.NotFittedError` if prediction methods or properties are accessed prior to fitting.

---

## 3. Common Evaluation Metrics Subsystem

The evaluation subsystem (`backend/evaluation/metrics.py`) provides the top-level evaluation function:

```python
def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    y_proba: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
    class_labels: Optional[List[Union[str, int]]] = None,
    positive_label: Optional[Union[str, int]] = None,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationMetricsResult:
```

### 3.1 Classification Metrics Suite

| Metric | Scope | Mathematical Formulation / Implementation | Convention / Averaging |
| :--- | :--- | :--- | :--- |
| **Accuracy** | Global | $\frac{\text{TP} + \text{TN}}{\text{Total Samples}}$ | Fraction of correct predictions over total. |
| **Balanced Accuracy** | Global | $\frac{1}{K}\sum_{k=1}^K \text{Recall}_k$ | Unweighted average of per-class recalls. Resilient to class imbalance. |
| **Precision** | Binary & Multi | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Binary: Positive class precision. Multi: Macro and weighted precision, plus per-class dict. |
| **Recall** | Binary & Multi | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Binary: Positive class recall. Multi: Macro and weighted recall, plus per-class dict. |
| **Macro F1** | Global | $\frac{1}{K}\sum_{k=1}^K \text{F1}_k$ | Unweighted harmonic mean across classes. Treats all classes with equal weight. |
| **Weighted F1** | Global | $\sum_{k=1}^K \frac{N_k}{N}\text{F1}_k$ | Class-size weighted harmonic mean. |
| **Positive F1** | Binary | $2 \cdot \frac{\text{Prec}_{\text{pos}} \cdot \text{Rec}_{\text{pos}}}{\text{Prec}_{\text{pos}} + \text{Rec}_{\text{pos}}}$ | Computed for explicit positive label in binary tasks. |
| **Confusion Matrix** | Global | $C_{i, j} = |\{s \in S \mid y_{\text{true}}(s) = i \land y_{\text{pred}}(s) = j\}|$ | Consistent row/column order matching `class_labels`. |
| **ROC-AUC** | Global | Binary / Multiclass OvR | Evaluated from class probabilities. Never fabricated. |

---

## 4. Binary vs. Multiclass Evaluation Conventions

### 4.1 Binary Classification Convention
- **Positive Label Resolution:**
  - If explicitly provided in `config` or parameter, verified against `class_labels`.
  - If omitted, resolves through canonical heuristics: `{0, 1} \to 1`, `{-1, 1} \to 1`, `{False, True} \to True`, or recognized string tokens (`"malignant"`, `"positive"`, `"yes"`, `"1"`).
  - Emits `precision_positive`, `recall_positive`, and `f1_positive`.
- **ROC-AUC:**
  - Evaluated against positive-class probabilities.
  - If 2D probability matrix `(N, 2)` is provided, extracts the specific column matching `positive_label`.

### 4.2 Multiclass Classification Convention
- **Task Detection:** Auto-detected when `len(class_labels) > 2`.
- **Positive Label:** Set to `None`; binary-specific metrics evaluate to `None`.
- **Averaging Strategies:**
  - Computes both **Macro** (unweighted mean) and **Weighted** (sample-weighted mean) for precision, recall, and F1.
  - Generates per-class breakdown dictionaries: `precision_per_class`, `recall_per_class`, and `f1_per_class`.

---

## 5. ROC-AUC Requirements & Boundary Handling

ROC-AUC measures class separability across all possible decision thresholds:

1. **Probability Requirement:**
   - ROC-AUC requires predicted probability distributions (`y_proba`) or continuous decision scores.
   - Predicted discrete labels alone are insufficient. If `y_proba` is `None`, `roc_auc` evaluates to `None` and an explicit diagnostic warning is appended.
2. **Multiclass One-vs-Rest (OvR):**
   - Each class is treated as positive in turn against all remaining classes.
   - Computes both `roc_auc_ovr_macro` (unweighted average of valid per-class OvR scores) and `roc_auc_ovr_weighted` (weighted by class support).
3. **Missing Classes & Single-Class Ground Truth:**
   - If only one unique class is present in test ground truth `y_true`, ROC-AUC is mathematically undefined. Returns `None` with diagnostic warning: `"ROC-AUC undefined: only one class is present in ground truth y_true."`
   - If a class is absent from the test split, its individual OvR score evaluates to `None`. Macro OvR calculates the mean of present classes without crashing.
4. **No Score Fabrication:**
   - The system **never** substitutes a fabricated default value (e.g. 0.5 or 0.0) when probabilities are missing or mathematically invalid.

---

## 6. Undefined Metrics, Degenerate Models & Zero Division

A `DummyClassifier` with `strategy="most_frequent"` predicts only the majority class. Consequently:
- Predictions for all other classes are zero ($\text{TP} = 0, \text{FP} = 0$).
- Precision denominator ($\text{TP} + \text{FP} = 0$) encounters division by zero.

### Safeguards Implemented:
1. **Configurable Zero Division:**
   - `zero_division=0.0` (default) ensures division-by-zero yields `0.0` rather than crashing with unhandled runtime errors.
2. **Dimension-Preserving Confusion Matrices:**
   - By passing explicit `labels=class_labels` to `confusion_matrix()`, the output matrix maintains full $K \times K$ dimensions.
   - Columns corresponding to unpredicted classes contain zeros.
3. **Diagnostic Warning System:**
   - Degenerate prediction distributions append explicit warnings to `EvaluationMetricsResult.warnings`:
     ```text
     "Class(es) [0] were never predicted by model (unique predicted classes: [1])."
     ```

---

## 7. Result Schema & Serialization

The evaluation result is encapsulated in `EvaluationMetricsResult` (`backend/evaluation/schemas.py`).

### 7.1 JSON Serializability
- Implements `to_dict()` and `to_json()`.
- Recursively sanitizes NumPy integer types (`np.int64`), floating types (`np.float64`), and boolean types (`np.bool_`) into native Python primitives.
- Replaces non-finite values (`NaN`, `Inf`) with `None` (`null` in JSON).

### 7.2 Experiment Schema Compatibility
Provides `to_experiment_metrics_dict()`, mapping directly to `docs/schemas/experiment_result.schema.json`:

```json
{
  "accuracy": 0.631579,
  "precision_macro": 0.315789,
  "recall_macro": 0.5,
  "f1_macro": 0.387097,
  "f1_weighted": 0.488965,
  "balanced_accuracy": 0.5,
  "roc_auc": 0.5
}
```

---

## 8. Integration With Classical & Quantum Models

Future models seamlessly use the common evaluation interface:

```python
# 1. Classical Model (e.g. SVM or Random Forest)
model = RandomForestClassifier(...)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

metrics_rf = evaluate_predictions(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    class_labels=bundle.target_classes,
)

# 2. Quantum Model (e.g. VQC via PennyLane)
vqc = VariationalQuantumClassifier(...)
vqc.fit(X_train_reduced, y_train)
y_pred_q = vqc.predict(X_test_reduced)
y_proba_q = vqc.predict_proba(X_test_reduced)

metrics_vqc = evaluate_predictions(
    y_true=y_test,
    y_pred=y_pred_q,
    y_proba=y_proba_q,
    class_labels=bundle.target_classes,
)

# Both emit identical schema structures for comparative analysis
assert set(metrics_rf.to_dict().keys()) == set(metrics_vqc.to_dict().keys())
```
