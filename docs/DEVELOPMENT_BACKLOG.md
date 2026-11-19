# Development Backlog — QML Benchmark Studio

> **Status:** Frozen — Day 2, 2026-11-19
> **Project:** 6th-Semester Research Project — 8-week schedule
> **Scope reference:** [docs/MVP_SCOPE.md](./MVP_SCOPE.md)

This backlog translates the frozen MVP scope into a prioritized, actionable task list. Each task has a unique ID, priority, description, dependencies, and acceptance criteria. Tasks are organized into Epics covering the full eight-week development plan.

**Priority levels:**

| Level | Meaning |
|---|---|
| P0 | Essential foundation — blocks other work. Must be done first. |
| P1 | Core MVP functionality — required for a working product. |
| P2 | Important supporting functionality — required for quality and correctness. |
| P3 | Future improvements — deferred beyond MVP. |

**Task ID format:** `EPIC-##` (e.g., `F01-01` = Epic 1, task 01)

---

## Table of Contents

- [EPIC 1 — Project Foundation and Architecture](#epic-1--project-foundation-and-architecture)
- [EPIC 2 — Dataset Management and Validation](#epic-2--dataset-management-and-validation)
- [EPIC 3 — Preprocessing and Feature Selection](#epic-3--preprocessing-and-feature-selection)
- [EPIC 4 — Classical ML Implementation](#epic-4--classical-ml-implementation)
- [EPIC 5 — Quantum VQC Implementation](#epic-5--quantum-vqc-implementation)
- [EPIC 6 — Quantum Kernel SVM Implementation](#epic-6--quantum-kernel-svm-implementation)
- [EPIC 7 — Benchmarking and Evaluation](#epic-7--benchmarking-and-evaluation)
- [EPIC 8 — Trainability and Barren Plateau Analysis](#epic-8--trainability-and-barren-plateau-analysis)
- [EPIC 9 — Experiment Management and Reproducibility](#epic-9--experiment-management-and-reproducibility)
- [EPIC 10 — FastAPI Backend](#epic-10--fastapi-backend)
- [EPIC 11 — React Frontend](#epic-11--react-frontend)
- [EPIC 12 — Integration, Testing, and Reporting](#epic-12--integration-testing-and-reporting)
- [Dependency Summary](#dependency-summary)
- [Weekly Schedule Mapping](#weekly-schedule-mapping)

---

## EPIC 1 — Project Foundation and Architecture

> Goal: Establish the structural and architectural foundations that all other epics depend on.

---

### F01-01 — Project configuration system

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 1 |
| **Description** | Define a configuration schema (YAML or JSON) for experiment parameters: dataset choice, model type, feature selection mode, qubit count, circuit depth, optimizer, random seeds, and output paths. The configuration must be loadable, validatable, and serializable from Python. |
| **Dependencies** | None |
| **Acceptance criteria** | A config file can be loaded from `configs/`; invalid fields raise clear validation errors; the config object is serializable back to disk; a default config is provided. |
| **Deliverable** | `configs/default_config.yaml`, `backend/config_schema.py` or equivalent |

---

### F01-02 — Logging infrastructure

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 1 |
| **Description** | Set up structured logging across the backend using Python's `logging` module. Log experiment steps, timing, errors, and warnings. Each experiment run should produce a timestamped log file in `reports/logs/`. |
| **Dependencies** | F01-01 |
| **Acceptance criteria** | Backend logs to both console and file; log level is configurable; experiment runs produce distinct log files; no uncaught exceptions silently swallowed. |
| **Deliverable** | `backend/logger.py`, log output in `reports/logs/` |

---

### F01-03 — Shared utilities module

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 1 |
| **Description** | Implement shared utility functions used across the backend: timing context manager, random seed setter, results serializer (JSON/CSV), and directory setup helpers. |
| **Dependencies** | None |
| **Acceptance criteria** | Timing wrapper measures wall-clock time accurately; seed setter works for numpy, random, and sklearn; results serializer writes valid JSON/CSV; unit tests pass. |
| **Deliverable** | `backend/utils.py`, `tests/test_utils.py` |

---

### F01-04 — Backend package structure

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 1 |
| **Description** | Add `__init__.py` files to all backend subdirectories to make them proper Python packages. Verify all imports work correctly from the project root. |
| **Dependencies** | None |
| **Acceptance criteria** | `import backend.data`, `import backend.classical`, etc., all succeed from the project root with the venv active; no circular imports. |
| **Deliverable** | `__init__.py` files in all `backend/` subdirectories |

---

## EPIC 2 — Dataset Management and Validation

> Goal: Load, validate, and prepare datasets for downstream processing.

---

### F02-01 — Built-in dataset loader

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 2 |
| **Description** | Implement a dataset loader that returns a standardized `DatasetBundle` object (features DataFrame, target Series, feature names, class names, dataset metadata) for the two built-in datasets: Breast Cancer Wisconsin and Wine. |
| **Dependencies** | F01-03, F01-04 |
| **Acceptance criteria** | `load_dataset("breast_cancer")` and `load_dataset("wine")` both return correctly typed DatasetBundles; feature and target shapes match scikit-learn documentation; class names are present; unit tests pass for both datasets. |
| **Deliverable** | `backend/data/loader.py`, `tests/test_loader.py` |

---

### F02-02 — CSV upload and parsing

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 2 |
| **Description** | Implement CSV parsing that accepts a file path or file-like object, returns a pandas DataFrame, and allows the user to specify the target column name. Handle encoding issues, header detection, and mixed-type columns gracefully. |
| **Dependencies** | F02-01 |
| **Acceptance criteria** | Valid CSV files are parsed correctly; target column is separated from features; malformed files raise descriptive errors; encoding errors are caught; unit tests with sample CSVs pass. |
| **Deliverable** | `backend/data/csv_loader.py`, sample test CSV files, `tests/test_csv_loader.py` |

---

### F02-03 — Dataset validation

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 2 |
| **Description** | Implement a validation module that inspects a DatasetBundle and produces a structured validation report including: missing value counts per column, duplicate row count, constant feature detection, class distribution, and class imbalance warning (if minority class < 20% of samples). |
| **Dependencies** | F02-01, F02-02 |
| **Acceptance criteria** | Validation report is a structured object (not just a print statement); all six checks are implemented; warnings are classified as errors or warnings; unit tests cover edge cases (empty columns, all-constant columns, heavily imbalanced classes). |
| **Deliverable** | `backend/data/validator.py`, `tests/test_validator.py` |

---

### F02-04 — Stratified train/test split

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 2 |
| **Description** | Implement a splitting function that produces stratified 80/20 train/test splits. The function must accept a random seed parameter and return consistently reproducible splits. Must validate that stratified splitting is applicable (target must be categorical). |
| **Dependencies** | F02-01 |
| **Acceptance criteria** | Splits are reproducible given the same seed; class proportions are maintained within ±2% of original distribution; function raises a clear error if the target cannot be stratified; unit tests for both datasets pass. |
| **Deliverable** | `backend/data/splitter.py`, `tests/test_splitter.py` |

---

## EPIC 3 — Preprocessing and Feature Selection

> Goal: Implement all preprocessing and feature selection operations in a leakage-free pipeline.

---

### F03-01 — Preprocessing pipeline (standardization and imputation)

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 2–3 |
| **Description** | Implement a preprocessing pipeline using scikit-learn Pipeline objects. The pipeline must include StandardScaler and SimpleImputer (configurable strategy). The pipeline must be fitted on training data only and then applied to both train and test sets. The fitted pipeline must be serializable. |
| **Dependencies** | F02-04 |
| **Acceptance criteria** | Pipeline is fit on X_train only; X_test is transformed (not fitted) using the fitted pipeline; scaling parameters are logged; unit tests confirm no test-time leakage (verify scaler parameters match training distribution); pipeline serializes and deserializes correctly. |
| **Deliverable** | `backend/data/preprocessor.py`, `tests/test_preprocessor.py` |

---

### F03-02 — Mutual Information feature selection

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Implement SelectKBest with mutual_info_classif, fitted on training data only. Accept k (number of features to select) as a parameter. Return selected feature names and MI scores. |
| **Dependencies** | F03-01 |
| **Acceptance criteria** | Selector is fitted on X_train only; feature names are correctly tracked; selected features and MI scores are logged; k can be set to match qubit count for Matched Reduced mode; unit tests pass on both built-in datasets. |
| **Deliverable** | `backend/data/feature_selection.py` (MI section), `tests/test_feature_selection.py` |

---

### F03-03 — ANOVA F-test feature selection

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Implement SelectKBest with f_classif, fitted on training data only. Accept k and return selected feature names and F-scores. Share implementation infrastructure with F03-02. |
| **Dependencies** | F03-01 |
| **Acceptance criteria** | Same leakage-free requirements as F03-02; F-scores are logged; selector can be swapped with MI selector via configuration parameter; unit tests pass. |
| **Deliverable** | `backend/data/feature_selection.py` (ANOVA section) |

---

### F03-04 — Correlation-based feature filtering

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Implement a correlation filter that computes the absolute Pearson correlation matrix on training data and removes features with pairwise correlation above a configurable threshold (default: 0.95). Keep the lower-indexed feature of each correlated pair. |
| **Dependencies** | F03-01 |
| **Acceptance criteria** | Correlation matrix is computed on X_train only; dropped features are logged by name; threshold is configurable; unit test with a manually constructed correlated dataset passes. |
| **Deliverable** | `backend/data/feature_selection.py` (correlation section) |

---

### F03-05 — PCA dimensionality reduction

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Implement PCA as an optional preprocessing step, fitted on training data only. Accept n_components as a parameter. Log explained variance ratios. PCA must be applicable after scaling. |
| **Dependencies** | F03-01 |
| **Acceptance criteria** | PCA is fitted on X_train only; explained variance ratios are reported; n_components is configurable; PCA can be used in Matched Reduced mode to match qubit count; unit tests pass. |
| **Deliverable** | `backend/data/preprocessor.py` (PCA extension) |

---

### F03-06 — Comparison mode coordinator

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Implement a coordinator that takes the configured comparison mode (Original / Matched Reduced / Quantum-limited) and produces the correct input tensors for classical and quantum models. When reducing, it must log the reduction method and feature count explicitly. Never silently reduce features. |
| **Dependencies** | F03-02, F03-03, F03-04, F03-05 |
| **Acceptance criteria** | All three modes produce correct shapes; Matched Reduced mode logs the method and resulting feature count; any feature reduction is explicitly recorded in the experiment output; unit tests for each mode pass. |
| **Deliverable** | `backend/data/comparison_modes.py`, `tests/test_comparison_modes.py` |

---

## EPIC 4 — Classical ML Implementation

> Goal: Implement all four classical ML models with configurable hyperparameters.

---

### F04-01 — DummyClassifier baseline

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Wrap scikit-learn's DummyClassifier with the project's model interface. Configurable strategy (most_frequent, stratified, uniform). Measure training time and inference time using the shared timing utility. |
| **Dependencies** | F01-03, F03-06 |
| **Acceptance criteria** | DummyClassifier trains and predicts on both datasets; timing is recorded; model interface is consistent with other classical models; unit tests pass. |
| **Deliverable** | `backend/classical/dummy.py`, `tests/test_classical_models.py` |

---

### F04-02 — Support Vector Machine

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Wrap scikit-learn's SVC with the project's model interface. Configurable kernel (linear, rbf), C, and gamma. Enable probability estimates (probability=True) for ROC-AUC computation. Measure training and inference time. |
| **Dependencies** | F04-01 |
| **Acceptance criteria** | SVM trains and predicts correctly on binary and multiclass tasks; probability estimates are available; timing is recorded; hyperparameters are passed through config; unit tests pass. |
| **Deliverable** | `backend/classical/svm.py` |

---

### F04-03 — Multi-Layer Perceptron

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Wrap scikit-learn's MLPClassifier. Configurable hidden_layer_sizes, activation, learning_rate_init, max_iter. Measure training and inference time. |
| **Dependencies** | F04-01 |
| **Acceptance criteria** | MLP trains and predicts correctly on binary and multiclass tasks; timing is recorded; unit tests pass. |
| **Deliverable** | `backend/classical/mlp.py` |

---

### F04-04 — Random Forest

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 3 |
| **Description** | Wrap scikit-learn's RandomForestClassifier. Configurable n_estimators, max_depth, min_samples_split, random_state. Measure training and inference time. |
| **Dependencies** | F04-01 |
| **Acceptance criteria** | Random Forest trains and predicts correctly on binary and multiclass tasks; timing is recorded; unit tests pass. |
| **Deliverable** | `backend/classical/random_forest.py` |

---

### F04-05 — Classical model interface

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 3 |
| **Description** | Define an abstract base class (or protocol) that all classical models implement: `fit(X_train, y_train)`, `predict(X_test)`, `predict_proba(X_test)`, `get_config() -> dict`, `get_timing() -> dict`. This ensures evaluation code works uniformly across all models. |
| **Dependencies** | F01-03 |
| **Acceptance criteria** | All four classical models implement this interface; a type check or unit test confirms compliance; calling `get_timing()` returns non-zero values after fit and predict. |
| **Deliverable** | `backend/classical/base.py` |

---

## EPIC 5 — Quantum VQC Implementation

> Goal: Implement the Variational Quantum Classifier using PennyLane.

---

### F05-01 — VQC circuit builder

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 4 |
| **Description** | Implement a PennyLane QNode factory that creates a parameterised VQC circuit. Accept: n_qubits (2–10), depth (1–4), entanglement (linear/circular). Apply angle encoding followed by RX/RY/RZ layers and the configured entanglement pattern. |
| **Dependencies** | F01-01, F01-04 |
| **Acceptance criteria** | Circuit is created correctly for all combinations of n_qubits ∈ {2,4,6,8,10} and depth ∈ {1,2,3,4}; both entanglement modes produce correct entanglement gates; n_qubits > 10 raises an error; circuit structure is printed/logged correctly. |
| **Deliverable** | `backend/quantum/vqc_circuit.py`, `tests/test_vqc_circuit.py` |

---

### F05-02 — VQC training loop (Adam)

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 4 |
| **Description** | Implement VQC training using PennyLane's Adam optimizer. Accept: learning rate, number of epochs, batch size. Compute cross-entropy or margin loss. Record training loss per epoch. Measure total training time and per-circuit evaluation count where measurable. |
| **Dependencies** | F05-01 |
| **Acceptance criteria** | Training loss decreases over epochs on a simple binary dataset; training time is measured; epoch losses are stored; random seed produces reproducible training trajectories; unit test passes. |
| **Deliverable** | `backend/quantum/vqc_trainer.py` (Adam section) |

---

### F05-03 — VQC training loop (COBYLA)

| Field | Value |
|---|---|
| **Priority** | P2 |
| **Week** | 4 |
| **Description** | Implement VQC training using COBYLA (from scipy.optimize or PennyLane's COBYLA interface) as an alternative optimizer. Record function evaluations, convergence status, and training time. |
| **Dependencies** | F05-02 |
| **Acceptance criteria** | COBYLA optimizer trains a VQC successfully on a small dataset; result object records convergence status and function evaluations; timing is measured; unit test passes. |
| **Deliverable** | `backend/quantum/vqc_trainer.py` (COBYLA section) |

---

### F05-04 — VQC multiclass extension

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 4 |
| **Description** | Extend the VQC to support multiclass classification (specifically for the Wine dataset with 3 classes). Implement One-vs-Rest or softmax output strategy. Ensure predictions produce class probabilities for fair metric computation. |
| **Dependencies** | F05-02 |
| **Acceptance criteria** | VQC trains and predicts 3-class Wine dataset; output is a probability vector of length 3; accuracy on a held-out test set is better than random; unit test passes. |
| **Deliverable** | `backend/quantum/vqc_multiclass.py` or extension of `vqc_trainer.py` |

---

### F05-05 — VQC model interface

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 4 |
| **Description** | Wrap the VQC trainer in the same model interface as classical models (fit, predict, predict_proba, get_config, get_timing). This allows the evaluation engine to treat VQC identically to classical models. |
| **Dependencies** | F04-05, F05-02 |
| **Acceptance criteria** | VQC satisfies the shared model interface; `get_timing()` returns training time, inference time, and circuit evaluation count; `get_config()` returns circuit configuration dict; unit tests pass. |
| **Deliverable** | `backend/quantum/vqc_model.py` |

---

## EPIC 6 — Quantum Kernel SVM Implementation

> Goal: Implement the Quantum Kernel SVM using PennyLane fidelity kernels.

---

### F06-01 — Quantum kernel function

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 4–5 |
| **Description** | Implement a fidelity-based quantum kernel function using PennyLane. Support two feature maps: angle encoding and ZZ feature map. The kernel K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|² must return values in [0, 1]. |
| **Dependencies** | F01-04 |
| **Acceptance criteria** | Kernel function returns values in [0, 1]; K(x, x) = 1 for all x (diagonal check); kernel matrix is symmetric; unit tests for both feature maps pass. |
| **Deliverable** | `backend/quantum/quantum_kernel.py`, `tests/test_quantum_kernel.py` |

---

### F06-02 — Kernel matrix construction with timing

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 5 |
| **Description** | Implement a matrix builder that computes the full NxN (train) and MxN (test) kernel matrices by calling the kernel function for all sample pairs. Measure and record total kernel construction time. Enforce the 200-sample training cap (default). |
| **Dependencies** | F06-01 |
| **Acceptance criteria** | Train matrix is NxN symmetric positive semi-definite; test matrix is MxN; construction time is recorded; 200-sample cap is enforced with a warning/error; matrices are optionally cached to disk; unit test passes on small datasets. |
| **Deliverable** | `backend/quantum/kernel_matrix.py`, `tests/test_kernel_matrix.py` |

---

### F06-03 — Quantum Kernel SVM model

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 5 |
| **Description** | Wrap the quantum kernel SVM in the shared model interface. Use scikit-learn SVC with precomputed kernel. Configurable C. Report kernel construction time separately from SVM training time. |
| **Dependencies** | F04-05, F06-02 |
| **Acceptance criteria** | QKSVM trains and predicts on both datasets; timing breakdown includes: kernel construction time + SVM training time + inference time; model interface is fully satisfied; unit tests pass. |
| **Deliverable** | `backend/quantum/qksvm_model.py`, `tests/test_qksvm_model.py` |

---

## EPIC 7 — Benchmarking and Evaluation

> Goal: Implement the full evaluation pipeline that produces standardized results for all models.

---

### F07-01 — Metrics calculator

| Field | Value |
|---|---|
| **Priority** | P0 |
| **Week** | 5 |
| **Description** | Implement a metrics module that accepts y_true, y_pred, y_prob and computes: accuracy, precision (macro), recall (macro), macro F1, weighted F1, balanced accuracy, ROC-AUC (binary and multiclass OvR), and confusion matrix. Return a structured `MetricsResult` object. |
| **Dependencies** | None |
| **Acceptance criteria** | All eight metrics are computed; multiclass ROC-AUC uses OvR macro; confusion matrix is a 2D array; unit tests with known inputs produce correct outputs; edge cases (single class prediction) are handled gracefully. |
| **Deliverable** | `backend/evaluation/metrics.py`, `tests/test_metrics.py` |

---

### F07-02 — Benchmark runner (single-seed)

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 5 |
| **Description** | Implement a runner that takes a model, a dataset, a comparison mode, and a configuration, and produces a full `BenchmarkResult`: metrics, timing breakdown, feature mode used, random seed, and model config. |
| **Dependencies** | F07-01, F04-05, F03-06 |
| **Acceptance criteria** | Runner produces correct metrics for all four classical models on both datasets; BenchmarkResult is serializable to JSON; feature mode is recorded; timing is included; unit tests pass. |
| **Deliverable** | `backend/evaluation/runner.py`, `tests/test_runner.py` |

---

### F07-03 — Repeated experiment runner (multi-seed)

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 5–6 |
| **Description** | Extend the benchmark runner to repeat experiments across multiple random seeds (default: 5; maximum: 30). Aggregate results to compute mean ± standard deviation for all metrics and timings. |
| **Dependencies** | F07-02 |
| **Acceptance criteria** | With 5 seeds, produces 5 independent results; mean and std are correctly computed; all seeds are stored in the aggregate result; results are serializable; unit test passes. |
| **Deliverable** | `backend/evaluation/repeated_runner.py`, `tests/test_repeated_runner.py` |

---

### F07-04 — Cross-validation module

| Field | Value |
|---|---|
| **Priority** | P2 |
| **Week** | 6 |
| **Description** | Implement 3-fold stratified cross-validation for hyperparameter evaluation. Accept a model factory, dataset, and parameter grid. Return fold-level metrics and aggregated scores. |
| **Dependencies** | F07-02 |
| **Acceptance criteria** | 3-fold CV runs correctly on classical models; scores match manual fold computation; results are serializable; unit test passes. |
| **Deliverable** | `backend/evaluation/cross_validation.py` |

---

### F07-05 — Comparison report generator

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 6 |
| **Description** | Implement a report generator that takes BenchmarkResults from multiple models and produces a structured comparison table: metrics side-by-side, timing side-by-side, comparison mode used, and any fairness warnings (e.g., input mismatch). Export to JSON and CSV. |
| **Dependencies** | F07-03 |
| **Acceptance criteria** | Report contains all models run in an experiment; fairness warnings are included where applicable; JSON and CSV exports are well-formed; unit test passes. |
| **Deliverable** | `backend/evaluation/report.py`, `tests/test_report.py` |

---

## EPIC 8 — Trainability and Barren Plateau Analysis

> Goal: Implement mandatory quantum trainability experiments and visualizations.

---

### F08-01 — Gradient measurement module

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 6 |
| **Description** | Implement a gradient measurement function using PennyLane's parameter-shift rule (or automatic differentiation). For a given circuit configuration and random initialization, compute the full gradient vector, its L2 norm, mean, and variance. |
| **Dependencies** | F05-01 |
| **Acceptance criteria** | Gradient vector shape matches number of trainable parameters; norm, mean, and variance are correctly computed; function works for all qubit counts (2, 4, 6, 8, 10) and depths (1–4); unit tests pass. |
| **Deliverable** | `backend/quantum/gradients.py`, `tests/test_gradients.py` |

---

### F08-02 — Trainability experiment runner

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 6 |
| **Description** | Implement an experiment runner that sweeps over: qubit counts (2, 4, 6, 8, 10), circuit depths (1–4), entanglement strategies (linear, circular), and multiple random initializations (default: 10; options: 20, 30). For each configuration, record gradient statistics and training/validation loss curves. |
| **Dependencies** | F08-01, F05-02 |
| **Acceptance criteria** | Runner produces results for all 5×4×2×10 = 400 default configurations; each result includes: qubit count, depth, entanglement, initialization index, gradient norm, mean, variance, and loss per epoch; results are serializable; runtime is tracked. |
| **Deliverable** | `backend/training/trainability_runner.py`, `tests/test_trainability_runner.py` |

---

### F08-03 — Gradient variance visualizations

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 6–7 |
| **Description** | Implement visualization functions (using matplotlib or Plotly) that produce: (1) gradient variance vs qubit count (one line per depth), (2) gradient variance vs circuit depth (one line per qubit count). Save figures to `reports/figures/`. |
| **Dependencies** | F08-02 |
| **Acceptance criteria** | Both plots are produced correctly from experiment results; axes are labelled; figures are saved as PNG and HTML (Plotly); unit test using synthetic data passes. |
| **Deliverable** | `backend/evaluation/trainability_plots.py`, figures in `reports/figures/` |

---

### F08-04 — Gradient distribution visualizations

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement histogram or violin plots showing the distribution of gradient components across all initializations, for each circuit configuration. |
| **Dependencies** | F08-02 |
| **Acceptance criteria** | Distribution plots are produced for each (qubit_count, depth) combination; plots clearly label configuration; saved as PNG and HTML. |
| **Deliverable** | `backend/evaluation/trainability_plots.py` (distribution section) |

---

### F08-05 — Training and validation loss curves

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement loss curve plots showing training loss and validation loss per epoch for each experimental configuration. Must display mean ± std across initializations. |
| **Dependencies** | F08-02 |
| **Acceptance criteria** | Loss curves are plotted with mean line and ±std shading; both train and validation loss are shown; plots are saved as PNG and HTML. |
| **Deliverable** | `backend/evaluation/trainability_plots.py` (loss curves section) |

---

## EPIC 9 — Experiment Management and Reproducibility

> Goal: Ensure experiments are logged, reproducible, and exportable.

---

### F09-01 — Experiment result storage

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 5–6 |
| **Description** | Implement a storage module that saves full experiment results (config, metrics, timing, raw predictions, model config) to JSON files in `reports/experiments/`. Each experiment has a unique timestamped ID. |
| **Dependencies** | F07-02, F01-02 |
| **Acceptance criteria** | Each experiment produces a uniquely named JSON file; file is human-readable; file contains full config, metrics, timings, and metadata; storage module is importable; unit test passes. |
| **Deliverable** | `backend/storage/experiment_store.py`, `tests/test_experiment_store.py` |

---

### F09-02 — Experiment history and listing

| Field | Value |
|---|---|
| **Priority** | P2 |
| **Week** | 6–7 |
| **Description** | Implement a module that lists all saved experiments (from `reports/experiments/`), loads a specific experiment by ID, and returns a summary table of all experiments. |
| **Dependencies** | F09-01 |
| **Acceptance criteria** | Listing returns all experiments with their IDs, timestamps, datasets, and models; a specific experiment can be loaded by ID; summary table is serializable; unit test passes. |
| **Deliverable** | `backend/storage/experiment_history.py` |

---

### F09-03 — Export module (CSV / JSON)

| Field | Value |
|---|---|
| **Priority** | P2 |
| **Week** | 7 |
| **Description** | Implement an export module that converts experiment results into user-friendly CSV tables (one row per model per experiment) and JSON summaries. Exports are saved to `reports/exports/`. |
| **Dependencies** | F09-01 |
| **Acceptance criteria** | CSV export is correctly formatted; JSON export is valid; exports include all benchmark metrics and timing; unit test passes. |
| **Deliverable** | `backend/storage/exporter.py` |

---

## EPIC 10 — FastAPI Backend

> Goal: Expose all backend functionality through a REST API.

---

### F10-01 — FastAPI application setup

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Create the FastAPI application entry point with CORS middleware configured for the React frontend. Set up health check endpoint (`GET /health`). Configure uvicorn startup. |
| **Dependencies** | F01-01 |
| **Acceptance criteria** | Server starts on `uvicorn backend.api.main:app`; `GET /health` returns `{"status": "ok"}`; CORS is configured for `localhost:3000`; no import errors. |
| **Deliverable** | `backend/api/main.py` |

---

### F10-02 — Dataset routes

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement routes: `GET /datasets` (list available datasets), `GET /datasets/{name}` (load built-in dataset metadata), `POST /datasets/upload` (accept CSV file, return validation report and column list). |
| **Dependencies** | F10-01, F02-03 |
| **Acceptance criteria** | All three routes return correct responses; CSV upload returns validation report JSON; malformed CSV returns a 422 with a descriptive error; routes are tested with FastAPI TestClient. |
| **Deliverable** | `backend/api/routes/datasets.py`, `tests/test_api_datasets.py` |

---

### F10-03 — Experiment routes

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement routes: `POST /experiments/run` (accept experiment config, run benchmark, return result ID), `GET /experiments` (list all), `GET /experiments/{id}` (fetch single result), `GET /experiments/{id}/export` (download CSV). |
| **Dependencies** | F10-01, F07-05, F09-01 |
| **Acceptance criteria** | POST run triggers a benchmark and returns a result ID; GET experiment returns stored result JSON; export returns valid CSV; routes are tested with FastAPI TestClient. |
| **Deliverable** | `backend/api/routes/experiments.py`, `tests/test_api_experiments.py` |

---

### F10-04 — Trainability analysis routes

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement routes: `POST /trainability/run` (accept trainability config, run analysis, return result ID), `GET /trainability/{id}` (fetch results), `GET /trainability/{id}/plots` (return plot file paths or base64-encoded images). |
| **Dependencies** | F10-01, F08-02 |
| **Acceptance criteria** | POST run triggers a trainability experiment; results are stored and retrievable; plot paths/data are returned; routes are tested with FastAPI TestClient. |
| **Deliverable** | `backend/api/routes/trainability.py`, `tests/test_api_trainability.py` |

---

### F10-05 — API error handling

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Implement global exception handlers for the FastAPI app. All 400/422/500 errors must return a structured JSON error body: `{"error": "...", "detail": "..."}`. Unhandled exceptions must not leak stack traces to the client. |
| **Dependencies** | F10-01 |
| **Acceptance criteria** | Invalid request bodies return 422 with structured error; unhandled exceptions return 500 with generic message (no stack trace); all tested with TestClient. |
| **Deliverable** | `backend/api/error_handlers.py` |

---

## EPIC 11 — React Frontend

> Goal: Implement the React + TypeScript web dashboard.

---

### F11-01 — React project initialization

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7 |
| **Description** | Initialize the React + TypeScript application in `frontend/` using Vite or Create React App. Install Bootstrap, Framer Motion, and Plotly.js. Set up the API service layer with the backend base URL. |
| **Dependencies** | F10-01 |
| **Acceptance criteria** | `npm run dev` starts the frontend; TypeScript compilation succeeds with no errors; Bootstrap styles are applied; API service can call `GET /health` successfully. |
| **Deliverable** | `frontend/` initialized project, `frontend/src/services/api.ts` |

---

### F11-02 — Dataset selection page

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 7–8 |
| **Description** | Implement a page that allows the user to select a built-in dataset or upload a CSV file. Display the validation report (missing values, duplicates, class distribution) on submission. Allow target column selection for uploaded CSVs. |
| **Dependencies** | F11-01, F10-02 |
| **Acceptance criteria** | Both built-in datasets can be selected and their metadata displayed; CSV upload sends file to API and displays validation report; target column dropdown works; loading state shown during API calls. |
| **Deliverable** | `frontend/src/pages/DatasetPage.tsx` |

---

### F11-03 — Experiment configuration page

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Implement a page that allows the user to configure and launch a benchmark experiment: select models, set hyperparameters, choose comparison mode, set number of seeds. Submit config to `POST /experiments/run`. |
| **Dependencies** | F11-02, F10-03 |
| **Acceptance criteria** | All models are configurable from the UI; comparison mode selector works; form validation prevents invalid qubit counts (>10); experiment submits successfully and shows the result ID. |
| **Deliverable** | `frontend/src/pages/ExperimentPage.tsx` |

---

### F11-04 — Results dashboard

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Implement a results dashboard page that displays benchmark results: comparison table (metrics + timing for all models), confusion matrices, and bar charts (Plotly). Allow CSV export of results. |
| **Dependencies** | F11-03, F10-03 |
| **Acceptance criteria** | Comparison table shows all metrics for all models; Plotly charts render correctly; CSV export downloads a valid file; loading and error states are handled. |
| **Deliverable** | `frontend/src/pages/ResultsPage.tsx`, chart components in `frontend/src/components/` |

---

### F11-05 — Trainability analysis page

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Implement a page for configuring and viewing trainability experiments. Allow setting qubit range, depth range, entanglement, and initialization count. Display the four required visualizations (gradient variance vs qubits, vs depth, distributions, loss curves) using Plotly. |
| **Dependencies** | F11-04, F10-04 |
| **Acceptance criteria** | All four visualization types render from experiment data; qubit and depth selectors work; Plotly charts are interactive; loading and error states are handled. |
| **Deliverable** | `frontend/src/pages/TrainabilityPage.tsx` |

---

### F11-06 — Experiment history page

| Field | Value |
|---|---|
| **Priority** | P2 |
| **Week** | 8 |
| **Description** | Implement a page that lists all past experiments with their dataset, models, date, and summary metrics. Allow clicking to view full results or export CSV. |
| **Dependencies** | F11-04, F10-03 |
| **Acceptance criteria** | All past experiments are listed; clicking opens the full result; export works; empty state is handled gracefully. |
| **Deliverable** | `frontend/src/pages/HistoryPage.tsx` |

---

## EPIC 12 — Integration, Testing, and Reporting

> Goal: End-to-end integration, full test coverage, and final project documentation.

---

### F12-01 — End-to-end integration test

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Implement an end-to-end pytest test that runs a complete benchmark pipeline: load Breast Cancer dataset → preprocess → run all four classical models → compute metrics → generate report → save to disk. Verify all outputs are correct. |
| **Dependencies** | F07-05, F09-01 |
| **Acceptance criteria** | Test passes without errors; all four classical models are run; output JSON is valid; metrics are within expected ranges for Breast Cancer dataset; test is deterministic given fixed seeds. |
| **Deliverable** | `tests/test_integration_benchmark.py` |

---

### F12-02 — Quantum integration test

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Implement a reduced-scale integration test for the VQC and QKSVM: use a small subset of Breast Cancer (n=50) with 2 qubits, 1 layer, 2 seeds. Verify the full pipeline runs without errors and produces valid metrics. |
| **Dependencies** | F05-05, F06-03 |
| **Acceptance criteria** | Test passes without errors; VQC and QKSVM both produce predictions; metrics are valid floats; test completes in under 2 minutes; test is deterministic given fixed seeds. |
| **Deliverable** | `tests/test_integration_quantum.py` |

---

### F12-03 — Full test suite audit

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Review all unit and integration tests. Ensure all epics (1–11) have at least one passing test per deliverable. Run the full pytest suite and fix any failing tests. |
| **Dependencies** | All prior tasks |
| **Acceptance criteria** | `pytest tests/ -v` passes with zero failures; test count is at least 30; no skipped tests (unless explicitly documented). |
| **Deliverable** | Passing `pytest` output, any fixes applied |

---

### F12-04 — Final documentation

| Field | Value |
|---|---|
| **Priority** | P1 |
| **Week** | 8 |
| **Description** | Update README.md to reflect the completed project. Write a final experiment report (PDF or Markdown) summarizing: datasets used, models evaluated, benchmark results, trainability findings, limitations, and conclusions. |
| **Dependencies** | All epics |
| **Acceptance criteria** | README is accurate and complete; final report covers all three research objectives; limitations section prominently states the simulator limitation; no claims of quantum speedup. |
| **Deliverable** | Updated `README.md`, `reports/final_report.md` |

---

## Dependency Summary

```
F01-01 ──► F01-02
F01-01 ──► F01-03
                F01-03 ──► F02-01 ──► F02-02 ──► F02-03
                           F02-01 ──► F02-04 ──► F03-01 ──► F03-02 ──► F03-06
                                                           ──► F03-03 ──► F03-06
                                                           ──► F03-04 ──► F03-06
                                                           ──► F03-05 ──► F03-06
F04-05 ──► F04-01 ──► F04-02
                 ──► F04-03
                 ──► F04-04
F03-06 ──► F04-01
F05-01 ──► F05-02 ──► F05-03
                 ──► F05-04
F04-05 ──► F05-05 ◄── F05-02
F06-01 ──► F06-02 ──► F06-03 ◄── F04-05
F07-01 ──► F07-02 ──► F07-03 ──► F07-05
F07-02 ──► F07-04
F08-01 ──► F08-02 ──► F08-03
                 ──► F08-04
                 ──► F08-05
F07-02 ──► F09-01 ──► F09-02
                 ──► F09-03
F10-01 ──► F10-02 / F10-03 / F10-04 / F10-05
F11-01 ──► F11-02 ──► F11-03 ──► F11-04 ──► F11-05
                                        ──► F11-06
```

---

## Weekly Schedule Mapping

| Week | Epics | Key deliverables |
|---|---|---|
| 1 (Done) | EPIC 1 partial | Environment, scaffolding, smoke tests |
| 2 | EPIC 1 complete, EPIC 2 | Config, logging, utils, dataset loader, validator, splitter |
| 3 | EPIC 3, EPIC 4 | Preprocessing, feature selection, all 4 classical models |
| 4 | EPIC 5, EPIC 6 partial | VQC circuit, training, multiclass; Quantum kernel function |
| 5 | EPIC 6 complete, EPIC 7, EPIC 9 partial | QKSVM, metrics, benchmark runner, experiment storage |
| 6 | EPIC 7 complete, EPIC 8, EPIC 9 | Multi-seed runner, CV, trainability experiments, history |
| 7 | EPIC 10, EPIC 11 partial | FastAPI routes, React app setup, dataset + experiment pages |
| 8 | EPIC 11 complete, EPIC 12 | Results dashboard, trainability page, integration tests, final report |
