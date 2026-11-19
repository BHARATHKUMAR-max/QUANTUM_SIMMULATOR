# MVP Acceptance Checklist — QML Benchmark Studio

> **Status:** Frozen — Day 2, 2026-11-19  
> **Project:** 6th-Semester Research Project  
> **Scope Document:** [docs/MVP_SCOPE.md](./MVP_SCOPE.md)  
> **Backlog Document:** [docs/DEVELOPMENT_BACKLOG.md](./DEVELOPMENT_BACKLOG.md)

This checklist serves as the formal acceptance gate at the conclusion of MVP development. Each item is independently verifiable with concrete verification procedures.

---

## 1. Dataset Management & Validation

- [ ] **CHK-DATA-01: Built-in Dataset Loading (Breast Cancer)**
  - *Verification:* Calling `load_dataset("breast_cancer")` loads 569 samples with 30 features, binary labels (0/1 or Malignant/Benign), metadata dictionary, and class names.
- [ ] **CHK-DATA-02: Built-in Dataset Loading (Wine)**
  - *Verification:* Calling `load_dataset("wine")` loads 178 samples with 13 features across 3 classes with valid class names and metadata.
- [ ] **CHK-DATA-03: User CSV Upload**
  - *Verification:* Uploading a valid CSV file via API/UI parses rows/columns, presents available columns, and allows target column selection without corrupting data types.
- [ ] **CHK-DATA-04: Dataset Validation & Diagnostics**
  - *Verification:* Uploading datasets with missing entries, duplicate rows, constant features, or imbalanced distributions triggers structured validation output with appropriate warnings/errors.
- [ ] **CHK-DATA-05: Stratified Train/Test Splitting**
  - *Verification:* 80/20 train/test splits maintain class distributions within ±2% margin across random seeds.

---

## 2. Preprocessing & Leakage Prevention

- [ ] **CHK-PREP-01: Zero Data Leakage Enforcement**
  - *Verification:* StandardScaler and SimpleImputer fit exclusively on `X_train`. Transformation of `X_test` uses training statistics only (verified via assertions comparing scaler means to `X_train` vs `X_test`).
- [ ] **CHK-PREP-02: Feature Selection Fit Isolation**
  - *Verification:* Mutual Information, ANOVA F-test, and Correlation filtering fit exclusively on `X_train`. Selected feature indices apply downstream without testing set access during selection.
- [ ] **CHK-PREP-03: PCA Dimensionality Reduction**
  - *Verification:* PCA fits on `X_train` only; explained variance ratio is logged; transforms `X_test` accurately.
- [ ] **CHK-PREP-04: Comparison Modes Execution**
  - *Verification:* System supports all three comparison modes (`Original features`, `Matched reduced features`, and `Quantum-limited features`).
- [ ] **CHK-PREP-05: Explicit Dimensionality Reduction Logging**
  - *Verification:* System never silently truncates features; feature counts and reduction techniques are explicitly recorded in experiment logs.

---

## 3. Classical Machine Learning Models

- [ ] **CHK-CLASS-01: DummyClassifier Baseline**
  - *Verification:* Executes baseline strategies (`most_frequent`, `stratified`, `uniform`) with recorded training and inference latencies.
- [ ] **CHK-CLASS-02: Support Vector Machine (Linear & RBF)**
  - *Verification:* SVM trains with user-specified `C`, `gamma`, and kernel options on binary and multiclass targets with probability outputs.
- [ ] **CHK-CLASS-03: Multi-Layer Perceptron**
  - *Verification:* MLP trains with configurable layer structures, activations, and learning rates on both datasets.
- [ ] **CHK-CLASS-04: Random Forest**
  - *Verification:* Random Forest runs with configurable `n_estimators`, `max_depth`, and reproducible seeds.
- [ ] **CHK-CLASS-05: Unified Model Interface**
  - *Verification:* All classical models conform to the standardized interface (`fit`, `predict`, `predict_proba`, `get_config`, `get_timing`).

---

## 4. Quantum ML Models (PennyLane Simulation)

- [ ] **CHK-QML-01: Classical Simulator Execution & Transparency**
  - *Verification:* All circuits execute via PennyLane `default.qubit` simulator. System explicitly notes execution is on classical hardware and disclaims quantum speedup.
- [ ] **CHK-QML-02: Qubit Range Boundary Enforcement (2–10 Qubits)**
  - *Verification:* VQC and Quantum Kernel configurations strictly enforce qubit counts between 2 and 10. Attempting > 10 qubits raises validation errors.
- [ ] **CHK-QML-03: VQC Angle Encoding & Parametric Gates**
  - *Verification:* VQC applies angle encoding, parameterised rotations (RX, RY, RZ), and configurable depth (1–4 layers).
- [ ] **CHK-QML-04: VQC Entanglement Strategies**
  - *Verification:* Supports both linear and circular CNOT entanglement topologies.
- [ ] **CHK-QML-05: VQC Optimizers**
  - *Verification:* VQC trains with Adam optimizer, with optional COBYLA support.
- [ ] **CHK-QML-06: VQC Multiclass Classification**
  - *Verification:* VQC accurately evaluates multiclass targets (Wine dataset) and outputs class probability distributions.
- [ ] **CHK-QML-07: Quantum Kernel Construction & Fidelity**
  - *Verification:* Fidelity-based quantum kernel computes inner products in [0, 1] with symmetric Gram matrix and diagonal identity (`K(x, x) = 1.0`).
- [ ] **CHK-QML-08: Quantum Kernel Feature Maps**
  - *Verification:* Supports Angle encoding and ZZ feature maps.
- [ ] **CHK-QML-09: Quantum Kernel SVM (QKSVM)**
  - *Verification:* Computes Gram matrix and fits scikit-learn precomputed SVC with configurable regularization `C`.
- [ ] **CHK-QML-10: QKSVM Sample Cap & Caching**
  - *Verification:* Default maximum training sample cap of 200 enforced with warning if exceeded; kernel matrices can be cached to disk.

---

## 5. Evaluation, Benchmarking & Reproducibility

- [ ] **CHK-EVAL-01: Predictive Metrics Suite**
  - *Verification:* Standardized calculation of Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Balanced Accuracy, ROC-AUC, and Confusion Matrix.
- [ ] **CHK-EVAL-02: Computational Cost Metrics**
  - *Verification:* Wall-clock measurement of training time, inference time, preprocessing time, feature selection time, and kernel construction time.
- [ ] **CHK-EVAL-03: Multi-Seed Experimentation**
  - *Verification:* Runs experiments across 5 repeated random seeds by default (configurable up to 30), reporting mean and standard deviation.
- [ ] **CHK-EVAL-04: Cross-Validation Support**
  - *Verification:* 3-fold stratified cross-validation for hyperparameter tuning.
- [ ] **CHK-EVAL-05: Fair Comparison Protocol**
  - *Verification:* Classical models and quantum models receive identical preprocessed inputs under `Matched reduced features` mode.
- [ ] **CHK-EVAL-06: Experiment History & Reproducibility**
  - *Verification:* All runs save full parameters, seeds, environment details, and metrics as JSON in `reports/experiments/`.

---

## 6. Quantum Trainability & Barren Plateau Research

- [ ] **CHK-TRAIN-01: Mandatory Trainability Module**
  - *Verification:* Trainability analysis module runs independently of standard benchmark runs.
- [ ] **CHK-TRAIN-02: Gradient Statistics Measurement**
  - *Verification:* Parameter-shift/autodiff computes gradient vectors, L2 norms, gradient mean, and gradient variance across parameters.
- [ ] **CHK-TRAIN-03: Parameter Scaling Grid**
  - *Verification:* Experiments scale across 2, 4, 6, 8, and 10 qubits and circuit depths 1, 2, 3, and 4.
- [ ] **CHK-TRAIN-04: Multi-Initialization Sampling**
  - *Verification:* Default 10 random parameter initializations per configuration (with options for 20 and 30).
- [ ] **CHK-TRAIN-05: Gradient Variance Plots**
  - *Verification:* Generates variance vs. qubit count and variance vs. circuit depth plots.
- [ ] **CHK-TRAIN-06: Gradient Distribution Plots**
  - *Verification:* Generates distribution plots (histograms/violins) of gradient magnitudes across configurations.
- [ ] **CHK-TRAIN-07: Training & Validation Loss Dynamics**
  - *Verification:* Generates epoch loss curves showing mean ± std trajectory.
- [ ] **CHK-TRAIN-08: Objective Scientific Interpretation**
  - *Verification:* Output reports include cautionary interpretation stating that small gradients alone do not prove barren plateaus.

---

## 7. API, UI & System Integration

- [ ] **CHK-SYS-01: FastAPI REST Endpoints**
  - *Verification:* `GET /health`, dataset management routes, experiment execution routes, and trainability endpoints function with TestClient test coverage.
- [ ] **CHK-SYS-02: Error Handling & Security**
  - *Verification:* Malformed requests receive structured 400/422 JSON errors; internal errors return 500 without leaking stack traces.
- [ ] **CHK-SYS-03: React Dashboard Workflows**
  - *Verification:* Frontend permits end-to-end dataset selection, configuration setup, job execution, and interactive Plotly chart visualization.
- [ ] **CHK-SYS-04: Data Export**
  - *Verification:* Allows downloading benchmark comparison tables and experiment logs as valid CSV/JSON files.
- [ ] **CHK-SYS-05: End-to-End Automated Testing**
  - *Verification:* Automated pytest suite runs both classical and reduced quantum integration flows without failures.
- [ ] **CHK-SYS-06: Documentation & Disclaimers**
  - *Verification:* README and generated reports prominently display simulation disclaimers, limitations, and reproducible run instructions.
