# MVP Scope — QML Benchmark Studio

> **Status:** Frozen — Day 2, 2026-11-19
> **Project:** 6th-Semester Research Project
> **Repository:** https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR

This document formally records the agreed Minimum Viable Product (MVP) scope for QML Benchmark Studio. No requirement in this document may be silently modified. Changes require an explicit revision entry at the bottom of this file.

---

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [Technology Stack](#2-technology-stack)
3. [MVP Feature Scope](#3-mvp-feature-scope)
   - [A. Dataset Support](#a-dataset-support)
   - [B. Feature Selection and Preprocessing](#b-feature-selection-and-preprocessing)
   - [C. Classical ML Models](#c-classical-ml-models)
   - [D. Quantum ML Models](#d-quantum-ml-models)
   - [E. Evaluation and Benchmarking](#e-evaluation-and-benchmarking)
   - [F. Quantum Trainability Analysis](#f-quantum-trainability-analysis)
4. [Out of Scope — Future Work](#4-out-of-scope--future-work)
5. [Research Limitations](#5-research-limitations)
6. [Revision History](#6-revision-history)

---

## 1. Research Objectives

The MVP must address these three research objectives:

| # | Objective |
|---|---|
| O1 | Compare predictive performance (accuracy, F1, AUC) of QML and classical ML models on the same datasets and inputs. |
| O2 | Compare computational costs (training time, inference time, kernel construction time) across model types. |
| O3 | Investigate quantum circuit trainability, including the detection and characterisation of barren plateau behaviour in gradient landscapes. |

---

## 2. Technology Stack

The following technology stack is **frozen**. Do not substitute or extend it without a formal scope change.

| Layer | Technology | Version / Notes |
|---|---|---|
| Quantum ML | PennyLane | 0.45.x, `default.qubit` simulator only |
| Classical ML | Scikit-learn | 1.9.x |
| Numerical computing | NumPy, Pandas | 2.5.x, 3.1.x |
| Backend API | FastAPI + Uvicorn | 0.165.x, 0.54.x |
| Frontend | React + TypeScript | To be initialised in Day 3+ |
| UI components | Bootstrap | — |
| Animations | Framer Motion | — |
| Data visualisation | Plotly.js | — |
| Testing | pytest | 9.2.x |
| Python runtime | Python 3.13.1 | 3.11+ compatible |

---

## 3. MVP Feature Scope

### A. Dataset Support

**Supported built-in datasets (via scikit-learn):**

| # | Dataset | Task | Classes | Notes |
|---|---|---|---|---|
| 1 | Breast Cancer Wisconsin (Diagnostic) | Binary classification | 2 (Malignant / Benign) | 30 features, 569 samples |
| 2 | Wine | Multiclass classification | 3 | 13 features, 178 samples |

**User-uploaded datasets:**

- Accept CSV file upload.
- Let the user select the target column from the uploaded file.
- Support numerical and categorical columns.

**Dataset validation requirements:**

- Validate that the uploaded file is well-formed CSV with at least one feature column and one target column.
- Validate column data types and report inconsistencies.
- Identify and report: missing values, duplicate rows, constant features, and class imbalance.
- Do not silently drop columns or rows without user acknowledgement.

**Train/test splitting:**

- Use stratified 80/20 split as the default.
- Stratified splitting must be applied where the target variable allows it (classification tasks).
- Preserve random seeds for reproducibility.

**Data leakage prevention (mandatory):**

- All preprocessing transformers (scalers, encoders, imputers) must be fitted exclusively on training data.
- Fitted transformers are then applied to test data.
- Feature selection procedures must be fitted on training data only.
- PCA must be fitted on training data only.
- Violation of this requirement is a defect, not a design choice.

---

### B. Feature Selection and Preprocessing

**Preprocessing operations (must all be implemented):**

| Operation | Notes |
|---|---|
| Standardization / scaling | StandardScaler or equivalent |
| Missing-value handling | SimpleImputer (mean/median/most_frequent strategies) |
| Mutual Information feature selection | SelectKBest with mutual_info_classif |
| ANOVA F-test feature selection | SelectKBest with f_classif |
| Correlation-based feature filtering | Drop highly correlated features (configurable threshold) |
| PCA dimensionality reduction | Configurable number of components |

**Comparison modes:**

| Mode | Description | Notes |
|---|---|---|
| Original features | Full feature set, no reduction | Baseline mode |
| Matched reduced features | Feature set reduced to match the quantum model's qubit count using the selected feature selection method | **Primary fair-comparison mode** |
| Quantum-limited features | Only the features the quantum model directly uses, without matching classical models to the same count | Secondary mode |

> **Rule:** The system must NEVER silently discard features to satisfy a quantum model's input dimension. When dimensionality reduction is applied, it must be explicitly logged and reported to the user.

---

### C. Classical ML Models

All four models must be implemented and testable:

| Model | Key configurable hyperparameters |
|---|---|
| DummyClassifier | Strategy (most_frequent, stratified, uniform) |
| Support Vector Machine | Kernel (linear, RBF), C, gamma |
| Multi-Layer Perceptron | Hidden layer sizes, activation, learning rate, max iterations |
| Random Forest | n_estimators, max_depth, min_samples_split, random_state |

Requirements:
- All classical models use scikit-learn implementations.
- Hyperparameters must be configurable by the user (not hard-coded constants).
- Models must support both binary and multiclass classification.

---

### D. Quantum ML Models

#### D.1 — Variational Quantum Classifier (VQC)

| Parameter | Specification |
|---|---|
| Framework | PennyLane |
| Simulator | `default.qubit` only |
| Qubit range | **2 to 10 qubits (maximum: 10)** |
| Encoding | Angle encoding |
| Gates | RX, RY, RZ rotations (where applicable) |
| Entanglement | Linear and circular (configurable) |
| Circuit depth | 1 to 4 layers (configurable) |
| Optimizers | Adam and COBYLA (where compatible) |
| Classification | Binary and multiclass |

Requirements:
- The qubit count must never exceed 10 in any VQC experiment.
- Circuit structure (depth, entanglement) must be logged with every experiment result.
- Output must be mapped to class probabilities for fair comparison with classical models.

#### D.2 — Quantum Kernel SVM

| Parameter | Specification |
|---|---|
| Kernel type | Fidelity-based quantum kernel |
| Feature maps | Angle encoding, ZZ feature map |
| Backend | Precomputed kernel passed to scikit-learn SVC |
| C parameter | Configurable by user |
| Training sample cap | Default maximum: **200 samples** |
| Kernel matrix caching | Cache computed kernel matrices where practical |

Requirements:
- The kernel matrix construction time must be measured and reported separately.
- The quadratic scaling of kernel construction (O(n²) circuit evaluations) must be documented in experiment outputs.
- The 200-sample cap is a default; it must be configurable.

---

### E. Evaluation and Benchmarking

#### E.1 — Predictive Performance Metrics

All of the following must be computed and reported for every model run:

| Metric | Notes |
|---|---|
| Accuracy | Standard accuracy |
| Precision | Macro-averaged |
| Recall | Macro-averaged |
| Macro F1 | Unweighted mean per class |
| Weighted F1 | Weighted by support |
| Balanced accuracy | Accounts for class imbalance |
| ROC-AUC | Where applicable (binary: standard; multiclass: OvR macro) |
| Confusion matrix | Full matrix, stored and displayed |

#### E.2 — Computational Performance Metrics

All of the following must be measured and reported:

| Metric | Notes |
|---|---|
| Training time | Wall-clock time, in seconds |
| Inference time | Wall-clock time for test-set prediction |
| Preprocessing time | Time for all preprocessing steps |
| Feature transformation time | Time for feature selection / PCA |
| Quantum kernel construction time | QKSVM only |
| Circuit evaluation count | VQC: number of circuit calls where measurable |

#### E.3 — Experimental Protocol

| Parameter | Default | Configurable |
|---|---|---|
| Train/test split | 80/20 stratified | No |
| Repeated experiments | 5 random seeds | Yes (up to 30) |
| Cross-validation | 3-fold stratified (for tuning) | Yes |
| Reporting | Mean ± standard deviation | — |
| Random seed preservation | Required | — |
| Experiment config logging | Required | — |

**Fairness requirements:**
- When comparing QML and classical models, they must use identical datasets, preprocessing pipelines, train/test splits, and input features (Matched Reduced mode).
- Deviations from matched inputs must be explicitly flagged in results output.

---

### F. Quantum Trainability Analysis

> **This is a mandatory research component. It must be implemented as part of the MVP. It is not optional.**

#### F.1 — Quantities to Measure

| Quantity | Notes |
|---|---|
| Gradient norms | L2 norm of the full gradient vector |
| Gradient mean | Mean of gradient components |
| Gradient variance | Variance of gradient components |
| Gradient distribution | Distribution across components and initializations |
| Training loss | Per epoch / iteration |
| Validation loss | Per epoch / iteration |

#### F.2 — Experimental Axes

| Axis | Values |
|---|---|
| Number of qubits | 2, 4, 6, 8, 10 |
| Circuit depth | 1, 2, 3, 4 |
| Entanglement strategy | Linear, circular |
| Random initializations | Default: 10; options: 20, 30 |

#### F.3 — Required Visualizations

| Visualization | Description |
|---|---|
| Gradient variance vs qubit count | Line/scatter plot per depth |
| Gradient variance vs circuit depth | Line/scatter plot per qubit count |
| Gradient distributions | Histogram or violin plot per configuration |
| Training loss curves | Loss vs epoch, per configuration |
| Validation loss curves | Loss vs epoch, per configuration |

#### F.4 — Interpretation Requirements

- Do **not** conclude that barren plateaus exist solely from small gradient values.
- Interpret results with reference to circuit architecture, initialization, depth, entanglement, and cost function structure.
- Distinguish between barren plateau behaviour and other trainability challenges (e.g., local minima, vanishing gradients from insufficient depth).
- Results are descriptive; do not overstate findings.

---

## 4. Out of Scope — Future Work

The following items are **explicitly excluded from the MVP**. They must not be introduced as current requirements. They may be considered after the MVP is complete.

| Item | Reason for exclusion |
|---|---|
| Real quantum hardware execution | Requires QPU access; not available in this research environment |
| Claims of practical quantum speedup | Simulator results cannot support such claims |
| Amplitude encoding | Out of scope for this study; angle encoding is sufficient |
| Advanced data re-uploading extensions | Out of scope; standard angle encoding is used |
| Quantum error correction | Hardware noise not modelled |
| Noise-aware hardware experiments | Noise models deferred to future work |
| Distributed or cloud quantum computing | Not needed for simulator-based research |
| Authentication and user accounts | Single-user academic tool |
| Multi-user collaboration features | Not required |
| Docker and Kubernetes | Deployment infrastructure out of scope |
| Redis or separate database servers | JSON/CSV file storage is sufficient |
| Mobile application | Web-only interface |
| AI-based model recommendation | Automated model search not in scope |
| Complex hyperparameter optimization | Grid/random search not in scope; agreed strategy only |

---

## 5. Research Limitations

> **These limitations must be prominently stated in all experiment reports and in the final project report.**

### 5.1 — Simulator Limitation (Mandatory Statement)

> *"PennyLane `default.qubit` is a classical quantum circuit simulator. Quantum circuits are simulated on classical computing hardware. Therefore, experimental results from this project cannot establish practical quantum computational speedup over classical hardware."*

### 5.2 — Additional Documented Limitations

| Limitation | Explanation |
|---|---|
| Classical simulation overhead | Circuit execution time includes the overhead of Python-based classical simulation, not pure quantum gate time. |
| Structural incomparability | Quantum and classical models have different computational structures. Direct timing comparisons must be interpreted carefully. |
| Quadratic kernel scaling | Quantum Kernel SVM requires O(n²) circuit evaluations for kernel matrix construction. This constrains the usable training set size. |
| Fair comparison requirements | Fair comparisons require identical datasets, preprocessing pipelines, train/test splits, and input features. Deviations invalidate direct comparisons. |
| Barren plateau interpretation | A small dataset does not automatically imply the absence of barren plateaus. Gradient variance must be analysed across multiple initializations and architectural configurations. |
| Barren plateau causation | Small gradients alone do not prove barren plateaus. The analysis must consider circuit architecture, depth, initialization, entanglement, and cost function structure. |
| Generalization | Findings on the two included datasets (Breast Cancer, Wine) may not generalize to other dataset sizes, feature distributions, or classification tasks. |

---

## 6. Revision History

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-11-19 | 1.0 | Day 2 setup | Initial freeze of MVP scope |
