# QML Benchmark Studio

A research-oriented platform for benchmarking **Quantum Machine Learning (QML)** against **Classical Machine Learning** models. Built as a 6th-semester academic research project.

---

## Project Description

QML Benchmark Studio provides a reproducible experimental environment to train, evaluate, and compare quantum-enhanced and classical machine learning pipelines on structured datasets. Results are visualised through an interactive web dashboard, enabling side-by-side analysis of predictive performance, computational cost, and training behaviour.

> **Note:** Quantum circuits are simulated using PennyLane's `default.qubit` simulator and therefore execute on **classical hardware**. No quantum speedup is claimed or expected.

---

## Research Objectives

1. **Performance comparison** — Evaluate and compare the predictive accuracy, precision, recall, and F1-score of QML models against classical ML baselines (e.g., SVM, Random Forest, Logistic Regression) on the same datasets.
2. **Computational cost analysis** — Measure and compare training time, inference latency, and resource usage between quantum-circuit-based and classical models.
3. **Trainability investigation** — Study the trainability of parameterised quantum circuits, including the detection and characterisation of *barren plateau* behaviour in gradient landscapes.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Quantum ML | [PennyLane](https://pennylane.ai/) |
| Classical ML | [Scikit-learn](https://scikit-learn.org/) |
| Numerical computing | [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Frontend framework | [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/) |
| UI components | [Bootstrap](https://getbootstrap.com/) |
| Animations | [Framer Motion](https://www.framer.com/motion/) |
| Data visualisation | [Plotly.js](https://plotly.com/javascript/) |
| Testing | [pytest](https://pytest.org/) |

---

## Project Structure

```text
QUANTUM_SIMMULATOR/
│
├── frontend/               # React + TypeScript web dashboard
│   ├── public/
│   └── src/
│       ├── components/     # Reusable UI components
│       ├── pages/          # Route-level page components
│       ├── layouts/        # Page layout wrappers
│       ├── animations/     # Framer Motion animation definitions
│       ├── services/       # API client and service functions
│       ├── hooks/          # Custom React hooks
│       ├── context/        # React context providers
│       └── styles/         # Global and module CSS/SCSS
│
├── backend/                # Python backend
│   ├── api/
│   │   └── routes/         # FastAPI route handlers
│   ├── data/               # Ingestion, schemas, validation, and profiling
│   │   ├── csv_validator.py
│   │   ├── dataset_loader.py
│   │   ├── dataset_profiler.py
│   │   └── schemas.py
│   ├── classical/          # Classical ML model implementations
│   ├── quantum/            # PennyLane quantum circuit implementations
│   ├── training/           # Training loop orchestration
│   ├── evaluation/         # Metrics and evaluation utilities
│   └── storage/            # Result persistence helpers
│
├── configs/                # Experiment configuration files (YAML/JSON)
├── docs/                   # Formal scope, backlog, architecture, schemas, and rules
│   ├── ARCHITECTURE.md
│   ├── CSV_VALIDATION_RULES.md
│   ├── DATA_FLOW.md
│   ├── MVP_SCOPE.md
│   ├── DEVELOPMENT_BACKLOG.md
│   ├── MVP_ACCEPTANCE_CHECKLIST.md
│   └── schemas/
│       ├── experiment_config.schema.json
│       ├── experiment_result.schema.json
│       └── experiment_state.schema.json
├── tests/                  # Pytest test suite (58 automated tests)
├── reports/                # Generated dataset profiles, tables, and figures
│   └── dataset_profiles/
│       ├── breast_cancer/
│       └── wine/
│
├── requirements.txt        # Python dependencies (pinned versions)
├── README.md
└── .gitignore
```

---

## Project Documentation

Detailed project planning, technical architecture, and acceptance criteria are maintained in `docs/`:

- [Architecture Specification](docs/ARCHITECTURE.md) — 8-layer architecture, module responsibilities, SQLite schema, and engineering rules.
- [Data Flow Specification](docs/DATA_FLOW.md) — 11-step experiment lifecycle, state machine, model dispatching, and reproducibility.
- [CSV Validation Rules](docs/CSV_VALIDATION_RULES.md) — Specification for CSV file health checks, target selection, error catalogue, and response schema.
- [JSON Schemas](docs/schemas/) — Formal Draft 2020-12 validation schemas for configuration, results, and execution state.
- [MVP Scope](docs/MVP_SCOPE.md) — Formal specifications for datasets, models, trainability analysis, fair comparison modes, and documented research limitations.
- [Development Backlog](docs/DEVELOPMENT_BACKLOG.md) — Actionable 12-epic development task backlog prioritized across the 8-week schedule.
- [MVP Acceptance Checklist](docs/MVP_ACCEPTANCE_CHECKLIST.md) — Independent verification criteria for final MVP completion.

---

## Getting Started

### Prerequisites

- Python 3.11+ (project developed on Python 3.13)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/BHARATHKUMAR-max/QUANTUM_SIMMULATOR.git
cd QUANTUM_SIMMULATOR
```

### 2. Create the virtual environment

```bash
# Windows
python -m venv .venv

# macOS / Linux
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the test suite

```bash
python -m pytest tests/ -v
```

---

## Dataset Loading, Validation & Profiling

QML Benchmark Studio provides a unified, reusable dataset ingestion pipeline that supports:
1. **Built-in Breast Cancer Wisconsin (Diagnostic)** benchmark (binary classification, 569 instances, 30 features).
2. **Built-in Wine Recognition** benchmark (multiclass classification, 178 instances, 13 features, 3 cultivars).
3. **User-provided CSV datasets** (binary and multiclass, UTF-8 / UTF-8 BOM, numerical and categorical features).

All datasets are encapsulated inside a standardized `DatasetBundle` container with strict target isolation ($X$ contains no target leakage) and semantic feature type identification.

### Python API Usage Examples

#### 1. Loading Built-in Benchmarks

```python
from backend.data import load_dataset, load_breast_cancer_data, load_wine_data

# Load Breast Cancer Wisconsin
bc_bundle = load_dataset("breast_cancer")
print(bc_bundle.X.shape)       # (569, 30)
print(bc_bundle.target_names)  # ['malignant', 'benign']

# Load Wine Recognition
wine_bundle = load_dataset("wine")
print(wine_bundle.X.shape)       # (178, 13)
print(wine_bundle.target_names)  # ['class_0', 'class_1', 'class_2']
```

#### 2. Validating a User-Provided CSV Dataset

Before loading a custom CSV, audit its structural integrity and data health using `validate_csv_dataset`:

```python
from backend.data import validate_csv_dataset

result = validate_csv_dataset(
    file_path="data/my_customer_churn.csv",
    target_column="churn",  # Mandatory explicit target selection
    min_rows=15            # Minimum 15 rows required for train/test splitting
)

print("Status:", result.validation_status)  # 'VALID', 'VALID_WITH_WARNINGS', or 'INVALID'
print("Rows:", result.row_count, "Columns:", result.column_count)
print("Numerical Features:", result.numerical_columns)
print("Categorical Features:", result.categorical_columns)

if not result.is_valid:
    for error in result.errors:
        print(f"[{error.code}] {error.message} -> Action: {error.suggested_action}")
```

#### 3. Loading a Validated User CSV Dataset

Load the CSV into a standardized `DatasetBundle` with original labels and feature types preserved:

```python
from backend.data import load_csv_dataset, load_dataset

# Direct loader
bundle = load_csv_dataset(
    file_path="data/my_customer_churn.csv",
    target_column="churn",
    dataset_name="customer_churn"  # Optional, defaults to file stem
)

# Or via the unified dispatch interface
bundle = load_dataset(
    dataset_name="csv",
    file_path="data/my_customer_churn.csv",
    target_column="churn"
)

print(bundle.dataset_name)     # 'customer_churn'
print(bundle.feature_types)    # {'tenure': 'numerical', 'contract': 'categorical', ...}
print(bundle.target_classes)   # Original labels preserved: ['No', 'Yes']
```

#### 4. Generating and Exporting Dataset Profiles

Compute comprehensive descriptive statistics, class distributions, missingness audits, variance analysis, and Pearson correlation matrices:

```python
from backend.data import profile_dataset, export_profile

# Compute profile
profile = profile_dataset(bundle)

# Export complete report suite (profile.json, CSV summaries, findings.md, 5 PNG plots)
artifacts = export_profile(
    profile=profile,
    bundle=bundle,
    output_dir="reports/dataset_profiles/customer_churn"
)

for filename, filepath in artifacts.items():
    print(f"Generated: {filename} -> {filepath}")
```

### Exported Artifacts

Reports and analytical summaries are exported to `reports/dataset_profiles/<dataset_name>/`:
- `profile.json` — Comprehensive, JSON-serializable dataset metadata and metrics.
- `descriptive_statistics.csv` — Feature-by-feature numerical summaries (mean, std, min, quantiles, max, variance, range).
- `categorical_statistics.csv` — Categorical summaries (counts, unique categories, mode, frequency) when non-numeric columns exist.
- `class_distribution.csv` — Sample counts and proportions across target classes.
- `correlation_matrix.csv` — Pairwise Pearson correlation coefficients on numerical features.
- `findings.md` — Human-readable markdown research brief detailing key distributions, collinearity, and non-leakage guidance.
- Visualization figures (`.png`):
  1. `class_distribution.png` — Target class distribution bar chart with percentages.
  2. `feature_distributions.png` — Class-conditioned histograms across representative features.
  3. `correlation_heatmap.png` — Pearson correlation coefficient heatmap.
  4. `feature_boxplots.png` — Standardized (z-scored) distribution boxplots.
  5. `feature_variance.png` — Logarithmic feature variance comparisons.

---

## Simulator Note

All quantum circuits in this project use PennyLane's **`default.qubit`** simulator. This is a pure-state statevector simulator that runs entirely on classical CPU/GPU hardware. It is chosen for its simplicity, reproducibility, and suitability for small-scale academic benchmarking. No access to real quantum hardware is required.

---

## Development Roadmap

| Week | Milestone |
|---|---|
| 1 | Environment setup and project scaffolding (completed) |
| 2 | Dataset pipeline, CSV validation, and profiling (completed) |
| 3 | Classical ML baseline models |
| 4 | Quantum circuit design and QML models |
| 5 | Training orchestration and benchmarking |
| 6 | Evaluation metrics and barren plateau analysis |
| 7 | FastAPI backend and React dashboard |
| 8 | Final report, documentation, and polish |

---

## License

Academic research project — 6th Semester, 2026.
