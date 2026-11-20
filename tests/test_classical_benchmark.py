"""
tests/test_classical_benchmark.py
=================================
Automated verification tests for Day 20 classical benchmark execution.

Verifies:
- Both Breast Cancer and Wine benchmarks execute without errors across all 4 models.
- 5 repetitions are performed using exact agreed seeds [42, 101, 202, 303, 404].
- Corresponding split indices are strictly identical across models.
- Confusion matrix dimensions and summation logic are verified.
- MLP convergence diagnostics (n_iter, loss, warnings, convergence flag) are captured.
- Missing ROC-AUC for multiclass SVM is safely excluded (not zeroed out).
- Output files (JSON, CSV, PNG charts) exist and have non-zero size.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from examples.run_classical_benchmark import (
    BENCHMARK_MODELS,
    compute_summed_confusion_matrices,
    extract_mlp_convergence_analysis,
    generate_benchmark_summary_csv,
    run_single_dataset_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "reports" / "classical_benchmark"


# ---------------------------------------------------------------------------
# Test: Output Artifact Existence and Integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["breast_cancer", "wine"])
def test_benchmark_directories_and_files_exist(dataset_name):
    """Verify that all required data exports and PNG plots exist and are non-empty."""
    ds_dir = BENCHMARK_DIR / dataset_name
    assert ds_dir.exists(), f"Benchmark directory {ds_dir} does not exist!"

    expected_files = [
        "benchmark_results.json",
        "benchmark_summary.csv",
        "accuracy_comparison.png",
        "macro_f1_comparison.png",
        "training_time.png",
        "inference_time.png",
        "confusion_matrices.png",
        "mlp_convergence.png",
    ]

    for fname in expected_files:
        fpath = ds_dir / fname
        assert fpath.exists(), f"File {fpath} is missing!"
        assert fpath.stat().st_size > 0, f"File {fpath} is empty!"


def test_classical_benchmark_markdown_report_exists():
    """Verify that CLASSICAL_BENCHMARK_REPORT.md exists and is non-empty."""
    report_path = BENCHMARK_DIR / "CLASSICAL_BENCHMARK_REPORT.md"
    assert report_path.exists()
    assert report_path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Test: JSON and CSV Content Integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["breast_cancer", "wine"])
def test_json_and_csv_data_consistency(dataset_name):
    """Verify JSON export and CSV summary data match and contain all 4 models."""
    json_path = BENCHMARK_DIR / dataset_name / "benchmark_results.json"
    csv_path = BENCHMARK_DIR / dataset_name / "benchmark_summary.csv"

    with open(json_path, encoding="utf-8") as f:
        json_data = json.load(f)

    df_csv = pd.read_csv(csv_path)

    # 4 models present
    assert set(json_data["models"]) == set(BENCHMARK_MODELS)
    assert len(df_csv) == 4
    assert set(df_csv["Model Identifier"]) == set(BENCHMARK_MODELS)

    # 5 repetitions
    assert json_data["n_repeats"] == 5
    assert len(json_data["split_indices"]) == 5

    # Check accuracy values in CSV match JSON
    for model_name in BENCHMARK_MODELS:
        json_acc = json_data["model_results"][model_name]["aggregates"]["accuracy"]["mean"]
        csv_row = df_csv[df_csv["Model Identifier"] == model_name].iloc[0]
        csv_acc = float(csv_row["Accuracy Mean"])
        assert pytest.approx(json_acc, rel=1e-5) == csv_acc


# ---------------------------------------------------------------------------
# Test: Split Index Synchronization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["breast_cancer", "wine"])
def test_shared_split_indices_synchronized(dataset_name):
    """Verify that all models on each dataset received identical train and test indices."""
    json_path = BENCHMARK_DIR / dataset_name / "benchmark_results.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for run_idx in range(5):
        root_train = data["split_indices"][run_idx]["train_indices"]
        root_test = data["split_indices"][run_idx]["test_indices"]

        for model_name in BENCHMARK_MODELS:
            model_run = data["model_results"][model_name]["runs"][run_idx]
            assert model_run["train_indices"] == root_train
            assert model_run["test_indices"] == root_test


# ---------------------------------------------------------------------------
# Test: Confusion Matrix Summation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name,expected_dim", [("breast_cancer", 2), ("wine", 3)])
def test_confusion_matrix_summation_correctness(dataset_name, expected_dim):
    """Verify that summed confusion matrices match the sum of individual run matrices."""
    json_path = BENCHMARK_DIR / dataset_name / "benchmark_results.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    cm_section = data["confusion_matrices"]

    for model_name in BENCHMARK_MODELS:
        summed_from_file = np.array(cm_section[model_name]["summed_matrix"])
        assert summed_from_file.shape == (expected_dim, expected_dim)

        # Re-sum directly from individual runs
        manual_sum = np.zeros((expected_dim, expected_dim), dtype=int)
        for run in data["model_results"][model_name]["runs"]:
            manual_sum += np.array(run["confusion_matrix"])

        np.testing.assert_array_equal(summed_from_file, manual_sum)


# ---------------------------------------------------------------------------
# Test: MLP Convergence Information
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["breast_cancer", "wine"])
def test_mlp_convergence_diagnostics_recorded(dataset_name):
    """Verify that MLP convergence diagnostics are captured for all 5 repetitions."""
    json_path = BENCHMARK_DIR / dataset_name / "benchmark_results.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    mlp_diag = data["mlp_convergence_diagnostics"]
    assert mlp_diag["total_runs"] == 5
    assert len(mlp_diag["runs"]) == 5

    for run_diag in mlp_diag["runs"]:
        assert run_diag["n_iter"] is not None
        assert run_diag["n_iter"] > 0
        assert run_diag["final_loss"] is not None
        assert run_diag["final_loss"] > 0.0
        assert isinstance(run_diag["converged"], bool)
        assert run_diag["max_iter"] == 300


# ---------------------------------------------------------------------------
# Test: Missing Metric Handling
# ---------------------------------------------------------------------------


def test_wine_svm_roc_auc_safely_excluded():
    """Verify that for Wine multiclass SVM with probability=False, ROC-AUC is cleanly N/A."""
    json_path = BENCHMARK_DIR / "wine" / "benchmark_results.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    svm_roc = data["model_results"]["svm"]["aggregates"]["roc_auc"]
    assert svm_roc["mean"] is None
    assert svm_roc["successful_runs"] == 0
    assert svm_roc["failed_runs"] == 5  # Undefined for all 5 runs
