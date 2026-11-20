"""
tests/test_result_aggregation.py
================================
Unit tests for repeated-experiment result aggregation and statistical calculations.

Verifies:
- Sample mean is accurately computed.
- Sample standard deviation strictly uses Bessel's correction (ddof=1).
- Single successful run returns std=None, not 0.0.
- Zero successful runs returns mean=None, std=None, min=None, max=None.
- Min and Max reflect true bounds.
- Missing metric values (e.g. None in ROC-AUC) are excluded, not converted to 0.0.
- Failed runs are tracked and do not corrupt valid metric statistics.
- MetricAggregate and aggregate dictionaries are JSON serializable.
"""

import json
import numpy as np
import pytest

from backend.training.aggregation import aggregate_metric, compute_aggregates
from backend.training.schemas import MetricAggregate, RepeatedRunResult


def _create_mock_run(run_number: int, accuracy: float, roc_auc=None, failed=False) -> RepeatedRunResult:
    """Helper to create minimal RepeatedRunResult instances for aggregation testing."""
    return RepeatedRunResult(
        experiment_id="exp_test",
        run_number=run_number,
        random_state=42 + run_number,
        train_indices=[0, 1],
        test_indices=[2],
        n_train=2,
        n_test=1,
        train_class_distribution={"0": 1, "1": 1},
        test_class_distribution={"1": 1},
        model_name="test_model",
        model_config={},
        accuracy=accuracy if not failed else None,
        f1_macro=accuracy if not failed else None,
        roc_auc=roc_auc if not failed else None,
        failed=failed,
        error_message="Simulated run failure" if failed else None,
    )


# ---------------------------------------------------------------------------
# Test: Statistical Correctness
# ---------------------------------------------------------------------------


def test_sample_mean_and_ddof1_std():
    """Mean and sample standard deviation (ddof=1) match numpy calculations."""
    values = [0.80, 0.85, 0.90, 0.75, 0.95]
    total_runs = 5
    agg = aggregate_metric(values, metric_name="accuracy", model_name="svm", total_runs=total_runs)

    expected_mean = float(np.mean(values))
    expected_std = float(np.std(values, ddof=1))
    expected_min = float(np.min(values))
    expected_max = float(np.max(values))

    assert pytest.approx(agg.mean, rel=1e-5) == expected_mean
    assert pytest.approx(agg.std, rel=1e-5) == expected_std
    assert agg.min == expected_min
    assert agg.max == expected_max
    assert agg.successful_runs == 5
    assert agg.failed_runs == 0


def test_single_successful_run_returns_none_std():
    """For sample size N=1, sample standard deviation is undefined and must be None."""
    values = [0.85]
    agg = aggregate_metric(values, metric_name="accuracy", model_name="svm", total_runs=1)

    assert agg.mean == 0.85
    assert agg.std is None
    assert agg.min == 0.85
    assert agg.max == 0.85
    assert agg.successful_runs == 1
    assert agg.failed_runs == 0


def test_all_runs_failed_or_missing():
    """When all runs fail or values are None, all aggregate stats must be None."""
    values = [None, None, None]
    agg = aggregate_metric(values, metric_name="roc_auc", model_name="svm", total_runs=3)

    assert agg.mean is None
    assert agg.std is None
    assert agg.min is None
    assert agg.max is None
    assert agg.successful_runs == 0
    assert agg.failed_runs == 3


def test_missing_metrics_not_imputed_as_zero():
    """Missing metric values (e.g. None for linear SVM ROC-AUC) are excluded, not converted to 0.0."""
    values = [0.90, None, 0.92, None, 0.94]
    agg = aggregate_metric(values, metric_name="roc_auc", model_name="svm", total_runs=5)

    valid_subset = [0.90, 0.92, 0.94]
    assert pytest.approx(agg.mean, rel=1e-5) == float(np.mean(valid_subset))
    assert pytest.approx(agg.std, rel=1e-5) == float(np.std(valid_subset, ddof=1))
    assert agg.successful_runs == 3
    assert agg.failed_runs == 2


# ---------------------------------------------------------------------------
# Test: compute_aggregates across runs
# ---------------------------------------------------------------------------


def test_compute_aggregates_with_partial_failures():
    """Aggregating across runs with partial failures preserves valid runs and tallies failures."""
    runs = [
        _create_mock_run(0, accuracy=0.80, roc_auc=0.88),
        _create_mock_run(1, accuracy=0.84, roc_auc=0.90),
        _create_mock_run(2, accuracy=0.0, failed=True),  # Failed run
        _create_mock_run(3, accuracy=0.86, roc_auc=None),  # Valid run, undefined ROC-AUC
        _create_mock_run(4, accuracy=0.90, roc_auc=0.92),
    ]

    aggs = compute_aggregates(runs, model_name="test_model")

    # Accuracy was valid in 4 runs
    acc_agg = aggs["accuracy"]
    assert acc_agg.successful_runs == 4
    assert acc_agg.failed_runs == 1
    assert pytest.approx(acc_agg.mean, rel=1e-5) == np.mean([0.80, 0.84, 0.86, 0.90])

    # ROC-AUC was valid in 3 runs
    roc_agg = aggs["roc_auc"]
    assert roc_agg.successful_runs == 3
    assert roc_agg.failed_runs == 2
    assert pytest.approx(roc_agg.mean, rel=1e-5) == np.mean([0.88, 0.90, 0.92])


def test_metric_aggregate_to_dict_serializable():
    """MetricAggregate.to_dict() must produce a JSON-serializable structure."""
    values = [0.81, 0.83, 0.85]
    agg = aggregate_metric(values, metric_name="f1_macro", model_name="mlp", total_runs=3)
    d = agg.to_dict()

    serialized = json.dumps(d)
    assert "mean" in d
    assert "std" in d
    assert d["successful_runs"] == 3
    assert d["failed_runs"] == 0
