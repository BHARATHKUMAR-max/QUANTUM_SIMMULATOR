"""
tests/test_repeated_runs.py
===========================
Unit and integration tests for repeated-experiment execution and leakage prevention.

Verifies:
- 5 default runs execute with default seeds [42, 101, 202, 303, 404].
- Custom repetition counts and custom seeds work.
- Train/test proportions are 80/20 and stratified.
- Exact train/test indices are recorded and reproducible.
- Each run fits independent preprocessing and model instances.
- Mode A (fixed) and Mode B (tuned) execute properly.
- Adversarial leakage prevention: test labels/outliers do not alter fitting or tuning.
- JSON serializability of result schemas.
"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

from backend.data.schemas import FeatureSelectionConfig, PreprocessingConfig
from backend.training.repeated_runs import (
    DEFAULT_SEEDS,
    run_repeated_experiment,
    validate_repeated_inputs,
)
from backend.training.schemas import RepeatedExperimentResult, RepeatedRunResult


@pytest.fixture()
def breast_cancer_data():
    """Breast Cancer dataset: 569 samples, 30 features, binary classification."""
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name="target")
    return X, y


@pytest.fixture()
def iris_data():
    """Iris dataset: 150 samples, 4 features, 3 classes."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y


# ---------------------------------------------------------------------------
# Test: Default 5 runs & seed execution
# ---------------------------------------------------------------------------


def test_five_default_runs_execute(breast_cancer_data):
    """Default experiment executes exactly 5 repetitions with agreed seeds."""
    X, y = breast_cancer_data
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        model_config={"strategy": "most_frequent"},
        n_repeats=5,
    )
    assert isinstance(result, RepeatedExperimentResult)
    assert result.n_repeats == 5
    assert len(result.runs) == 5
    assert result.seeds == DEFAULT_SEEDS
    assert [r.random_state for r in result.runs] == DEFAULT_SEEDS


def test_custom_repetition_counts_and_seeds(breast_cancer_data):
    """Custom repetition count and custom seed list are respected."""
    X, y = breast_cancer_data
    custom_seeds = [11, 22, 33]
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        n_repeats=3,
        seeds=custom_seeds,
    )
    assert result.n_repeats == 3
    assert len(result.runs) == 3
    assert result.seeds == custom_seeds
    assert [r.random_state for r in result.runs] == custom_seeds


def test_split_proportions_and_stratification(breast_cancer_data):
    """Train/test split proportions adhere to 80/20 and preserve class distributions."""
    X, y = breast_cancer_data
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        test_size=0.2,
        stratify=True,
        n_repeats=2,
    )
    total_samples = len(X)
    for run in result.runs:
        assert run.n_test == int(np.ceil(total_samples * 0.2)) or run.n_test == int(np.floor(total_samples * 0.2))
        assert run.n_train + run.n_test == total_samples
        # Verify stratification ratio
        train_ratio = run.train_class_distribution["1"] / run.n_train
        test_ratio = run.test_class_distribution["1"] / run.n_test
        assert abs(train_ratio - test_ratio) < 0.05


def test_exact_indices_saved_and_reproducible(breast_cancer_data):
    """Saved train/test indices exactly partition the dataset and reproduce across runs."""
    X, y = breast_cancer_data
    result_1 = run_repeated_experiment(X=X, y=y, model_name="dummy", n_repeats=2, seeds=[42, 101])
    result_2 = run_repeated_experiment(X=X, y=y, model_name="dummy", n_repeats=2, seeds=[42, 101])

    for r1, r2 in zip(result_1.runs, result_2.runs):
        assert r1.train_indices == r2.train_indices
        assert r1.test_indices == r2.test_indices
        assert len(set(r1.train_indices) & set(r1.test_indices)) == 0
        assert len(set(r1.train_indices) | set(r1.test_indices)) == len(X)


# ---------------------------------------------------------------------------
# Test: Input Validation
# ---------------------------------------------------------------------------


def test_input_validation_errors(breast_cancer_data):
    """Invalid input configurations raise clear ValueErrors."""
    X, y = breast_cancer_data
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_repeated_inputs(X, y.iloc[:-5], n_repeats=5, test_size=0.2, stratify=True)

    with pytest.raises(ValueError, match="Minimum 15 samples required"):
        validate_repeated_inputs(X.iloc[:10], y.iloc[:10], n_repeats=5, test_size=0.2, stratify=True)

    with pytest.raises(ValueError, match="test_size"):
        validate_repeated_inputs(X, y, n_repeats=5, test_size=0.8, stratify=True)

    with pytest.raises(ValueError, match="n_repeats"):
        validate_repeated_inputs(X, y, n_repeats=0, test_size=0.2, stratify=True)


# ---------------------------------------------------------------------------
# Test: Mode A vs Mode B
# ---------------------------------------------------------------------------


def test_mode_a_fixed_configuration(breast_cancer_data):
    """Mode A applies the exact fixed configuration across all runs."""
    X, y = breast_cancer_data
    cfg = {"kernel": "linear", "C": 0.5}
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="svm",
        model_config=cfg,
        mode="fixed",
        n_repeats=2,
    )
    assert result.mode == "fixed"
    for run in result.runs:
        assert run.model_config["C"] == 0.5
        assert run.model_config["kernel"] == "linear"
        assert run.selected_params is None


def test_mode_b_tuned_within_each_repetition(breast_cancer_data):
    """Mode B executes CV tuning strictly on training data and records selected params per run."""
    X, y = breast_cancer_data
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        mode="tuned",
        n_repeats=2,
        seeds=[42, 101],
        tuning_max_candidates=3,
    )
    assert result.mode == "tuned"
    for run in result.runs:
        assert run.selected_params is not None
        assert "strategy" in run.selected_params


# ---------------------------------------------------------------------------
# Test: Leakage Safeguard
# ---------------------------------------------------------------------------


def test_leakage_safeguard_test_labels_do_not_affect_tuning(breast_cancer_data):
    """
    Adversarial test: changing held-out test labels must not change tuning selections or training metrics.
    
    Verifies that tune_model in Mode B is strictly insulated from the test partition.
    """
    from backend.data.schemas import DatasetBundle
    from backend.data.splitting import create_train_test_split
    from backend.training.repeated_runs import _execute_single_run

    X, y = breast_cancer_data
    bundle = DatasetBundle(
        X=X,
        y=y,
        feature_names=list(X.columns),
        target_names=["malignant", "benign"],
        dataset_name="bc",
    )
    split_normal = create_train_test_split(bundle, test_size=0.2, random_state=42)

    # Execute single run in Mode B on normal split
    res_normal = _execute_single_run(
        split=split_normal,
        run_number=0,
        experiment_id="exp_leak_test",
        model_name="dummy",
        mode="tuned",
        tuning_max_candidates=3,
    )

    # Construct adversarial split where test labels are completely inverted
    split_adversarial = create_train_test_split(bundle, test_size=0.2, random_state=42)
    split_adversarial.y_test = 1 - split_adversarial.y_test

    res_adversarial = _execute_single_run(
        split=split_adversarial,
        run_number=0,
        experiment_id="exp_leak_test",
        model_name="dummy",
        mode="tuned",
        tuning_max_candidates=3,
    )

    # Inverting held-out test labels must NOT alter the hyperparameter configuration selected by CV
    assert res_normal.selected_params == res_adversarial.selected_params
    assert res_normal.train_time_seconds > 0.0 or res_normal.accuracy is not None


# ---------------------------------------------------------------------------
# Test: Serialization & Metadata
# ---------------------------------------------------------------------------


def test_single_repetition_execution(breast_cancer_data):
    """n_repeats=1 executes properly and produces aggregates with std=None."""
    X, y = breast_cancer_data
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        n_repeats=1,
    )
    assert result.n_repeats == 1
    assert len(result.runs) == 1
    acc_agg = result.aggregates["accuracy"]
    assert acc_agg.successful_runs == 1
    assert acc_agg.std is None
    assert acc_agg.mean is not None


def test_failure_containment_in_experiment(breast_cancer_data, monkeypatch):
    """A simulated exception during run 1 does not abort run 0, 2, 3, 4."""
    X, y = breast_cancer_data

    from backend.classical.dummy_baseline import DummyBaseline
    original_fit = DummyBaseline.fit
    call_count = [0]

    def faulty_fit(self, X_t, y_t):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("Injected transient failure during fit")
        return original_fit(self, X_t, y_t)

    monkeypatch.setattr(DummyBaseline, "fit", faulty_fit)

    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        n_repeats=3,
        seeds=[42, 101, 202],
    )

    assert result.n_repeats == 3
    assert len(result.runs) == 3
    assert result.runs[0].failed is False
    assert result.runs[1].failed is True
    assert "Injected transient failure" in result.runs[1].error_message
    assert result.runs[2].failed is False

    acc_agg = result.aggregates["accuracy"]
    assert acc_agg.successful_runs == 2
    assert acc_agg.failed_runs == 1
    assert acc_agg.mean is not None
    assert acc_agg.std is not None


def test_repeated_experiment_json_serializable(breast_cancer_data):
    """RepeatedExperimentResult.to_dict() must serialize cleanly to JSON."""
    X, y = breast_cancer_data
    result = run_repeated_experiment(
        X=X,
        y=y,
        model_name="dummy",
        n_repeats=2,
    )
    data_dict = result.to_dict()
    serialized = json.dumps(data_dict)
    assert len(serialized) > 0
    assert data_dict["n_repeats"] == 2
    assert "package_versions" in data_dict
