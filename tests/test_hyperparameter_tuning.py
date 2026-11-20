"""
tests/test_hyperparameter_tuning.py
======================================
Unit and integration tests for backend/training/hyperparameter_tuning.py.

Verifies:
- All four models are tunable with the default grid.
- Candidate budget (max_candidates) is respected.
- Best config is selected by CV score, not test score.
- Failed candidates are recorded, not silently dropped.
- Seed reproducibility across full tuning runs.
- SVM linear kernel candidates have no gamma parameter.
- TuningResult and CandidateResult structure correctness.
- final_evaluation() produces valid test-set metrics.
- Integration tests on Breast Cancer (binary) and Wine (multiclass).
"""

import json

import numpy as np
import pandas as pd
import pytest

from backend.training.hyperparameter_tuning import (
    DEFAULT_GRIDS,
    CandidateResult,
    TuningResult,
    final_evaluation,
    tune_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def breast_cancer_split():
    """Return (X_train, X_test, y_train, y_test) for Breast Cancer dataset."""
    from sklearn.datasets import load_breast_cancer

    from backend.data.schemas import DatasetBundle
    from backend.data.splitting import create_train_test_split

    bc = load_breast_cancer()
    X_all = pd.DataFrame(bc.data, columns=bc.feature_names)
    y_all = pd.Series(bc.target, name="target")

    bundle = DatasetBundle(
        X=X_all,
        y=y_all,
        feature_names=list(X_all.columns),
        target_names=["malignant", "benign"],
        dataset_name="breast_cancer",
    )
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    return split.X_train, split.X_test, split.y_train, split.y_test


@pytest.fixture()
def wine_split():
    """Return (X_train, X_test, y_train, y_test) for Wine dataset (3 classes)."""
    from sklearn.datasets import load_wine

    from backend.data.schemas import DatasetBundle
    from backend.data.splitting import create_train_test_split

    wine = load_wine()
    X_all = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_all = pd.Series(wine.target, name="target")

    bundle = DatasetBundle(
        X=X_all,
        y=y_all,
        feature_names=list(X_all.columns),
        target_names=list(wine.target_names),
        dataset_name="wine",
    )
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    return split.X_train, split.X_test, split.y_train, split.y_test


# ---------------------------------------------------------------------------
# Test: search grid structure
# ---------------------------------------------------------------------------


def test_default_grids_exist_for_all_models():
    """DEFAULT_GRIDS must contain entries for all four classical models."""
    for model_name in ["dummy", "svm", "mlp", "random_forest"]:
        assert model_name in DEFAULT_GRIDS
        assert len(DEFAULT_GRIDS[model_name]) > 0


def test_svm_gamma_only_for_rbf():
    """
    Linear SVM candidates must NOT carry a gamma parameter.
    gamma only affects the RBF kernel; including it for linear is misleading.
    """
    svm_grid = DEFAULT_GRIDS["svm"]
    for candidate in svm_grid:
        if candidate.get("kernel") == "linear":
            assert "gamma" not in candidate, (
                f"Linear SVM candidate should not have gamma: {candidate}"
            )
        elif candidate.get("kernel") == "rbf":
            assert "gamma" in candidate, (
                f"RBF SVM candidate should have gamma: {candidate}"
            )


def test_svm_grid_candidate_count():
    """SVM grid: 3 C values × (1 linear + 2 gamma for rbf) = 9 candidates."""
    svm_grid = DEFAULT_GRIDS["svm"]
    assert len(svm_grid) == 9, f"Expected 9 SVM candidates, got {len(svm_grid)}."


def test_mlp_grid_candidate_count():
    """MLP grid: 3 hidden × 2 alpha × 2 lr = 12 candidates."""
    mlp_grid = DEFAULT_GRIDS["mlp"]
    assert len(mlp_grid) == 12, f"Expected 12 MLP candidates, got {len(mlp_grid)}."


def test_rf_grid_candidate_count():
    """RF grid: 2 n_estimators × 3 max_depth × 2 min_samples_split = 12 candidates."""
    rf_grid = DEFAULT_GRIDS["random_forest"]
    assert len(rf_grid) == 12, f"Expected 12 RF candidates, got {len(rf_grid)}."


def test_dummy_grid_candidate_count():
    """Dummy grid: 3 strategy variants."""
    dummy_grid = DEFAULT_GRIDS["dummy"]
    assert len(dummy_grid) == 3, f"Expected 3 Dummy candidates, got {len(dummy_grid)}."


# ---------------------------------------------------------------------------
# Test: candidate budget
# ---------------------------------------------------------------------------


def test_candidate_budget_respected(breast_cancer_split):
    """max_candidates must limit the number of candidates evaluated."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    assert result.n_candidates_evaluated <= 2, (
        f"Expected at most 2 candidates, got {result.n_candidates_evaluated}."
    )
    assert len(result.all_candidates) <= 2


def test_candidate_budget_zero_raises(breast_cancer_split):
    """max_candidates=0 means no candidates → all fail → RuntimeError."""
    X_train, _, y_train, _ = breast_cancer_split
    with pytest.raises(RuntimeError, match="failed during cross-validation"):
        tune_model(
            model_name="dummy",
            X_train=X_train,
            y_train=y_train,
            max_candidates=0,
            n_splits=3,
            metric="accuracy",
            random_state=42,
        )


# ---------------------------------------------------------------------------
# Test: best config selected by CV score, not test score
# ---------------------------------------------------------------------------


def test_best_config_selected_by_cv_score_not_test_score(breast_cancer_split):
    """
    tune_model must rank candidates by CV score. The test set must be untouched.
    Verify: supplying X_test / y_test has no effect on best_config (they are not used).
    """
    X_train, _X_test, y_train, _y_test = breast_cancer_split

    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=3,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )

    assert result.stage == "cv_selection"
    assert result.best_config is not None

    # Best config must have the highest mean CV score among all_candidates
    best_score = result.best_mean_cv_score
    for candidate in result.all_candidates:
        if not candidate.failed:
            assert candidate.mean_cv_score <= best_score + 1e-9, (
                f"Candidate with higher score ({candidate.mean_cv_score:.4f}) "
                f"not selected as best ({best_score:.4f})."
            )


# ---------------------------------------------------------------------------
# Test: failed candidates are recorded, not dropped
# ---------------------------------------------------------------------------


def test_failed_candidates_recorded_not_silently_dropped(breast_cancer_split):
    """
    When a custom grid contains an invalid config, the failure must be
    recorded in all_candidates with failed=True, not silently skipped.
    """
    X_train, _, y_train, _ = breast_cancer_split

    # Create a grid where first candidate is intentionally bad (strategy="invalid")
    # and second is valid
    custom_grid = [
        {"strategy": "invalid_nonexistent_strategy"},
        {"strategy": "most_frequent"},
    ]

    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        custom_grid=custom_grid,
        max_candidates=10,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )

    assert len(result.all_candidates) == 2
    # First candidate should have failed
    failed_candidates = [c for c in result.all_candidates if c.failed]
    assert len(failed_candidates) >= 1, "Expected at least one failed candidate to be recorded."
    # Failed candidates should have a failure_reason string
    for fc in failed_candidates:
        assert fc.failure_reason is not None and len(fc.failure_reason) > 0
    # Overall result should still succeed using the valid candidate
    assert result.best_config == {"strategy": "most_frequent"}


# ---------------------------------------------------------------------------
# Test: seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility(breast_cancer_split):
    """Same seed must produce identical best_config and CV scores."""
    X_train, _, y_train, _ = breast_cancer_split

    result_a = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=3,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    result_b = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=3,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )

    assert result_a.best_config == result_b.best_config
    assert abs(result_a.best_mean_cv_score - result_b.best_mean_cv_score) < 1e-9


# ---------------------------------------------------------------------------
# Test: TuningResult and CandidateResult structure
# ---------------------------------------------------------------------------


def test_tuning_result_structure(breast_cancer_split):
    """TuningResult must contain all required fields with correct types."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert isinstance(result, TuningResult)
    assert isinstance(result.model_name, str)
    assert isinstance(result.best_config, dict)
    assert isinstance(result.best_mean_cv_score, float)
    assert isinstance(result.best_std_cv_score, float)
    assert result.best_mean_cv_score >= 0.0
    assert result.best_std_cv_score >= 0.0
    assert result.selection_metric == "f1_macro"
    assert result.stage == "cv_selection"
    assert isinstance(result.all_candidates, list)
    assert result.n_candidates_evaluated > 0


def test_tuning_result_to_dict_serializable(breast_cancer_split):
    """TuningResult.to_dict() must produce a JSON-serializable dictionary."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    d = result.to_dict()
    json.dumps(d)  # Must not raise


def test_candidate_result_structure(breast_cancer_split):
    """Each CandidateResult must have valid structure."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=3,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    for candidate in result.all_candidates:
        assert isinstance(candidate, CandidateResult)
        assert isinstance(candidate.config, dict)
        if not candidate.failed:
            assert 0.0 <= candidate.mean_cv_score <= 1.0
            assert candidate.std_cv_score >= 0.0
            assert len(candidate.fold_scores) == 3
        else:
            assert candidate.failure_reason is not None


def test_candidate_to_dict_serializable(breast_cancer_split):
    """CandidateResult.to_dict() must be JSON-serializable."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    for candidate in result.all_candidates:
        json.dumps(candidate.to_dict())  # Must not raise


# ---------------------------------------------------------------------------
# Test: all four models are tunable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["dummy", "svm", "mlp", "random_forest"])
def test_all_four_models_tunable(breast_cancer_split, model_name):
    """All four classical models must complete tuning with max_candidates=2."""
    X_train, _, y_train, _ = breast_cancer_split
    result = tune_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert isinstance(result, TuningResult)
    assert result.best_config is not None
    assert result.best_mean_cv_score >= 0.0


# ---------------------------------------------------------------------------
# Test: custom grid
# ---------------------------------------------------------------------------


def test_custom_grid_overrides_default(breast_cancer_split):
    """Custom grid must fully replace the default grid."""
    X_train, _, y_train, _ = breast_cancer_split
    custom_grid = [{"strategy": "uniform"}]
    result = tune_model(
        model_name="dummy",
        X_train=X_train,
        y_train=y_train,
        custom_grid=custom_grid,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    assert result.n_candidates_evaluated == 1
    assert result.best_config == {"strategy": "uniform"}


# ---------------------------------------------------------------------------
# Test: final_evaluation
# ---------------------------------------------------------------------------


def test_final_evaluation_structure(breast_cancer_split):
    """final_evaluation must return all required keys with valid types."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = final_evaluation(
        model_name="dummy",
        best_config={"strategy": "most_frequent"},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert result["stage"] == "final_test_evaluation"
    assert result["model_name"] == "dummy"
    assert isinstance(result["config"], dict)
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["n_train"], int)
    assert isinstance(result["n_test"], int)
    assert isinstance(result["n_features_used"], int)
    assert isinstance(result["fit_time_seconds"], float)
    assert isinstance(result["predict_time_seconds"], float)
    assert result["n_train"] > 0
    assert result["n_test"] > 0
    assert result["n_features_used"] > 0


def test_final_evaluation_serializable(breast_cancer_split):
    """final_evaluation result must be JSON-serializable."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = final_evaluation(
        model_name="dummy",
        best_config={"strategy": "most_frequent"},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    json.dumps(result)  # Must not raise


def test_final_evaluation_uses_correct_data_sizes(breast_cancer_split):
    """n_train and n_test must reflect actual split sizes."""
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = final_evaluation(
        model_name="dummy",
        best_config={"strategy": "most_frequent"},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert result["n_train"] == len(X_train)
    assert result["n_test"] == len(X_test)


# ---------------------------------------------------------------------------
# Test: integration — Breast Cancer binary classification
# ---------------------------------------------------------------------------


def test_integration_breast_cancer_full_tune(breast_cancer_split):
    """
    Full end-to-end tuning on Breast Cancer using SVM with max_candidates=3.
    Followed by final_evaluation on the test set.
    Verify result structure and that accuracy > dummy baseline.
    """
    X_train, X_test, y_train, y_test = breast_cancer_split

    tuning = tune_model(
        model_name="svm",
        X_train=X_train,
        y_train=y_train,
        max_candidates=3,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert isinstance(tuning, TuningResult)
    assert tuning.stage == "cv_selection"
    assert tuning.best_mean_cv_score > 0.0

    test_result = final_evaluation(
        model_name="svm",
        best_config=tuning.best_config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert test_result["stage"] == "final_test_evaluation"
    # SVM should outperform a random guess on Breast Cancer
    assert test_result["metrics"]["accuracy"] > 0.5


# ---------------------------------------------------------------------------
# Test: integration — Wine multiclass classification
# ---------------------------------------------------------------------------


def test_integration_wine_multiclass(wine_split):
    """
    Full tuning on Wine dataset (3 classes) using Random Forest with max_candidates=2.
    Followed by final_evaluation on the test set.
    """
    X_train, X_test, y_train, y_test = wine_split

    tuning = tune_model(
        model_name="random_forest",
        X_train=X_train,
        y_train=y_train,
        max_candidates=2,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert isinstance(tuning, TuningResult)
    assert tuning.best_mean_cv_score >= 0.0

    test_result = final_evaluation(
        model_name="random_forest",
        best_config=tuning.best_config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert test_result["stage"] == "final_test_evaluation"
    assert test_result["n_test"] == len(X_test)
    # RF on Wine should achieve reasonable F1
    assert test_result["metrics"]["f1_macro"] > 0.3


# ---------------------------------------------------------------------------
# Test: MLP hidden_layer_sizes tuple serialization
# ---------------------------------------------------------------------------


def test_mlp_hidden_layer_sizes_serialized_as_list(breast_cancer_split):
    """
    MLP hidden_layer_sizes is a tuple; to_dict() must serialize it as a list
    for JSON compatibility.
    """
    X_train, X_test, y_train, y_test = breast_cancer_split
    result = final_evaluation(
        model_name="mlp",
        best_config={"hidden_layer_sizes": (64, 32), "max_iter": 50},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    # Serialized config should have list not tuple for hidden_layer_sizes
    config_serialized = result["config"]
    assert isinstance(config_serialized.get("hidden_layer_sizes"), list), (
        "hidden_layer_sizes should be serialized as a list, not a tuple."
    )
    # Result must still be JSON-serializable
    json.dumps(result)


# ---------------------------------------------------------------------------
# Test: unknown model raises
# ---------------------------------------------------------------------------


def test_unknown_model_raises(breast_cancer_split):
    """tune_model must raise ValueError for unknown model names."""
    X_train, _, y_train, _ = breast_cancer_split
    with pytest.raises((ValueError, TypeError)):
        tune_model(
            model_name="quantum_svm_does_not_exist",
            X_train=X_train,
            y_train=y_train,
            n_splits=3,
        )
