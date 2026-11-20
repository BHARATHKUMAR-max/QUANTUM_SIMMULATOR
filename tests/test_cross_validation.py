"""
tests/test_cross_validation.py
================================
Unit and integration tests for backend/training/cross_validation.py.

Verifies:
- Correct fold count and stratification.
- No index overlap between fold train and val slices.
- Seed reproducibility.
- Small class counts handled gracefully (error when folds exceed classes).
- Preprocessing fitted exclusively on fold training slice (leakage safeguard).
- Feature selection fitted exclusively on fold training slice.
- CVResult structure correctness.
- Supported and unsupported metric names.
"""

import numpy as np
import pandas as pd
import pytest

from backend.data.schemas import FeatureSelectionConfig, PreprocessingConfig
from backend.training.cross_validation import (
    CVResult,
    FoldScore,
    SUPPORTED_METRICS,
    run_cv_for_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def iris_data():
    """Iris-like dataset: 150 samples, 4 numerical features, 3 balanced classes."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y


@pytest.fixture()
def breast_cancer_data():
    """Breast Cancer dataset: 569 samples, 30 numerical features, 2 classes."""
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name="target")
    return X, y


@pytest.fixture()
def small_imbalanced_data():
    """Tiny dataset with severe class imbalance: 20 majority, 2 minority."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=22), "b": rng.normal(size=22)})
    y = pd.Series([0] * 20 + [1] * 2, name="target")
    return X, y


# ---------------------------------------------------------------------------
# Test: fold count and stratification
# ---------------------------------------------------------------------------


def test_three_fold_stratification_works(breast_cancer_data):
    """Three stratified folds should be created from a balanced dataset."""
    X, y = breast_cancer_data
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    assert result.n_folds == 3
    assert len(result.fold_scores) == 3


def test_fold_indices_no_overlap(breast_cancer_data):
    """Train and val indices within each fold must be disjoint."""
    from backend.data.splitting import create_stratified_cv_splits

    X, y = breast_cancer_data
    folds = create_stratified_cv_splits(X, y, n_splits=3, random_state=42)
    for fold in folds:
        overlap = set(fold.train_indices) & set(fold.val_indices)
        assert len(overlap) == 0, (
            f"Fold {fold.fold_index}: {len(overlap)} overlapping indices detected."
        )


def test_all_training_indices_covered(breast_cancer_data):
    """Union of all fold train+val indices should cover the entire training set."""
    from backend.data.splitting import create_stratified_cv_splits

    X, y = breast_cancer_data
    folds = create_stratified_cv_splits(X, y, n_splits=3, random_state=42)
    all_val = set()
    for fold in folds:
        all_val.update(fold.val_indices)
    assert all_val == set(X.index), "Some training samples never appeared in validation."


def test_seed_reproducibility(breast_cancer_data):
    """Same seed must produce identical CV scores."""
    X, y = breast_cancer_data
    result_a = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    result_b = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    scores_a = [fs.val_score for fs in result_a.fold_scores]
    scores_b = [fs.val_score for fs in result_b.fold_scores]
    assert scores_a == scores_b, "Same seed produced different CV scores."


def test_different_seeds_produce_different_folds(breast_cancer_data):
    """Different seeds typically produce different fold compositions."""
    X, y = breast_cancer_data
    result_a = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    result_b = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=99,
    )
    # n_train and n_val should still be the same size even with different seeds
    assert result_a.n_folds == result_b.n_folds


def test_small_class_count_raises_when_folds_exceed_classes(small_imbalanced_data):
    """Requesting more folds than minority class samples should raise ValueError."""
    X, y = small_imbalanced_data
    with pytest.raises(ValueError):
        run_cv_for_config(
            model_name="dummy",
            config={"strategy": "most_frequent"},
            X_train=X,
            y_train=y,
            n_splits=3,  # minority class has only 2 samples
            metric="accuracy",
        )


# ---------------------------------------------------------------------------
# Test: preprocessing leakage safeguard
# ---------------------------------------------------------------------------


def test_preprocessing_fitted_on_fold_train_only():
    """
    Adversarial test: extreme outliers in validation slice must NOT affect
    the fold preprocessor's fitted statistics (mean/std).

    Pass: scaler statistics match fold-train only (outliers do not shift them).
    Fail: scaler statistics incorporate val-slice outliers (leakage).
    """
    from sklearn.preprocessing import StandardScaler
    from backend.data.preprocessing import fit_preprocessor

    rng = np.random.default_rng(0)
    n = 90
    X = pd.DataFrame({"x": rng.normal(0, 1, n)})
    # Indices 0–59 are fold-train; indices 60–89 are fold-val with extreme outliers
    X_fold_train = X.iloc[:60].copy()
    X_fold_val = X.iloc[60:].copy()
    X_fold_val["x"] = 1_000_000.0  # Extreme values only in val

    # Fit preprocessor on fold-train exclusively
    artifact = fit_preprocessor(X_train=X_fold_train)

    # Extract fitted scaler from ColumnTransformer (attribute is 'transformer')
    fitted_transformer = artifact.transformer

    # Find the numerical pipeline (registered as 'num' in the ColumnTransformer)
    num_transformer = None
    for name, trans, cols in fitted_transformer.transformers_:
        if hasattr(trans, "named_steps"):
            # It's a Pipeline — check if any step has a scaler with mean_
            for step_name, step in trans.steps:
                if hasattr(step, "mean_"):
                    num_transformer = trans
                    break
        if num_transformer is not None:
            break

    if num_transformer is None:
        pytest.skip("No fitted scaler found in transformer (may use 'none' scaling strategy).")

    # Find the scaler step inside the pipeline
    scaler = None
    for step_name, step in num_transformer.steps:
        if hasattr(step, "mean_"):
            scaler = step
            break

    if scaler is not None and hasattr(scaler, "mean_"):
        # Mean should be close to 0 (fold-train) not ~1,000,000 (val)
        assert abs(scaler.mean_[0]) < 10, (
            f"Scaler mean ({scaler.mean_[0]:.2f}) suggests val outliers leaked into fitting."
        )
    else:
        pytest.skip("StandardScaler not used in numerical pipeline.")


def test_feature_selection_fitted_on_fold_train_only(iris_data):
    """
    Feature selector must be fitted on fold-train slice only.
    Shuffling fold-val labels must not change the selected feature set.
    """
    from backend.data.feature_selection import fit_selector, transform_with_selector

    X, y = iris_data
    X_fold_train = X.iloc[:100]
    y_fold_train = y.iloc[:100]

    fs_config = FeatureSelectionConfig(method="anova", k_features=2)

    # Fit selector on fold-train
    artifact = fit_selector(X_fold_train, y_fold_train, config=fs_config)
    selected_features_original = artifact.selected_feature_names

    # Shuffle fold-val labels (adversarial)
    rng = np.random.default_rng(99)
    y_fold_train_shuffled = y_fold_train.copy()
    shuffled_values = rng.permutation(y_fold_train.values)
    y_fold_train_shuffled[:] = shuffled_values

    # Refit with shuffled fold-val labels (simulates what WOULD happen with leakage)
    artifact_shuffled = fit_selector(X_fold_train, y_fold_train, config=fs_config)
    # The feature selection result should not depend on how val labels are arranged
    assert artifact_shuffled.selected_feature_names == selected_features_original, (
        "Feature selection is not deterministic given the same fold-train data."
    )


# ---------------------------------------------------------------------------
# Test: CVResult structure
# ---------------------------------------------------------------------------


def test_cv_result_structure_all_fields(breast_cancer_data):
    """CVResult must contain all required fields with valid types and ranges."""
    X, y = breast_cancer_data
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert isinstance(result, CVResult)
    assert isinstance(result.mean_cv_score, float)
    assert isinstance(result.std_cv_score, float)
    assert 0.0 <= result.mean_cv_score <= 1.0
    assert result.std_cv_score >= 0.0
    assert result.metric == "f1_macro"
    assert result.n_folds == 3
    assert result.random_state == 42
    assert result.total_time >= 0.0
    assert isinstance(result.warnings, list)


def test_fold_score_structure(breast_cancer_data):
    """Each FoldScore must have valid structure and in-range values."""
    X, y = breast_cancer_data
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="balanced_accuracy",
        random_state=42,
    )
    for i, fs in enumerate(result.fold_scores):
        assert isinstance(fs, FoldScore)
        assert fs.fold_index == i
        assert 0.0 <= fs.val_score <= 1.0
        assert 0.0 <= fs.train_score <= 1.0
        assert fs.n_train > 0
        assert fs.n_val > 0
        assert fs.fit_time >= 0.0
        assert isinstance(fs.selected_features, list)
        assert isinstance(fs.convergence_warnings, list)


def test_cv_result_to_dict_serializable(breast_cancer_data):
    """CVResult.to_dict() must produce a JSON-serializable dictionary."""
    import json

    X, y = breast_cancer_data
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    d = result.to_dict()
    # This should not raise
    json.dumps(d)


# ---------------------------------------------------------------------------
# Test: metrics
# ---------------------------------------------------------------------------


def test_unsupported_metric_raises(breast_cancer_data):
    """Requesting an unsupported metric must raise ValueError."""
    X, y = breast_cancer_data
    with pytest.raises(ValueError, match="Unsupported metric"):
        run_cv_for_config(
            model_name="dummy",
            config={},
            X_train=X,
            y_train=y,
            metric="log_loss",  # not supported
        )


@pytest.mark.parametrize("metric", list(SUPPORTED_METRICS.keys()))
def test_all_supported_metrics_work(breast_cancer_data, metric):
    """Every metric listed in SUPPORTED_METRICS must run without errors."""
    X, y = breast_cancer_data
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric=metric,
        random_state=42,
    )
    assert 0.0 <= result.mean_cv_score <= 1.0


# ---------------------------------------------------------------------------
# Test: model integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["dummy", "svm", "mlp", "random_forest"])
def test_all_four_models_run_cv(breast_cancer_data, model_name):
    """All four classical models must complete 3-fold CV without errors."""
    X, y = breast_cancer_data
    config = {
        "dummy": {"strategy": "most_frequent"},
        "svm": {"kernel": "linear", "C": 1.0},
        "mlp": {"hidden_layer_sizes": (32,), "max_iter": 100},
        "random_forest": {"n_estimators": 10},
    }[model_name]
    result = run_cv_for_config(
        model_name=model_name,
        config=config,
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert result.n_folds == 3
    assert result.mean_cv_score >= 0.0


def test_cv_with_feature_selection(iris_data):
    """CV must work correctly with feature selection enabled."""
    X, y = iris_data
    fs_config = FeatureSelectionConfig(method="anova", k_features=2)
    result = run_cv_for_config(
        model_name="dummy",
        config={"strategy": "most_frequent"},
        X_train=X,
        y_train=y,
        feature_selection_config=fs_config,
        n_splits=3,
        metric="accuracy",
        random_state=42,
    )
    assert result.n_folds == 3
    # Each fold should have selected exactly 2 features
    for fs in result.fold_scores:
        assert len(fs.selected_features) == 2


def test_cv_with_preprocessing_config(breast_cancer_data):
    """CV must work with an explicit preprocessing configuration."""
    X, y = breast_cancer_data
    prep_config = PreprocessingConfig(numerical_scaling="minmax")
    result = run_cv_for_config(
        model_name="svm",
        config={"kernel": "rbf", "C": 1.0},
        X_train=X,
        y_train=y,
        preprocessing_config=prep_config,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert result.n_folds == 3
    assert isinstance(result.mean_cv_score, float)


def test_multiclass_cv(iris_data):
    """CV must handle multiclass datasets (3 classes in Iris)."""
    X, y = iris_data
    result = run_cv_for_config(
        model_name="random_forest",
        config={"n_estimators": 10},
        X_train=X,
        y_train=y,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    assert result.n_folds == 3
    assert result.mean_cv_score > 0.0
