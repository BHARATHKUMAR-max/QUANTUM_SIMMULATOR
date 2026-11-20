"""
tests/test_random_forest.py
===========================
Comprehensive test suite for the Classical RandomForestClassifier in QML Benchmark Studio.

Tests cover:
- Section A: Configuration and hyperparameter validation (n_estimators, max_depth, min_samples_split, max_features)
- Section B: Binary classification on Breast Cancer (predictions, probabilities, class labels, evaluation integration)
- Section C: Multiclass classification on Wine (predictions, probability distribution, class ordering, OvR ROC-AUC)
- Section D: Feature importance extraction, name mapping, order preservation, sum-to-one, and disabled mode
- Section E: Deterministic reproducibility with fixed random seed
- Section F: Timing measurements, hardware metadata, NotFittedError guards, and JSON serialization
- Section G: Data leakage safety (train-only preprocessing fitting, quarantine of test data)
"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from backend.classical.random_forest import RandomForestClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


class TestRandomForestConfiguration:
    """Test hyperparameter validation, defaults, and error checking."""

    def test_default_configuration(self):
        """Verify default configuration attributes."""
        clf = RandomForestClassifier()
        assert clf.n_estimators == 100
        assert clf.max_depth is None
        assert clf.min_samples_split == 2
        assert clf.random_state == 42
        assert clf.class_weight is None
        assert clf.max_features == "sqrt"
        assert clf.n_jobs == -1
        assert clf.compute_feature_importances is True
        assert clf.execution_device == "CPU"
        assert clf.is_fitted is False

    def test_custom_valid_configuration(self):
        """Verify custom trees, depth, and split parameters."""
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            max_features="log2",
            class_weight="balanced",
            compute_feature_importances=False,
        )
        assert clf.n_estimators == 200
        assert clf.max_depth == 10
        assert clf.min_samples_split == 5
        assert clf.max_features == "log2"
        assert clf.class_weight == "balanced"
        assert clf.compute_feature_importances is False

    def test_invalid_n_estimators_raises_value_error(self):
        """Verify non-positive or non-integer n_estimators raises ValueError."""
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            RandomForestClassifier(n_estimators=0)

        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            RandomForestClassifier(n_estimators=-50)

        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            RandomForestClassifier(n_estimators=10.5)

    def test_invalid_max_depth_raises_value_error(self):
        """Verify non-positive or non-integer max_depth raises ValueError."""
        with pytest.raises(ValueError, match="max_depth must be None or a positive integer"):
            RandomForestClassifier(max_depth=0)

        with pytest.raises(ValueError, match="max_depth must be None or a positive integer"):
            RandomForestClassifier(max_depth=-5)

        with pytest.raises(ValueError, match="max_depth must be None or a positive integer"):
            RandomForestClassifier(max_depth="invalid")

    def test_invalid_min_samples_split_raises_value_error(self):
        """Verify min_samples_split below 2 (for int) or outside (0, 1] (for float) raises ValueError."""
        with pytest.raises(ValueError, match="min_samples_split"):
            RandomForestClassifier(min_samples_split=1)

        with pytest.raises(ValueError, match="min_samples_split"):
            RandomForestClassifier(min_samples_split=-2)

        with pytest.raises(ValueError, match="min_samples_split"):
            RandomForestClassifier(min_samples_split=0.0)

        with pytest.raises(ValueError, match="min_samples_split"):
            RandomForestClassifier(min_samples_split=1.5)

    def test_invalid_max_features_raises_value_error(self):
        """Verify unsupported max_features raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported string max_features"):
            RandomForestClassifier(max_features="invalid_option")

        with pytest.raises(ValueError, match="Integer max_features must be positive"):
            RandomForestClassifier(max_features=0)

        with pytest.raises(ValueError, match="Float max_features must be in"):
            RandomForestClassifier(max_features=1.5)

    def test_invalid_class_weight_raises_value_error(self):
        """Verify unsupported class_weight raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported class_weight"):
            RandomForestClassifier(class_weight="unsupported_mode")


class TestRandomForestBinaryClassification:
    """Test binary classification using Breast Cancer dataset."""

    @pytest.fixture
    def breast_cancer_data(self):
        """Prepare standardized Breast Cancer train and test splits."""
        bundle = load_dataset("breast_cancer")
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train, bundle.feature_types, PreprocessingConfig(numerical_scaling="standard"))
        X_train_proc = transform_with_preprocessor(prep, split.X_train)
        X_test_proc = transform_with_preprocessor(prep, split.X_test)
        return X_train_proc, split.y_train, X_test_proc, split.y_test

    def test_binary_fit_predict_and_probabilities(self, breast_cancer_data):
        """Verify Random Forest binary classification on Breast Cancer."""
        X_train, y_train, X_test, y_test = breast_cancer_data

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert list(clf.classes_) == [0, 1]
        assert clf.n_features_in_ == X_train.shape[1]
        assert clf.n_samples_in_ == len(X_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert np.isin(y_pred, [0, 1]).all()

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(np.sum(y_proba, axis=1), np.ones(len(X_test)), atol=1e-5)
        assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()

        # High accuracy on Breast Cancer test set (> 90%)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.90

    def test_binary_evaluation_integration(self, breast_cancer_data):
        """Verify binary evaluation integration computes Accuracy and ROC-AUC."""
        X_train, y_train, X_test, y_test = breast_cancer_data

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        res = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=[0, 1],
            positive_label=1,
        )

        assert res.accuracy > 0.90
        assert res.balanced_accuracy > 0.90
        assert res.f1_macro > 0.90
        assert res.roc_auc is not None
        assert res.roc_auc > 0.95
        assert res.metadata["probabilities_provided"] is True


class TestRandomForestMulticlassClassification:
    """Test multiclass classification using Wine dataset."""

    @pytest.fixture
    def wine_data(self):
        """Prepare standardized Wine train and test splits."""
        bundle = load_dataset("wine")
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train, bundle.feature_types, PreprocessingConfig(numerical_scaling="standard"))
        X_train_proc = transform_with_preprocessor(prep, split.X_train)
        X_test_proc = transform_with_preprocessor(prep, split.X_test)
        return X_train_proc, split.y_train, X_test_proc, split.y_test

    def test_multiclass_predictions_and_probabilities(self, wine_data):
        """Verify multiclass class ordering, prediction shapes, and OvR probabilities."""
        X_train, y_train, X_test, y_test = wine_data

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert list(clf.classes_) == [0, 1, 2]

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(np.unique(y_pred)).issubset({0, 1, 2})

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(np.sum(y_proba, axis=1), np.ones(len(X_test)), atol=1e-5)

        # High accuracy on Wine test set (> 85%)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.85

    def test_multiclass_evaluation_integration(self, wine_data):
        """Verify multiclass evaluation integration computes Macro F1, OvR ROC-AUC, and Confusion Matrix."""
        X_train, y_train, X_test, y_test = wine_data

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        res = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=[0, 1, 2],
        )

        assert res.task_type == "multiclass"
        assert res.accuracy > 0.85
        assert res.f1_macro > 0.85
        assert res.roc_auc is not None
        assert res.roc_auc > 0.90
        assert len(res.confusion_matrix) == 3
        assert len(res.confusion_matrix[0]) == 3


class TestFeatureImportance:
    """Test feature importance extraction, name mapping, and order preservation."""

    def test_feature_importance_enabled_and_properties(self):
        """Verify importance extraction when compute_feature_importances=True."""
        df_X = pd.DataFrame({
            "col_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "col_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "col_c": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        })
        y = np.array([0, 0, 0, 1, 1, 1])

        clf = RandomForestClassifier(n_estimators=20, compute_feature_importances=True, random_state=42)
        clf.fit(df_X, y)

        importances = clf.feature_importances_
        assert importances is not None
        assert isinstance(importances, dict)
        assert list(importances.keys()) == ["col_a", "col_b", "col_c"]

        # Sum of importances should be approximately 1.0
        total_importance = sum(importances.values())
        assert total_importance == pytest.approx(1.0, abs=1e-5)

        # Raw importances
        raw = clf.feature_importances_raw_
        assert raw is not None
        assert len(raw) == 3

        # Sorted top_features list
        top_list = clf.top_features_
        assert top_list is not None
        assert len(top_list) == 3
        assert top_list[0]["importance"] >= top_list[1]["importance"] >= top_list[2]["importance"]

    def test_feature_importance_disabled(self):
        """Verify importance properties return None when compute_feature_importances=False."""
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)

        clf = RandomForestClassifier(n_estimators=10, compute_feature_importances=False, random_state=42)
        clf.fit(X, y)

        assert clf.feature_importances_ is None
        assert clf.feature_importances_raw_ is None
        assert clf.top_features_ is None

        data = clf.to_dict()
        assert data["feature_importances"] is None
        assert data["top_features"] is None

    def test_numpy_fallback_feature_names(self):
        """Verify fallback feature names feature_0, feature_1, etc. for NumPy inputs."""
        X = np.random.randn(20, 4)
        y = np.array([0, 1] * 10)

        clf = RandomForestClassifier(n_estimators=10, compute_feature_importances=True, random_state=42)
        clf.fit(X, y)

        expected_names = ["feature_0", "feature_1", "feature_2", "feature_3"]
        assert clf.feature_names_in_ == expected_names
        assert list(clf.feature_importances_.keys()) == expected_names


class TestReproducibility:
    """Verify determinism across independent runs with fixed random state."""

    def test_identical_seed_produces_identical_predictions_and_importances(self):
        """Verify identical configuration and seed yield bitwise identical outputs."""
        X = np.random.randn(40, 5)
        y = np.array([0, 1] * 20)

        clf1 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)
        proba1 = clf1.predict_proba(X)
        imp1 = clf1.feature_importances_raw_

        clf2 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)
        proba2 = clf2.predict_proba(X)
        imp2 = clf2.feature_importances_raw_

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_allclose(proba1, proba2, atol=1e-7)
        np.testing.assert_allclose(imp1, imp2, atol=1e-7)


class TestTimingAndSerialization:
    """Verify execution profiling, hardware metadata, and JSON compliance."""

    def test_unfitted_model_raises_not_fitted_error(self):
        """Verify accessing properties or prediction prior to fit raises NotFittedError."""
        clf = RandomForestClassifier()

        with pytest.raises(NotFittedError):
            _ = clf.classes_

        with pytest.raises(NotFittedError):
            _ = clf.n_features_in_

        with pytest.raises(NotFittedError):
            _ = clf.feature_importances_

        with pytest.raises(NotFittedError):
            _ = clf.feature_names_in_

        with pytest.raises(NotFittedError):
            clf.predict([[1.0, 2.0]])

        with pytest.raises(NotFittedError):
            clf.predict_proba([[1.0, 2.0]])

    def test_timing_and_hardware_reporting(self):
        """Verify training and inference timings are non-negative and device is CPU."""
        X = np.random.randn(30, 4)
        y = np.array([0, 1] * 15)

        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)

        assert clf.training_time_seconds is not None
        assert clf.training_time_seconds >= 0.0

        _ = clf.predict(X[:5])
        assert clf.inference_time_seconds is not None
        assert clf.inference_time_seconds >= 0.0

        data = clf.to_dict()
        assert data["execution_device"] == "CPU"
        assert data["training_time_seconds"] == clf.training_time_seconds
        assert data["inference_time_seconds"] == clf.inference_time_seconds

    def test_to_dict_json_serializable(self):
        """Verify model metadata dictionary serializes cleanly to JSON."""
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        model_dict = clf.to_dict()
        json_str = json.dumps(model_dict, indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "random_forest"
        assert parsed["n_estimators"] == 10
        assert parsed["execution_device"] == "CPU"
        assert isinstance(parsed["feature_importances"], dict)


class TestLeakageSafety:
    """Verify that preprocessing is strictly fitted on training data and quarantined from test data."""

    def test_random_forest_pipeline_leakage_safety(self):
        """Verify test data does not contaminate preprocessing or model parameters."""
        bundle = load_dataset("breast_cancer")
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)

        # Scaler fitted exclusively on training data
        prep_cfg = PreprocessingConfig(numerical_scaling="standard")
        fitted_prep = fit_preprocessor(split.X_train, bundle.feature_types, prep_cfg)

        X_train_proc = transform_with_preprocessor(fitted_prep, split.X_train)
        X_test_proc = transform_with_preprocessor(fitted_prep, split.X_test)

        # Record scaler means
        scaler = fitted_prep.transformer.named_transformers_["num"].named_steps["scaler"]
        train_means = scaler.mean_.copy()

        # Fit Random Forest
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train_proc, split.y_train)

        # Ensure scaler parameters were unchanged by test transformation or model fitting
        np.testing.assert_array_equal(scaler.mean_, train_means)
        assert clf.n_samples_in_ == len(split.X_train)
