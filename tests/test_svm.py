"""
tests/test_svm.py
=================
Comprehensive test suite for the Classical SVMClassifier in QML Benchmark Studio.

Tests cover:
- Section A: Configuration and hyperparameter validation (kernel, C, gamma, probability, class_weight)
- Section B: Binary classification on Breast Cancer (predictions, probabilities, decision scores)
- Section C: Multiclass classification on Wine (predictions, OvR decision scores, probability alignment)
- Section D: Defensive validation against unfitted calls, empty data, and shape mismatches
- Section E: Integration with the shared evaluation module (ROC-AUC via decision scores vs probabilities, JSON compliance)
- Section F: Deterministic reproducibility across repeated trials
- Section G: Timing profiling, CPU device reporting, and metadata export
"""

import json
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from backend.classical.svm import SVMClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


class TestSVMConfiguration:
    """Test hyperparameter validation and configuration defaults."""

    def test_default_configuration(self):
        """Verify default configuration attributes."""
        clf = SVMClassifier()
        assert clf.kernel == "rbf"
        assert clf.C == 1.0
        assert clf.gamma == "scale"
        assert clf.probability is False
        assert clf.random_state == 42
        assert clf.class_weight is None
        assert clf.execution_device == "CPU"
        assert clf.is_fitted is False

    def test_linear_kernel_configuration(self):
        """Verify linear kernel initialization."""
        clf = SVMClassifier(kernel="linear", C=0.5, class_weight="balanced")
        assert clf.kernel == "linear"
        assert clf.C == 0.5
        assert clf.class_weight == "balanced"

    def test_valid_gamma_configurations_for_rbf(self):
        """Verify valid gamma values for RBF kernel."""
        clf_scale = SVMClassifier(kernel="rbf", gamma="scale")
        assert clf_scale.gamma == "scale"

        clf_auto = SVMClassifier(kernel="rbf", gamma="auto")
        assert clf_auto.gamma == "auto"

        clf_numeric = SVMClassifier(kernel="rbf", gamma=0.05)
        assert clf_numeric.gamma == 0.05

    def test_invalid_kernel_raises_value_error(self):
        """Verify unsupported kernels raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported kernel"):
            SVMClassifier(kernel="poly")

        with pytest.raises(ValueError, match="Unsupported kernel"):
            SVMClassifier(kernel="sigmoid")

    def test_invalid_C_raises_value_error(self):
        """Verify non-positive or non-numerical C raises ValueError."""
        with pytest.raises(ValueError, match="Regularization parameter C"):
            SVMClassifier(C=0.0)

        with pytest.raises(ValueError, match="Regularization parameter C"):
            SVMClassifier(C=-2.5)

        with pytest.raises(ValueError, match="Regularization parameter C"):
            SVMClassifier(C="invalid")

    def test_invalid_gamma_for_rbf_raises_value_error(self):
        """Verify invalid gamma for RBF kernel raises ValueError."""
        with pytest.raises(ValueError, match="String gamma"):
            SVMClassifier(kernel="rbf", gamma="invalid_str")

        with pytest.raises(ValueError, match="Numerical gamma"):
            SVMClassifier(kernel="rbf", gamma=-0.1)

        with pytest.raises(ValueError, match="Numerical gamma"):
            SVMClassifier(kernel="rbf", gamma=0.0)

    def test_invalid_probability_type_raises_value_error(self):
        """Verify non-boolean probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be a boolean"):
            SVMClassifier(probability="true")

    def test_invalid_class_weight_raises_value_error(self):
        """Verify invalid class_weight raises ValueError."""
        with pytest.raises(ValueError, match="class_weight"):
            SVMClassifier(class_weight="unsupported_mode")


class TestSVMBinaryClassification:
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

    def test_binary_linear_svm_predict_and_decision_function(self, breast_cancer_data):
        """Verify Linear SVM prediction shapes, classes, and decision function."""
        X_train, y_train, X_test, y_test = breast_cancer_data

        clf = SVMClassifier(kernel="linear", C=1.0, probability=False)
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert set(clf.classes_) == {0, 1}
        assert clf.n_features_in_ == X_train.shape[1]
        assert clf.n_samples_in_ == len(X_train)
        assert clf.support_vectors_count_ > 0

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert np.isin(y_pred, [0, 1]).all()

        # Decision function for binary classification is 1D signed distance
        scores = clf.decision_function(X_test)
        assert scores.shape == (len(X_test),)
        assert isinstance(scores, np.ndarray)

        # High accuracy on Breast Cancer test set (> 90%)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.90

    def test_binary_rbf_svm_with_probabilities(self, breast_cancer_data):
        """Verify RBF SVM with probability estimates enabled."""
        X_train, y_train, X_test, y_test = breast_cancer_data

        clf = SVMClassifier(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        assert y_proba.shape == (len(X_test), 2)
        # Probabilities must sum to 1.0 across classes
        np.testing.assert_allclose(np.sum(y_proba, axis=1), np.ones(len(X_test)), atol=1e-5)
        # Values must be bounded in [0, 1]
        assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()

        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.90

    def test_predict_proba_disabled_raises_attribute_error(self, breast_cancer_data):
        """Verify predict_proba raises AttributeError when probability=False."""
        X_train, y_train, X_test, _ = breast_cancer_data
        clf = SVMClassifier(kernel="rbf", probability=False)
        clf.fit(X_train, y_train)

        with pytest.raises(AttributeError, match="predict_proba is not available"):
            clf.predict_proba(X_test)


class TestSVMMulticlassClassification:
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

    def test_multiclass_rbf_svm_predictions_and_classes(self, wine_data):
        """Verify multiclass class ordering, prediction shapes, and decision function."""
        X_train, y_train, X_test, y_test = wine_data

        clf = SVMClassifier(kernel="rbf", C=1.0, gamma="scale", probability=False)
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert list(clf.classes_) == [0, 1, 2]

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(np.unique(y_pred)).issubset({0, 1, 2})

        # Multiclass OvR decision function has shape (n_samples, n_classes)
        decision_scores = clf.decision_function(X_test)
        assert decision_scores.shape == (len(X_test), 3)

        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.85

    def test_multiclass_probability_distribution(self, wine_data):
        """Verify multiclass probability estimates sum to 1.0 across 3 classes."""
        X_train, y_train, X_test, _ = wine_data

        clf = SVMClassifier(kernel="linear", C=0.5, probability=True, random_state=42)
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(np.sum(y_proba, axis=1), np.ones(len(X_test)), atol=1e-5)


class TestDefensiveValidation:
    """Verify input validation, boundary checking, and unfitted errors."""

    def test_unfitted_model_raises_not_fitted_error(self):
        """Verify calling prediction methods prior to fit raises NotFittedError."""
        clf = SVMClassifier()

        with pytest.raises(NotFittedError):
            _ = clf.classes_

        with pytest.raises(NotFittedError):
            _ = clf.n_features_in_

        with pytest.raises(NotFittedError):
            _ = clf.support_vectors_count_

        with pytest.raises(NotFittedError):
            clf.predict([[1.0, 2.0]])

        with pytest.raises(NotFittedError):
            clf.decision_function([[1.0, 2.0]])

    def test_empty_training_data_raises_value_error(self):
        """Verify passing empty matrices raises descriptive ValueError."""
        clf = SVMClassifier()

        with pytest.raises(ValueError, match="empty"):
            clf.fit(np.empty((0, 4)), np.empty(0))

        with pytest.raises(ValueError, match="None"):
            clf.fit(None, np.array([0, 1]))

    def test_mismatched_samples_raises_value_error(self):
        """Verify sample count mismatch between features and labels raises ValueError."""
        clf = SVMClassifier()
        with pytest.raises(ValueError, match="Sample count mismatch"):
            clf.fit(np.zeros((5, 3)), np.zeros(4))

    def test_feature_count_mismatch_raises_value_error(self):
        """Verify predicting on different feature count raises ValueError."""
        clf = SVMClassifier()
        clf.fit(np.zeros((6, 3)), np.array([0, 0, 0, 1, 1, 1]))

        with pytest.raises(ValueError, match="Feature count mismatch"):
            clf.predict(np.zeros((2, 4)))

        with pytest.raises(ValueError, match="Feature count mismatch"):
            clf.decision_function(np.zeros((2, 2)))


class TestEvaluationIntegration:
    """Test end-to-end integration with the shared evaluation module."""

    @pytest.fixture
    def breast_cancer_split(self):
        """Prepare standardized Breast Cancer split and preprocessed matrices."""
        bundle = load_dataset("breast_cancer")
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train, bundle.feature_types, PreprocessingConfig(numerical_scaling="standard"))
        X_train_proc = transform_with_preprocessor(prep, split.X_train)
        X_test_proc = transform_with_preprocessor(prep, split.X_test)
        return X_train_proc, split.y_train, X_test_proc, split.y_test

    def test_binary_roc_auc_with_decision_scores(self, breast_cancer_split):
        """Verify binary ROC-AUC computed using decision scores when probability=False."""
        X_train, y_train, X_test, y_test = breast_cancer_split

        clf = SVMClassifier(kernel="rbf", probability=False)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_scores = clf.decision_function(X_test)

        res = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_scores,
            class_labels=[0, 1],
            positive_label=1,
        )

        assert res.accuracy > 0.90
        assert res.roc_auc is not None
        assert res.roc_auc > 0.90
        assert res.metadata["decision_scores_provided"] is True
        assert res.metadata["score_type"] == "decision_scores"

    def test_binary_roc_auc_with_calibrated_probabilities(self, breast_cancer_split):
        """Verify binary ROC-AUC computed using probabilities when probability=True."""
        X_train, y_train, X_test, y_test = breast_cancer_split

        clf = SVMClassifier(kernel="rbf", probability=True, random_state=42)
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
        assert res.roc_auc is not None
        assert res.roc_auc > 0.90
        assert res.metadata["probabilities_provided"] is True
        assert res.metadata["score_type"] == "probabilities"

    def test_multiclass_decision_scores_do_not_produce_uncalibrated_auc(self):
        """Verify multiclass evaluation returns roc_auc=None when only decision scores are provided."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        decision_scores = np.array([
            [2.5, 0.1, -1.2],
            [-0.5, 3.2, 0.1],
            [-1.0, 0.2, 2.8],
            [1.8, -0.2, -0.8],
            [-0.2, 2.1, 0.3],
            [-1.5, 0.1, 2.4],
        ])

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_score=decision_scores,
            class_labels=[0, 1, 2],
        )

        assert res.task_type == "multiclass"
        assert res.roc_auc is None
        assert any("requires calibrated class probabilities" in w for w in res.warnings)

    def test_svm_result_json_serializability(self, breast_cancer_split):
        """Verify complete SVM evaluation result serializes to valid JSON."""
        X_train, y_train, X_test, y_test = breast_cancer_split

        clf = SVMClassifier(kernel="linear", probability=True, random_state=42)
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

        # Include model metadata
        full_payload = {
            "model": clf.to_dict(),
            "evaluation": res.to_dict(),
            "experiment_metrics": res.to_experiment_metrics_dict(),
        }

        json_str = json.dumps(full_payload, indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["model"]["kernel"] == "linear"
        assert parsed["model"]["execution_device"] == "CPU"
        assert parsed["evaluation"]["accuracy"] > 0.90


class TestReproducibilityAndTiming:
    """Test determinism, execution timing, and metadata reporting."""

    def test_reproducible_predictions_with_fixed_seed(self):
        """Verify identical configuration and seed yield bitwise identical predictions."""
        X = np.random.randn(30, 4)
        y = np.array([0, 1, 0, 1] * 7 + [0, 1])

        clf1 = SVMClassifier(kernel="rbf", C=1.5, gamma=0.1, probability=True, random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)
        proba1 = clf1.predict_proba(X)

        clf2 = SVMClassifier(kernel="rbf", C=1.5, gamma=0.1, probability=True, random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)
        proba2 = clf2.predict_proba(X)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_allclose(proba1, proba2, atol=1e-7)

    def test_timing_and_hardware_metadata(self):
        """Verify training and inference durations are recorded and execution device is CPU."""
        X = np.random.randn(40, 5)
        y = np.array([0, 1] * 20)

        clf = SVMClassifier(kernel="linear")
        clf.fit(X, y)

        assert clf.training_time_seconds is not None
        assert clf.training_time_seconds >= 0.0

        _ = clf.predict(X[:5])
        assert clf.inference_time_seconds is not None
        assert clf.inference_time_seconds >= 0.0

        model_dict = clf.to_dict()
        assert model_dict["execution_device"] == "CPU"
        assert model_dict["training_time_seconds"] == clf.training_time_seconds
        assert model_dict["inference_time_seconds"] == clf.inference_time_seconds
