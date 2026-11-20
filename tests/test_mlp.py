"""
tests/test_mlp.py
=================
Comprehensive test suite for the Classical MLPClassifier in QML Benchmark Studio.

Tests cover:
- Section A: Configuration and hyperparameter validation (hidden_layer_sizes, activation, solver, lr, max_iter)
- Section B: Binary classification on Breast Cancer (predictions, probabilities, class labels, evaluation integration)
- Section C: Multiclass classification on Wine (predictions, probability distribution, class ordering, OvR ROC-AUC)
- Section D: Deterministic reproducibility with fixed random seed
- Section E: Convergence diagnostics and ConvergenceWarning capture (deliberately low max_iter)
- Section F: Timing measurements, hardware metadata, NotFittedError guards, and JSON serialization
- Section G: Data leakage safety (train-only scaler fitting, quarantine of test data)
"""

import json
import warnings
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from backend.classical.mlp import MLPClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


class TestMLPConfiguration:
    """Test hyperparameter validation, defaults, and configuration error handling."""

    def test_default_configuration(self):
        """Verify default configuration attributes."""
        clf = MLPClassifier()
        assert clf.hidden_layer_sizes == (64, 32)
        assert clf.activation == "relu"
        assert clf.solver == "adam"
        assert clf.alpha == 0.0001
        assert clf.learning_rate_init == 0.001
        assert clf.max_iter == 200
        assert clf.random_state == 42
        assert clf.early_stopping is False
        assert clf.validation_fraction == 0.1
        assert clf.n_iter_no_change == 10
        assert clf.execution_device == "CPU"
        assert clf.is_fitted is False

    def test_custom_valid_configurations(self):
        """Verify custom hidden layers and optimizer configurations."""
        clf_single = MLPClassifier(hidden_layer_sizes=32)
        assert clf_single.hidden_layer_sizes == (32,)

        clf_tuple = MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", solver="sgd")
        assert clf_tuple.hidden_layer_sizes == (128, 64)
        assert clf_tuple.activation == "tanh"
        assert clf_tuple.solver == "sgd"

    def test_invalid_hidden_layer_sizes_raises_value_error(self):
        """Verify invalid layer configurations raise descriptive ValueError."""
        with pytest.raises(ValueError, match="Hidden layer neuron count must be positive"):
            MLPClassifier(hidden_layer_sizes=0)

        with pytest.raises(ValueError, match="Hidden layer neuron count must be positive"):
            MLPClassifier(hidden_layer_sizes=-10)

        with pytest.raises(ValueError, match="at least one hidden layer"):
            MLPClassifier(hidden_layer_sizes=())

        with pytest.raises(ValueError, match="positive integer"):
            MLPClassifier(hidden_layer_sizes=(64, -32))

        with pytest.raises(ValueError, match="positive integer"):
            MLPClassifier(hidden_layer_sizes=(64, 0))

    def test_invalid_activation_raises_value_error(self):
        """Verify unsupported activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPClassifier(activation="sigmoid_unsupported")

        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPClassifier(activation="leaky_relu")

    def test_invalid_solver_raises_value_error(self):
        """Verify unsupported solver raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported solver"):
            MLPClassifier(solver="rmsprop")

    def test_invalid_alpha_raises_value_error(self):
        """Verify negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            MLPClassifier(alpha=-0.01)

    def test_invalid_learning_rate_raises_value_error(self):
        """Verify non-positive learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate_init must be positive"):
            MLPClassifier(learning_rate_init=0.0)

        with pytest.raises(ValueError, match="learning_rate_init must be positive"):
            MLPClassifier(learning_rate_init=-0.05)

    def test_invalid_max_iter_raises_value_error(self):
        """Verify non-positive max_iter raises ValueError."""
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            MLPClassifier(max_iter=0)

        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            MLPClassifier(max_iter=-50)


class TestMLPBinaryClassification:
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
        """Verify MLP binary classification on Breast Cancer."""
        X_train, y_train, X_test, y_test = breast_cancer_data

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42)
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert list(clf.classes_) == [0, 1]
        assert clf.n_features_in_ == X_train.shape[1]
        assert clf.n_samples_in_ == len(X_train)
        assert clf.total_hidden_neurons_ == 96

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

        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
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


class TestMLPMulticlassClassification:
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

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
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
        """Verify multiclass evaluation integration computes Macro F1 and OvR ROC-AUC."""
        X_train, y_train, X_test, y_test = wine_data

        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
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


class TestReproducibility:
    """Verify determinism across independent runs with fixed random state."""

    def test_identical_seed_produces_identical_weights_and_predictions(self):
        """Verify identical configuration and seed yield bitwise identical outputs."""
        X = np.random.randn(50, 6)
        y = np.array([0, 1, 2, 0, 1] * 10)

        clf1 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)
        proba1 = clf1.predict_proba(X)

        clf2 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)
        proba2 = clf2.predict_proba(X)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_allclose(proba1, proba2, atol=1e-6)
        assert clf1.loss_ == pytest.approx(clf2.loss_, abs=1e-5)
        assert clf1.n_iter_ == clf2.n_iter_


class TestConvergenceDiagnostics:
    """Test convergence capture and reporting under low iteration limits."""

    def test_low_max_iter_triggers_convergence_warning(self):
        """Verify deliberate low max_iter records convergence failure truthfully."""
        X = np.random.randn(60, 8)
        y = np.array([0, 1] * 30)

        # Deliberately low max_iter to force non-convergence
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2, random_state=42)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            clf.fit(X, y)

        assert clf.is_fitted
        # Model ran exactly max_iter iterations
        assert clf.n_iter_ == 2
        # Convergence status must be False
        assert clf.converged_ is False
        assert clf.convergence_warning_ is not None
        assert "Maximum iterations" in clf.convergence_warning_

        # Model must still predict despite non-convergence
        preds = clf.predict(X)
        assert preds.shape == (60,)

        # Metadata dictionary records convergence failure truthfully
        model_dict = clf.to_dict()
        assert model_dict["converged"] is False
        assert model_dict["convergence_warning"] == clf.convergence_warning_
        assert model_dict["max_iter"] == 2
        assert model_dict["n_iter"] == 2


class TestTimingAndSerialization:
    """Verify execution profiling, hardware metadata, and JSON compliance."""

    def test_unfitted_model_raises_not_fitted_error(self):
        """Verify accessing properties or prediction prior to fit raises NotFittedError."""
        clf = MLPClassifier()

        with pytest.raises(NotFittedError):
            _ = clf.classes_

        with pytest.raises(NotFittedError):
            _ = clf.n_features_in_

        with pytest.raises(NotFittedError):
            _ = clf.loss_

        with pytest.raises(NotFittedError):
            _ = clf.converged_

        with pytest.raises(NotFittedError):
            clf.predict([[1.0, 2.0]])

        with pytest.raises(NotFittedError):
            clf.predict_proba([[1.0, 2.0]])

    def test_timing_and_hardware_reporting(self):
        """Verify training and inference timings are non-negative and device is CPU."""
        X = np.random.randn(30, 4)
        y = np.array([0, 1] * 15)

        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
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
        """Verify model metadata dictionary serializes cleanly to JSON without errors."""
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)

        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=50, random_state=42)
        clf.fit(X, y)

        model_dict = clf.to_dict()
        json_str = json.dumps(model_dict, indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "mlp"
        assert parsed["hidden_layer_sizes"] == [16, 8]
        assert parsed["execution_device"] == "CPU"


class TestLeakageSafety:
    """Verify that preprocessing is strictly fitted on training data and quarantined from test data."""

    def test_mlp_pipeline_leakage_safety(self):
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

        # Fit MLP
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=42)
        clf.fit(X_train_proc, split.y_train)

        # Ensure scaler parameters were unchanged by test transformation or model fitting
        np.testing.assert_array_equal(scaler.mean_, train_means)
        assert clf.n_samples_in_ == len(split.X_train)
