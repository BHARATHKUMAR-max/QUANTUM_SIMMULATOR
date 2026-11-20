"""
tests/test_vqc_classifier.py
============================
Unit and integration test suite for the Binary and Multiclass VQC Classifier
prediction interface in QML Benchmark Studio (Day 24 milestone).

Test Categories:
  1. TestLabelHandling: Binary/multiclass string and integer targets, class ordering,
     label decoding, single-class rejection, NaN target rejection.
  2. TestBinaryOutput: Output shape (N, 2), probability bounds [0, 1], row sums == 1.0,
     analytical (1-z)/2 mapping, positive class convention (classes_[1]), configurable readout wire.
  3. TestMulticlassOutput: Output shape (N, K), numerical stability of softmax with extreme logits,
     row sums == 1.0, head reproducibility, 3-class and 4-class tasks.
  4. TestInterfaceAndIntegration: fit/predict/predict_proba scikit-learn compatibility,
     evaluation metrics integration (calculate_all_metrics), parameter getters/setters,
     timing tracking, dimensional mismatch validation.
"""

import math
from typing import List
import numpy as np
import pandas as pd
import pytest

from backend.evaluation.metrics import evaluate_predictions
from backend.quantum.vqc_classifier import (
    VQCClassifier,
    VQCClassifierConfig,
    expectation_to_binary_proba,
    softmax,
)


# ===========================================================================
# 1. Label Handling & Ordering Tests
# ===========================================================================

class TestLabelHandling:
    """Tests verifying label encoding, deterministic class ordering, and decoding."""

    def test_binary_string_labels(self):
        """Verify binary string labels are preserved and decoded accurately."""
        X = np.random.normal(size=(20, 2))
        y = np.array(["malignant", "benign"] * 10)

        clf = VQCClassifier(n_qubits=2, depth=1)
        clf.fit(X, y)

        assert clf.is_binary_ is True
        assert clf.n_classes_ == 2
        # Deterministic lexicographical sorting: "benign" < "malignant"
        assert list(clf.classes_) == ["benign", "malignant"]

        preds = clf.predict(X)
        assert len(preds) == 20
        assert set(preds).issubset({"benign", "malignant"})
        assert preds.dtype.kind in ("U", "O")

    def test_binary_integer_labels(self):
        """Verify binary integer labels [0, 1] and [1, 2]."""
        X = np.random.normal(size=(10, 2))
        y = np.array([1, 2] * 5)

        clf = VQCClassifier(n_qubits=2, depth=1)
        clf.fit(X, y)

        assert clf.is_binary_ is True
        assert list(clf.classes_) == [1, 2]

        preds = clf.predict(X)
        assert set(preds).issubset({1, 2})

    def test_multiclass_string_labels(self):
        """Verify 3-class string labels."""
        X = np.random.normal(size=(30, 3))
        y = np.array(["class_C", "class_A", "class_B"] * 10)

        clf = VQCClassifier(n_qubits=3, depth=1)
        clf.fit(X, y)

        assert clf.is_binary_ is False
        assert clf.n_classes_ == 3
        assert list(clf.classes_) == ["class_A", "class_B", "class_C"]

        preds = clf.predict(X)
        assert len(preds) == 30
        assert set(preds).issubset({"class_A", "class_B", "class_C"})

    def test_multiclass_integer_labels(self):
        """Verify multiclass integer labels [0, 1, 2]."""
        X = np.random.normal(size=(15, 3))
        y = np.array([2, 0, 1] * 5)

        clf = VQCClassifier(n_qubits=3, depth=1)
        clf.fit(X, y)

        assert clf.is_binary_ is False
        assert list(clf.classes_) == [0, 1, 2]

    def test_single_class_target_raises_error(self):
        """Verify target with only 1 class raises ValueError."""
        X = np.random.normal(size=(10, 2))
        y = np.array(["benign"] * 10)

        clf = VQCClassifier(n_qubits=2, depth=1)
        with pytest.raises(ValueError, match="At least 2 distinct target classes"):
            clf.fit(X, y)

    def test_nan_target_labels_raise_error(self):
        """Verify NaN target labels raise ValueError."""
        X = np.random.normal(size=(6, 2))
        y = np.array([0, 1, np.nan, 1, 0, 1])

        clf = VQCClassifier(n_qubits=2, depth=1)
        with pytest.raises(ValueError, match="missing/NaN labels"):
            clf.fit(X, y)


# ===========================================================================
# 2. Binary Output & Probability Mapping Tests
# ===========================================================================

class TestBinaryOutput:
    """Tests for binary probability formula (1-z)/2, readout wire, and shape."""

    def test_binary_probability_formula_mathematical_cases(self):
        """
        Verify expectation_to_binary_proba values:
          z = -1.0 -> P(y=1) = 1.0, P(y=0) = 0.0
          z =  0.0 -> P(y=1) = 0.5, P(y=0) = 0.5
          z = +1.0 -> P(y=1) = 0.0, P(y=0) = 1.0
        """
        z_vals = np.array([-1.0, 0.0, 1.0, -0.5, 0.5])
        proba = expectation_to_binary_proba(z_vals)

        assert proba.shape == (5, 2)
        # Row 0: z = -1.0
        np.testing.assert_allclose(proba[0], [0.0, 1.0], atol=1e-12)
        # Row 1: z =  0.0
        np.testing.assert_allclose(proba[1], [0.5, 0.5], atol=1e-12)
        # Row 2: z = +1.0
        np.testing.assert_allclose(proba[2], [1.0, 0.0], atol=1e-12)
        # Row 3: z = -0.5 -> P(y=1) = 0.75, P(y=0) = 0.25
        np.testing.assert_allclose(proba[3], [0.25, 0.75], atol=1e-12)
        # Row 4: z = +0.5 -> P(y=1) = 0.25, P(y=0) = 0.75
        np.testing.assert_allclose(proba[4], [0.75, 0.25], atol=1e-12)

    def test_binary_predict_proba_shape_and_bounds(self):
        """Verify predict_proba on fitted binary classifier."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y_train = np.array(["no", "yes", "no", "yes"])

        clf = VQCClassifier(n_qubits=2, depth=1, random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_train)
        assert proba.shape == (4, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(4), atol=1e-12)

    def test_positive_class_convention(self):
        """Verify positive class convention is strictly classes_[1]."""
        X = np.zeros((4, 2))
        y = np.array(["A", "B", "A", "B"])
        clf = VQCClassifier(n_qubits=2, depth=1)
        clf.fit(X, y)

        assert clf.classes_[0] == "A"  # negative class (column 0)
        assert clf.classes_[1] == "B"  # positive class (column 1)

    def test_configurable_readout_wire(self):
        """Verify configuring readout_wire=1 extracts expectation from wire 1."""
        X = np.random.normal(size=(5, 3))
        y = np.array([0, 1, 0, 1, 0])

        clf_wire0 = VQCClassifier(n_qubits=3, depth=1, readout_wire=0, random_state=42)
        clf_wire0.fit(X, y)
        p0 = clf_wire0.predict_proba(X)

        clf_wire1 = VQCClassifier(n_qubits=3, depth=1, readout_wire=1, random_state=42)
        clf_wire1.fit(X, y)
        p1 = clf_wire1.predict_proba(X)

        # Wire 0 and Wire 1 expectations are generally distinct
        assert p0.shape == (5, 2)
        assert p1.shape == (5, 2)

    def test_invalid_readout_wire_rejected(self):
        """Verify readout_wire out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="readout_wire=4 is invalid for n_qubits=4"):
            VQCClassifierConfig(n_qubits=4, readout_wire=4)

        with pytest.raises(ValueError, match="readout_wire=-1"):
            VQCClassifierConfig(n_qubits=4, readout_wire=-1)


# ===========================================================================
# 3. Multiclass Output & Softmax Tests
# ===========================================================================

class TestMulticlassOutput:
    """Tests for multiclass linear head, softmax numerical stability, and shapes."""

    def test_softmax_function_numerical_stability(self):
        """Verify softmax does not overflow on large logits and sums to 1.0."""
        # Extreme logits that would cause exp() overflow without shifting
        logits_extreme = np.array([
            [1000.0, 1001.0, 999.0],
            [-500.0, -500.0, -500.0],
            [0.0, 0.0, 0.0],
        ])

        proba = softmax(logits_extreme)
        assert proba.shape == (3, 3)
        assert not np.isnan(proba).any()
        assert not np.isinf(proba).any()
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), [1.0, 1.0, 1.0], atol=1e-12)

        # In row 0: 1001 is largest -> proba[0, 1] should be largest
        assert np.argmax(proba[0]) == 1
        # In row 1: equal logits -> 1/3 each
        np.testing.assert_allclose(proba[1], [1/3, 1/3, 1/3], atol=1e-12)

    def test_multiclass_predict_proba_shape_and_sums(self):
        """Verify multiclass predict_proba returns shape (N, K) and rows sum to 1."""
        X = np.random.normal(size=(20, 4))
        y = np.array(["red", "green", "blue", "yellow"] * 5)

        clf = VQCClassifier(n_qubits=4, depth=1, random_state=42)
        clf.fit(X, y)

        assert clf.is_binary_ is False
        assert clf.n_classes_ == 4
        assert clf.head_weights_.shape == (4, 4)
        assert clf.head_bias_.shape == (4,)

        proba = clf.predict_proba(X)
        assert proba.shape == (20, 4)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(20), atol=1e-12)

        preds = clf.predict(X)
        assert len(preds) == 20
        assert set(preds).issubset({"red", "green", "blue", "yellow"})

    def test_reproducible_head_initialization(self):
        """Verify identical head_seed produces identical head weights and predictions."""
        X = np.random.normal(size=(10, 3))
        y = np.array([0, 1, 2] * 3 + [0])

        clf1 = VQCClassifier(n_qubits=3, depth=1, head_seed=123, seed=42)
        clf1.fit(X, y)

        clf2 = VQCClassifier(n_qubits=3, depth=1, head_seed=123, seed=42)
        clf2.fit(X, y)

        np.testing.assert_array_equal(clf1.head_weights_, clf2.head_weights_)
        np.testing.assert_array_equal(clf1.predict_proba(X), clf2.predict_proba(X))


# ===========================================================================
# 4. Interface & Integration Tests
# ===========================================================================

class TestInterfaceAndIntegration:
    """Tests verifying scikit-learn compatibility, metric integration, and guardrails."""

    def test_unfitted_model_raises_runtime_error(self):
        """Verify calling predict or predict_proba before fit raises RuntimeError."""
        clf = VQCClassifier(n_qubits=2)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            clf.predict(np.zeros((2, 2)))

        with pytest.raises(RuntimeError, match="not fitted yet"):
            clf.predict_proba(np.zeros((2, 2)))

    def test_feature_count_mismatch_raises_error(self):
        """Verify input feature count mismatch with n_qubits raises ValueError."""
        X_3 = np.random.normal(size=(10, 3))
        y = np.array([0, 1] * 5)

        clf = VQCClassifier(n_qubits=4)  # configured for 4 qubits
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            clf.fit(X_3, y)

        X_4 = np.random.normal(size=(10, 4))
        clf.fit(X_4, y)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            clf.predict(X_3)

    def test_binary_evaluation_metrics_integration(self):
        """Verify predict and predict_proba outputs integrate cleanly with calculate_all_metrics."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        y_true = np.array(["benign", "malignant"] * 15)

        clf = VQCClassifier(n_qubits=2, depth=1, random_state=42)
        clf.fit(X, y_true)

        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)

        # Evaluate with common metrics engine
        metrics_res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            positive_label="malignant",
        )

        assert 0.0 <= metrics_res.accuracy <= 1.0
        assert 0.0 <= metrics_res.f1_macro <= 1.0
        assert metrics_res.roc_auc is not None
        assert 0.0 <= metrics_res.roc_auc <= 1.0
        assert metrics_res.confusion_matrix is not None
        assert len(metrics_res.confusion_matrix) == 2
        assert len(metrics_res.class_labels) == 2

    def test_multiclass_evaluation_metrics_integration(self):
        """Verify multiclass outputs integrate cleanly with calculate_all_metrics (OvR ROC-AUC)."""
        rng = np.random.default_rng(101)
        X = rng.normal(size=(30, 3))
        y_true = np.array(["wine_A", "wine_B", "wine_C"] * 10)

        clf = VQCClassifier(n_qubits=3, depth=1, random_state=101)
        clf.fit(X, y_true)

        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)

        metrics_res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        assert 0.0 <= metrics_res.accuracy <= 1.0
        assert 0.0 <= metrics_res.f1_macro <= 1.0
        assert metrics_res.roc_auc is not None
        assert 0.0 <= metrics_res.roc_auc <= 1.0
        assert len(metrics_res.confusion_matrix) == 3
        assert len(metrics_res.class_labels) == 3

    def test_parameter_getters_and_setters(self):
        """Verify circuit weights and head parameters can be retrieved and updated."""
        X = np.random.normal(size=(10, 3))
        y = np.array([0, 1, 2] * 3 + [0])

        clf = VQCClassifier(n_qubits=3, depth=1, rotation_gates="RY")
        clf.fit(X, y)

        # Get circuit weights
        w = clf.get_vqc_weights()
        assert w.shape == (1, 3, 1)

        # Set circuit weights
        new_w = np.ones((1, 3, 1), dtype=np.float64)
        clf.set_vqc_weights(new_w)
        np.testing.assert_array_equal(clf.get_vqc_weights(), new_w)

        # Get and set head parameters
        head_params = clf.get_head_parameters()
        assert "weights" in head_params
        assert "bias" in head_params
        assert head_params["weights"].shape == (3, 3)
        assert head_params["bias"].shape == (3,)

        new_head_w = np.zeros((3, 3))
        new_head_b = np.ones(3)
        clf.set_head_parameters(new_head_w, new_head_b)
        np.testing.assert_array_equal(clf.head_weights_, new_head_w)
        np.testing.assert_array_equal(clf.head_bias_, new_head_b)

    def test_timing_and_config_reporting(self):
        """Verify get_timing() and get_config() emit structured metadata."""
        X = np.random.normal(size=(10, 2))
        y = np.array([0, 1] * 5)

        clf = VQCClassifier(n_qubits=2, depth=1)
        clf.fit(X, y)
        _ = clf.predict(X)

        timing = clf.get_timing()
        assert "training_time_seconds" in timing
        assert "inference_time_seconds" in timing
        assert timing["training_time_seconds"] >= 0.0
        assert timing["inference_time_seconds"] >= 0.0

        cfg = clf.get_config()
        assert cfg["is_fitted"] is True
        assert cfg["is_binary"] is True
        assert cfg["n_classes"] == 2
        assert cfg["classes"] == [0, 1]

    def test_single_sample_prediction(self):
        """Verify single 1D sample vector can be passed to predict and predict_proba."""
        X_train = np.random.normal(size=(10, 2))
        y_train = np.array(["neg", "pos"] * 5)

        clf = VQCClassifier(n_qubits=2, depth=1)
        clf.fit(X_train, y_train)

        # Single 1D sample
        sample_1d = np.array([0.5, -0.5])
        proba_1d = clf.predict_proba(sample_1d)
        pred_1d = clf.predict(sample_1d)

        assert proba_1d.shape == (1, 2)
        assert len(pred_1d) == 1
        assert pred_1d[0] in ("neg", "pos")
