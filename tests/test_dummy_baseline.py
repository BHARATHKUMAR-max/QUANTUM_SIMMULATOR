"""
tests/test_dummy_baseline.py
============================
Unit and behavioural tests for DummyBaseline in QML Benchmark Studio.

Tests cover:
- Binary classification
- Multiclass classification
- Most-frequent prediction strategy
- Training-only fitting invariants
- Prediction shapes (1D labels, 2D probabilities)
- Preservation of original label types (int, string, bool)
- Robust validation against empty/invalid inputs and unfitted calls
- Serialization via to_dict()
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from backend.classical.dummy_baseline import DummyBaseline


class TestDummyBaselineCore:
    """Core functionality and prediction logic tests."""

    def test_binary_most_frequent_prediction(self):
        """Verify DummyBaseline predicts the majority class in binary data."""
        # 3 samples of class 0, 7 samples of class 1
        X_train = np.arange(20).reshape(10, 2)
        y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        clf = DummyBaseline()
        clf.fit(X_train, y_train)

        assert clf.is_fitted
        assert clf.most_frequent_class_ == 1
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

        X_test = np.array([[100, 200], [300, 400], [500, 600]])
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Output shapes
        assert y_pred.shape == (3,)
        assert y_proba.shape == (3, 2)

        # All predictions must be class 1
        np.testing.assert_array_equal(y_pred, np.array([1, 1, 1]))

        # Probabilities: 0.0 for class 0, 1.0 for class 1
        expected_proba = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(y_proba, expected_proba)

    def test_multiclass_most_frequent_prediction(self):
        """Verify DummyBaseline predicts the majority class in multiclass data."""
        # Class 0: 2, Class 1: 5, Class 2: 3
        X_train = np.zeros((10, 4))
        y_train = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])

        clf = DummyBaseline()
        clf.fit(X_train, y_train)

        assert clf.most_frequent_class_ == 1
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))

        X_test = np.ones((4, 4))
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        assert y_pred.shape == (4,)
        assert y_proba.shape == (4, 3)
        np.testing.assert_array_equal(y_pred, np.array([1, 1, 1, 1]))

        expected_proba = np.zeros((4, 3))
        expected_proba[:, 1] = 1.0
        np.testing.assert_allclose(y_proba, expected_proba)

    def test_string_labels_preserved(self):
        """Verify original string labels are preserved without coercion to integers."""
        X_train = np.random.randn(8, 3)
        y_train = np.array(["benign", "malignant", "malignant", "malignant", "benign", "malignant", "benign", "malignant"])

        clf = DummyBaseline()
        clf.fit(X_train, y_train)

        assert clf.most_frequent_class_ == "malignant"
        assert list(clf.classes_) == ["benign", "malignant"]

        X_test = np.random.randn(3, 3)
        y_pred = clf.predict(X_test)
        assert y_pred.dtype.kind in ("U", "O", "S")
        np.testing.assert_array_equal(y_pred, np.array(["malignant", "malignant", "malignant"]))

    def test_pandas_dataframe_and_series_support(self):
        """Verify compatibility with pandas DataFrame features and Series targets."""
        df_X = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [10.0, 20.0, 30.0, 40.0]})
        s_y = pd.Series(["A", "B", "B", "B"], name="target")

        clf = DummyBaseline()
        clf.fit(df_X, s_y)

        assert clf.n_features_in_ == 2
        assert clf.n_samples_in_ == 4
        assert clf.most_frequent_class_ == "B"

        df_test = pd.DataFrame({"feat_a": [5.0, 6.0], "feat_b": [50.0, 60.0]})
        y_pred = clf.predict(df_test)
        np.testing.assert_array_equal(y_pred, np.array(["B", "B"]))


class TestDummyBaselineInvariants:
    """Tests ensuring training-only invariants and strict encapsulation."""

    def test_test_data_never_used_during_fit(self):
        """Verify model predictions do not depend on test data distribution."""
        X_train = np.zeros((6, 2))
        y_train = np.array([0, 0, 0, 0, 1, 1])  # Class 0 is majority (4 vs 2)

        clf = DummyBaseline()
        clf.fit(X_train, y_train)

        # Even if test data is 100% class 1, prediction must strictly be class 0
        X_test_all_ones = np.ones((10, 2))
        y_pred = clf.predict(X_test_all_ones)
        np.testing.assert_array_equal(y_pred, np.zeros(10))

    def test_repeated_predictions_deterministic_and_idempotent(self):
        """Verify predict calls are purely functional and do not alter estimator state."""
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array(["cat", "dog", "dog", "dog"])

        clf = DummyBaseline(random_state=42)
        clf.fit(X_train, y_train)

        pred1 = clf.predict([[10], [20]])
        pred2 = clf.predict([[10], [20]])
        np.testing.assert_array_equal(pred1, pred2)
        assert clf.most_frequent_class_ == "dog"


class TestDummyBaselineInputValidation:
    """Defensive programming tests for invalid, empty, or unfitted inputs."""

    def test_unfitted_model_raises_not_fitted_error(self):
        """Verify calling predict, predict_proba, or properties before fit raises NotFittedError."""
        clf = DummyBaseline()

        assert not clf.is_fitted
        with pytest.raises(NotFittedError):
            _ = clf.classes_

        with pytest.raises(NotFittedError):
            _ = clf.most_frequent_class_

        with pytest.raises(NotFittedError):
            _ = clf.n_features_in_

        with pytest.raises(NotFittedError):
            clf.predict([[1, 2]])

        with pytest.raises(NotFittedError):
            clf.predict_proba([[1, 2]])

    def test_empty_training_data_raises_value_error(self):
        """Verify passing empty matrices raises descriptive ValueError."""
        clf = DummyBaseline()

        with pytest.raises(ValueError, match="empty"):
            clf.fit(np.empty((0, 2)), np.empty(0))

        with pytest.raises(ValueError, match="None"):
            clf.fit(None, np.array([1, 2]))

        with pytest.raises(ValueError, match="None"):
            clf.fit(np.array([[1, 2]]), None)

    def test_mismatched_sample_count_raises_value_error(self):
        """Verify sample count mismatch between X and y raises ValueError."""
        clf = DummyBaseline()
        with pytest.raises(ValueError, match="Sample count mismatch"):
            clf.fit(np.zeros((5, 2)), np.zeros(4))

    def test_mismatched_test_features_raises_value_error(self):
        """Verify predicting on feature count different from training raises ValueError."""
        clf = DummyBaseline()
        clf.fit(np.zeros((5, 3)), np.array([0, 1, 1, 1, 0]))

        with pytest.raises(ValueError, match="Feature count mismatch"):
            clf.predict(np.zeros((2, 4)))

        with pytest.raises(ValueError, match="Feature count mismatch"):
            clf.predict_proba(np.zeros((2, 2)))

    def test_empty_test_matrix_raises_value_error(self):
        """Verify predicting on empty test array raises ValueError."""
        clf = DummyBaseline()
        clf.fit(np.zeros((5, 2)), np.array([0, 1, 1, 1, 0]))

        with pytest.raises(ValueError, match="empty"):
            clf.predict(np.empty((0, 2)))


class TestDummyBaselineSerialization:
    """Test to_dict serialization for logging and experiment persistence."""

    def test_to_dict_unfitted_and_fitted(self):
        """Verify to_dict returns clean, JSON-serializable dictionaries."""
        clf = DummyBaseline(strategy="most_frequent", random_state=123)
        unfitted_dict = clf.to_dict()
        assert unfitted_dict["model_name"] == "dummy"
        assert unfitted_dict["is_fitted"] is False

        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = np.array([10, 20, 20])
        clf.fit(X, y)

        fitted_dict = clf.to_dict()
        assert fitted_dict["is_fitted"] is True
        assert fitted_dict["most_frequent_class"] == 20
        assert fitted_dict["classes"] == [10, 20]
        assert fitted_dict["n_features_in"] == 2
        assert fitted_dict["n_samples_in"] == 3
        assert fitted_dict["feature_names_in"] == ["f1", "f2"]
