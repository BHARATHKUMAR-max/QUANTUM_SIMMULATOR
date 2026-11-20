"""
tests/test_evaluation_metrics.py
================================
Unit, edge-case, and integration tests for the common evaluation metrics module.

Tests cover:
- Section B: Basic classification metric accuracy against hand-computed ground truth
- Section C: Missing-class scenarios, degenerate single-class predictions, zero division
- Section D: ROC-AUC calculations (binary, multiclass OvR, probability alignment, edge cases)
- Section E: Schema integrity, NumPy type sanitization, and JSON serialization
- Section F: End-to-end integration with Breast Cancer and Wine through the data pipeline
"""

import json
import math
import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.classical.dummy_baseline import DummyBaseline
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import (
    PreprocessingConfig,
    fit_preprocessor,
    transform_with_preprocessor,
)
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import (
    compute_confusion_matrix,
    evaluate_predictions,
)
from backend.evaluation.schemas import (
    EvaluationConfig,
    EvaluationMetricsResult,
)


class TestBasicMetricsCorrectness:
    """Verify standard classification metrics against hand-calculated values."""

    def test_hand_calculated_binary_metrics(self):
        """
        Verify metric values on a small, hand-verifiable binary dataset.
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        Positive class = 1
        """
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=[0, 1],
            positive_label=1,
        )

        # Expected calculations:
        # TN=1, FP=1, FN=0, TP=2
        assert res.confusion_matrix == [[1, 1], [0, 2]]
        assert res.sample_count == 4
        assert res.accuracy == pytest.approx(0.75, abs=1e-5)
        assert res.balanced_accuracy == pytest.approx(0.75, abs=1e-5)

        # Positive class (1): TP=2, FP=1, FN=0 -> Prec = 2/3, Rec = 1.0, F1 = 0.8
        assert res.precision_positive == pytest.approx(2 / 3, abs=1e-5)
        assert res.recall_positive == pytest.approx(1.0, abs=1e-5)
        assert res.f1_positive == pytest.approx(0.8, abs=1e-5)

        # Class 0: TN=1, FP=1, FN=0 -> Prec = 1/1 = 1.0, Rec = 1/2 = 0.5, F1 = 2/3
        assert res.precision_per_class["0"] == pytest.approx(1.0, abs=1e-5)
        assert res.recall_per_class["0"] == pytest.approx(0.5, abs=1e-5)

        # Macro averages: Prec = (1.0 + 2/3)/2 = 5/6, Rec = (0.5 + 1.0)/2 = 0.75, F1 = (2/3 + 4/5)/2 = 11/15
        assert res.precision_macro == pytest.approx(5 / 6, abs=1e-5)
        assert res.recall_macro == pytest.approx(0.75, abs=1e-5)
        assert res.f1_macro == pytest.approx(11 / 15, abs=1e-5)

    def test_multiclass_metrics_against_scikit_learn(self):
        """Verify multiclass metrics match direct scikit-learn computations."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2, 2])
        classes = [0, 1, 2]

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=classes,
        )

        assert res.task_type == "multiclass"
        assert res.positive_label is None
        assert res.accuracy == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-5)
        assert res.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred), abs=1e-5)
        assert res.precision_macro == pytest.approx(precision_score(y_true, y_pred, average="macro"), abs=1e-5)
        assert res.precision_weighted == pytest.approx(precision_score(y_true, y_pred, average="weighted"), abs=1e-5)
        assert res.recall_macro == pytest.approx(recall_score(y_true, y_pred, average="macro"), abs=1e-5)
        assert res.recall_weighted == pytest.approx(recall_score(y_true, y_pred, average="weighted"), abs=1e-5)
        assert res.f1_macro == pytest.approx(f1_score(y_true, y_pred, average="macro"), abs=1e-5)
        assert res.f1_weighted == pytest.approx(f1_score(y_true, y_pred, average="weighted"), abs=1e-5)

    def test_string_labels_binary_classification(self):
        """Verify binary evaluation works with string labels and explicit positive label."""
        y_true = ["benign", "malignant", "benign", "malignant"]
        y_pred = ["benign", "benign", "benign", "malignant"]

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            positive_label="malignant",
        )

        assert res.task_type == "binary"
        assert res.positive_label == "malignant"
        assert res.recall_positive == pytest.approx(0.5, abs=1e-5)
        assert res.precision_positive == pytest.approx(1.0, abs=1e-5)


class TestMissingClassesAndEdgeCases:
    """Test defensive handling of missing classes, degenerate models, and zero division."""

    def test_class_in_true_never_predicted_by_model(self):
        """Verify that when a model never predicts a class, confusion matrix preserves dimensions."""
        y_true = [0, 1, 2, 0, 1, 2]
        # Model predicts only class 0 (e.g. DummyClassifier)
        y_pred = [0, 0, 0, 0, 0, 0]
        labels = [0, 1, 2]

        res = evaluate_predictions(y_true=y_true, y_pred=y_pred, class_labels=labels)

        # Confusion matrix must be 3x3
        assert len(res.confusion_matrix) == 3
        assert len(res.confusion_matrix[0]) == 3
        # Class 1 and 2 columns must be all zeros
        assert res.confusion_matrix == [
            [2, 0, 0],
            [2, 0, 0],
            [2, 0, 0],
        ]

        # Precision for classes 1 and 2 must be 0.0 per zero_division policy
        assert res.precision_per_class["1"] == 0.0
        assert res.precision_per_class["2"] == 0.0

        # Diagnostics warning must record absent predictions
        assert any("were never predicted by model" in w for w in res.warnings)

    def test_class_in_labels_absent_from_test_partition(self):
        """Verify that when a class is in class_labels but absent from y_true, matrix preserves shape."""
        # Dataset has classes 0, 1, 2, but test partition only received 0 and 1
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        labels = [0, 1, 2]

        res = evaluate_predictions(y_true=y_true, y_pred=y_pred, class_labels=labels)

        # Confusion matrix must still be 3x3
        assert len(res.confusion_matrix) == 3
        assert len(res.confusion_matrix[0]) == 3
        # Row 2 (true class 2) must be all zeros
        assert res.confusion_matrix[2] == [0, 0, 0]

        # Warning recorded for absent class in ground truth
        assert any("absent from ground truth" in w for w in res.warnings)

    def test_single_class_degenerate_prediction(self):
        """Verify binary model predicting only the negative class does not crash."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 0]  # Never predicts positive class

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=[0, 1],
            positive_label=1,
        )

        assert res.precision_positive == 0.0
        assert res.recall_positive == 0.0
        assert res.f1_positive == 0.0
        assert res.accuracy == 0.5
        assert res.confusion_matrix == [[2, 0], [2, 0]]

    def test_empty_or_mismatched_inputs_raise_value_error(self):
        """Verify empty arrays or mismatched lengths raise descriptive ValueError."""
        with pytest.raises(ValueError, match="empty"):
            evaluate_predictions([], [])

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_predictions([0, 1], [0, 1, 0])

        with pytest.raises(ValueError, match="None"):
            evaluate_predictions(None, [0, 1])


class TestROCAUCCalculation:
    """Verify ROC-AUC calculations, probability column alignment, and edge cases."""

    def test_binary_roc_auc_with_2d_probabilities(self):
        """Verify binary ROC-AUC matches scikit-learn when 2D probabilities are passed."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        # Probabilities for [class 0, class 1]
        y_proba = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.9, 0.1],
        ])

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=[0, 1, 0, 1, 1, 0],
            y_proba=y_proba,
            class_labels=[0, 1],
            positive_label=1,
        )

        expected_auc = roc_auc_score(y_true, y_proba[:, 1])
        assert res.roc_auc == pytest.approx(expected_auc, abs=1e-5)

    def test_binary_roc_auc_positive_class_at_first_column(self):
        """Verify column alignment when positive label is at index 0."""
        y_true = np.array(["pos", "neg", "pos", "neg"])
        labels = ["pos", "neg"]
        # Col 0 = 'pos', Col 1 = 'neg'
        y_proba = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.1, 0.9],
        ])

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=["pos", "neg", "pos", "neg"],
            y_proba=y_proba,
            class_labels=labels,
            positive_label="pos",
        )

        # High probabilities for 'pos' when true is 'pos' -> AUC should be 1.0
        assert res.roc_auc == pytest.approx(1.0, abs=1e-5)

    def test_multiclass_ovr_roc_auc(self):
        """Verify multiclass OvR ROC-AUC matches scikit-learn calculations."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.8, 0.1, 0.1],
            [0.15, 0.7, 0.15],
            [0.1, 0.1, 0.8],
        ])
        labels = [0, 1, 2]

        res = evaluate_predictions(
            y_true=y_true,
            y_pred=[0, 1, 2, 0, 1, 2],
            y_proba=y_proba,
            class_labels=labels,
        )

        expected_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        assert res.roc_auc == pytest.approx(expected_macro, abs=1e-5)
        assert res.roc_auc_ovr_macro == pytest.approx(expected_macro, abs=1e-5)

    def test_no_probabilities_provided_returns_none(self):
        """Verify missing probabilities results in roc_auc=None without raising error."""
        res = evaluate_predictions(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1],
            y_proba=None,
        )

        assert res.roc_auc is None
        assert res.roc_auc_ovr_macro is None
        assert any("no probability scores" in w for w in res.warnings)

    def test_single_class_ground_truth_returns_none_safely(self):
        """Verify single class in y_true returns roc_auc=None with explanatory warning."""
        res = evaluate_predictions(
            y_true=[1, 1, 1, 1],
            y_pred=[1, 1, 1, 1],
            y_proba=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            class_labels=[0, 1],
            positive_label=1,
        )

        assert res.roc_auc is None
        assert any("only one class is present in ground truth" in w for w in res.warnings)


class TestSerializationAndSchemaCompliance:
    """Verify strict JSON serializability and schema conformity."""

    def test_full_result_serializable_to_json(self):
        """Verify to_dict returns native Python types and serializes cleanly with json.dumps."""
        y_true = np.array([0, 1, 0, 1], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 1], dtype=np.int64)
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]], dtype=np.float64)

        res = evaluate_predictions(y_true, y_pred, y_proba=y_proba, class_labels=[0, 1], positive_label=1)
        res_dict = res.to_dict()

        # Must serialize without TypeError
        json_str = json.dumps(res_dict)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Re-parse to verify structure
        parsed = json.loads(json_str)
        assert parsed["sample_count"] == 4
        assert parsed["accuracy"] == 0.75
        assert isinstance(parsed["confusion_matrix"], list)
        assert parsed["class_labels"] == [0, 1]

    def test_experiment_metrics_dict_conforms_to_schema(self):
        """Verify to_experiment_metrics_dict matches experiment_result.schema.json metrics field."""
        res = evaluate_predictions(
            y_true=[0, 1, 1, 0],
            y_pred=[0, 1, 0, 0],
            class_labels=[0, 1],
            positive_label=1,
        )

        metrics_dict = res.to_experiment_metrics_dict()
        expected_keys = {
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "balanced_accuracy",
            "roc_auc",
        }
        assert set(metrics_dict.keys()) == expected_keys
        # roc_auc must be None (serializes to null in JSON)
        assert metrics_dict["roc_auc"] is None


class TestFullDataPipelineIntegration:
    """Integration tests running DummyBaseline across the complete Day 1–12 data pipeline."""

    def test_breast_cancer_dummy_baseline_pipeline_workflow(self):
        """
        Execute full data pipeline with DummyBaseline on Breast Cancer:
        Loading -> Split (80/20) -> Preprocessing -> Dummy Fit -> Predict -> Evaluate.
        """
        # 1. Ingestion
        bundle = load_dataset("breast_cancer")
        assert bundle.dataset_name == "breast_cancer"

        # 2. Stratified Split
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        assert len(split.X_train) == 455
        assert len(split.X_test) == 114

        # 3. Leakage-Safe Preprocessing
        prep_config = PreprocessingConfig(numerical_scaling="standard")
        fitted_prep = fit_preprocessor(split.X_train, bundle.feature_types, prep_config)
        X_train_proc = transform_with_preprocessor(fitted_prep, split.X_train)
        X_test_proc = transform_with_preprocessor(fitted_prep, split.X_test)

        # 4. DummyBaseline Fitting (Training only)
        baseline = DummyBaseline()
        baseline.fit(X_train_proc, split.y_train)

        assert baseline.is_fitted
        # Breast Cancer has 357 benign (1) and 212 malignant (0); majority is class 1
        assert baseline.most_frequent_class_ == 1

        # 5. Prediction
        y_test_pred = baseline.predict(X_test_proc)
        y_test_proba = baseline.predict_proba(X_test_proc)

        assert len(y_test_pred) == 114
        assert y_test_proba.shape == (114, 2)
        # All predictions must be class 1
        assert np.all(y_test_pred == 1)

        # 6. Common Evaluation
        result = evaluate_predictions(
            y_true=split.y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            class_labels=[0, 1],
            positive_label=1,
        )

        # In test set (114 samples): 72 benign (class 1) and 42 malignant (class 0)
        # Dummy classifier predicting all 1s gets exactly 72/114 accuracy ≈ 0.631579
        expected_accuracy = 72 / 114
        assert result.accuracy == pytest.approx(expected_accuracy, abs=1e-4)

        # Recall for positive class (1) is 1.0 (all positive cases predicted positive)
        assert result.recall_positive == pytest.approx(1.0, abs=1e-5)
        # Precision for positive class is 72 / 114
        assert result.precision_positive == pytest.approx(expected_accuracy, abs=1e-4)

        # For constant dummy probabilities, ROC-AUC is exactly 0.5
        assert result.roc_auc == pytest.approx(0.5, abs=1e-5)

        # Confusion Matrix check:
        # True 0: 42 (all predicted as 1) -> [0, 42]
        # True 1: 72 (all predicted as 1) -> [0, 72]
        assert result.confusion_matrix == [[0, 42], [0, 72]]

        # Schema conformity
        exp_metrics = result.to_experiment_metrics_dict()
        assert exp_metrics["accuracy"] == pytest.approx(expected_accuracy, abs=1e-4)
        assert exp_metrics["roc_auc"] == 0.5

    def test_wine_multiclass_dummy_baseline_pipeline_workflow(self):
        """
        Execute full data pipeline with DummyBaseline on Wine multiclass dataset:
        Loading -> Split (80/20) -> Preprocessing -> Dummy Fit -> Predict -> Evaluate.
        """
        # 1. Ingestion
        bundle = load_dataset("wine")
        assert bundle.dataset_name == "wine"
        assert len(bundle.target_classes) == 3

        # 2. Stratified Split
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        assert len(split.X_train) == 142
        assert len(split.X_test) == 36

        # 3. Leakage-Safe Preprocessing
        prep_config = PreprocessingConfig(numerical_scaling="standard")
        fitted_prep = fit_preprocessor(split.X_train, bundle.feature_types, prep_config)
        X_train_proc = transform_with_preprocessor(fitted_prep, split.X_train)
        X_test_proc = transform_with_preprocessor(fitted_prep, split.X_test)

        # 4. DummyBaseline Fit
        baseline = DummyBaseline()
        baseline.fit(X_train_proc, split.y_train)

        assert baseline.is_fitted
        # Class distribution in wine: class 1 is most frequent (59/178 in dataset)
        most_freq = baseline.most_frequent_class_
        assert most_freq in [0, 1, 2]

        # 5. Prediction
        y_test_pred = baseline.predict(X_test_proc)
        y_test_proba = baseline.predict_proba(X_test_proc)

        assert len(y_test_pred) == 36
        assert y_test_proba.shape == (36, 3)

        # 6. Evaluation
        result = evaluate_predictions(
            y_true=split.y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            class_labels=[0, 1, 2],
        )

        assert result.task_type == "multiclass"
        assert result.sample_count == 36
        assert len(result.confusion_matrix) == 3
        assert len(result.confusion_matrix[0]) == 3

        # Dummy classifier with constant predictions yields OvR ROC-AUC of 0.5
        assert result.roc_auc == pytest.approx(0.5, abs=1e-5)
        assert result.roc_auc_ovr_macro == pytest.approx(0.5, abs=1e-5)

        # JSON serializability check
        json_payload = result.to_json(indent=2)
        assert len(json_payload) > 0
