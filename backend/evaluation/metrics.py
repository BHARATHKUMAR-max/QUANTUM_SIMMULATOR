"""
backend/evaluation/metrics.py
=============================
Common evaluation engine for QML Benchmark Studio.

Computes standardized, model-agnostic classification metrics for both classical
models (DummyClassifier, SVM, Random Forest, MLP) and quantum models (VQC, QK-SVM).

Key Invariants:
- Evaluates purely from (y_true, y_pred) and optional probabilities (y_proba).
- Fully independent of model architecture or implementation details.
- Handles binary and multiclass tasks with explicit, documented averaging conventions.
- Preserves full confusion matrix dimensions even when models fail to predict certain classes.
- Safely handles undefined metrics, zero division, and missing classes without unhandled exceptions.
- Emits structured, JSON-serializable results with zero lingering NumPy types.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.evaluation.schemas import (
    ConfusionMatrixData,
    EvaluationConfig,
    EvaluationMetricsResult,
    _sanitize_for_json,
)


def _resolve_class_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    explicit_labels: Optional[List[Union[str, int]]] = None,
) -> List[Union[str, int]]:
    """
    Determine the deterministic, ordered list of unique class labels.

    If explicit labels are provided, they are preserved as the authoritative list.
    Otherwise, labels are derived from the union of y_true and y_pred with deterministic sorting.
    """
    if explicit_labels is not None and len(explicit_labels) > 0:
        return list(explicit_labels)

    # Extract unique values from y_true and y_pred
    unique_true = np.unique(y_true).tolist()
    unique_pred = np.unique(y_pred).tolist()
    all_unique = list(dict.fromkeys(unique_true + unique_pred))

    # Convert any NumPy scalar types to native Python types
    cleaned_unique = [
        int(x) if isinstance(x, (np.integer, int)) else str(x) if isinstance(x, str) else x
        for x in all_unique
    ]

    # Sort deterministically
    try:
        # If all numerical, sort numerically
        if all(isinstance(x, (int, float)) for x in cleaned_unique):
            return sorted(cleaned_unique)
        return sorted(cleaned_unique, key=str)
    except TypeError:
        return sorted(cleaned_unique, key=lambda x: str(x))


def _resolve_positive_label(
    class_labels: List[Union[str, int]],
    explicit_pos_label: Optional[Union[str, int]] = None,
    task_type: str = "binary",
) -> Tuple[Optional[Union[str, int]], Optional[str]]:
    """
    Resolve the positive class label for binary classification.

    Returns:
        Tuple of (positive_label, warning_message_if_any)
    """
    if task_type != "binary":
        return None, None

    if explicit_pos_label is not None:
        if explicit_pos_label in class_labels:
            return explicit_pos_label, None
        # Check string representation match
        for lbl in class_labels:
            if str(lbl) == str(explicit_pos_label):
                return lbl, None
        warning = (
            f"Specified positive_label '{explicit_pos_label}' was not found in class labels "
            f"{class_labels}; falling back to default convention."
        )
    else:
        warning = None

    # Default heuristic for binary classification
    # 1. Standard {0, 1} integer coding -> 1
    if set(class_labels).issubset({0, 1, np.int64(0), np.int64(1), np.int32(0), np.int32(1)}):
        return 1 if 1 in class_labels else class_labels[-1], warning

    # 2. Standard {-1, 1} bipolar coding -> 1
    if set(class_labels).issubset({-1, 1}):
        return 1, warning

    # 3. Boolean coding -> True
    if all(isinstance(c, (bool, np.bool_)) for c in class_labels):
        return True, warning

    # 4. Canonical affirmative string keywords
    affirmative = {"1", "positive", "malignant", "yes", "true", "presence"}
    for lbl in class_labels:
        if str(lbl).strip().lower() in affirmative:
            return lbl, warning

    # 5. Fallback: last element in sorted class order
    fallback_pos = class_labels[-1]
    fallback_warning = (
        f"Binary positive_label was not explicitly specified; "
        f"defaulted to '{fallback_pos}' according to canonical class ordering."
    )
    if warning:
        fallback_warning = f"{warning} {fallback_warning}"
    return fallback_pos, fallback_warning


def compute_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    class_labels: Optional[List[Union[str, int]]] = None,
) -> ConfusionMatrixData:
    """
    Compute structured confusion matrix with guaranteed dimensions and label alignment.

    Args:
        y_true: Ground truth target array.
        y_pred: Model predictions array.
        class_labels: Optional explicit ordering of class labels.

    Returns:
        ConfusionMatrixData with 2D integer matrix and corresponding label list.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = _resolve_class_labels(y_true_arr, y_pred_arr, class_labels)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    matrix_list = [[int(cell) for cell in row] for row in cm]
    return ConfusionMatrixData(matrix=matrix_list, labels=labels)


def _compute_roc_auc(
    y_true_arr: np.ndarray,
    y_proba: Optional[Union[pd.DataFrame, np.ndarray, List]],
    class_labels: List[Union[str, int]],
    task_type: str,
    positive_label: Optional[Union[str, int]],
    multiclass_average: str = "macro",
    y_score: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Dict[str, Optional[float]],
    List[str],
    Optional[str],
]:
    """
    Compute ROC-AUC with comprehensive boundary handling for probabilities and decision scores.

    Returns:
        Tuple of (overall_roc_auc, ovr_macro, ovr_weighted, per_class_dict, warnings, score_type)
    """
    warnings: List[str] = []

    # Case 1: Both y_proba and y_score are None
    if y_proba is None and y_score is None:
        warnings.append("ROC-AUC not calculated: no probability scores (y_proba) or decision scores (y_score) provided.")
        return None, None, None, {}, warnings, None

    # Case 2: y_proba is provided (preferred for both binary and multiclass)
    if y_proba is not None:
        score_type = "probabilities"
        if isinstance(y_proba, pd.DataFrame):
            score_arr = y_proba.to_numpy()
        else:
            score_arr = np.asarray(y_proba)
    else:
        # Case 3: Only y_score is provided
        score_type = "decision_scores"
        if task_type == "multiclass":
            warnings.append(
                "Multiclass ROC-AUC not calculated from decision scores: multiclass ROC-AUC "
                "requires calibrated class probabilities (probability=True); decision scores "
                "cannot be treated as normalized probabilities."
            )
            return None, None, None, {}, warnings, score_type

        # Binary decision scores
        if isinstance(y_score, pd.DataFrame):
            score_arr = y_score.to_numpy()
        else:
            score_arr = np.asarray(y_score)

    if score_arr.size == 0:
        warnings.append(f"ROC-AUC not calculated: {score_type} array is empty.")
        return None, None, None, {}, warnings, score_type

    if len(score_arr) != len(y_true_arr):
        warnings.append(
            f"ROC-AUC not calculated: sample count mismatch between {score_type} ({len(score_arr)}) "
            f"and y_true ({len(y_true_arr)})."
        )
        return None, None, None, {}, warnings, score_type

    unique_true = np.unique(y_true_arr)
    if len(unique_true) < 2:
        warnings.append("ROC-AUC undefined: only one class is present in ground truth y_true.")
        return None, None, None, {}, warnings, score_type

    # Binary Classification ROC-AUC
    if task_type == "binary":
        if score_arr.ndim == 1:
            score_vec = score_arr
        elif score_arr.ndim == 2:
            if score_arr.shape[1] == 2:
                # Find column corresponding to positive_label
                if positive_label in class_labels:
                    pos_idx = class_labels.index(positive_label)
                else:
                    pos_idx = 1
                score_vec = score_arr[:, pos_idx]
            elif score_arr.shape[1] == 1:
                score_vec = score_arr[:, 0]
            else:
                warnings.append(
                    f"ROC-AUC not calculated: binary classification requires 1 or 2 score columns, "
                    f"got shape {score_arr.shape}."
                )
                return None, None, None, {}, warnings, score_type
        else:
            warnings.append(
                f"ROC-AUC not calculated: unsupported score array dimension {score_arr.ndim}."
            )
            return None, None, None, {}, warnings, score_type

        try:
            # Map y_true to binary 0/1 array for positive_label
            y_binary = (y_true_arr == positive_label).astype(int)
            if len(np.unique(y_binary)) < 2:
                warnings.append(
                    f"ROC-AUC undefined: ground truth binary projection for positive class "
                    f"'{positive_label}' contains only one class."
                )
                return None, None, None, {}, warnings, score_type

            auc_score = float(roc_auc_score(y_binary, score_vec))
            if math.isnan(auc_score) or math.isinf(auc_score):
                warnings.append("ROC-AUC evaluated to non-finite NaN/Inf value.")
                return None, None, None, {}, warnings, score_type

            auc_score = round(auc_score, 6)
            return auc_score, auc_score, auc_score, {str(positive_label): auc_score}, warnings, score_type
        except Exception as exc:
            warnings.append(f"Binary ROC-AUC calculation failed: {str(exc)}")
            return None, None, None, {}, warnings, score_type

    # Multiclass Classification ROC-AUC (One-vs-Rest)
    if score_arr.ndim != 2:
        warnings.append(
            f"ROC-AUC not calculated: multiclass evaluation requires a 2D probability matrix, "
            f"got dimension {score_arr.ndim}."
        )
        return None, None, None, {}, warnings, score_type

    if score_arr.shape[1] != len(class_labels):
        warnings.append(
            f"ROC-AUC not calculated: probability matrix column count ({score_arr.shape[1]}) "
            f"does not match class count ({len(class_labels)})."
        )
        return None, None, None, {}, warnings, score_type

    per_class_auc: Dict[str, Optional[float]] = {}
    valid_scores: List[float] = []
    class_weights: List[int] = []

    for idx, cls_name in enumerate(class_labels):
        y_binary = (y_true_arr == cls_name).astype(int)
        count_positive = int(np.sum(y_binary))
        if count_positive == 0 or count_positive == len(y_true_arr):
            # Class completely absent from y_true or only class present
            per_class_auc[str(cls_name)] = None
            warnings.append(
                f"OvR ROC-AUC undefined for class '{cls_name}': class has {count_positive} true samples."
            )
        else:
            try:
                score = float(roc_auc_score(y_binary, score_arr[:, idx]))
                if math.isnan(score) or math.isinf(score):
                    per_class_auc[str(cls_name)] = None
                else:
                    score = round(score, 6)
                    per_class_auc[str(cls_name)] = score
                    valid_scores.append(score)
                    class_weights.append(count_positive)
            except Exception as e:
                per_class_auc[str(cls_name)] = None
                warnings.append(f"OvR ROC-AUC calculation failed for class '{cls_name}': {str(e)}")

    if not valid_scores:
        warnings.append("Multiclass OvR ROC-AUC could not be computed for any class.")
        return None, None, None, per_class_auc, warnings, score_type

    macro_ovr = round(float(np.mean(valid_scores)), 6)
    if sum(class_weights) > 0:
        weighted_ovr = round(float(np.average(valid_scores, weights=class_weights)), 6)
    else:
        weighted_ovr = macro_ovr

    overall = weighted_ovr if multiclass_average == "weighted" else macro_ovr
    return overall, macro_ovr, weighted_ovr, per_class_auc, warnings, score_type


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    y_proba: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
    class_labels: Optional[List[Union[str, int]]] = None,
    positive_label: Optional[Union[str, int]] = None,
    config: Optional[EvaluationConfig] = None,
    y_score: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
) -> EvaluationMetricsResult:
    """
    Perform standardized evaluation of predictions against ground truth labels.

    Args:
        y_true: Ground truth target labels (n_samples,).
        y_pred: Predicted class labels (n_samples,).
        y_proba: Optional predicted probabilities (n_samples, n_classes) or (n_samples,).
        class_labels: Optional explicit ordering of class labels.
        positive_label: Optional explicit positive class label for binary classification.
        config: Optional EvaluationConfig instance overriding individual parameters.

    Returns:
        EvaluationMetricsResult instance with all classification metrics, confusion matrix,
        ROC-AUC, and diagnostic metadata.

    Raises:
        ValueError: If y_true or y_pred is None, empty, or has mismatched lengths.
    """
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must not be None.")

    # Convert to 1D numpy arrays
    if isinstance(y_true, (pd.Series, pd.Index)):
        y_true_arr = y_true.to_numpy()
    else:
        y_true_arr = np.asarray(y_true)

    if isinstance(y_pred, (pd.Series, pd.Index)):
        y_pred_arr = y_pred.to_numpy()
    else:
        y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} observations, "
            f"but y_pred has {len(y_pred_arr)} observations."
        )

    # Resolve configuration options
    cfg = config or EvaluationConfig()
    zero_division = cfg.zero_division
    multiclass_avg = cfg.roc_auc_multiclass_average

    # Resolve labels from arguments or config
    effective_labels = class_labels if class_labels is not None else cfg.class_labels
    labels = _resolve_class_labels(y_true_arr, y_pred_arr, effective_labels)

    # Determine task type
    if cfg.task_type is not None:
        task_type = cfg.task_type
    else:
        task_type = "binary" if len(labels) <= 2 else "multiclass"

    # Resolve positive label for binary tasks
    effective_pos_label = positive_label if positive_label is not None else cfg.positive_label
    pos_label, pos_warning = _resolve_positive_label(labels, effective_pos_label, task_type)

    warnings: List[str] = []
    if pos_warning:
        warnings.append(pos_warning)

    # Diagnostic checks for missing classes
    unique_true_set = set(y_true_arr)
    unique_pred_set = set(y_pred_arr)
    absent_from_pred = [
        int(c) if isinstance(c, (np.integer, int)) else str(c)
        for c in labels if c not in unique_pred_set
    ]
    absent_from_true = [
        int(c) if isinstance(c, (np.integer, int)) else str(c)
        for c in labels if c not in unique_true_set
    ]
    unique_pred_clean = [
        int(c) if isinstance(c, (np.integer, int)) else str(c)
        for c in sorted(list(unique_pred_set), key=str)
    ]

    if absent_from_pred:
        warnings.append(
            f"Class(es) {absent_from_pred} were never predicted by model "
            f"(unique predicted classes: {unique_pred_clean})."
        )
    if absent_from_true:
        warnings.append(
            f"Class(es) {absent_from_true} are absent from ground truth test partition."
        )

    # 1. Accuracy
    acc = round(float(accuracy_score(y_true_arr, y_pred_arr)), 6)

    # 2. Balanced Accuracy
    balanced_acc = round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 6)

    # 3. Macro and Weighted Metrics
    prec_macro = round(
        float(precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=zero_division)),
        6,
    )
    prec_weighted = round(
        float(precision_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=zero_division)),
        6,
    )

    rec_macro = round(
        float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=zero_division)),
        6,
    )
    rec_weighted = round(
        float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=zero_division)),
        6,
    )

    f1_macro = round(
        float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=zero_division)),
        6,
    )
    f1_weighted = round(
        float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=zero_division)),
        6,
    )

    # 4. Per-Class Metrics
    prec_per_class_arr = precision_score(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=zero_division
    )
    rec_per_class_arr = recall_score(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=zero_division
    )
    f1_per_class_arr = f1_score(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=zero_division
    )

    prec_per_class = {str(c): round(float(v), 6) for c, v in zip(labels, prec_per_class_arr)}
    rec_per_class = {str(c): round(float(v), 6) for c, v in zip(labels, rec_per_class_arr)}
    f1_per_class = {str(c): round(float(v), 6) for c, v in zip(labels, f1_per_class_arr)}

    # 5. Binary-Specific Metrics
    if task_type == "binary" and pos_label is not None:
        prec_positive = round(
            float(precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, average="binary", zero_division=zero_division)),
            6,
        )
        rec_positive = round(
            float(recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, average="binary", zero_division=zero_division)),
            6,
        )
        f1_positive = round(
            float(f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, average="binary", zero_division=zero_division)),
            6,
        )
    else:
        prec_positive = None
        rec_positive = None
        f1_positive = None

    # 6. Confusion Matrix
    cm_obj = compute_confusion_matrix(y_true_arr, y_pred_arr, class_labels=labels)

    # 7. ROC-AUC
    roc_auc_val, roc_ovr_macro, roc_ovr_weighted, roc_per_class, auc_warnings, score_type = _compute_roc_auc(
        y_true_arr=y_true_arr,
        y_proba=y_proba,
        class_labels=labels,
        task_type=task_type,
        positive_label=pos_label,
        multiclass_average=multiclass_avg,
        y_score=y_score,
    )
    warnings.extend(auc_warnings)

    # Compile metadata
    metadata: Dict[str, Any] = {
        "zero_division_policy": zero_division,
        "probabilities_provided": y_proba is not None,
        "decision_scores_provided": y_score is not None,
        "score_type": score_type,
        "labels_source": "explicit" if effective_labels is not None else "inferred",
    }

    return EvaluationMetricsResult(
        accuracy=acc,
        precision_macro=prec_macro,
        precision_weighted=prec_weighted,
        precision_positive=prec_positive,
        precision_per_class=prec_per_class,
        recall_macro=rec_macro,
        recall_weighted=rec_weighted,
        recall_positive=rec_positive,
        recall_per_class=rec_per_class,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_positive=f1_positive,
        f1_per_class=f1_per_class,
        balanced_accuracy=balanced_acc,
        roc_auc=roc_auc_val,
        roc_auc_ovr_macro=roc_ovr_macro,
        roc_auc_ovr_weighted=roc_ovr_weighted,
        roc_auc_per_class=roc_per_class,
        confusion_matrix=cm_obj.matrix,
        class_labels=labels,
        sample_count=len(y_true_arr),
        task_type=task_type,
        positive_label=pos_label,
        warnings=warnings,
        metadata=metadata,
    )
