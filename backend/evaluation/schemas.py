"""
backend/evaluation/schemas.py
=============================
Data models and result schemas for model evaluation in QML Benchmark Studio.

These schemas provide standardized, model-agnostic evaluation result structures
for classical baselines (DummyClassifier, SVM, Random Forest, MLP) and quantum
models (VQC, QK-SVM).

Key Invariants:
- Impartial calculation across classical and quantum models.
- Fully JSON serializable: no NumPy scalars, ndarrays, or NaNs left in outputs.
- Explicit nulls/None for non-applicable or mathematically undefined metrics.
- Comprehensive confusion matrix preservation with explicit class label ordering.
"""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects into native JSON-serializable Python structures.
    Converts NumPy scalar types, handles NaN/Inf by converting to None, and unwraps arrays.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    return str(obj)


@dataclass
class ConfusionMatrixData:
    """
    Structured confusion matrix with explicit class label alignment.

    Invariants:
        matrix[i][j] represents observations known to be in class labels[i]
        and predicted to be in class labels[j].
    """
    matrix: List[List[int]]
    labels: List[Union[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert confusion matrix to JSON-serializable dict."""
        return {
            "matrix": [[int(cell) for cell in row] for row in self.matrix],
            "labels": [
                int(lbl) if isinstance(lbl, (np.integer, int)) else str(lbl)
                for lbl in self.labels
            ],
            "num_classes": len(self.labels),
        }


@dataclass
class EvaluationConfig:
    """
    Configuration options for model evaluation.

    Attributes:
        task_type: Task nature ('binary', 'multiclass', or None for auto-detection).
        positive_label: Explicit positive class label for binary classification.
        class_labels: Explicit full list of valid class labels to preserve order and missing classes.
        zero_division: Value to return when precision/recall/F1 calculation encounters zero division (default: 0.0).
        roc_auc_multiclass_average: Averaging strategy for multiclass ROC-AUC ('macro' or 'weighted').
        include_per_class: Whether to compute per-class precision, recall, F1, and ROC-AUC.
    """
    task_type: Optional[str] = None
    positive_label: Optional[Union[str, int]] = None
    class_labels: Optional[List[Union[str, int]]] = None
    zero_division: float = 0.0
    roc_auc_multiclass_average: str = "macro"
    include_per_class: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dict."""
        return {
            "task_type": self.task_type,
            "positive_label": (
                int(self.positive_label)
                if isinstance(self.positive_label, (np.integer, int))
                else str(self.positive_label) if self.positive_label is not None else None
            ),
            "class_labels": (
                [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in self.class_labels]
                if self.class_labels is not None else None
            ),
            "zero_division": float(self.zero_division),
            "roc_auc_multiclass_average": self.roc_auc_multiclass_average,
            "include_per_class": self.include_per_class,
        }


@dataclass
class EvaluationMetricsResult:
    """
    Standardized result structure returned by the common evaluation module.

    Model-independent container for classification metrics, confusion matrices,
    ROC-AUC evaluations, warnings, and diagnostic metadata.
    """
    accuracy: Optional[float]
    precision_macro: Optional[float]
    precision_weighted: Optional[float]
    precision_positive: Optional[float]
    precision_per_class: Dict[str, Optional[float]]
    recall_macro: Optional[float]
    recall_weighted: Optional[float]
    recall_positive: Optional[float]
    recall_per_class: Dict[str, Optional[float]]
    f1_macro: Optional[float]
    f1_weighted: Optional[float]
    f1_positive: Optional[float]
    f1_per_class: Dict[str, Optional[float]]
    balanced_accuracy: Optional[float]
    roc_auc: Optional[float]
    roc_auc_ovr_macro: Optional[float]
    roc_auc_ovr_weighted: Optional[float]
    roc_auc_per_class: Dict[str, Optional[float]]
    confusion_matrix: List[List[int]]
    class_labels: List[Union[str, int]]
    sample_count: int
    task_type: str
    positive_label: Optional[Union[str, int]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_experiment_metrics_dict(self) -> Dict[str, Optional[float]]:
        """
        Produce a flat metrics dictionary strictly matching docs/schemas/experiment_result.schema.json.

        Returns:
            Dictionary with keys: accuracy, precision_macro, recall_macro, f1_macro,
            f1_weighted, balanced_accuracy, roc_auc.
        """
        return {
            "accuracy": _sanitize_for_json(self.accuracy),
            "precision_macro": _sanitize_for_json(self.precision_macro),
            "recall_macro": _sanitize_for_json(self.recall_macro),
            "f1_macro": _sanitize_for_json(self.f1_macro),
            "f1_weighted": _sanitize_for_json(self.f1_weighted),
            "balanced_accuracy": _sanitize_for_json(self.balanced_accuracy),
            "roc_auc": _sanitize_for_json(self.roc_auc),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the complete evaluation result to a standard, JSON-serializable dictionary.
        All NumPy scalars and arrays are recursively converted to native Python types.
        """
        raw_dict = {
            "accuracy": self.accuracy,
            "precision": {
                "macro": self.precision_macro,
                "weighted": self.precision_weighted,
                "positive": self.precision_positive,
                "per_class": self.precision_per_class,
            },
            "recall": {
                "macro": self.recall_macro,
                "weighted": self.recall_weighted,
                "positive": self.recall_positive,
                "per_class": self.recall_per_class,
            },
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "f1": {
                "macro": self.f1_macro,
                "weighted": self.f1_weighted,
                "positive": self.f1_positive,
                "per_class": self.f1_per_class,
            },
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": {
                "value": self.roc_auc,
                "ovr_macro": self.roc_auc_ovr_macro,
                "ovr_weighted": self.roc_auc_ovr_weighted,
                "per_class": self.roc_auc_per_class,
            },
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
            "sample_count": self.sample_count,
            "task_type": self.task_type,
            "positive_label": self.positive_label,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
        return _sanitize_for_json(raw_dict)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize evaluation result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
