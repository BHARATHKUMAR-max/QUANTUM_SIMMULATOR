"""
backend/evaluation/__init__.py
==============================
Model evaluation subsystem for QML Benchmark Studio.

Exposes model-independent classification metrics, confusion matrices, and ROC-AUC evaluation.
"""

from backend.evaluation.metrics import (
    compute_confusion_matrix,
    evaluate_predictions,
)
from backend.evaluation.schemas import (
    ConfusionMatrixData,
    EvaluationConfig,
    EvaluationMetricsResult,
)

__all__ = [
    "compute_confusion_matrix",
    "evaluate_predictions",
    "ConfusionMatrixData",
    "EvaluationConfig",
    "EvaluationMetricsResult",
]
