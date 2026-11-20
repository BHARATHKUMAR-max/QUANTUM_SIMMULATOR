"""
backend/training/schemas.py
===========================
Data schemas and containers for repeated-run experiments, multi-model
benchmarking, and result aggregation in QML Benchmark Studio.

These schemas ensure full JSON serializability, strict separation of
per-run metadata, reproducible dataset indexing, and comprehensive
tracking of metrics across repetitions.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np


def _sanitize_value(val: Any) -> Any:
    """Recursively convert NumPy scalars and collections into JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 6)
    if isinstance(val, np.ndarray):
        return [_sanitize_value(v) for v in val.tolist()]
    if isinstance(val, dict):
        return {str(k): _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_value(v) for v in val]
    return str(val)


@dataclass
class RepeatedRunResult:
    """
    Detailed results, metadata, and metrics from a single train/test repetition.

    Attributes:
        experiment_id: Unique identifier for the experiment campaign.
        run_number: Zero-based index of this repetition (e.g. 0 to 4).
        random_state: Deterministic seed used to generate the split.
        train_indices: Preserved dataset row indices for training.
        test_indices: Preserved dataset row indices for test evaluation.
        n_train: Number of training samples.
        n_test: Number of test samples.
        train_class_distribution: Class sample distribution in training partition.
        test_class_distribution: Class sample distribution in test partition.
        model_name: Canonical model identifier (e.g. 'svm', 'random_forest').
        model_config: Config applied for this run (or base config in tuned mode).
        selected_params: Hyperparameters selected via CV (if tuning was enabled).
        preprocessing_config: Preprocessing configuration dictionary.
        feature_selection_config: Feature selection configuration dictionary if enabled.
        comparison_mode: Active comparison mode if applicable.
        accuracy: Test set classification accuracy.
        balanced_accuracy: Test set balanced accuracy.
        precision_macro: Test set macro-averaged precision.
        precision_weighted: Test set weighted-averaged precision.
        recall_macro: Test set macro-averaged recall.
        recall_weighted: Test set weighted-averaged recall.
        f1_macro: Test set macro-averaged F1 score.
        f1_weighted: Test set weighted-averaged F1 score.
        roc_auc: Test set ROC-AUC (None if unavailable, e.g. non-probabilistic models).
        confusion_matrix: Test set confusion matrix [[int]].
        train_time_seconds: Model training duration in seconds.
        inference_time_seconds: Model test prediction duration in seconds.
        preprocessing_time_seconds: Preprocessing and feature selection fitting time.
        total_time_seconds: Total duration for this repetition.
        failed: Whether this repetition failed with an error.
        error_message: Error description if failed is True.
        warnings: List of diagnostic or non-fatal warning messages.
        metrics: Full dictionary output from evaluate_predictions.
    """
    experiment_id: str
    run_number: int
    random_state: int
    train_indices: List[Any]
    test_indices: List[Any]
    n_train: int
    n_test: int
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    model_name: str
    model_config: Dict[str, Any]
    selected_params: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    feature_selection_config: Optional[Dict[str, Any]] = None
    comparison_mode: Optional[str] = None
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_macro: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_weighted: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    train_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    preprocessing_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    convergence_info: Optional[Dict[str, Any]] = None
    failed: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert run result to a JSON-serializable dictionary."""
        d = asdict(self)
        return _sanitize_value(d)


@dataclass
class MetricAggregate:
    """
    Statistical summary of a single metric across repetitions.

    Attributes:
        model_name: Model identifier.
        metric: Metric name (e.g. 'accuracy', 'f1_macro', 'roc_auc').
        mean: Sample mean across valid runs (None if 0 valid runs).
        std: Sample standard deviation (ddof=1, None if < 2 valid runs).
        min: Minimum value across valid runs.
        max: Maximum value across valid runs.
        successful_runs: Number of runs with valid numeric metric values.
        failed_runs: Number of failed runs or runs where metric was undefined.
        total_runs: Total repetitions attempted.
    """
    model_name: str
    metric: str
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    successful_runs: int
    failed_runs: int
    total_runs: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric aggregate to a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "metric": self.metric,
            "mean": round(self.mean, 6) if self.mean is not None else None,
            "std": round(self.std, 6) if self.std is not None else None,
            "min": round(self.min, 6) if self.min is not None else None,
            "max": round(self.max, 6) if self.max is not None else None,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "total_runs": self.total_runs,
        }


@dataclass
class RepeatedExperimentResult:
    """
    Aggregated result container for a single model across multiple repetitions.

    Attributes:
        experiment_id: Unique experiment campaign identifier.
        model_name: Canonical model identifier.
        mode: Experiment configuration mode ('fixed' or 'tuned').
        n_repeats: Number of repetitions attempted.
        seeds: List of random seeds used for partitioning.
        test_size: Proportion allocated to held-out test partition.
        stratified: Whether stratified partitioning was used.
        runs: List of individual RepeatedRunResult objects.
        aggregates: Dictionary mapping metric names to MetricAggregate objects.
        total_time_seconds: Cumulative wall-clock execution time.
        model_config: Base model configuration dictionary.
        preprocessing_config: Base preprocessing configuration dictionary.
        feature_selection_config: Base feature selection configuration dictionary.
        comparison_mode: Comparison mode name if applicable.
        package_versions: Environment package provenance metadata.
    """
    experiment_id: str
    model_name: str
    mode: str
    n_repeats: int
    seeds: List[int]
    test_size: float
    stratified: bool
    runs: List[RepeatedRunResult]
    aggregates: Dict[str, MetricAggregate]
    total_time_seconds: float
    model_config: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    feature_selection_config: Optional[Dict[str, Any]] = None
    comparison_mode: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire experiment result to a JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "mode": self.mode,
            "n_repeats": self.n_repeats,
            "seeds": self.seeds,
            "test_size": self.test_size,
            "stratified": self.stratified,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "model_config": _sanitize_value(self.model_config),
            "preprocessing_config": _sanitize_value(self.preprocessing_config),
            "feature_selection_config": _sanitize_value(self.feature_selection_config),
            "comparison_mode": self.comparison_mode,
            "package_versions": self.package_versions,
            "aggregates": {k: agg.to_dict() for k, agg in self.aggregates.items()},
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class MultiModelComparisonResult:
    """
    Container for benchmarking multiple models across identical shared splits.

    Attributes:
        experiment_id: Unique experiment campaign identifier.
        mode: Experiment configuration mode ('fixed' or 'tuned').
        n_repeats: Number of repetitions.
        seeds: Shared seeds applied across all models.
        models: List of model identifiers compared.
        split_indices: Shared train/test partition indices for each repetition.
        model_results: Mapping of model_name to RepeatedExperimentResult.
        comparison_mode: Optional comparison mode string.
        total_time_seconds: Cumulative execution duration.
        package_versions: Environment package provenance metadata.
    """
    experiment_id: str
    mode: str
    n_repeats: int
    seeds: List[int]
    models: List[str]
    split_indices: List[Dict[str, List[Any]]]
    model_results: Dict[str, RepeatedExperimentResult]
    comparison_mode: Optional[str] = None
    total_time_seconds: float = 0.0
    package_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert multi-model comparison to a JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "n_repeats": self.n_repeats,
            "seeds": self.seeds,
            "models": self.models,
            "comparison_mode": self.comparison_mode,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "package_versions": self.package_versions,
            "split_indices": _sanitize_value(self.split_indices),
            "model_results": {m: res.to_dict() for m, res in self.model_results.items()},
        }
