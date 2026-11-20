"""
backend/training/aggregation.py
===============================
Result aggregation and statistical summarization for repeated runs in QML Benchmark Studio.

Computes sample mean, sample standard deviation (ddof=1), minimum, maximum,
and execution counts across independent experiment repetitions.

Critical Statistical Conventions:
---------------------------------
1. Sample Standard Deviation: ddof=1 is strictly used when valid sample count >= 2.
2. Single-Run Standard Deviation: For exactly one valid run, std is None (not 0.0).
3. Missing Metric Handling: Non-applicable metrics (such as ROC-AUC for models
   without probability outputs) are excluded from calculations; they are NEVER
   silently imputed as zero.
4. Transparent Failure Tracking: Failed runs or runs where an individual metric
   is undefined are explicitly counted and reported.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.training.schemas import MetricAggregate, RepeatedRunResult


TRACKED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "train_time_seconds",
    "inference_time_seconds",
    "preprocessing_time_seconds",
    "total_time_seconds",
]


def aggregate_metric(
    values: List[Optional[float]],
    metric_name: str,
    model_name: str,
    total_runs: int,
) -> MetricAggregate:
    """
    Calculate summary statistics for a sequence of metric values.

    Args:
        values: List of metric values (floats or None).
        metric_name: Identifier for the metric.
        model_name: Identifier of the evaluated model.
        total_runs: Total repetitions attempted in the experiment.

    Returns:
        MetricAggregate dataclass with mean, std, min, max, and run counts.
    """
    # Filter valid numeric values (non-None, non-NaN, non-Inf)
    valid_vals: List[float] = [
        float(v) for v in values if v is not None and not np.isnan(v) and not np.isinf(v)
    ]
    valid_count = len(valid_vals)
    failed_count = total_runs - valid_count

    if valid_count == 0:
        return MetricAggregate(
            model_name=model_name,
            metric=metric_name,
            mean=None,
            std=None,
            min=None,
            max=None,
            successful_runs=0,
            failed_runs=failed_count,
            total_runs=total_runs,
        )

    min_val = float(np.min(valid_vals))
    max_val = float(np.max(valid_vals))
    mean_val = float(np.mean(valid_vals))

    if valid_count == 1:
        # Sample standard deviation undefined for sample size 1
        std_val = None
    else:
        # Unbiased sample standard deviation (Bessel's correction, ddof=1)
        std_val = float(np.std(valid_vals, ddof=1))

    return MetricAggregate(
        model_name=model_name,
        metric=metric_name,
        mean=mean_val,
        std=std_val,
        min=min_val,
        max=max_val,
        successful_runs=valid_count,
        failed_runs=failed_count,
        total_runs=total_runs,
    )


def compute_aggregates(
    runs: List[RepeatedRunResult],
    model_name: str,
    custom_metrics: Optional[List[str]] = None,
) -> Dict[str, MetricAggregate]:
    """
    Aggregate all tracked metrics across a list of repeated run results.

    Args:
        runs: List of RepeatedRunResult instances.
        model_name: Model identifier.
        custom_metrics: Optional explicit list of metric attribute names to aggregate.

    Returns:
        Dictionary mapping metric names to MetricAggregate objects.
    """
    total_runs = len(runs)
    metric_keys = custom_metrics or TRACKED_METRICS
    aggregates: Dict[str, MetricAggregate] = {}

    for metric_name in metric_keys:
        values: List[Optional[float]] = []
        for run in runs:
            if run.failed:
                values.append(None)
                continue

            # Check direct attribute on RepeatedRunResult
            if hasattr(run, metric_name):
                val = getattr(run, metric_name)
                # If val is a dict (like ROC-AUC nested dict), extract 'value' if present
                if isinstance(val, dict):
                    val = val.get("value")
                values.append(val)
            elif metric_name in run.metrics:
                m_val = run.metrics[metric_name]
                if isinstance(m_val, dict):
                    m_val = m_val.get("value")
                values.append(m_val)
            else:
                values.append(None)

        aggregates[metric_name] = aggregate_metric(
            values=values,
            metric_name=metric_name,
            model_name=model_name,
            total_runs=total_runs,
        )

    return aggregates
