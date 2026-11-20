"""
backend/training/__init__.py
============================
Training orchestration package for QML Benchmark Studio.

Provides leakage-safe cross-validation, hyperparameter tuning, repeated runs,
and multi-model benchmarking workflows for classical models.

Public API:
    from backend.training.cross_validation import run_cv_for_config, FoldScore, CVResult
    from backend.training.hyperparameter_tuning import tune_model, final_evaluation, TuningResult, CandidateResult, DEFAULT_GRIDS
    from backend.training.schemas import RepeatedRunResult, MetricAggregate, RepeatedExperimentResult, MultiModelComparisonResult
    from backend.training.aggregation import compute_aggregates, aggregate_metric
    from backend.training.repeated_runs import run_repeated_experiment, run_repeated_comparison, DEFAULT_SEEDS
"""

from backend.training.aggregation import (
    aggregate_metric,
    compute_aggregates,
)
from backend.training.cross_validation import (
    CVResult,
    FoldScore,
    run_cv_for_config,
)
from backend.training.hyperparameter_tuning import (
    CandidateResult,
    DEFAULT_GRIDS,
    TuningResult,
    final_evaluation,
    tune_model,
)
from backend.training.repeated_runs import (
    DEFAULT_SEEDS,
    run_repeated_comparison,
    run_repeated_experiment,
)
from backend.training.schemas import (
    MetricAggregate,
    MultiModelComparisonResult,
    RepeatedExperimentResult,
    RepeatedRunResult,
)

__all__ = [
    # Cross-validation
    "FoldScore",
    "CVResult",
    "run_cv_for_config",
    # Hyperparameter tuning
    "CandidateResult",
    "TuningResult",
    "DEFAULT_GRIDS",
    "tune_model",
    "final_evaluation",
    # Repeated runs & multi-model benchmarking
    "DEFAULT_SEEDS",
    "RepeatedRunResult",
    "MetricAggregate",
    "RepeatedExperimentResult",
    "MultiModelComparisonResult",
    "aggregate_metric",
    "compute_aggregates",
    "run_repeated_experiment",
    "run_repeated_comparison",
]
