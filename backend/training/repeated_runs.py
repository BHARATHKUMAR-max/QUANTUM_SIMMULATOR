"""
backend/training/repeated_runs.py
=================================
Repeated experiment runner and multi-model benchmarking engine for QML Benchmark Studio.

Executes models across multiple reproducible, stratified train/test partitions to assess
performance stability and variability without making unjustified statistical claims.

Key Engineering Invariants:
---------------------------
1. Strict Estimator Isolation: Preprocessors, feature selectors, and models are instantiated
   and fitted strictly on that specific run's training partition. No fitted state leaks.
2. Fair Shared Partitions: When evaluating multiple models, partitions are generated once
   per seed and reused identically across all models.
3. Absolute Test Quarantine: Held-out test partitions are evaluated exactly once at the end
   of each repetition; they never participate in preprocessing fitting or tuning.
4. Transparent Failure Containment: Failures in individual runs are isolated, recorded, and
   do not invalidate successful runs.
5. Dual Experiment Modes:
   - Mode A ('fixed'): Uses a pre-specified validated configuration across all runs.
   - Mode B ('tuned'): Performs hyperparameter tuning (via CV) exclusively on each run's
     training partition; selected parameters are recorded per run.
"""

from datetime import datetime, timezone
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
import sklearn

from backend.classical.model_configs import BaseModelConfig
from backend.classical.model_registry import create_model, validate_model_config
from backend.data.comparison_modes import prepare_comparison_mode
from backend.data.feature_selection import fit_selector, transform_with_selector
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor
from backend.data.schemas import (
    ComparisonModeConfig,
    DatasetBundle,
    FeatureSelectionConfig,
    PreprocessingConfig,
    TrainTestSplitResult,
)
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions
from backend.training.aggregation import compute_aggregates
from backend.training.hyperparameter_tuning import tune_model
from backend.training.schemas import (
    MultiModelComparisonResult,
    RepeatedExperimentResult,
    RepeatedRunResult,
    _sanitize_value,
)


DEFAULT_SEEDS = [42, 101, 202, 303, 404]


def get_provenance_metadata() -> Dict[str, str]:
    """Capture environment package versions and hardware platform provenance."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_repeated_inputs(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, List[Any]],
    n_repeats: int,
    test_size: float,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate input matrices, label dimensions, and class balance requirements.

    Args:
        X: Feature matrix.
        y: Target label vector.
        n_repeats: Number of experiment repetitions.
        test_size: Proportion allocated to held-out test partition.
        stratify: Whether stratification is required.

    Returns:
        Tuple of validated (pd.DataFrame, pd.Series).

    Raises:
        ValueError: If dimensions mismatch, sample count is insufficient, or class counts violate stratification.
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None.")

    # Convert to DataFrame and Series if needed
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(np.asarray(X).shape[1])])
    else:
        X_df = X.copy()

    if not isinstance(y, pd.Series):
        y_sr = pd.Series(y, name="target")
    else:
        y_sr = y.copy()

    if len(X_df) != len(y_sr):
        raise ValueError(f"Length mismatch: X has {len(X_df)} rows, y has {len(y_sr)} labels.")

    if len(X_df) < 15:
        raise ValueError(f"Dataset has only {len(X_df)} samples. Minimum 15 samples required.")

    if not (0.05 <= test_size <= 0.5):
        raise ValueError(f"test_size={test_size} must be between 0.05 and 0.5.")

    if n_repeats < 1:
        raise ValueError(f"n_repeats={n_repeats} must be at least 1.")

    if stratify:
        class_counts = y_sr.value_counts()
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise ValueError(
                f"Stratification impossible: minority class has only {min_class_count} sample(s). Minimum 2 required."
            )

    return X_df, y_sr


def generate_repeated_splits(
    dataset_bundle: DatasetBundle,
    seeds: List[int],
    test_size: float = 0.2,
    stratify: bool = True,
) -> List[TrainTestSplitResult]:
    """
    Generate reproducible, stratified train/test splits for each provided seed.

    Args:
        dataset_bundle: DatasetBundle container.
        seeds: List of integer seeds.
        test_size: Test proportion.
        stratify: Whether to enforce stratified class sampling.

    Returns:
        List of TrainTestSplitResult objects.
    """
    splits: List[TrainTestSplitResult] = []
    for seed in seeds:
        split = create_train_test_split(
            dataset_bundle=dataset_bundle,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
            shuffle=True,
        )
        splits.append(split)
    return splits


def _execute_single_run(
    split: TrainTestSplitResult,
    run_number: int,
    experiment_id: str,
    model_name: str,
    model_config: Optional[Union[Dict[str, Any], BaseModelConfig]] = None,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    comparison_mode_config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
    mode: str = "fixed",
    tuning_metric: str = "f1_macro",
    tuning_max_candidates: int = 50,
) -> RepeatedRunResult:
    """
    Execute one single train-fit-evaluate repetition with strict leakage safety.

    Args:
        split: TrainTestSplitResult for this repetition.
        run_number: Zero-based repetition counter.
        experiment_id: Unique experiment campaign UUID.
        model_name: Canonical model identifier.
        model_config: Model configuration (base or fixed).
        preprocessing_config: Preprocessing options.
        feature_selection_config: Feature selection options.
        comparison_mode_config: Comparison mode options.
        mode: 'fixed' (Mode A) or 'tuned' (Mode B).
        tuning_metric: Target metric for CV candidate ranking.
        tuning_max_candidates: Maximum candidates to evaluate in tuning.

    Returns:
        RepeatedRunResult container.
    """
    run_start = time.monotonic()
    warnings_list: List[str] = []
    selected_params: Optional[Dict[str, Any]] = None

    # Base resolved config dictionary
    validated_cfg = validate_model_config(model_name, model_config)
    effective_model_cfg = validated_cfg.to_dict()

    try:
        prep_start = time.monotonic()

        # 1. Fit preprocessor strictly on this repetition's training partition
        prep_artifact = fit_preprocessor(X_train=split.X_train, config=preprocessing_config)
        X_train_proc = transform_with_preprocessor(prep_artifact, split.X_train)
        X_test_proc = transform_with_preprocessor(prep_artifact, split.X_test)

        # 2. Fit feature selector strictly on this repetition's training partition (if enabled)
        if feature_selection_config is not None:
            fs_artifact = fit_selector(
                X_train=X_train_proc,
                y_train=split.y_train,
                config=feature_selection_config,
            )
            X_train_proc = transform_with_selector(fs_artifact, X_train_proc)
            X_test_proc = transform_with_selector(fs_artifact, X_test_proc)
            warnings_list.extend(fs_artifact.warnings)

        # 3. Optional Comparison Mode application
        comp_mode_str = None
        if comparison_mode_config is not None:
            comp_res = prepare_comparison_mode(
                X_train_processed=X_train_proc,
                y_train=split.y_train,
                X_test_processed=X_test_proc,
                y_test=split.y_test,
                config=comparison_mode_config,
            )
            X_train_proc = comp_res.X_train_classical
            X_test_proc = comp_res.X_test_classical
            comp_mode_str = comp_res.mode
            warnings_list.extend(comp_res.validation_warnings)

        prep_time = time.monotonic() - prep_start

        # 4. Mode Branching: Fixed (Mode A) vs Tuned (Mode B)
        if mode == "tuned":
            # Tune using only this repetition's training partition via CV
            tuning_res = tune_model(
                model_name=model_name,
                X_train=split.X_train,
                y_train=split.y_train,
                preprocessing_config=preprocessing_config,
                feature_selection_config=feature_selection_config,
                max_candidates=tuning_max_candidates,
                metric=tuning_metric,
                random_state=split.random_state,
            )
            selected_params = tuning_res.best_config
            model_to_fit_config = selected_params
        else:
            # Mode A: fixed configuration
            model_to_fit_config = effective_model_cfg

        # 5. Fit Model on full training partition
        fit_start = time.monotonic()
        model_instance = create_model(model_name, model_to_fit_config)
        model_instance.fit(X_train_proc, split.y_train)
        train_time = time.monotonic() - fit_start

        # 6. Predict on held-out test partition
        predict_start = time.monotonic()
        y_pred = model_instance.predict(X_test_proc)
        inference_time = time.monotonic() - predict_start

        # 7. Collect probabilities or decision scores if available
        y_proba = None
        y_score = None
        if hasattr(model_instance, "predict_proba"):
            try:
                y_proba = model_instance.predict_proba(X_test_proc)
            except Exception:
                y_proba = None

        if y_proba is None and hasattr(model_instance, "decision_function"):
            try:
                y_score = model_instance.decision_function(X_test_proc)
            except Exception:
                y_score = None

        # 8. Evaluate using model-agnostic evaluation engine
        eval_metrics = evaluate_predictions(
            y_true=split.y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            y_score=y_score,
        )

        total_time = time.monotonic() - run_start

        # Collect convergence diagnostics if model is MLP
        convergence_info = None
        if model_name == "mlp":
            convergence_info = {
                "n_iter": getattr(model_instance, "n_iter_", None),
                "max_iter": getattr(model_instance, "max_iter", None),
                "loss": getattr(model_instance, "loss_", None),
                "converged": getattr(model_instance, "converged_", None),
                "convergence_warning": getattr(model_instance, "convergence_warning_", None),
                "early_stopping": getattr(model_instance, "early_stopping", False),
                "loss_curve": getattr(getattr(model_instance, "_clf", None), "loss_curve_", None),
            }

        # Extract scalar ROC-AUC safely
        roc_auc_val: Optional[float] = None
        if eval_metrics.roc_auc is not None:
            if isinstance(eval_metrics.roc_auc, dict):
                roc_auc_val = eval_metrics.roc_auc.get("value")
            else:
                roc_auc_val = float(eval_metrics.roc_auc)

        return RepeatedRunResult(
            experiment_id=experiment_id,
            run_number=run_number,
            random_state=split.random_state,
            train_indices=list(split.train_indices),
            test_indices=list(split.test_indices),
            n_train=split.n_train,
            n_test=split.n_test,
            train_class_distribution={str(k): int(v) for k, v in split.train_class_distribution.items()},
            test_class_distribution={str(k): int(v) for k, v in split.test_class_distribution.items()},
            model_name=model_name,
            model_config=_sanitize_value(model_to_fit_config),
            selected_params=_sanitize_value(selected_params) if selected_params else None,
            preprocessing_config=_sanitize_value(prep_artifact.config),
            feature_selection_config=_sanitize_value(feature_selection_config) if feature_selection_config else None,
            comparison_mode=comp_mode_str,
            accuracy=eval_metrics.accuracy,
            balanced_accuracy=eval_metrics.balanced_accuracy,
            precision_macro=eval_metrics.precision_macro,
            precision_weighted=eval_metrics.precision_weighted,
            recall_macro=eval_metrics.recall_macro,
            recall_weighted=eval_metrics.recall_weighted,
            f1_macro=eval_metrics.f1_macro,
            f1_weighted=eval_metrics.f1_weighted,
            roc_auc=roc_auc_val,
            confusion_matrix=eval_metrics.confusion_matrix,
            train_time_seconds=train_time,
            inference_time_seconds=inference_time,
            preprocessing_time_seconds=prep_time,
            total_time_seconds=total_time,
            convergence_info=_sanitize_value(convergence_info),
            failed=False,
            error_message=None,
            warnings=warnings_list + eval_metrics.warnings,
            metrics=eval_metrics.to_dict(),
        )

    except Exception as exc:
        total_time = time.monotonic() - run_start
        return RepeatedRunResult(
            experiment_id=experiment_id,
            run_number=run_number,
            random_state=split.random_state,
            train_indices=list(split.train_indices),
            test_indices=list(split.test_indices),
            n_train=split.n_train,
            n_test=split.n_test,
            train_class_distribution={str(k): int(v) for k, v in split.train_class_distribution.items()},
            test_class_distribution={str(k): int(v) for k, v in split.test_class_distribution.items()},
            model_name=model_name,
            model_config=_sanitize_value(effective_model_cfg),
            selected_params=None,
            preprocessing_config=None,
            feature_selection_config=None,
            comparison_mode=None,
            accuracy=None,
            balanced_accuracy=None,
            precision_macro=None,
            precision_weighted=None,
            recall_macro=None,
            recall_weighted=None,
            f1_macro=None,
            f1_weighted=None,
            roc_auc=None,
            confusion_matrix=None,
            train_time_seconds=0.0,
            inference_time_seconds=0.0,
            preprocessing_time_seconds=0.0,
            total_time_seconds=total_time,
            convergence_info=None,
            failed=True,
            error_message=str(exc),
            warnings=warnings_list,
            metrics={},
        )


def run_repeated_experiment(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, List[Any]],
    model_name: str,
    model_config: Optional[Union[Dict[str, Any], BaseModelConfig]] = None,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    comparison_mode_config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
    n_repeats: int = 5,
    test_size: float = 0.2,
    seeds: Optional[List[int]] = None,
    stratify: bool = True,
    mode: str = "fixed",
    dataset_name: str = "dataset",
    experiment_id: Optional[str] = None,
    tuning_metric: str = "f1_macro",
    tuning_max_candidates: int = 50,
) -> RepeatedExperimentResult:
    """
    Execute repeated stratified experiment for a single model across independent splits.

    Args:
        X: Feature matrix.
        y: Target label vector.
        model_name: Canonical model name ('dummy', 'svm', 'mlp', 'random_forest').
        model_config: Fixed model configuration (Mode A) or base overrides.
        preprocessing_config: Optional PreprocessingConfig or dict.
        feature_selection_config: Optional FeatureSelectionConfig or dict.
        comparison_mode_config: Optional ComparisonModeConfig or dict.
        n_repeats: Number of experiment repetitions (default 5).
        test_size: Proportion allocated to held-out test partition (default 0.2).
        seeds: Optional explicit list of integer seeds. Defaults to DEFAULT_SEEDS[:n_repeats].
        stratify: Whether to enforce stratified class sampling (default True).
        mode: 'fixed' (Mode A, default) or 'tuned' (Mode B).
        dataset_name: Identifier for dataset in reporting.
        experiment_id: Optional UUID string. If None, generated automatically.
        tuning_metric: Selection metric if mode='tuned'.
        tuning_max_candidates: Candidate evaluation limit if mode='tuned'.

    Returns:
        RepeatedExperimentResult container with per-run and aggregate summaries.
    """
    exp_start = time.monotonic()
    exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:12]}"

    if mode not in ("fixed", "tuned"):
        raise ValueError(f"Invalid experiment mode '{mode}'. Allowed: 'fixed', 'tuned'.")

    # Validate inputs
    X_df, y_sr = validate_repeated_inputs(X, y, n_repeats, test_size, stratify)

    # Resolve seeds
    if seeds is None:
        effective_seeds = DEFAULT_SEEDS[:n_repeats]
        if len(effective_seeds) < n_repeats:
            # Deterministically extend if n_repeats > 5
            last_seed = effective_seeds[-1] if effective_seeds else 42
            effective_seeds += [last_seed + (i + 1) * 100 for i in range(n_repeats - len(effective_seeds))]
    else:
        if len(seeds) != n_repeats:
            raise ValueError(f"Length of seeds ({len(seeds)}) must match n_repeats ({n_repeats}).")
        effective_seeds = list(seeds)

    # Wrap in DatasetBundle for unified splitting
    bundle = DatasetBundle(
        X=X_df,
        y=y_sr,
        feature_names=list(X_df.columns),
        target_names=[str(c) for c in np.unique(y_sr)],
        dataset_name=dataset_name,
    )

    # Generate splits
    splits = generate_repeated_splits(
        dataset_bundle=bundle,
        seeds=effective_seeds,
        test_size=test_size,
        stratify=stratify,
    )

    # Execute runs sequentially
    runs: List[RepeatedRunResult] = []
    for i, split in enumerate(splits):
        run_res = _execute_single_run(
            split=split,
            run_number=i,
            experiment_id=exp_id,
            model_name=model_name,
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            feature_selection_config=feature_selection_config,
            comparison_mode_config=comparison_mode_config,
            mode=mode,
            tuning_metric=tuning_metric,
            tuning_max_candidates=tuning_max_candidates,
        )
        runs.append(run_res)

    # Compute aggregates
    aggregates = compute_aggregates(runs, model_name=model_name)
    total_duration = time.monotonic() - exp_start

    return RepeatedExperimentResult(
        experiment_id=exp_id,
        model_name=model_name,
        mode=mode,
        n_repeats=n_repeats,
        seeds=effective_seeds,
        test_size=test_size,
        stratified=stratify,
        runs=runs,
        aggregates=aggregates,
        total_time_seconds=total_duration,
        model_config=_sanitize_value(model_config) if model_config else None,
        preprocessing_config=_sanitize_value(preprocessing_config) if preprocessing_config else None,
        feature_selection_config=_sanitize_value(feature_selection_config) if feature_selection_config else None,
        comparison_mode=comparison_mode_config.mode if isinstance(comparison_mode_config, ComparisonModeConfig) else None,
        package_versions=get_provenance_metadata(),
    )


def run_repeated_comparison(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, List[Any]],
    models: List[str],
    model_configs: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    comparison_mode_config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
    n_repeats: int = 5,
    test_size: float = 0.2,
    seeds: Optional[List[int]] = None,
    stratify: bool = True,
    mode: str = "fixed",
    dataset_name: str = "dataset",
    experiment_id: Optional[str] = None,
    tuning_metric: str = "f1_macro",
    tuning_max_candidates: int = 50,
) -> MultiModelComparisonResult:
    """
    Execute fair repeated comparison across multiple models sharing IDENTICAL splits.

    Every model receives the exact same dataset partition in corresponding runs.

    Args:
        X: Feature matrix.
        y: Target label vector.
        models: List of model identifiers to compare (e.g. ['dummy', 'svm', 'mlp', 'random_forest']).
        model_configs: Optional dictionary mapping model_name to configuration dictionary.
        preprocessing_config: Preprocessing options shared across models.
        feature_selection_config: Feature selection options shared across models.
        comparison_mode_config: Comparison mode options.
        n_repeats: Number of repetitions (default 5).
        test_size: Held-out test proportion (default 0.2).
        seeds: Random seed list.
        stratify: Whether to enforce stratified class sampling.
        mode: 'fixed' or 'tuned'.
        dataset_name: Dataset identifier.
        experiment_id: Campaign ID.
        tuning_metric: Selection metric if mode='tuned'.
        tuning_max_candidates: Candidate evaluation limit if mode='tuned'.

    Returns:
        MultiModelComparisonResult containing synchronized per-model results.
    """
    comp_start = time.monotonic()
    exp_id = experiment_id or f"comp_{uuid.uuid4().hex[:12]}"
    configs_map = model_configs or {}

    # Validate inputs
    X_df, y_sr = validate_repeated_inputs(X, y, n_repeats, test_size, stratify)

    # Resolve seeds once
    if seeds is None:
        effective_seeds = DEFAULT_SEEDS[:n_repeats]
        if len(effective_seeds) < n_repeats:
            last_seed = effective_seeds[-1] if effective_seeds else 42
            effective_seeds += [last_seed + (i + 1) * 100 for i in range(n_repeats - len(effective_seeds))]
    else:
        if len(seeds) != n_repeats:
            raise ValueError(f"Length of seeds ({len(seeds)}) must match n_repeats ({n_repeats}).")
        effective_seeds = list(seeds)

    # Wrap in DatasetBundle and generate splits ONCE
    bundle = DatasetBundle(
        X=X_df,
        y=y_sr,
        feature_names=list(X_df.columns),
        target_names=[str(c) for c in np.unique(y_sr)],
        dataset_name=dataset_name,
    )
    shared_splits = generate_repeated_splits(
        dataset_bundle=bundle,
        seeds=effective_seeds,
        test_size=test_size,
        stratify=stratify,
    )

    split_indices: List[Dict[str, List[Any]]] = [
        {"run_number": i, "random_state": s.random_state, "train_indices": list(s.train_indices), "test_indices": list(s.test_indices)}
        for i, s in enumerate(shared_splits)
    ]

    model_results: Dict[str, RepeatedExperimentResult] = {}

    for model_name in models:
        m_start = time.monotonic()
        m_cfg = configs_map.get(model_name, None)
        runs: List[RepeatedRunResult] = []

        # Iterate over the identical shared splits
        for i, split in enumerate(shared_splits):
            run_res = _execute_single_run(
                split=split,
                run_number=i,
                experiment_id=exp_id,
                model_name=model_name,
                model_config=m_cfg,
                preprocessing_config=preprocessing_config,
                feature_selection_config=feature_selection_config,
                comparison_mode_config=comparison_mode_config,
                mode=mode,
                tuning_metric=tuning_metric,
                tuning_max_candidates=tuning_max_candidates,
            )
            runs.append(run_res)

        aggs = compute_aggregates(runs, model_name=model_name)
        m_duration = time.monotonic() - m_start

        model_results[model_name] = RepeatedExperimentResult(
            experiment_id=exp_id,
            model_name=model_name,
            mode=mode,
            n_repeats=n_repeats,
            seeds=effective_seeds,
            test_size=test_size,
            stratified=stratify,
            runs=runs,
            aggregates=aggs,
            total_time_seconds=m_duration,
            model_config=_sanitize_value(m_cfg) if m_cfg else None,
            preprocessing_config=_sanitize_value(preprocessing_config) if preprocessing_config else None,
            feature_selection_config=_sanitize_value(feature_selection_config) if feature_selection_config else None,
            comparison_mode=comparison_mode_config.mode if isinstance(comparison_mode_config, ComparisonModeConfig) else None,
            package_versions=get_provenance_metadata(),
        )

    total_duration = time.monotonic() - comp_start

    return MultiModelComparisonResult(
        experiment_id=exp_id,
        mode=mode,
        n_repeats=n_repeats,
        seeds=effective_seeds,
        models=models,
        split_indices=split_indices,
        model_results=model_results,
        comparison_mode=comparison_mode_config.mode if isinstance(comparison_mode_config, ComparisonModeConfig) else None,
        total_time_seconds=total_duration,
        package_versions=get_provenance_metadata(),
    )
