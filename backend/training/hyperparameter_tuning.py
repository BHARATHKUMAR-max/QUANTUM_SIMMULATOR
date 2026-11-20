"""
backend/training/hyperparameter_tuning.py
==========================================
Deterministic grid-search hyperparameter tuning for QML Benchmark Studio.

Implements a custom grid-search that integrates fold-local preprocessing and
feature selection, ensuring zero information leakage from the held-out test set.

Architecture
------------
1. For each candidate configuration in the search grid, run_cv_for_config() is
   called to obtain a mean CV score (using training data only).
2. The candidate with the highest mean CV score is selected as best_config.
3. final_evaluation() trains the best config on all training data, then evaluates
   on the held-out test set exactly once.

No test-set data is used during candidate selection (stage='cv_selection').
Test evaluation is performed only during stage='final_test_evaluation'.

Default Search Grids
--------------------
- dummy:         3 candidates  (strategy variants)
- svm:           9 candidates  (kernel × C × gamma for rbf only)
- mlp:           12 candidates (hidden_layers × alpha × learning_rate_init)
- random_forest: 12 candidates (n_estimators × max_depth × min_samples_split)

Usage
-----
    from backend.training.hyperparameter_tuning import tune_model, final_evaluation

    tuning_result = tune_model(
        model_name="svm",
        X_train=X_train,
        y_train=y_train,
        preprocessing_config=prep_cfg,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    print(tuning_result.best_config, tuning_result.best_mean_cv_score)

    test_result = final_evaluation(
        model_name="svm",
        best_config=tuning_result.best_config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessing_config=prep_cfg,
    )
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backend.classical.model_registry import create_model
from backend.data.feature_selection import fit_selector, transform_with_selector
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor
from backend.data.schemas import FeatureSelectionConfig, PreprocessingConfig
from backend.evaluation.metrics import evaluate_predictions
from backend.training.cross_validation import run_cv_for_config


# ---------------------------------------------------------------------------
# Default search grids
# ---------------------------------------------------------------------------

def _build_svm_grid() -> List[Dict[str, Any]]:
    """
    Build SVM search grid: gamma is included only for 'rbf' kernel.

    Linear SVM candidates never carry a gamma parameter (it has no effect for
    linear kernels and would create misleading parameter parity).
    """
    grid = []
    for C in [0.1, 1.0, 10.0]:
        grid.append({"kernel": "linear", "C": C})
        for gamma in ["scale", "auto"]:
            grid.append({"kernel": "rbf", "C": C, "gamma": gamma})
    return grid


def _build_mlp_grid() -> List[Dict[str, Any]]:
    """Build MLP grid: hidden_layer_sizes × alpha × learning_rate_init."""
    grid = []
    for hidden in [(32,), (64,), (64, 32)]:
        for alpha in [0.0001, 0.001]:
            for lr in [0.001, 0.01]:
                grid.append({
                    "hidden_layer_sizes": hidden,
                    "alpha": alpha,
                    "learning_rate_init": lr,
                    "max_iter": 300,
                })
    return grid


def _build_rf_grid() -> List[Dict[str, Any]]:
    """Build Random Forest grid: n_estimators × max_depth × min_samples_split."""
    grid = []
    for n in [100, 200]:
        for depth in [None, 10, 20]:
            for mss in [2, 5]:
                grid.append({
                    "n_estimators": n,
                    "max_depth": depth,
                    "min_samples_split": mss,
                })
    return grid


#: Default per-model hyperparameter search grids.
DEFAULT_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "dummy": [
        {"strategy": "most_frequent"},
        {"strategy": "stratified"},
        {"strategy": "uniform"},
    ],
    "svm": _build_svm_grid(),
    "mlp": _build_mlp_grid(),
    "random_forest": _build_rf_grid(),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """
    Result for a single hyperparameter candidate configuration.

    Attributes:
        config: Candidate parameter dictionary evaluated.
        fold_scores: Per-fold validation scores.
        mean_cv_score: Mean of fold_scores (primary selection criterion).
        std_cv_score: Standard deviation of fold_scores.
        n_folds_completed: Number of successfully completed folds.
        fit_times: Per-fold fit times in seconds.
        total_time: Total wall-clock seconds for evaluating this candidate.
        failed: Whether the candidate evaluation encountered an error.
        failure_reason: Description of failure if failed=True.
        convergence_warnings: Collected convergence warnings across all folds.
    """
    config: Dict[str, Any]
    fold_scores: List[float]
    mean_cv_score: float
    std_cv_score: float
    n_folds_completed: int
    fit_times: List[float]
    total_time: float
    failed: bool = False
    failure_reason: Optional[str] = None
    convergence_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        config_serialized = {}
        for k, v in self.config.items():
            # Serialize tuples (e.g. hidden_layer_sizes) as lists
            config_serialized[k] = list(v) if isinstance(v, tuple) else v

        return {
            "config": config_serialized,
            "mean_cv_score": round(self.mean_cv_score, 6),
            "std_cv_score": round(self.std_cv_score, 6),
            "fold_scores": [round(s, 6) for s in self.fold_scores],
            "n_folds_completed": self.n_folds_completed,
            "fit_times": [round(t, 4) for t in self.fit_times],
            "total_time": round(self.total_time, 4),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "convergence_warnings": self.convergence_warnings,
        }


@dataclass
class TuningResult:
    """
    Result of a full hyperparameter tuning run for one model.

    Attributes:
        model_name: Canonical model identifier.
        best_config: Configuration with highest mean CV score.
        best_mean_cv_score: Mean CV score of the best configuration.
        best_std_cv_score: Standard deviation of CV scores for the best configuration.
        selection_metric: Metric used for ranking candidates.
        all_candidates: Ordered list of all evaluated CandidateResult objects.
        n_candidates_evaluated: Total candidates evaluated (excluding budget-trimmed ones).
        random_state: Random state used throughout.
        stage: Evaluation stage identifier. Always 'cv_selection' here.
    """
    model_name: str
    best_config: Dict[str, Any]
    best_mean_cv_score: float
    best_std_cv_score: float
    selection_metric: str
    all_candidates: List[CandidateResult]
    n_candidates_evaluated: int
    random_state: int
    stage: str = "cv_selection"

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        best_config_serialized = {}
        for k, v in self.best_config.items():
            best_config_serialized[k] = list(v) if isinstance(v, tuple) else v

        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "best_config": best_config_serialized,
            "best_mean_cv_score": round(self.best_mean_cv_score, 6),
            "best_std_cv_score": round(self.best_std_cv_score, 6),
            "selection_metric": self.selection_metric,
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "n_candidates_total": len(self.all_candidates),
            "random_state": self.random_state,
        }


# ---------------------------------------------------------------------------
# Main tuning function
# ---------------------------------------------------------------------------


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    custom_grid: Optional[List[Dict[str, Any]]] = None,
    max_candidates: int = 50,
    n_splits: int = 3,
    metric: str = "f1_macro",
    random_state: int = 42,
) -> TuningResult:
    """
    Run deterministic grid-search hyperparameter tuning using cross-validation.

    For each candidate configuration:
      1. run_cv_for_config() is called with training data only.
      2. The mean CV score is used to rank candidates.

    The test set is never used here. Call final_evaluation() separately.

    Args:
        model_name: Canonical or alias model identifier.
        X_train: Training features (test set completely excluded).
        y_train: Training labels.
        preprocessing_config: Optional PreprocessingConfig or dict.
        feature_selection_config: Optional FeatureSelectionConfig or dict.
        custom_grid: Optional list of parameter dicts. Overrides DEFAULT_GRIDS.
        max_candidates: Maximum number of candidates to evaluate (budget guard).
        n_splits: Number of CV folds per candidate.
        metric: Scoring metric for candidate ranking.
        random_state: Seed for fold generation.

    Returns:
        TuningResult with best_config, best_mean_cv_score, and all_candidates.

    Raises:
        ValueError: If model_name is unknown or metric is unsupported.
        RuntimeError: If all candidates fail evaluation.
    """
    # Resolve search grid
    from backend.classical.model_registry import _resolve_model_name  # noqa: PLC0415
    canonical_name = _resolve_model_name(model_name)

    if custom_grid is not None:
        grid = custom_grid
    elif canonical_name in DEFAULT_GRIDS:
        grid = DEFAULT_GRIDS[canonical_name]
    else:
        # Fallback: single default-config candidate
        grid = [{}]

    # Apply candidate budget
    grid = grid[:max_candidates]

    all_candidates: List[CandidateResult] = []

    for candidate_config in grid:
        candidate_start = time.monotonic()
        try:
            cv_result = run_cv_for_config(
                model_name=model_name,
                config=candidate_config,
                X_train=X_train,
                y_train=y_train,
                preprocessing_config=preprocessing_config,
                feature_selection_config=feature_selection_config,
                n_splits=n_splits,
                metric=metric,
                random_state=random_state,
            )
            fold_scores_vals = [fs.val_score for fs in cv_result.fold_scores]
            fit_times = [fs.fit_time for fs in cv_result.fold_scores]
            all_conv_warnings = []
            for fs in cv_result.fold_scores:
                all_conv_warnings.extend(fs.convergence_warnings)

            candidate = CandidateResult(
                config=candidate_config,
                fold_scores=fold_scores_vals,
                mean_cv_score=cv_result.mean_cv_score,
                std_cv_score=cv_result.std_cv_score,
                n_folds_completed=cv_result.n_folds,
                fit_times=fit_times,
                total_time=time.monotonic() - candidate_start,
                failed=False,
                failure_reason=None,
                convergence_warnings=all_conv_warnings,
            )

        except Exception as exc:  # noqa: BLE001
            candidate = CandidateResult(
                config=candidate_config,
                fold_scores=[],
                mean_cv_score=float("-inf"),
                std_cv_score=0.0,
                n_folds_completed=0,
                fit_times=[],
                total_time=time.monotonic() - candidate_start,
                failed=True,
                failure_reason=str(exc),
                convergence_warnings=[],
            )

        all_candidates.append(candidate)

    # Select best candidate by mean CV score (failed candidates have -inf score)
    successful = [c for c in all_candidates if not c.failed]
    if not successful:
        raise RuntimeError(
            f"All {len(all_candidates)} candidate(s) for model '{model_name}' "
            f"failed during cross-validation. See all_candidates for details."
        )

    best_candidate = max(successful, key=lambda c: c.mean_cv_score)

    return TuningResult(
        model_name=model_name,
        best_config=best_candidate.config,
        best_mean_cv_score=best_candidate.mean_cv_score,
        best_std_cv_score=best_candidate.std_cv_score,
        selection_metric=metric,
        all_candidates=all_candidates,
        n_candidates_evaluated=len(all_candidates),
        random_state=random_state,
        stage="cv_selection",
    )


# ---------------------------------------------------------------------------
# Final test-set evaluation (called once, after tuning is complete)
# ---------------------------------------------------------------------------


def final_evaluation(
    model_name: str,
    best_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
) -> Dict[str, Any]:
    """
    Fit best configuration on all training data and evaluate on the held-out test set.

    This function is the ONLY entry-point that touches X_test / y_test.
    It must be called exactly once after tuning is complete.

    Pipeline:
      1. Fit preprocessor on all X_train (no folds — full training set).
      2. Transform X_train and X_test with the fitted preprocessor.
      3. Optionally fit feature selector on transformed X_train; apply to both.
      4. Fit model on transformed X_train.
      5. Predict on transformed X_test.
      6. Evaluate using evaluate_predictions() (the standard evaluation engine).

    Args:
        model_name: Canonical or alias model identifier.
        best_config: Best configuration selected by tune_model().
        X_train: Full training features.
        y_train: Full training labels.
        X_test: Held-out test features.
        y_test: Held-out test labels.
        preprocessing_config: Optional preprocessing configuration.
        feature_selection_config: Optional feature selection configuration.

    Returns:
        Dictionary with keys:
          - "stage": "final_test_evaluation"
          - "model_name": canonical model name
          - "config": serialized best_config
          - "metrics": EvaluationMetricsResult.to_dict()
          - "n_train": training sample count
          - "n_test": test sample count
          - "n_features_used": feature count used in final model
          - "fit_time_seconds": model training time
          - "predict_time_seconds": model prediction time
    """
    eval_start = time.monotonic()

    # 1. Fit preprocessor on full training set
    prep_artifact = fit_preprocessor(X_train=X_train, config=preprocessing_config)
    X_train_proc = transform_with_preprocessor(prep_artifact, X_train)
    X_test_proc = transform_with_preprocessor(prep_artifact, X_test)

    # 2. Optional feature selection on full training set
    if feature_selection_config is not None:
        fs_artifact = fit_selector(
            X_train=X_train_proc,
            y_train=y_train,
            config=feature_selection_config,
        )
        X_train_proc = transform_with_selector(fs_artifact, X_train_proc)
        X_test_proc = transform_with_selector(fs_artifact, X_test_proc)

    n_features_used = X_train_proc.shape[1]

    # 3. Fit model on full transformed training set
    fit_start = time.monotonic()
    model = create_model(model_name, best_config)
    model.fit(X_train_proc, y_train)
    fit_time = time.monotonic() - fit_start

    # 4. Predict on held-out test set
    predict_start = time.monotonic()
    y_pred = model.predict(X_test_proc)
    predict_time = time.monotonic() - predict_start

    # 5. Optionally obtain probabilities (for ROC-AUC)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_proc)
        except Exception:  # noqa: BLE001
            y_proba = None

    # 6. Evaluate using the standard evaluation engine
    metrics_result = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Serialize best_config (handle tuple values like hidden_layer_sizes)
    config_serialized: Dict[str, Any] = {}
    for k, v in best_config.items():
        config_serialized[k] = list(v) if isinstance(v, tuple) else v

    return {
        "stage": "final_test_evaluation",
        "model_name": model_name,
        "config": config_serialized,
        "metrics": metrics_result.to_dict(),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features_used": n_features_used,
        "fit_time_seconds": round(fit_time, 4),
        "predict_time_seconds": round(predict_time, 4),
        "total_time_seconds": round(time.monotonic() - eval_start, 4),
    }
