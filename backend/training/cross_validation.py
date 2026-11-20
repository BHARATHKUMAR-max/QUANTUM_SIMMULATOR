"""
backend/training/cross_validation.py
=====================================
Leakage-safe stratified cross-validation workflow for QML Benchmark Studio.

Integrates with the existing data pipeline (splitting, preprocessing, feature selection)
to provide reproducible fold-level evaluation for classical models.

Critical Leakage Safeguard
--------------------------
Every preprocessing artifact (ColumnTransformer) and feature-selector artifact
(SelectKBest) is fitted **exclusively** on the fold training slice. The fold
validation slice is transformed—never refitted on. The held-out test partition
never participates in any CV operation.

Usage
-----
    from backend.training.cross_validation import run_cv_for_config

    result = run_cv_for_config(
        model_name="svm",
        config={"kernel": "rbf", "C": 1.0},
        X_train=X_train,
        y_train=y_train,
        preprocessing_config=preprocessing_config,
        feature_selection_config=feature_selection_config,
        n_splits=3,
        metric="f1_macro",
        random_state=42,
    )
    print(result.mean_cv_score, result.std_cv_score)
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from backend.classical.model_registry import create_model
from backend.data.feature_selection import fit_selector, transform_with_selector
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor
from backend.data.schemas import FeatureSelectionConfig, PreprocessingConfig
from backend.data.splitting import create_stratified_cv_splits


# ---------------------------------------------------------------------------
# Supported evaluation metrics
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1_macro": lambda y_t, y_p: f1_score(y_t, y_p, average="macro", zero_division=0),
    "f1_weighted": lambda y_t, y_p: f1_score(y_t, y_p, average="weighted", zero_division=0),
}


def _compute_scalar_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute a single numeric metric from predictions.

    Args:
        metric: Metric name. Must be one of SUPPORTED_METRICS.
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Metric score as float.

    Raises:
        ValueError: If metric name is unrecognised.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported: {sorted(SUPPORTED_METRICS.keys())}"
        )
    return float(SUPPORTED_METRICS[metric](y_true, y_pred))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldScore:
    """
    Per-fold evaluation result for one cross-validation fold.

    Attributes:
        fold_index: Zero-based fold index.
        train_score: Training partition metric score for this fold (informational only).
        val_score: Validation partition metric score for this fold (the CV signal).
        metric: Metric name used for scoring.
        fit_time: Wall-clock seconds to fit preprocessor + (optionally) feature selector + model.
        n_train: Number of fold training samples.
        n_val: Number of fold validation samples.
        selected_features: Feature names after preprocessing (and optional selection).
        convergence_warnings: Any convergence warnings captured during model fitting.
    """
    fold_index: int
    train_score: float
    val_score: float
    metric: str
    fit_time: float
    n_train: int
    n_val: int
    selected_features: List[str]
    convergence_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        return {
            "fold_index": self.fold_index,
            "train_score": round(self.train_score, 6),
            "val_score": round(self.val_score, 6),
            "metric": self.metric,
            "fit_time": round(self.fit_time, 4),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_selected_features": len(self.selected_features),
            "selected_features": self.selected_features,
            "convergence_warnings": self.convergence_warnings,
        }


@dataclass
class CVResult:
    """
    Aggregated cross-validation result across all folds.

    Attributes:
        model_name: Canonical model identifier.
        config: Configuration dictionary passed to the model.
        fold_scores: Per-fold FoldScore instances.
        mean_cv_score: Mean validation score across all folds.
        std_cv_score: Standard deviation of validation scores across all folds.
        metric: Metric name used throughout evaluation.
        n_folds: Number of folds.
        random_state: Random state used for fold generation.
        total_time: Total wall-clock seconds for the full CV run.
        warnings: Aggregated cross-fold warnings.
    """
    model_name: str
    config: Dict[str, Any]
    fold_scores: List[FoldScore]
    mean_cv_score: float
    std_cv_score: float
    metric: str
    n_folds: int
    random_state: int
    total_time: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "mean_cv_score": round(self.mean_cv_score, 6),
            "std_cv_score": round(self.std_cv_score, 6),
            "metric": self.metric,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "total_time": round(self.total_time, 4),
            "fold_scores": [fs.to_dict() for fs in self.fold_scores],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Main CV function
# ---------------------------------------------------------------------------


def run_cv_for_config(
    model_name: str,
    config: Optional[Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    n_splits: int = 3,
    metric: str = "f1_macro",
    random_state: int = 42,
) -> CVResult:
    """
    Run stratified k-fold cross-validation for one model configuration.

    Per-fold pipeline (strictly leakage-safe):
      1. Subset X_train / y_train using fold train_indices and val_indices.
      2. fit_preprocessor() on fold training slice only.
      3. transform_with_preprocessor() on both fold train and val slices.
      4. Optionally: fit_selector() on fold training slice; apply to both.
      5. create_model() → model.fit(fold_train) → model.predict(fold_val).
      6. Compute scalar metric and record FoldScore.

    The held-out test partition is never touched.

    Args:
        model_name: Canonical or alias model identifier (e.g. "svm", "random_forest").
        config: Parameter dictionary passed to create_model(). None uses defaults.
        X_train: Full training feature DataFrame (test set excluded).
        y_train: Full training target Series.
        preprocessing_config: PreprocessingConfig or dict. None uses defaults.
        feature_selection_config: FeatureSelectionConfig or dict. None skips selection.
        n_splits: Number of CV folds. Minimum class count must be >= n_splits.
        metric: Validation metric. One of: "accuracy", "balanced_accuracy",
                "f1_macro", "f1_weighted".
        random_state: Seed for stratified fold generation.

    Returns:
        CVResult with per-fold scores and aggregate statistics.

    Raises:
        ValueError: If metric is unsupported or fold generation fails.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported: {sorted(SUPPORTED_METRICS.keys())}"
        )

    overall_start = time.monotonic()
    all_warnings: List[str] = []

    # Generate fold indices from training data only
    folds = create_stratified_cv_splits(
        X_train=X_train,
        y_train=y_train,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
    )

    fold_scores: List[FoldScore] = []

    for fold in folds:
        fold_start = time.monotonic()
        fold_warnings: List[str] = []

        # --- Step 1: Subset data using fold indices ---
        X_fold_train_raw = X_train.loc[fold.train_indices]
        X_fold_val_raw = X_train.loc[fold.val_indices]
        y_fold_train = y_train.loc[fold.train_indices]
        y_fold_val = y_train.loc[fold.val_indices]

        # --- Step 2: Fit preprocessor exclusively on fold training slice ---
        prep_artifact = fit_preprocessor(
            X_train=X_fold_train_raw,
            config=preprocessing_config,
        )

        # --- Step 3: Transform both slices with the fold-fitted preprocessor ---
        X_fold_train_proc = transform_with_preprocessor(prep_artifact, X_fold_train_raw)
        X_fold_val_proc = transform_with_preprocessor(prep_artifact, X_fold_val_raw)

        # --- Step 4: Optional feature selection (fold-train only) ---
        if feature_selection_config is not None:
            fs_artifact = fit_selector(
                X_train=X_fold_train_proc,
                y_train=y_fold_train,
                config=feature_selection_config,
            )
            X_fold_train_proc = transform_with_selector(fs_artifact, X_fold_train_proc)
            X_fold_val_proc = transform_with_selector(fs_artifact, X_fold_val_proc)
            fold_warnings.extend(fs_artifact.warnings)
            selected_features = fs_artifact.selected_feature_names
        else:
            selected_features = list(X_fold_train_proc.columns)

        # --- Step 5: Fit model and predict on validation slice ---
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = create_model(model_name, config)
            model.fit(X_fold_train_proc, y_fold_train)

        convergence_warnings = []
        for w in captured:
            msg = str(w.message)
            if "ConvergenceWarning" in str(w.category) or "convergence" in msg.lower():
                convergence_warnings.append(msg)
            # Absorb other fitting warnings silently

        y_pred_train = model.predict(X_fold_train_proc)
        y_pred_val = model.predict(X_fold_val_proc)

        # --- Step 6: Score ---
        train_score = _compute_scalar_metric(metric, y_fold_train.to_numpy(), y_pred_train)
        val_score = _compute_scalar_metric(metric, y_fold_val.to_numpy(), y_pred_val)

        fold_time = time.monotonic() - fold_start

        fold_score = FoldScore(
            fold_index=fold.fold_index,
            train_score=train_score,
            val_score=val_score,
            metric=metric,
            fit_time=fold_time,
            n_train=len(y_fold_train),
            n_val=len(y_fold_val),
            selected_features=selected_features,
            convergence_warnings=convergence_warnings,
        )
        fold_scores.append(fold_score)
        all_warnings.extend(fold_warnings)

    val_scores = [fs.val_score for fs in fold_scores]
    mean_cv = float(np.mean(val_scores))
    std_cv = float(np.std(val_scores, ddof=0))
    total_time = time.monotonic() - overall_start

    return CVResult(
        model_name=model_name,
        config=config or {},
        fold_scores=fold_scores,
        mean_cv_score=mean_cv,
        std_cv_score=std_cv,
        metric=metric,
        n_folds=len(fold_scores),
        random_state=random_state,
        total_time=total_time,
        warnings=all_warnings,
    )
