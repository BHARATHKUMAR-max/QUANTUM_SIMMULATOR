"""
backend/data/feature_selection.py
==================================
Leakage-safe supervised feature selection for QML Benchmark Studio.

Implements two selection methods:
  - Mutual Information (mutual_info_classif): model-free, captures non-linear relationships.
  - ANOVA F-test (f_classif): parametric, fast, linear assumptions.

Both methods enforce strict training-only fitting to prevent data leakage.
Selectors are always fitted on X_train (or the fold training slice in CV mode) and
applied to X_test / fold validation data via transform-only operations.

Usage:
    from backend.data.feature_selection import fit_selector, transform_with_selector

    selector_artifact = fit_selector(X_train_processed, y_train, config)
    X_train_selected  = transform_with_selector(selector_artifact, X_train_processed)
    X_test_selected   = transform_with_selector(selector_artifact, X_test_processed)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from backend.data.schemas import (
    CVFoldResult,
    FeatureSelectionConfig,
    FeatureSelectorArtifact,
    FeatureSelectionPreview,
    PreprocessorArtifact,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_k(
    n_candidate: int,
    config: FeatureSelectionConfig,
) -> int:
    """
    Resolve the number of features to select given configuration and available features.

    Manual mode: validates and uses config.k_features directly.
    Auto mode:   uses config.k_features if set, else applies config.proportion,
                 else defaults to all available features.

    Args:
        n_candidate: Total number of candidate features.
        config: Validated FeatureSelectionConfig.

    Returns:
        Resolved k (int in [1, n_candidate]).

    Raises:
        ValueError: When k_features or proportion is invalid given actual feature count.
    """
    if n_candidate == 0:
        raise ValueError("Cannot select features from an empty feature matrix.")

    if config.mode == "manual":
        k = config.k_features  # Already validated > 0 in __post_init__
        if k > n_candidate:
            raise ValueError(
                f"Requested k_features={k} exceeds available candidate features ({n_candidate})."
            )
        return k

    # Auto mode
    if config.k_features is not None:
        k = config.k_features
        if k > n_candidate:
            k = n_candidate  # Silently cap at available; recorded in warnings
        return k

    if config.proportion is not None:
        k = max(1, int(np.floor(config.proportion * n_candidate)))
        return min(k, n_candidate)

    # Default: select all features
    return n_candidate


def _safe_scores_pvalues(
    selector: SelectKBest,
    n_candidate: int,
) -> Tuple[List[float], Optional[List[float]], List[str]]:
    """
    Extract scores and p-values from a fitted SelectKBest, replacing NaN/Inf safely.

    Returns:
        (scores_list, pvalues_list_or_None, warnings_list)
    """
    warnings_out: List[str] = []
    raw_scores = np.array(selector.scores_, dtype=float)

    # Replace NaN/Inf scores with 0.0 and emit a warning
    nan_mask = ~np.isfinite(raw_scores)
    if nan_mask.any():
        bad_count = int(nan_mask.sum())
        warnings_out.append(
            f"{bad_count} feature(s) produced NaN or infinite scores "
            "(likely constant features). Their scores were replaced with 0.0 "
            "and they will not be selected."
        )
        raw_scores[nan_mask] = 0.0

    scores_list = raw_scores.tolist()

    pvalues_list = None
    if selector.pvalues_ is not None:
        raw_pvals = np.array(selector.pvalues_, dtype=float)
        nan_pval_mask = ~np.isfinite(raw_pvals)
        if nan_pval_mask.any():
            raw_pvals[nan_pval_mask] = 1.0  # Treat undefined p-value as "not significant"
        pvalues_list = raw_pvals.tolist()

    return scores_list, pvalues_list, warnings_out


def _compute_rankings(scores: List[float]) -> List[int]:
    """
    Rank features from highest to lowest score (rank 1 = best).
    Ties receive the same rank (dense ranking).
    """
    arr = np.array(scores)
    # scipy-style: argsort descending, then assign ranks starting at 1
    sorted_indices = np.argsort(arr)[::-1]
    ranks = np.empty(len(arr), dtype=int)
    for rank_pos, idx in enumerate(sorted_indices):
        ranks[idx] = rank_pos + 1
    return ranks.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[Union[FeatureSelectionConfig, Dict[str, Any]]] = None,
) -> FeatureSelectorArtifact:
    """
    Fit a supervised feature selector exclusively on training data.

    The selector is fitted only on (X_train, y_train). No test or validation
    data is used in score computation. This enforces the same leakage-safe
    contract as the Day 7/8 preprocessing pipeline.

    Args:
        X_train: Preprocessed training features (pd.DataFrame).
                 Expected to be the output of transform_with_preprocessor().
        y_train: Training target labels (pd.Series).
        config: FeatureSelectionConfig instance or dict. Defaults to
                FeatureSelectionConfig() if None (mutual_info, auto, all features).

    Returns:
        FeatureSelectorArtifact: Contains fitted selector, feature names, scores,
                                  rankings, and preview metadata.

    Raises:
        ValueError: On invalid config, empty inputs, or k > n_candidate in manual mode.
    """
    # 1. Parse config
    sel_config = FeatureSelectionConfig.from_dict(config)

    # 2. Validate inputs
    if X_train.empty:
        raise ValueError("X_train is empty. Cannot fit a feature selector.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same number of rows. "
            f"Got X_train={len(X_train)}, y_train={len(y_train)}."
        )
    if y_train.isnull().any():
        raise ValueError("y_train contains missing values. All target labels must be provided.")

    candidate_names: List[str] = list(X_train.columns)
    n_candidate = len(candidate_names)

    # 3. Resolve k
    warnings_list: List[str] = []
    if sel_config.mode == "auto" and sel_config.k_features is not None:
        if sel_config.k_features > n_candidate:
            warnings_list.append(
                f"Requested k_features={sel_config.k_features} exceeds available features "
                f"({n_candidate}). Capping to {n_candidate}."
            )
    k = _resolve_k(n_candidate, sel_config)

    # 4. Build sklearn score function
    if sel_config.method == "mutual_info":
        def _mi_scorer(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return mutual_info_classif(X, y, random_state=sel_config.random_state)
        score_func = _mi_scorer
    elif sel_config.method == "anova":
        score_func = f_classif
    else:
        raise ValueError(f"Unsupported method: '{sel_config.method}'")

    # 5. Fit SelectKBest on training data only
    selector = SelectKBest(score_func=score_func, k=k)
    X_arr = X_train.to_numpy()
    y_arr = y_train.to_numpy()
    selector.fit(X_arr, y_arr)

    # 6. Extract and clean scores/p-values
    scores_list, pvalues_list, score_warnings = _safe_scores_pvalues(selector, n_candidate)
    warnings_list.extend(score_warnings)

    # 7. Compute rankings
    rankings = _compute_rankings(scores_list)

    # 8. Determine selected feature names (in original order, not score order)
    support_mask = selector.get_support()
    selected_names = [name for name, sel in zip(candidate_names, support_mask) if sel]

    artifact = FeatureSelectorArtifact(
        selector=selector,
        method=sel_config.method,
        mode=sel_config.mode,
        candidate_feature_names=candidate_names,
        selected_feature_names=selected_names,
        scores=scores_list,
        pvalues=pvalues_list,
        rankings=rankings,
        k_requested=k,
        k_selected=len(selected_names),
        random_state=sel_config.random_state,
        warnings=warnings_list,
        config=sel_config.to_dict(),
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    return artifact


def transform_with_selector(
    selector_artifact: FeatureSelectorArtifact,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply an already-fitted feature selector to a DataFrame (train or test).

    This is a transform-only operation. The selector's fitted state is never
    modified. Test data cannot influence the learned feature ranking.

    Args:
        selector_artifact: FeatureSelectorArtifact from fit_selector().
        X: DataFrame to reduce (must have the same columns as the training data
           passed to fit_selector()).

    Returns:
        pd.DataFrame: Reduced DataFrame containing only the selected features,
                      preserving original row index and selected column names.

    Raises:
        ValueError: If X columns do not match the fitted candidate feature names.
    """
    expected = selector_artifact.candidate_feature_names
    actual = list(X.columns)
    if actual != expected:
        raise ValueError(
            f"Column mismatch: expected {expected}, got {actual}. "
            "X must have the same columns as the training data used in fit_selector()."
        )

    X_arr = X.to_numpy()
    X_selected_arr = selector_artifact.selector.transform(X_arr)

    return pd.DataFrame(
        X_selected_arr,
        columns=selector_artifact.selected_feature_names,
        index=X.index,
    )


def build_selection_preview(artifact: FeatureSelectorArtifact) -> FeatureSelectionPreview:
    """
    Build a JSON-serializable FeatureSelectionPreview from a FeatureSelectorArtifact.

    Suitable for backend API responses and future frontend display.

    Args:
        artifact: Fitted FeatureSelectorArtifact.

    Returns:
        FeatureSelectionPreview with per-feature metadata and summary statistics.
    """
    features_meta = []
    for i, name in enumerate(artifact.candidate_feature_names):
        entry: Dict[str, Any] = {
            "name": name,
            "score": artifact.scores[i],
            "rank": artifact.rankings[i],
            "selected": name in artifact.selected_feature_names,
        }
        if artifact.pvalues is not None:
            entry["pvalue"] = artifact.pvalues[i]
        features_meta.append(entry)

    return FeatureSelectionPreview(
        method=artifact.method,
        mode=artifact.mode,
        k_requested=artifact.k_requested,
        k_selected=artifact.k_selected,
        features=features_meta,
        selected_feature_names=artifact.selected_feature_names,
        warnings=artifact.warnings,
    )


def prepare_cv_fold_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fold: CVFoldResult,
    config: Optional[Union[FeatureSelectionConfig, Dict[str, Any]]] = None,
) -> Tuple[FeatureSelectorArtifact, pd.DataFrame, pd.DataFrame]:
    """
    Fit an independent feature selector for a single CV fold and transform both slices.

    Critical leakage safeguard: the selector is fitted exclusively on the fold's
    training slice (fold.train_indices). The validation slice (fold.val_indices) is
    transformed using the already-fitted selector without any refitting.

    Args:
        X_train: Preprocessed training features (full training partition, pre-split).
        y_train: Training target labels (full training partition).
        fold: CVFoldResult with fold.train_indices and fold.val_indices.
        config: FeatureSelectionConfig or dict. Defaults to FeatureSelectionConfig().

    Returns:
        Tuple of:
          - FeatureSelectorArtifact (fitted on fold training slice only)
          - X_fold_train_selected (pd.DataFrame)
          - X_fold_val_selected (pd.DataFrame)
    """
    X_fold_train = X_train.loc[fold.train_indices]
    X_fold_val = X_train.loc[fold.val_indices]
    y_fold_train = y_train.loc[fold.train_indices]

    # Fit selector exclusively on the fold training slice
    fold_selector = fit_selector(X_fold_train, y_fold_train, config)

    X_fold_train_selected = transform_with_selector(fold_selector, X_fold_train)
    X_fold_val_selected = transform_with_selector(fold_selector, X_fold_val)

    return fold_selector, X_fold_train_selected, X_fold_val_selected
