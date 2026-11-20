"""
backend/data/correlation_filter.py
===================================
Leakage-safe Pearson correlation-based feature filtering for QML Benchmark Studio.

Removes redundant features whose absolute pairwise Pearson correlation exceeds a
configurable threshold. Correlations are computed exclusively on training data.
The resulting CorrelationFilterArtifact stores the retained column list, which is
applied to test/validation data via transform_with_filter() without recalculating.

Deterministic Removal Policy
-----------------------------
When features A and B are correlated above the threshold, the feature appearing
LATER in the established column order is marked for removal. This means:
  - Earlier features (lower column index) are always preferred.
  - Within any correlated group, only one feature is kept (the earliest).
  - Column order of retained features is never changed.
  - The policy is documented in every CorrelatedPairInfo record.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.data.schemas import (
    CorrelationFilterArtifact,
    CorrelationFilterConfig,
    CorrelatedPairInfo,
    CVFoldResult,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_correlation_filter(
    X_train: pd.DataFrame,
    config: Optional[Union[CorrelationFilterConfig, Dict[str, Any]]] = None,
) -> CorrelationFilterArtifact:
    """
    Compute pairwise Pearson correlations on training data and identify redundant features.

    Correlations are computed exclusively on X_train. No target labels are used
    (unsupervised method). The returned artifact encodes which features to retain,
    allowing leakage-free application to test/validation data.

    Args:
        X_train: Preprocessed training features (numerical recommended).
        config: CorrelationFilterConfig or dict. Defaults to threshold=0.95.

    Returns:
        CorrelationFilterArtifact with retained/removed feature names and pair metadata.

    Raises:
        ValueError: On empty input or invalid config.
    """
    cfg = CorrelationFilterConfig.from_dict(config)
    warnings_list: List[str] = []

    if X_train.empty:
        raise ValueError("X_train is empty. Cannot compute correlations.")
    if X_train.shape[1] == 0:
        raise ValueError("X_train has no columns.")

    original_names = list(X_train.columns)
    n_original = len(original_names)

    # Replace non-finite values with NaN before correlation calculation
    X_finite = X_train.copy()
    n_inf = int(np.isinf(X_finite.to_numpy(dtype=float, na_value=0.0)).sum())
    if n_inf > 0:
        warnings_list.append(
            f"{n_inf} non-finite value(s) detected and replaced with NaN before correlation computation."
        )
        X_finite = X_finite.replace([np.inf, -np.inf], np.nan)

    # Detect and warn about constant columns (correlation undefined)
    constant_cols = [c for c in original_names if X_finite[c].nunique(dropna=True) <= 1]
    if constant_cols:
        warnings_list.append(
            f"Constant or near-constant columns detected (correlation undefined): {constant_cols}. "
            f"They will be retained but produce NaN correlations with other features."
        )

    # Compute pairwise Pearson correlation matrix using training data only
    corr_matrix = X_finite.corr(method="pearson")

    # Greedy deterministic removal: iterate feature pairs in column order,
    # mark the LATER feature for removal when |corr| >= threshold.
    to_remove: set = set()
    correlated_pairs: List[CorrelatedPairInfo] = []

    for i in range(n_original):
        name_i = original_names[i]
        if name_i in to_remove:
            continue
        for j in range(i + 1, n_original):
            name_j = original_names[j]
            if name_j in to_remove:
                continue
            corr_val = corr_matrix.loc[name_i, name_j]
            if np.isnan(corr_val):
                continue  # Cannot determine correlation for constant features
            abs_corr = float(abs(corr_val))
            if abs_corr >= cfg.threshold:
                to_remove.add(name_j)
                correlated_pairs.append(
                    CorrelatedPairInfo(
                        feature_a=name_i,
                        feature_b=name_j,
                        correlation=round(float(corr_val), 8),
                        removed_feature=name_j,
                        reason=(
                            f"Absolute Pearson correlation |r|={abs_corr:.4f} >= threshold={cfg.threshold}. "
                            f"Policy: retain earlier column '{name_i}', remove later column '{name_j}'."
                        ),
                    )
                )

    retained = [n for n in original_names if n not in to_remove]
    removed = [n for n in original_names if n in to_remove]

    return CorrelationFilterArtifact(
        retained_feature_names=retained,
        removed_feature_names=removed,
        correlated_pairs=correlated_pairs,
        config=cfg.to_dict(),
        original_feature_names=original_names,
        n_original=n_original,
        n_retained=len(retained),
        n_removed=len(removed),
        warnings=warnings_list,
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def transform_with_filter(
    artifact: CorrelationFilterArtifact,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply an already-fitted correlation filter to a DataFrame.

    Selects only the columns listed in artifact.retained_feature_names.
    Correlations are NOT recalculated — this is a pure column-selection operation.

    Args:
        artifact: CorrelationFilterArtifact from fit_correlation_filter().
        X: DataFrame to filter. Must contain all retained feature columns.

    Returns:
        pd.DataFrame: Filtered DataFrame with retained columns in original order.

    Raises:
        ValueError: If retained columns are missing from X.
    """
    missing = [c for c in artifact.retained_feature_names if c not in X.columns]
    if missing:
        raise ValueError(
            f"The following retained columns are missing from X: {missing}"
        )
    return X[artifact.retained_feature_names].copy()


def prepare_cv_fold_filter(
    X_train: pd.DataFrame,
    fold: CVFoldResult,
    config: Optional[Union[CorrelationFilterConfig, Dict[str, Any]]] = None,
) -> Tuple[CorrelationFilterArtifact, pd.DataFrame, pd.DataFrame]:
    """
    Fit an independent correlation filter for a single CV fold.

    The filter is fitted exclusively on the fold's training slice.
    The validation slice is filtered using the already-fitted artifact (transform-only).

    Args:
        X_train: Full preprocessed training partition.
        fold: CVFoldResult with fold.train_indices and fold.val_indices.
        config: CorrelationFilterConfig or dict.

    Returns:
        Tuple of (CorrelationFilterArtifact, X_fold_train_filtered, X_fold_val_filtered).
    """
    X_fold_train = X_train.loc[fold.train_indices]
    X_fold_val = X_train.loc[fold.val_indices]

    fold_artifact = fit_correlation_filter(X_fold_train, config)
    X_fold_train_filtered = transform_with_filter(fold_artifact, X_fold_train)
    X_fold_val_filtered = transform_with_filter(fold_artifact, X_fold_val)

    return fold_artifact, X_fold_train_filtered, X_fold_val_filtered
