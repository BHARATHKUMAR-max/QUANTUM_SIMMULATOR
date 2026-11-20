"""
backend/data/dimensionality_reduction.py
=========================================
Leakage-safe PCA dimensionality reduction for QML Benchmark Studio.

PCA is fitted exclusively on training data. Test and validation data are transformed
using the already-fitted PCA instance. Two configuration modes are supported:

  Mode A — Explicit component count:
    PCAConfig(n_components=10)

  Mode B — Explained variance target:
    PCAConfig(variance_target=0.95)

Output components are named PC1, PC2, ... PCn.

IMPORTANT: PCA components are linear combinations of the original features —
they are NOT selected original features. Component names (PC1, PC2, ...) make
this distinction explicit. The original_feature_names field records the input
space for auditability.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from backend.data.schemas import (
    CVFoldResult,
    PCAConfig,
    PCAArtifact,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pca(
    X_train: pd.DataFrame,
    config: Union[PCAConfig, Dict[str, Any]],
) -> PCAArtifact:
    """
    Fit PCA exclusively on training data.

    Mode A (n_components): fits PCA with the specified number of components.
    Mode B (variance_target): fits a full PCA then selects the smallest number
      of components that reach the cumulative explained variance target.

    Args:
        X_train: Preprocessed training features (numerical; output of preprocessing pipeline).
                 Must contain only finite values — apply imputation and scaling first.
        config: PCAConfig or dict with 'n_components' or 'variance_target'.

    Returns:
        PCAArtifact containing the fitted PCA, component names, and variance metadata.

    Raises:
        ValueError: On invalid config, non-finite inputs, or insufficient dimensions.
    """
    cfg = PCAConfig.from_dict(config)
    warnings_list: List[str] = []

    if X_train.empty:
        raise ValueError("X_train is empty. Cannot fit PCA.")

    n_samples, n_features = X_train.shape
    if n_features == 0:
        raise ValueError("X_train has no features.")

    # Check for non-finite values
    X_arr = X_train.to_numpy(dtype=float)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError(
            "X_train contains non-finite values (NaN or Inf). "
            "Apply imputation and scaling before PCA."
        )

    max_components = min(n_samples, n_features)
    original_feature_names = list(X_train.columns)

    if cfg.n_components is not None:
        # Mode A: Explicit component count
        if cfg.n_components > max_components:
            raise ValueError(
                f"n_components={cfg.n_components} exceeds max allowed "
                f"({max_components} = min(n_samples={n_samples}, n_features={n_features}))."
            )
        pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state, whiten=cfg.whiten)
        pca.fit(X_arr)
        n_actual = cfg.n_components

    else:
        # Mode B: Explained variance target
        full_pca = PCA(n_components=max_components, random_state=cfg.random_state, whiten=cfg.whiten)
        full_pca.fit(X_arr)
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        candidates = np.where(cumvar >= cfg.variance_target)[0]
        if len(candidates) == 0:
            n_actual = max_components
            warnings_list.append(
                f"variance_target={cfg.variance_target} could not be reached with {max_components} components. "
                f"Using all {max_components} components (cumulative variance = {float(cumvar[-1]):.4f})."
            )
        else:
            n_actual = int(candidates[0]) + 1

        # Refit cleanly with the resolved n_actual
        pca = PCA(n_components=n_actual, random_state=cfg.random_state, whiten=cfg.whiten)
        pca.fit(X_arr)

    evr = [float(v) for v in pca.explained_variance_ratio_]
    cumulative_evr = [float(v) for v in np.cumsum(evr)]
    total_retained = float(sum(evr))
    component_names = [f"PC{i + 1}" for i in range(n_actual)]

    return PCAArtifact(
        pca=pca,
        original_feature_names=original_feature_names,
        component_names=component_names,
        n_components_actual=n_actual,
        explained_variance_ratio=evr,
        cumulative_explained_variance=cumulative_evr,
        total_variance_retained=total_retained,
        config=cfg.to_dict(),
        warnings=warnings_list,
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def transform_with_pca(
    artifact: PCAArtifact,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project data into the already-fitted PCA component space.

    This is a transform-only operation. The PCA is never refitted on X.

    Args:
        artifact: PCAArtifact from fit_pca().
        X: DataFrame to transform. Must have the same columns as training data.

    Returns:
        pd.DataFrame: Projected data with columns named PC1, PC2, ..., preserving row index.

    Raises:
        ValueError: If X columns do not match the expected feature names.
    """
    expected = artifact.original_feature_names
    actual = list(X.columns)
    if actual != expected:
        raise ValueError(
            f"Column mismatch: expected {expected}, got {actual}. "
            "X must have the same columns as the training data used in fit_pca()."
        )

    X_arr = X.to_numpy(dtype=float)
    X_pca = artifact.pca.transform(X_arr)

    return pd.DataFrame(
        X_pca,
        columns=artifact.component_names,
        index=X.index,
    )


def prepare_cv_fold_pca(
    X_train: pd.DataFrame,
    fold: CVFoldResult,
    config: Union[PCAConfig, Dict[str, Any]],
) -> Tuple[PCAArtifact, pd.DataFrame, pd.DataFrame]:
    """
    Fit an independent PCA for a single CV fold and transform both slices.

    The PCA is fitted exclusively on the fold's training slice.
    The validation slice is projected using the already-fitted PCA (transform-only).

    Args:
        X_train: Full preprocessed training partition.
        fold: CVFoldResult with fold.train_indices and fold.val_indices.
        config: PCAConfig or dict.

    Returns:
        Tuple of (PCAArtifact, X_fold_train_pca, X_fold_val_pca).
    """
    X_fold_train = X_train.loc[fold.train_indices]
    X_fold_val = X_train.loc[fold.val_indices]

    fold_pca_artifact = fit_pca(X_fold_train, config)
    X_fold_train_pca = transform_with_pca(fold_pca_artifact, X_fold_train)
    X_fold_val_pca = transform_with_pca(fold_pca_artifact, X_fold_val)

    return fold_pca_artifact, X_fold_train_pca, X_fold_val_pca
