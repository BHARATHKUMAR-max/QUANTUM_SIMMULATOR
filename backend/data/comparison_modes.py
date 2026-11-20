"""
backend/data/comparison_modes.py
=================================
Comparison mode orchestration and feature-to-qubit compatibility validation
for QML Benchmark Studio.

Implements the three core comparison modes:
  - Mode A (Matched Reduced Features, PRIMARY): Classical and quantum models receive
    the exact same reduced feature representation (same columns, order, values).
  - Mode B (Quantum-Limited Features): Feature count is strictly bounded by quantum
    hardware constraints (qubits). Models are benchmarked on quantum-compatible data.
  - Mode C (Best Practical Configuration): Classical and quantum models each use their
    respective best practical configurations; explicitly labeled as an unmatched comparison.

Core Guarantees:
  1. No silent feature loss (truncation like X[:, :k] and zero-padding are forbidden).
  2. Strict leakage-free execution (all reduction steps fit on training partitions only).
  3. Verifiable matched inputs (dimensional equality alone is insufficient; identities must match).
  4. Comprehensive JSON-serializable provenance metadata.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.data.schemas import (
    ComparisonModeConfig,
    ComparisonModeResult,
    CorrelatedPairInfo,
    CorrelationFilterArtifact,
    CorrelationFilterConfig,
    CVFoldResult,
    FeatureSelectionConfig,
    FeatureSelectorArtifact,
    MAX_QUBITS,
    MIN_QUBITS,
    ModelInputMetadata,
    PCAArtifact,
    PCAConfig,
    QubitCompatibilityResult,
    VALID_COMPARISON_MODES,
    VALID_ENCODING_STRATEGIES,
)
from backend.data.feature_selection import fit_selector, transform_with_selector
from backend.data.correlation_filter import fit_correlation_filter, transform_with_filter
from backend.data.dimensionality_reduction import fit_pca, transform_with_pca


# ---------------------------------------------------------------------------
# 1. Feature-to-Qubit Compatibility Validation
# ---------------------------------------------------------------------------

def validate_qubit_compatibility(
    n_features: int,
    n_qubits: int,
    encoding_strategy: str = "angle",
    feature_names: Optional[List[str]] = None,
) -> QubitCompatibilityResult:
    """
    Validate whether a candidate feature representation is compatible with quantum circuits.

    Rules:
      - Supported qubit count: [MIN_QUBITS, MAX_QUBITS] (2 to 10 qubits).
      - Primary 'angle' encoding: exactly 1 feature per qubit (n_features == n_qubits).
      - 'zz_feature_map' encoding: exactly 1 feature per qubit (n_features == n_qubits).
      - Silent feature truncation (e.g. slicing X[:, :n_qubits]) is strictly prohibited.
      - Silent zero-padding or feature duplication is strictly prohibited.

    Args:
        n_features: Number of input features to evaluate.
        n_qubits: Configured number of target qubits.
        encoding_strategy: Quantum data encoding method ('angle' or 'zz_feature_map').
        feature_names: Optional list of feature names for mapping documentation.

    Returns:
        QubitCompatibilityResult with validation status, deterministic mapping, and error details.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Standardize encoding strategy
    norm_strategy = encoding_strategy
    if norm_strategy in ("angle", "angle_embedding"):
        norm_strategy = "angle"
    elif norm_strategy in ("zz", "zz_feature_map"):
        norm_strategy = "zz_feature_map"
    else:
        errors.append(
            f"Invalid encoding strategy: '{encoding_strategy}'. Supported: {sorted(VALID_ENCODING_STRATEGIES)}"
        )
        return QubitCompatibilityResult(
            is_compatible=False,
            n_qubits=n_qubits,
            n_features=n_features,
            encoding_strategy=encoding_strategy,
            errors=errors,
            warnings=warnings,
        )

    # Validate qubit bounds
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        errors.append(f"Qubit count must be an integer, got: {type(n_qubits).__name__}.")
    elif not (MIN_QUBITS <= n_qubits <= MAX_QUBITS):
        errors.append(
            f"Qubit count {n_qubits} is outside the supported simulator range "
            f"[{MIN_QUBITS}, {MAX_QUBITS}]. PennyLane default.qubit benchmarks are constrained to 2-10 qubits."
        )

    # Validate feature count
    if not isinstance(n_features, int) or isinstance(n_features, bool):
        errors.append(f"Feature count must be an integer, got: {type(n_features).__name__}.")
    elif n_features <= 0:
        errors.append(f"Feature count must be a positive integer, got: {n_features}.")

    # Encoding-specific dimensional matching (1 feature per qubit for current MVP)
    if isinstance(n_qubits, int) and not isinstance(n_qubits, bool) and isinstance(n_features, int) and not isinstance(n_features, bool):
        if n_features > n_qubits:
            errors.append(
                f"Feature count ({n_features}) exceeds qubit count ({n_qubits}) for '{norm_strategy}' encoding. "
                "Explicit dimensionality reduction (e.g. PCA, Mutual Information, ANOVA) must be applied. "
                "Silent feature truncation (e.g. slicing X[:, :n_qubits]) is strictly prohibited."
            )
        elif n_features < n_qubits:
            errors.append(
                f"Feature count ({n_features}) is less than qubit count ({n_qubits}) for '{norm_strategy}' encoding. "
                "Silent zero-padding or feature duplication is strictly prohibited. "
                "Reduce the configured qubit count to match the feature dimensionality."
            )

    is_compatible = len(errors) == 0

    # Build deterministic feature-to-qubit mapping if compatible
    feature_to_qubit_map: Dict[str, str] = {}
    if is_compatible:
        for i in range(n_qubits):
            qubit_label = f"q{i}"
            if feature_names and i < len(feature_names):
                feature_to_qubit_map[qubit_label] = feature_names[i]
            else:
                feature_to_qubit_map[qubit_label] = f"feature_{i}"

    return QubitCompatibilityResult(
        is_compatible=is_compatible,
        n_qubits=n_qubits if isinstance(n_qubits, int) else 0,
        n_features=n_features if isinstance(n_features, int) else 0,
        encoding_strategy=norm_strategy,
        feature_to_qubit_map=feature_to_qubit_map,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 2. Strict Matched-Input Verification
# ---------------------------------------------------------------------------

def verify_matched_inputs(
    X_classical: pd.DataFrame,
    X_quantum: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """
    Verify whether classical and quantum input feature matrices are genuinely identical.

    Dimensional equality alone is insufficient; feature identities, ordering, sample
    count, index alignment, and numerical values must match completely.

    Args:
        X_classical: Feature DataFrame presented to classical models.
        X_quantum: Feature DataFrame presented to quantum circuits.

    Returns:
        Tuple of (is_matched, list_of_discrepancy_reasons).
    """
    errors: List[str] = []

    # 1. Dimensionality check
    if X_classical.shape != X_quantum.shape:
        errors.append(
            f"Shape mismatch: classical has shape {X_classical.shape}, "
            f"quantum has shape {X_quantum.shape}."
        )
        return False, errors

    # 2. Column names and order
    classical_cols = list(X_classical.columns)
    quantum_cols = list(X_quantum.columns)

    if classical_cols != quantum_cols:
        if set(classical_cols) == set(quantum_cols):
            errors.append(
                f"Column ordering differs: classical={classical_cols}, quantum={quantum_cols}. "
                "Strict matching requires identical column order."
            )
        else:
            errors.append(
                f"Feature identities differ: classical columns={classical_cols}, quantum columns={quantum_cols}. "
                "Equal dimensionality alone is insufficient for matched-input verification."
            )

    # 3. Row index alignment
    if list(X_classical.index) != list(X_quantum.index):
        errors.append("Sample row indices do not match between classical and quantum inputs.")

    # 4. Numerical value equivalence
    if not errors:
        arr_c = X_classical.to_numpy(dtype=float, na_value=np.nan)
        arr_q = X_quantum.to_numpy(dtype=float, na_value=np.nan)
        if not np.allclose(arr_c, arr_q, atol=1e-9, equal_nan=True):
            errors.append("Matrix values differ between classical and quantum inputs.")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# 3. Internal Modular Reduction Engine
# ---------------------------------------------------------------------------

def _apply_reduction_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    reduction_method: Optional[str],
    feature_selection_config: Optional[Union[Dict[str, Any], FeatureSelectionConfig]] = None,
    correlation_filter_config: Optional[Union[Dict[str, Any], CorrelationFilterConfig]] = None,
    pca_config: Optional[Union[Dict[str, Any], PCAConfig]] = None,
    target_k: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str, Dict[str, Any]]:
    """
    Apply a single modular reduction transformation fitted exclusively on X_train.

    Returns:
        (X_train_reduced, X_test_reduced, transformations_applied, feature_type, details)
    """
    if reduction_method is None or reduction_method == "none":
        return (
            X_train.copy(),
            X_test.copy(),
            [],
            "original_features",
            {"reduction_method": "none", "n_features": X_train.shape[1]},
        )

    norm_method = reduction_method.lower()

    if norm_method in ("mutual_info", "anova"):
        # Resolve config
        fs_cfg = FeatureSelectionConfig.from_dict(feature_selection_config)
        # If target_k is supplied (Mode B), enforce it strictly
        if target_k is not None:
            fs_cfg = FeatureSelectionConfig(
                method=norm_method,
                mode="manual",
                k_features=target_k,
                random_state=fs_cfg.random_state,
            )
        elif fs_cfg.method != norm_method:
            fs_cfg.method = norm_method

        selector = fit_selector(X_train, y_train, config=fs_cfg)
        X_tr = transform_with_selector(selector, X_train)
        X_te = transform_with_selector(selector, X_test)

        details = {
            "method": selector.method,
            "k_requested": selector.k_requested,
            "k_selected": selector.k_selected,
            "selected_feature_names": selector.selected_feature_names,
            "scores": selector.scores,
            "warnings": selector.warnings,
        }
        return X_tr, X_te, [f"feature_selection_{norm_method}"], "original_features", details

    elif norm_method == "correlation":
        cf_cfg = CorrelationFilterConfig.from_dict(correlation_filter_config)
        filter_art = fit_correlation_filter(X_train, config=cf_cfg)
        X_tr = transform_with_filter(filter_art, X_train)
        X_te = transform_with_filter(filter_art, X_test)

        details = {
            "method": "correlation_filter",
            "threshold": cf_cfg.threshold,
            "n_retained": filter_art.n_retained,
            "n_removed": filter_art.n_removed,
            "removed_features": filter_art.removed_feature_names,
            "warnings": filter_art.warnings,
        }
        return X_tr, X_te, ["correlation_filter"], "original_features", details

    elif norm_method == "pca":
        # Resolve config
        if target_k is not None:
            p_cfg = PCAConfig(n_components=target_k)
        elif pca_config is not None:
            p_cfg = PCAConfig.from_dict(pca_config)
        else:
            p_cfg = PCAConfig(variance_target=0.95)

        pca_art = fit_pca(X_train, config=p_cfg)
        X_tr = transform_with_pca(pca_art, X_train)
        X_te = transform_with_pca(pca_art, X_test)

        details = {
            "method": "pca",
            "n_components": pca_art.n_components_actual,
            "component_names": pca_art.component_names,
            "total_variance_retained": pca_art.total_variance_retained,
            "explained_variance_ratio": pca_art.explained_variance_ratio,
            "warnings": pca_art.warnings,
        }
        return X_tr, X_te, ["pca"], "transformed_components", details

    else:
        raise ValueError(
            f"Unsupported reduction method: '{reduction_method}'. "
            "Supported: 'mutual_info', 'anova', 'correlation', 'pca', 'none'."
        )


# ---------------------------------------------------------------------------
# 4. Mode A — Matched Reduced Features (PRIMARY SCIENTIFIC MODE)
# ---------------------------------------------------------------------------

def prepare_matched_reduced_features(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
) -> ComparisonModeResult:
    """
    Prepare data under Mode A (Matched Reduced Features).

    Both classical and quantum models receive the EXACT same reduced feature
    representation. Transformations are fitted exclusively on training data.
    If quantum qubit compatibility is evaluated and fails, validation errors are logged.

    Args:
        X_train_processed: Preprocessed training features.
        y_train: Training target labels.
        X_test_processed: Preprocessed test features.
        y_test: Test target labels.
        config: ComparisonModeConfig for Mode A.

    Returns:
        ComparisonModeResult with is_matched_input=True.
    """
    cfg = ComparisonModeConfig.from_dict(config)
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    # 1. Apply single shared reduction step
    X_tr_red, X_te_red, transforms, feat_type, details = _apply_reduction_step(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        reduction_method=cfg.reduction_method,
        feature_selection_config=cfg.feature_selection_config,
        correlation_filter_config=cfg.correlation_filter_config,
        pca_config=cfg.pca_config,
    )

    # 2. Both model families receive identical DataFrames
    X_tr_c = X_tr_red
    X_te_c = X_te_red
    X_tr_q = X_tr_red.copy()
    X_te_q = X_te_red.copy()

    # 3. Verify matched input invariance
    is_matched, match_errors = verify_matched_inputs(X_tr_c, X_tr_q)
    if not is_matched:
        validation_errors.extend(match_errors)

    # 4. Qubit compatibility check (if qubit_count is specified)
    qubit_compat: Optional[QubitCompatibilityResult] = None
    if cfg.qubit_count is not None:
        qubit_compat = validate_qubit_compatibility(
            n_features=X_tr_q.shape[1],
            n_qubits=cfg.qubit_count,
            encoding_strategy=cfg.encoding_strategy,
            feature_names=list(X_tr_q.columns),
        )
        if not qubit_compat.is_compatible:
            validation_errors.extend(qubit_compat.errors)
        validation_warnings.extend(qubit_compat.warnings)

    meta_c = ModelInputMetadata(
        model_family="classical",
        feature_names=list(X_tr_c.columns),
        n_features=X_tr_c.shape[1],
        feature_type=feat_type,
        transformations_applied=transforms,
        reduction_details=details,
    )
    meta_q = ModelInputMetadata(
        model_family="quantum",
        feature_names=list(X_tr_q.columns),
        n_features=X_tr_q.shape[1],
        feature_type=feat_type,
        transformations_applied=transforms,
        reduction_details=details,
    )

    return ComparisonModeResult(
        mode="matched_reduced",
        is_matched_input=is_matched and len(validation_errors) == 0,
        X_train_classical=X_tr_c,
        X_test_classical=X_te_c,
        X_train_quantum=X_tr_q,
        X_test_quantum=X_te_q,
        y_train=y_train,
        y_test=y_test,
        classical_metadata=meta_c,
        quantum_metadata=meta_q,
        qubit_compatibility=qubit_compat,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 5. Mode B — Quantum-Limited Features
# ---------------------------------------------------------------------------

def prepare_quantum_limited_features(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
    config: Union[Dict[str, Any], ComparisonModeConfig],
) -> ComparisonModeResult:
    """
    Prepare data under Mode B (Quantum-Limited Features).

    Both model families are restricted to a feature representation that matches
    the quantum hardware constraints (qubit_count). If original feature count exceeds
    qubit_count and no reduction is configured, a strict validation error is returned.
    Silent feature truncation or padding is prohibited.

    Args:
        X_train_processed: Preprocessed training features.
        y_train: Training target labels.
        X_test_processed: Preprocessed test features.
        y_test: Test target labels.
        config: ComparisonModeConfig for Mode B (qubit_count is mandatory).

    Returns:
        ComparisonModeResult with is_matched_input=True.
    """
    cfg = ComparisonModeConfig.from_dict(config)
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    if cfg.qubit_count is None:
        raise ValueError("qubit_count must be explicitly configured for 'quantum_limited' mode.")

    k = cfg.qubit_count
    n_original = X_train_processed.shape[1]

    # Verify requirement for reduction
    if n_original > k and (cfg.reduction_method is None or cfg.reduction_method == "none"):
        validation_errors.append(
            f"Original feature count ({n_original}) exceeds configured qubit_count ({k}). "
            "In Mode B (Quantum-Limited), explicit feature reduction (MI, ANOVA, PCA) is required. "
            "Silent feature truncation (e.g. slicing X[:, :n_qubits]) is strictly prohibited."
        )
        # Empty mock result indicating failure
        empty_df = pd.DataFrame()
        dummy_meta = ModelInputMetadata(
            model_family="unassigned",
            feature_names=[],
            n_features=0,
            feature_type="none",
        )
        return ComparisonModeResult(
            mode="quantum_limited",
            is_matched_input=False,
            X_train_classical=empty_df,
            X_test_classical=empty_df,
            X_train_quantum=empty_df,
            X_test_quantum=empty_df,
            y_train=y_train,
            y_test=y_test,
            classical_metadata=dummy_meta,
            quantum_metadata=dummy_meta,
            validation_errors=validation_errors,
            fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    # Apply reduction targeting exactly k features
    target_k = k if n_original != k else None
    X_tr_red, X_te_red, transforms, feat_type, details = _apply_reduction_step(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        reduction_method=cfg.reduction_method,
        feature_selection_config=cfg.feature_selection_config,
        correlation_filter_config=cfg.correlation_filter_config,
        pca_config=cfg.pca_config,
        target_k=target_k,
    )

    # Validate resulting feature count matches qubit_count
    qubit_compat = validate_qubit_compatibility(
        n_features=X_tr_red.shape[1],
        n_qubits=k,
        encoding_strategy=cfg.encoding_strategy,
        feature_names=list(X_tr_red.columns),
    )
    if not qubit_compat.is_compatible:
        validation_errors.extend(qubit_compat.errors)
    validation_warnings.extend(qubit_compat.warnings)

    X_tr_c = X_tr_red
    X_te_c = X_te_red
    X_tr_q = X_tr_red.copy()
    X_te_q = X_te_red.copy()

    is_matched, match_errors = verify_matched_inputs(X_tr_c, X_tr_q)
    if not is_matched:
        validation_errors.extend(match_errors)

    meta_c = ModelInputMetadata(
        model_family="classical",
        feature_names=list(X_tr_c.columns),
        n_features=X_tr_c.shape[1],
        feature_type=feat_type,
        transformations_applied=transforms,
        reduction_details=details,
    )
    meta_q = ModelInputMetadata(
        model_family="quantum",
        feature_names=list(X_tr_q.columns),
        n_features=X_tr_q.shape[1],
        feature_type=feat_type,
        transformations_applied=transforms,
        reduction_details=details,
    )

    return ComparisonModeResult(
        mode="quantum_limited",
        is_matched_input=is_matched and len(validation_errors) == 0,
        X_train_classical=X_tr_c,
        X_test_classical=X_te_c,
        X_train_quantum=X_tr_q,
        X_test_quantum=X_te_q,
        y_train=y_train,
        y_test=y_test,
        classical_metadata=meta_c,
        quantum_metadata=meta_q,
        qubit_compatibility=qubit_compat,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 6. Mode C — Best Practical Configuration
# ---------------------------------------------------------------------------

def prepare_best_practical_configuration(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
) -> ComparisonModeResult:
    """
    Prepare data under Mode C (Best Practical Configuration).

    Allows classical models to use an optimal representation (e.g. all features or
    unconstrained feature selection) while quantum circuits use a quantum-compatible
    representation (e.g. PCA to 4 components).

    Clearly recorded as an unmatched practical comparison (is_matched_input=False).

    Args:
        X_train_processed: Preprocessed training features.
        y_train: Training target labels.
        X_test_processed: Preprocessed test features.
        y_test: Test target labels.
        config: ComparisonModeConfig specifying classical and quantum overrides.

    Returns:
        ComparisonModeResult with separate classical and quantum inputs.
    """
    cfg = ComparisonModeConfig.from_dict(config)
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    # 1. Classical reduction pipeline (defaults to 'none' = all features for best practical classical)
    c_method = cfg.classical_reduction_method or cfg.reduction_method or "none"
    X_tr_c, X_te_c, c_transforms, c_type, c_details = _apply_reduction_step(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        reduction_method=c_method,
        feature_selection_config=cfg.classical_feature_selection_config or cfg.feature_selection_config,
        correlation_filter_config=cfg.classical_correlation_filter_config or cfg.correlation_filter_config,
        pca_config=cfg.classical_pca_config or cfg.pca_config,
    )

    # 2. Quantum reduction pipeline (targets qubit_count if specified)
    q_method = cfg.quantum_reduction_method or cfg.reduction_method or "pca"
    q_target_k = cfg.qubit_count
    X_tr_q, X_te_q, q_transforms, q_type, q_details = _apply_reduction_step(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        reduction_method=q_method,
        feature_selection_config=cfg.quantum_feature_selection_config or cfg.feature_selection_config,
        correlation_filter_config=cfg.quantum_correlation_filter_config or cfg.correlation_filter_config,
        pca_config=cfg.quantum_pca_config or cfg.pca_config,
        target_k=q_target_k,
    )

    # 3. Check quantum compatibility
    qubit_compat: Optional[QubitCompatibilityResult] = None
    if cfg.qubit_count is not None:
        qubit_compat = validate_qubit_compatibility(
            n_features=X_tr_q.shape[1],
            n_qubits=cfg.qubit_count,
            encoding_strategy=cfg.encoding_strategy,
            feature_names=list(X_tr_q.columns),
        )
        if not qubit_compat.is_compatible:
            validation_errors.extend(qubit_compat.errors)
        validation_warnings.extend(qubit_compat.warnings)

    # 4. In Mode C, check if inputs match (usually False, explicitly labeled as practical)
    is_matched, _ = verify_matched_inputs(X_tr_c, X_tr_q)

    meta_c = ModelInputMetadata(
        model_family="classical",
        feature_names=list(X_tr_c.columns),
        n_features=X_tr_c.shape[1],
        feature_type=c_type,
        transformations_applied=c_transforms,
        reduction_details=c_details,
    )
    meta_q = ModelInputMetadata(
        model_family="quantum",
        feature_names=list(X_tr_q.columns),
        n_features=X_tr_q.shape[1],
        feature_type=q_type,
        transformations_applied=q_transforms,
        reduction_details=q_details,
    )

    return ComparisonModeResult(
        mode="best_practical",
        is_matched_input=is_matched and len(validation_errors) == 0,
        X_train_classical=X_tr_c,
        X_test_classical=X_te_c,
        X_train_quantum=X_tr_q,
        X_test_quantum=X_te_q,
        y_train=y_train,
        y_test=y_test,
        classical_metadata=meta_c,
        quantum_metadata=meta_q,
        qubit_compatibility=qubit_compat,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 7. Unified Dispatcher & Cross-Validation Integration
# ---------------------------------------------------------------------------

def prepare_comparison_mode(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
) -> ComparisonModeResult:
    """
    High-level entry point: prepare dataset inputs under the requested comparison mode.

    Args:
        X_train_processed: Preprocessed training features.
        y_train: Training target labels.
        X_test_processed: Preprocessed test features.
        y_test: Test target labels.
        config: ComparisonModeConfig or dict specifying mode and reduction parameters.

    Returns:
        ComparisonModeResult with segregated classical and quantum DataFrames.
    """
    cfg = ComparisonModeConfig.from_dict(config)

    if cfg.mode == "matched_reduced":
        return prepare_matched_reduced_features(
            X_train_processed, y_train, X_test_processed, y_test, config=cfg
        )
    elif cfg.mode == "quantum_limited":
        return prepare_quantum_limited_features(
            X_train_processed, y_train, X_test_processed, y_test, config=cfg
        )
    elif cfg.mode == "best_practical":
        return prepare_best_practical_configuration(
            X_train_processed, y_train, X_test_processed, y_test, config=cfg
        )
    else:
        raise ValueError(
            f"Unsupported comparison mode: '{cfg.mode}'. Supported: {sorted(VALID_COMPARISON_MODES)}"
        )


def prepare_cv_fold_comparison(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    fold: CVFoldResult,
    config: Optional[Union[Dict[str, Any], ComparisonModeConfig]] = None,
) -> ComparisonModeResult:
    """
    Prepare a single cross-validation fold under the specified comparison mode.

    Guarantees strict fold-local fitting: all transformations are fitted exclusively
    on the fold's training slice (fold.train_indices) and applied to the validation
    slice (fold.val_indices) with zero leakage.

    Args:
        X_train_processed: Full training partition features.
        y_train: Full training partition target labels.
        fold: CVFoldResult containing train_indices and val_indices.
        config: ComparisonModeConfig.

    Returns:
        ComparisonModeResult containing fold training and validation DataFrames.
    """
    X_fold_train = X_train_processed.loc[fold.train_indices]
    y_fold_train = y_train.loc[fold.train_indices]
    X_fold_val = X_train_processed.loc[fold.val_indices]
    y_fold_val = y_train.loc[fold.val_indices]

    return prepare_comparison_mode(
        X_train_processed=X_fold_train,
        y_train=y_fold_train,
        X_test_processed=X_fold_val,
        y_test=y_fold_val,
        config=config,
    )
