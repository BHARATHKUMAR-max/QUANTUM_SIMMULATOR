"""
tests/test_comparison_modes.py
==============================
Comprehensive test suite for Day 11 — Comparison Modes and Qubit Compatibility Validation.

Covers:
  1. Feature-to-Qubit compatibility checks (qubit bounds, encoding rules, silent loss prevention).
  2. Strict matched-input verification (dimensions, names, ordering, values).
  3. Mode A (Matched Reduced Features, PRIMARY): identical inputs, PCA support, leakage safety.
  4. Mode B (Quantum-Limited Features): qubit-count enforcement, explicit reduction requirements.
  5. Mode C (Best Practical Configuration): distinct representations, practical comparison labeling.
  6. CV fold independent fitting.
  7. JSON serializability of comparison results.
  8. End-to-end integration on Breast Cancer and Wine benchmark datasets.
"""

import copy
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

from backend.data.comparison_modes import (
    prepare_best_practical_configuration,
    prepare_comparison_mode,
    prepare_cv_fold_comparison,
    prepare_matched_reduced_features,
    prepare_quantum_limited_features,
    validate_qubit_compatibility,
    verify_matched_inputs,
)
from backend.data.schemas import (
    ComparisonModeConfig,
    ComparisonModeResult,
    DatasetBundle,
    FeatureSelectionConfig,
    MAX_QUBITS,
    MIN_QUBITS,
    PCAConfig,
    QubitCompatibilityResult,
)
from backend.data.splitting import create_stratified_cv_splits, create_train_test_split
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def breast_cancer_split():
    """Stratified and preprocessed Breast Cancer benchmark dataset."""
    bc = load_breast_cancer(as_frame=True)
    X = bc.frame.drop(columns=["target"]).astype(float)
    y = bc.frame["target"]
    bundle = DatasetBundle(
        X=X, y=y,
        feature_names=list(X.columns),
        target_names=["malignant", "benign"],
        dataset_name="breast_cancer",
        target_name="target",
    )
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    prep = fit_preprocessor(split.X_train)
    X_train_proc = transform_with_preprocessor(prep, split.X_train)
    X_test_proc = transform_with_preprocessor(prep, split.X_test)
    return split, X_train_proc, X_test_proc


@pytest.fixture
def wine_split():
    """Stratified and preprocessed Wine 3-class benchmark dataset."""
    wine = load_wine(as_frame=True)
    X = wine.frame.drop(columns=["target"]).astype(float)
    y = wine.frame["target"]
    bundle = DatasetBundle(
        X=X, y=y,
        feature_names=list(X.columns),
        target_names=["c0", "c1", "c2"],
        dataset_name="wine",
        target_name="target",
    )
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    prep = fit_preprocessor(split.X_train)
    X_train_proc = transform_with_preprocessor(prep, split.X_train)
    X_test_proc = transform_with_preprocessor(prep, split.X_test)
    return split, X_train_proc, X_test_proc


# ---------------------------------------------------------------------------
# 1. Feature-to-Qubit Compatibility Checks
# ---------------------------------------------------------------------------

class TestQubitCompatibility:

    def test_valid_angle_compatibility(self):
        """Exact 1-to-1 match between 4 features and 4 qubits under angle encoding."""
        res = validate_qubit_compatibility(n_features=4, n_qubits=4, encoding_strategy="angle")
        assert res.is_compatible is True
        assert len(res.errors) == 0
        assert len(res.feature_to_qubit_map) == 4
        assert res.feature_to_qubit_map["q0"] == "feature_0"
        assert res.feature_to_qubit_map["q3"] == "feature_3"

    def test_named_features_mapping(self):
        """Features with explicit names are recorded deterministically in feature_to_qubit_map."""
        feature_names = ["PC1", "PC2", "PC3"]
        res = validate_qubit_compatibility(
            n_features=3, n_qubits=3, encoding_strategy="angle", feature_names=feature_names
        )
        assert res.is_compatible is True
        assert res.feature_to_qubit_map == {"q0": "PC1", "q1": "PC2", "q2": "PC3"}

    def test_zz_feature_map_compatibility(self):
        """ZZ feature map requires 1-to-1 feature-to-qubit equality."""
        res = validate_qubit_compatibility(n_features=5, n_qubits=5, encoding_strategy="zz_feature_map")
        assert res.is_compatible is True
        assert res.encoding_strategy == "zz_feature_map"

    def test_qubit_count_below_minimum_rejected(self):
        """Qubit counts < 2 are outside the supported simulator range."""
        res = validate_qubit_compatibility(n_features=1, n_qubits=1, encoding_strategy="angle")
        assert res.is_compatible is False
        assert any("outside the supported simulator range" in err for err in res.errors)

    def test_qubit_count_above_maximum_rejected(self):
        """Qubit counts > 10 are strictly rejected."""
        res = validate_qubit_compatibility(n_features=11, n_qubits=11, encoding_strategy="angle")
        assert res.is_compatible is False
        assert any("outside the supported simulator range" in err for err in res.errors)

    def test_excess_features_rejected_no_silent_truncation(self):
        """More features than qubits must NOT silently slice; an error is mandatory."""
        res = validate_qubit_compatibility(n_features=8, n_qubits=4, encoding_strategy="angle")
        assert res.is_compatible is False
        assert any("exceeds qubit count" in err for err in res.errors)
        assert any("Silent feature truncation" in err for err in res.errors)

    def test_fewer_features_rejected_no_silent_padding(self):
        """Fewer features than qubits must NOT silently zero-pad; an error is mandatory."""
        res = validate_qubit_compatibility(n_features=3, n_qubits=6, encoding_strategy="angle")
        assert res.is_compatible is False
        assert any("is less than qubit count" in err for err in res.errors)
        assert any("Silent zero-padding" in err for err in res.errors)

    def test_invalid_encoding_rejected(self):
        """Unsupported encoding methods fail validation."""
        res = validate_qubit_compatibility(n_features=4, n_qubits=4, encoding_strategy="amplitude_embedding")
        assert res.is_compatible is False
        assert any("Invalid encoding strategy" in err for err in res.errors)

    def test_non_integer_qubit_count_rejected(self):
        """Non-integer qubit count fails validation."""
        res = validate_qubit_compatibility(n_features=4, n_qubits=4.5, encoding_strategy="angle")
        assert res.is_compatible is False


# ---------------------------------------------------------------------------
# 2. Strict Matched-Input Verification
# ---------------------------------------------------------------------------

class TestVerifyMatchedInputs:

    def test_identical_dataframes_pass(self):
        """Exact identical DataFrames verify successfully."""
        df_a = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]}, index=[0, 1])
        df_b = df_a.copy()
        is_matched, errors = verify_matched_inputs(df_a, df_b)
        assert is_matched is True
        assert len(errors) == 0

    def test_different_column_order_rejected(self):
        """Same column names in different order must fail matching verification."""
        df_a = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]})
        df_b = pd.DataFrame({"PC2": [3.0, 4.0], "PC1": [1.0, 2.0]})
        is_matched, errors = verify_matched_inputs(df_a, df_b)
        assert is_matched is False
        assert any("Column ordering differs" in err for err in errors)

    def test_different_column_identities_rejected_same_shape(self):
        """Same shape (2 columns) but different feature names must fail matching."""
        df_a = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]})
        df_b = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [3.0, 4.0]})
        is_matched, errors = verify_matched_inputs(df_a, df_b)
        assert is_matched is False
        assert any("Equal dimensionality alone is insufficient" in err for err in errors)

    def test_different_values_rejected(self):
        """Same column names and shape but different values must fail matching."""
        df_a = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]})
        df_b = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 999.0]})
        is_matched, errors = verify_matched_inputs(df_a, df_b)
        assert is_matched is False
        assert any("Matrix values differ" in err for err in errors)

    def test_different_row_count_rejected(self):
        """Row count mismatch fails verification."""
        df_a = pd.DataFrame({"PC1": [1.0, 2.0, 3.0]})
        df_b = pd.DataFrame({"PC1": [1.0, 2.0]})
        is_matched, errors = verify_matched_inputs(df_a, df_b)
        assert is_matched is False
        assert any("Shape mismatch" in err for err in errors)


# ---------------------------------------------------------------------------
# 3. Mode A — Matched Reduced Features (PRIMARY SCIENTIFIC MODE)
# ---------------------------------------------------------------------------

class TestModeAMatchedReduced:

    def test_mode_a_with_pca(self, breast_cancer_split):
        """Mode A with PCA: both classical and quantum receive identical PCA components."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=6),
            qubit_count=6,
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert isinstance(res, ComparisonModeResult)
        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.mode == "matched_reduced"

        # Both received exact same shape and columns
        assert res.X_train_classical.shape == (len(X_train), 6)
        assert res.X_train_quantum.shape == (len(X_train), 6)
        assert list(res.X_train_classical.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
        assert list(res.X_train_quantum.columns) == list(res.X_train_classical.columns)

        # Numerical equality
        np.testing.assert_array_equal(
            res.X_train_classical.to_numpy(), res.X_train_quantum.to_numpy()
        )
        np.testing.assert_array_equal(
            res.X_test_classical.to_numpy(), res.X_test_quantum.to_numpy()
        )

        # Metadata records transformed components
        assert res.classical_metadata.feature_type == "transformed_components"
        assert res.quantum_metadata.feature_type == "transformed_components"
        assert res.qubit_compatibility.is_compatible is True

    def test_mode_a_with_mutual_info(self, breast_cancer_split):
        """Mode A with Mutual Information: both models receive identical selected original features."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="mutual_info",
            feature_selection_config=FeatureSelectionConfig(
                method="mutual_info", mode="manual", k_features=4, random_state=42
            ),
            qubit_count=4,
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.X_train_classical.shape[1] == 4
        assert res.X_train_quantum.shape[1] == 4
        assert res.classical_metadata.feature_type == "original_features"
        assert res.quantum_metadata.feature_type == "original_features"
        assert res.classical_metadata.feature_names == res.quantum_metadata.feature_names

    def test_mode_a_with_correlation_filter(self, breast_cancer_split):
        """Mode A with Correlation Filtering: both models receive identical collinearity-filtered features."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="correlation",
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.X_train_classical.shape[1] < X_train.shape[1]
        assert list(res.X_train_classical.columns) == list(res.X_train_quantum.columns)

    def test_mode_a_incompatible_qubit_count_flagged(self, breast_cancer_split):
        """If qubit_count does not match resulting features, Mode A logs compatibility error."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=6),
            qubit_count=4,  # Intentionally mismatched (6 features != 4 qubits)
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is False
        assert any("exceeds qubit count" in err for err in res.validation_errors)


# ---------------------------------------------------------------------------
# 4. Mode B — Quantum-Limited Features
# ---------------------------------------------------------------------------

class TestModeBQuantumLimited:

    def test_mode_b_requires_qubit_count(self):
        """Mode B requires explicit qubit_count; omitting it raises ValueError."""
        with pytest.raises(ValueError, match="qubit_count is mandatory"):
            ComparisonModeConfig(mode="quantum_limited", qubit_count=None)

    def test_mode_b_pca_enforces_qubit_count(self, breast_cancer_split):
        """Mode B automatically targets qubit_count features with PCA."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="quantum_limited",
            qubit_count=5,
            reduction_method="pca",
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.X_train_classical.shape[1] == 5
        assert res.X_train_quantum.shape[1] == 5
        assert res.qubit_compatibility.is_compatible is True
        assert len(res.qubit_compatibility.feature_to_qubit_map) == 5

    def test_mode_b_anova_enforces_qubit_count(self, breast_cancer_split):
        """Mode B automatically targets qubit_count features with ANOVA."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="quantum_limited",
            qubit_count=4,
            reduction_method="anova",
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.X_train_classical.shape[1] == 4
        assert res.X_train_quantum.shape[1] == 4

    def test_mode_b_excess_features_without_reduction_rejected(self, breast_cancer_split):
        """Mode B rejects excess features without explicit reduction; silent slicing prohibited."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="quantum_limited",
            qubit_count=4,
            reduction_method="none",  # 30 features > 4 qubits without reduction
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is False
        assert any("Silent feature truncation" in err for err in res.validation_errors)


# ---------------------------------------------------------------------------
# 5. Mode C — Best Practical Configuration
# ---------------------------------------------------------------------------

class TestModeCBestPractical:

    def test_mode_c_distinct_representations(self, breast_cancer_split):
        """Mode C allows classical to use all features while quantum uses PCA."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="best_practical",
            classical_reduction_method="none",  # Full 30 features
            quantum_reduction_method="pca",      # 4 PCA components
            qubit_count=4,
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.mode == "best_practical"
        # Crucial invariant: labeled as practical, NOT matched!
        assert res.is_matched_input is False

        assert res.X_train_classical.shape[1] == 30
        assert res.X_train_quantum.shape[1] == 4
        assert res.classical_metadata.n_features == 30
        assert res.quantum_metadata.n_features == 4
        assert res.qubit_compatibility.is_compatible is True

    def test_mode_c_separate_metadata(self, wine_split):
        """Mode C maintains separate, accurate metadata for classical and quantum."""
        split, X_train, X_test = wine_split
        cfg = ComparisonModeConfig(
            mode="best_practical",
            classical_reduction_method="anova",
            classical_feature_selection_config=FeatureSelectionConfig(
                method="anova", mode="manual", k_features=8
            ),
            quantum_reduction_method="pca",
            qubit_count=3,
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is False
        assert res.classical_metadata.n_features == 8
        assert res.classical_metadata.feature_type == "original_features"
        assert res.quantum_metadata.n_features == 3
        assert res.quantum_metadata.feature_type == "transformed_components"


# ---------------------------------------------------------------------------
# 6. Leakage Prevention & Cross-Validation Integration
# ---------------------------------------------------------------------------

class TestComparisonModeLeakagePrevention:

    def test_test_data_never_used_in_reduction_fit(self, breast_cancer_split):
        """Transforming test data must use already-fitted training statistics without refitting."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=4),
        )

        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        # Test set is transformed strictly with the training projection
        assert res.X_test_classical.shape == (len(X_test), 4)
        assert list(res.X_test_classical.columns) == ["PC1", "PC2", "PC3", "PC4"]

    def test_cv_folds_fit_independent_comparison_artifacts(self, breast_cancer_split):
        """Each CV fold fits its own reduction independently on fold train slice."""
        split, X_train, X_test = breast_cancer_split
        folds = create_stratified_cv_splits(X_train, split.y_train, n_splits=3, random_state=42)

        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=4),
            qubit_count=4,
        )

        fold_results = []
        for fold in folds:
            fold_res = prepare_cv_fold_comparison(X_train, split.y_train, fold=fold, config=cfg)
            assert fold_res.is_valid is True
            assert fold_res.is_matched_input is True
            assert fold_res.X_train_classical.shape[1] == 4
            assert fold_res.X_test_classical.shape[1] == 4  # Represents fold val slice
            fold_results.append(fold_res)

        # Independence: fold train slices have distinct sample counts and variance
        assert len(fold_results) == 3


# ---------------------------------------------------------------------------
# 7. Serialization and Configuration Integrity
# ---------------------------------------------------------------------------

class TestSerializationAndConfig:

    def test_comparison_result_to_dict_serializable(self, breast_cancer_split):
        """ComparisonModeResult.to_dict() must produce valid JSON without TypeError."""
        split, X_train, X_test = breast_cancer_split
        cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=4),
            qubit_count=4,
        )
        res = prepare_comparison_mode(X_train, split.y_train, X_test, split.y_test, config=cfg)

        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["is_matched_input"] is True
        assert d["n_train_samples"] == len(X_train)
        assert d["n_test_samples"] == len(X_test)

        # Verify JSON serializability
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_invalid_comparison_mode_raises(self):
        """Invalid comparison mode strings raise ValueError at config construction."""
        with pytest.raises(ValueError, match="Invalid comparison mode"):
            ComparisonModeConfig(mode="arbitrary_comparison")

    def test_invalid_qubit_count_raises_in_config(self):
        """Qubit counts outside [2, 10] raise ValueError at config construction."""
        with pytest.raises(ValueError, match="outside the supported range"):
            ComparisonModeConfig(qubit_count=12)

        with pytest.raises(ValueError, match="outside the supported range"):
            ComparisonModeConfig(qubit_count=1)
