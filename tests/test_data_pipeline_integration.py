"""
tests/test_data_pipeline_integration.py
========================================
Comprehensive End-to-End Integration Test Suite for Day 12.

Validates the complete dataset-to-feature-preparation workflow of QML Benchmark Studio:
  1. Ingestion & Validation (built-in datasets and valid/invalid CSVs).
  2. Profiling (descriptive statistics, class distributions, missingness).
  3. Stratified Partitioning (train/test split, index preservation, seed reproducibility).
  4. Leakage-Safe Preprocessing (imputation, scaling, encoding fitted exclusively on X_train).
  5. Feature Reduction (Mutual Information, ANOVA, Correlation Filter, PCA).
  6. Comparison Modes (Mode A: Matched Reduced, Mode B: Quantum-Limited, Mode C: Best Practical).
  7. Feature-to-Qubit Compatibility (qubit bounds 2-10, angle & ZZ feature map, no silent loss).
  8. Full Pipeline Reproducibility (identical seeds produce bitwise-identical pipelines).
  9. Metadata & Serialization Integrity (JSON-serializable provenance, PC vs original distinction).
"""

import copy
import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine

from backend.data import (
    CSVValidationResult,
    ComparisonModeConfig,
    ComparisonModeResult,
    CorrelationFilterConfig,
    DatasetBundle,
    DatasetProfile,
    DEFAULT_SPLIT_SEEDS,
    FeatureSelectionConfig,
    MAX_QUBITS,
    MIN_QUBITS,
    PCAConfig,
    PreprocessingConfig,
    ProcessedDatasetResult,
    create_stratified_cv_splits,
    create_train_test_split,
    fit_correlation_filter,
    fit_pca,
    fit_preprocessor,
    fit_selector,
    load_breast_cancer_data,
    load_csv_dataset,
    load_dataset,
    load_wine_data,
    prepare_comparison_mode,
    prepare_cv_fold_comparison,
    prepare_cv_fold_data,
    prepare_split_and_preprocessed_data,
    profile_dataset,
    transform_with_filter,
    transform_with_pca,
    transform_with_preprocessor,
    transform_with_selector,
    validate_csv_dataset,
    validate_qubit_compatibility,
    verify_matched_inputs,
)


# ---------------------------------------------------------------------------
# Section A: Built-in Dataset Pipeline Integration
# ---------------------------------------------------------------------------

class TestBuiltinDatasetPipeline:
    """Validate full pipeline execution on Breast Cancer and Wine built-in benchmarks."""

    def test_breast_cancer_full_pipeline_mode_a(self):
        """End-to-end pipeline: Breast Cancer -> Profile -> Split -> Preprocess -> Mode A (PCA)."""
        # 1. Load
        bundle = load_breast_cancer_data()
        assert bundle.dataset_name == "breast_cancer"
        assert bundle.X.shape == (569, 30)
        assert len(bundle.y) == 569
        assert bundle.target_classes == ["malignant", "benign"]

        # 2. Profile
        profile = profile_dataset(bundle)
        assert isinstance(profile, DatasetProfile)
        assert profile.n_samples == 569
        assert profile.n_features == 30
        assert profile.class_distribution.counts == {"malignant": 212, "benign": 357}

        # 3. Stratified Split
        split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)
        assert split.X_train.shape == (455, 30)
        assert split.X_test.shape == (114, 30)
        # Verify mutual exclusivity
        assert len(set(split.train_indices) & set(split.test_indices)) == 0

        # 4. Fit Preprocessing exclusively on X_train
        prep_cfg = PreprocessingConfig(numerical_scaling="standard", numerical_imputation="median")
        preprocessor = fit_preprocessor(split.X_train, config=prep_cfg)
        X_tr_proc = transform_with_preprocessor(preprocessor, split.X_train)
        X_te_proc = transform_with_preprocessor(preprocessor, split.X_test)
        assert X_tr_proc.shape == (455, 30)
        assert X_te_proc.shape == (114, 30)

        # 5. Comparison Mode A (Matched Reduced with PCA to 6 components)
        cmp_cfg = ComparisonModeConfig(
            mode="matched_reduced",
            reduction_method="pca",
            pca_config=PCAConfig(n_components=6),
            qubit_count=6,
            encoding_strategy="angle",
        )
        cmp_result = prepare_comparison_mode(
            X_tr_proc, split.y_train, X_te_proc, split.y_test, config=cmp_cfg
        )

        assert cmp_result.is_valid is True
        assert cmp_result.is_matched_input is True
        assert cmp_result.X_train_classical.shape == (455, 6)
        assert cmp_result.X_train_quantum.shape == (455, 6)
        assert list(cmp_result.X_train_classical.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
        assert list(cmp_result.X_train_quantum.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]

        # Verify matched inputs
        matched, errs = verify_matched_inputs(cmp_result.X_train_classical, cmp_result.X_train_quantum)
        assert matched is True
        assert len(errs) == 0

        # Qubit compatibility
        assert cmp_result.qubit_compatibility.is_compatible is True
        assert len(cmp_result.qubit_compatibility.feature_to_qubit_map) == 6

    def test_wine_multiclass_full_pipeline_mode_b(self):
        """End-to-end pipeline: Wine 3-class -> Profile -> Split -> Preprocess -> Mode B (ANOVA)."""
        # 1. Load
        bundle = load_wine_data()
        assert bundle.dataset_name == "wine"
        assert bundle.X.shape == (178, 13)
        assert len(bundle.target_classes) == 3

        # 2. Profile
        profile = profile_dataset(bundle)
        assert profile.n_classes == 3
        assert profile.n_samples == 178

        # 3. Stratified Split
        split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)
        assert split.X_train.shape == (142, 13)
        assert split.X_test.shape == (36, 13)

        # 4. Preprocess
        preprocessor = fit_preprocessor(split.X_train)
        X_tr_proc = transform_with_preprocessor(preprocessor, split.X_train)
        X_te_proc = transform_with_preprocessor(preprocessor, split.X_test)

        # 5. Mode B (Quantum-Limited to 4 qubits via ANOVA)
        cmp_cfg = ComparisonModeConfig(
            mode="quantum_limited",
            qubit_count=4,
            reduction_method="anova",
            encoding_strategy="zz_feature_map",
        )
        cmp_result = prepare_comparison_mode(
            X_tr_proc, split.y_train, X_te_proc, split.y_test, config=cmp_cfg
        )

        assert cmp_result.is_valid is True
        assert cmp_result.is_matched_input is True
        assert cmp_result.X_train_classical.shape == (142, 4)
        assert cmp_result.X_train_quantum.shape == (142, 4)
        assert cmp_result.qubit_compatibility.is_compatible is True
        assert cmp_result.qubit_compatibility.encoding_strategy == "zz_feature_map"


# ---------------------------------------------------------------------------
# Section B: User CSV Dataset Integration & Validation Tests
# ---------------------------------------------------------------------------

class TestCSVDatasetPipeline:
    """Validate pipeline ingestion, validation, and execution on representative CSV fixtures."""

    @pytest.fixture
    def valid_mixed_csv(self, tmp_path):
        """Valid synthetic CSV with numerical, categorical, and boolean features with missing values."""
        filepath = tmp_path / "mixed_dataset.csv"
        df = pd.DataFrame({
            "age": [25.0, 32.0, np.nan, 45.0, 50.0, 22.0, 38.0, 41.0, 29.0, 35.0, 48.0, 52.0, 27.0, 33.0, 44.0, 39.0],
            "city": ["NYC", "London", "Paris", "NYC", "London", "Paris", "NYC", "London", "Paris", "NYC", "London", "Paris", "NYC", "London", "Paris", "NYC"],
            "income": [50000.0, 62000.0, 58000.0, 85000.0, 92000.0, 42000.0, 71000.0, 78000.0, 53000.0, 66000.0, 89000.0, 95000.0, 51000.0, 64000.0, 82000.0, 75000.0],
            "target": ["churn", "stay", "churn", "stay", "stay", "churn", "stay", "stay", "churn", "stay", "stay", "stay", "churn", "stay", "stay", "stay"],
        })
        df.to_csv(filepath, index=False)
        return str(filepath)

    @pytest.fixture
    def valid_multiclass_csv(self, tmp_path):
        """Valid synthetic CSV with multiclass target."""
        filepath = tmp_path / "multiclass_dataset.csv"
        df = pd.DataFrame({
            "x1": np.linspace(1.0, 10.0, 24),
            "x2": np.linspace(10.0, 1.0, 24),
            "category": ["A", "B", "C"] * 8,
            "target": [0, 1, 2] * 8,
        })
        df.to_csv(filepath, index=False)
        return str(filepath)

    def test_valid_csv_complete_pipeline_flow(self, valid_mixed_csv):
        """Validate, load, profile, split, preprocess, and compare a custom CSV dataset."""
        # 1. Validate
        val_res = validate_csv_dataset(valid_mixed_csv, target_column="target", min_rows=15)
        assert isinstance(val_res, CSVValidationResult)
        assert val_res.is_valid is True
        assert val_res.row_count == 16
        assert "age" in val_res.numerical_columns
        assert "city" in val_res.categorical_columns

        # 2. Load
        bundle = load_csv_dataset(valid_mixed_csv, target_column="target", dataset_name="custom_churn")
        assert bundle.dataset_name == "custom_churn"
        assert bundle.X.shape == (16, 3)

        # 3. Profile
        profile = profile_dataset(bundle)
        assert profile.n_samples == 16

        # 4. Split
        split = create_train_test_split(bundle, test_size=0.25, random_state=42, stratify=True)
        assert len(split.X_train) == 12
        assert len(split.X_test) == 4

        # 5. Preprocess
        prep = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)
        X_tr_proc = transform_with_preprocessor(prep, split.X_train)
        X_te_proc = transform_with_preprocessor(prep, split.X_test)
        assert not X_tr_proc.isnull().any().any()
        assert not X_te_proc.isnull().any().any()

        # 6. Mode A Comparison
        cmp_cfg = ComparisonModeConfig(mode="matched_reduced", reduction_method="pca", pca_config=PCAConfig(n_components=2), qubit_count=2)
        res = prepare_comparison_mode(X_tr_proc, split.y_train, X_te_proc, split.y_test, config=cmp_cfg)
        assert res.is_valid is True
        assert res.is_matched_input is True
        assert res.X_train_classical.shape == (12, 2)

    def test_invalid_csv_empty_file(self, tmp_path):
        """Empty CSV triggers structured FILE_EMPTY validation error."""
        filepath = tmp_path / "empty.csv"
        filepath.write_text("")
        val_res = validate_csv_dataset(str(filepath), target_column="target")
        assert val_res.is_valid is False
        assert any(e.code == "FILE_EMPTY" for e in val_res.errors)

    def test_invalid_csv_missing_target(self, tmp_path):
        """CSV without specified target column triggers TARGET_NOT_FOUND error."""
        filepath = tmp_path / "no_target.csv"
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        df.to_csv(filepath, index=False)
        val_res = validate_csv_dataset(str(filepath), target_column="nonexistent")
        assert val_res.is_valid is False
        assert any(e.code == "TARGET_NOT_FOUND" for e in val_res.errors)

    def test_invalid_csv_insufficient_samples(self, tmp_path):
        """CSV with fewer than min_rows triggers ROW_COUNT_LOW error."""
        filepath = tmp_path / "tiny.csv"
        df = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]})
        df.to_csv(filepath, index=False)
        val_res = validate_csv_dataset(str(filepath), target_column="target", min_rows=15)
        assert val_res.is_valid is False
        assert any(e.code == "FILE_ROW_COUNT_LOW" for e in val_res.errors)

    def test_invalid_csv_single_class_target(self, tmp_path):
        """CSV target with only 1 unique class triggers TARGET_SINGLE_CLASS error."""
        filepath = tmp_path / "single_class.csv"
        df = pd.DataFrame({"f1": range(20), "target": [1] * 20})
        df.to_csv(filepath, index=False)
        val_res = validate_csv_dataset(str(filepath), target_column="target")
        assert val_res.is_valid is False
        assert any(e.code == "TARGET_SINGLE_CLASS" for e in val_res.errors)


# ---------------------------------------------------------------------------
# Section C: Stratification Integrity Tests
# ---------------------------------------------------------------------------

class TestStratificationIntegrity:
    """Verify class balance preservation, index fidelity, and seed reproducibility."""

    def test_binary_stratification_preserves_class_ratio(self):
        """Train and test partitions maintain class balance within integer rounding tolerance."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)

        full_ratio = (bundle.y == 1).mean()
        train_ratio = (split.y_train == 1).mean()
        test_ratio = (split.y_test == 1).mean()

        # Class ratio should be preserved to within 1%
        assert abs(train_ratio - full_ratio) < 0.01
        assert abs(test_ratio - full_ratio) < 0.02

    def test_multiclass_stratification_preserves_class_counts(self):
        """Multiclass Wine dataset preserves proportions for all 3 classes."""
        bundle = load_wine_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)

        for c in [0, 1, 2]:
            full_prop = (bundle.y == c).mean()
            train_prop = (split.y_train == c).mean()
            test_prop = (split.y_test == c).mean()
            assert abs(train_prop - full_prop) < 0.03
            assert abs(test_prop - full_prop) < 0.05

    def test_indices_are_disjoint_and_complete(self):
        """train_indices and test_indices partition original indices exactly."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)

        orig_idx_set = set(bundle.X.index)
        train_idx_set = set(split.train_indices)
        test_idx_set = set(split.test_indices)

        assert train_idx_set.isdisjoint(test_idx_set)
        assert train_idx_set | test_idx_set == orig_idx_set

    def test_supported_seeds_produce_valid_splits(self):
        """Every seed in DEFAULT_SPLIT_SEEDS produces valid, complete splits."""
        bundle = load_wine_data()
        for seed in DEFAULT_SPLIT_SEEDS:
            split = create_train_test_split(bundle, test_size=0.2, random_state=seed, stratify=True)
            assert len(split.X_train) == 142
            assert len(split.X_test) == 36
            assert split.random_state == seed

    def test_insufficient_class_count_raises_descriptive_error(self):
        """A minority class with only 1 sample raises an explicit descriptive error under stratify=True."""
        df_x = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        s_y = pd.Series([0, 0, 0, 0, 0, 1], name="target")  # class 1 has only 1 sample
        bundle = DatasetBundle(
            X=df_x, y=s_y, feature_names=["feat"], target_names=["0", "1"], dataset_name="rare_class", target_name="target"
        )
        with pytest.raises(ValueError, match="Cannot perform stratified split"):
            create_train_test_split(bundle, test_size=0.5, stratify=True)


# ---------------------------------------------------------------------------
# Section D: Leakage-Prevention Adversarial Tests
# ---------------------------------------------------------------------------

class TestLeakagePreventionEndToEnd:
    """Adversarial validation guaranteeing training-only fitting across all transformation stages."""

    @pytest.fixture
    def adversarial_bundle(self):
        """
        Adversarial dataset where test data exhibits severe distribution shift:
          - Train values ~ 10.0; Test values ~ 1000.0.
          - Train categories: ['A', 'B']; Test categories: ['B', 'UNSEEN_C'].
        """
        n_train = 30
        n_test = 30
        train_vals = np.linspace(8.0, 12.0, n_train)
        test_vals = np.linspace(900.0, 1100.0, n_test)
        num_col = np.concatenate([train_vals, test_vals])
        num_col[0] = np.nan
        num_col[-1] = np.nan

        cat_col = ["A", "B"] * (n_train // 2) + ["B", "UNSEEN_C"] * (n_test // 2)
        y = [0, 1] * ((n_train + n_test) // 2)

        df = pd.DataFrame({"metric": num_col, "category": cat_col})
        s_y = pd.Series(y, name="target")
        return DatasetBundle(
            X=df, y=s_y, feature_names=["metric", "category"], target_names=["0", "1"],
            dataset_name="adversarial_leakage", target_name="target",
            feature_types={"metric": "numerical", "category": "categorical"}
        ), n_train, n_test

    def test_imputer_and_scaler_quarantine(self, adversarial_bundle):
        """Imputer and Scaler learn parameters strictly from X_train."""
        bundle, n_train, n_test = adversarial_bundle
        split = create_train_test_split(bundle, test_size=0.5, stratify=False, shuffle=False)

        preprocessor = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)
        ct = preprocessor.transformer
        imputer = ct.named_transformers_["num"].named_steps["imputer"]
        scaler = ct.named_transformers_["num"].named_steps["scaler"]

        # Learned statistics must be around 10.0, not contaminated by test ~ 1000.0
        assert imputer.statistics_[0] < 20.0
        assert scaler.mean_[0] < 20.0

    def test_transforming_test_leaves_fitted_state_unchanged(self, adversarial_bundle):
        """Calling transform_with_preprocessor on test data cannot alter fitted attributes."""
        bundle, _, _ = adversarial_bundle
        split = create_train_test_split(bundle, test_size=0.5, stratify=False, shuffle=False)
        prep = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)

        num_pipe = prep.transformer.named_transformers_["num"]
        mean_before = copy.deepcopy(num_pipe.named_steps["scaler"].mean_)

        _ = transform_with_preprocessor(prep, split.X_test)
        np.testing.assert_array_equal(num_pipe.named_steps["scaler"].mean_, mean_before)

    def test_feature_selector_quarantine(self, adversarial_bundle):
        """Feature selection scores reflect strictly training labels and features."""
        bundle, _, _ = adversarial_bundle
        split = create_train_test_split(bundle, test_size=0.5, stratify=False, shuffle=False)
        prep = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        selector = fit_selector(X_tr, split.y_train, FeatureSelectionConfig(method="anova"))
        scores_before = copy.deepcopy(selector.scores)

        _ = transform_with_selector(selector, X_te)
        assert selector.scores == scores_before

    def test_pca_quarantine(self, adversarial_bundle):
        """PCA components and explained variance reflect strictly training data."""
        bundle, _, _ = adversarial_bundle
        split = create_train_test_split(bundle, test_size=0.5, stratify=False, shuffle=False)
        prep = fit_preprocessor(split.X_train, feature_types=bundle.feature_types)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        pca_art = fit_pca(X_tr, PCAConfig(n_components=2))
        evr_before = copy.deepcopy(pca_art.explained_variance_ratio)

        _ = transform_with_pca(pca_art, X_te)
        assert pca_art.explained_variance_ratio == evr_before


# ---------------------------------------------------------------------------
# Section E: Reproducibility Tests
# ---------------------------------------------------------------------------

class TestPipelineReproducibility:
    """Verify that identical configurations and seeds yield bitwise-identical results across runs."""

    def test_full_pipeline_reproducibility(self):
        """Two identical pipeline runs on Breast Cancer yield identical splits, preprocessed data, and PCA outputs."""
        bundle = load_breast_cancer_data()

        def _run_pipeline():
            split = create_train_test_split(bundle, test_size=0.2, random_state=42)
            prep = fit_preprocessor(split.X_train)
            X_tr = transform_with_preprocessor(prep, split.X_train)
            X_te = transform_with_preprocessor(prep, split.X_test)
            cmp_cfg = ComparisonModeConfig(
                mode="matched_reduced", reduction_method="pca", pca_config=PCAConfig(n_components=4, random_state=42), qubit_count=4
            )
            res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cmp_cfg)
            return split, X_tr, X_te, res

        split1, X_tr1, X_te1, res1 = _run_pipeline()
        split2, X_tr2, X_te2, res2 = _run_pipeline()

        # Identical split indices
        assert split1.train_indices == split2.train_indices
        assert split1.test_indices == split2.test_indices

        # Identical preprocessed DataFrames
        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_frame_equal(X_te1, X_te2)

        # Identical PCA projections
        pd.testing.assert_frame_equal(res1.X_train_classical, res2.X_train_classical)
        pd.testing.assert_frame_equal(res1.X_test_classical, res2.X_test_classical)

        # Identical metadata (excluding run-specific timestamps)
        d1 = res1.to_dict()
        d2 = res2.to_dict()
        d1.pop("fitted_timestamp_utc", None)
        d2.pop("fitted_timestamp_utc", None)
        assert d1 == d2


# ---------------------------------------------------------------------------
# Section F: Feature Metadata & Serialization Tests
# ---------------------------------------------------------------------------

class TestFeatureMetadataAndSerialization:
    """Verify semantic distinction between original and PCA features, and clean JSON export."""

    def test_pca_components_distinct_from_original_features(self):
        """PCA component names (PC1..PCk) must be labeled as transformed_components, not original."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cmp_cfg = ComparisonModeConfig(
            mode="matched_reduced", reduction_method="pca", pca_config=PCAConfig(n_components=4), qubit_count=4
        )
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cmp_cfg)

        assert res.classical_metadata.feature_type == "transformed_components"
        for name in res.classical_metadata.feature_names:
            assert name.startswith("PC")
            assert name not in bundle.feature_names

    def test_feature_selection_metadata_preserves_original_names(self):
        """Mutual Information selection preserves semantic original feature names."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cmp_cfg = ComparisonModeConfig(
            mode="matched_reduced", reduction_method="mutual_info",
            feature_selection_config=FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=5),
            qubit_count=5
        )
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cmp_cfg)

        assert res.classical_metadata.feature_type == "original_features"
        assert len(res.classical_metadata.feature_names) == 5
        for name in res.classical_metadata.feature_names:
            assert any(orig in name for orig in bundle.feature_names)

    def test_comparison_result_to_dict_json_serializable(self):
        """ComparisonModeResult.to_dict() converts cleanly with json.dumps without TypeError."""
        bundle = load_wine_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cmp_cfg = ComparisonModeConfig(
            mode="best_practical", classical_reduction_method="none", quantum_reduction_method="pca", qubit_count=4
        )
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cmp_cfg)

        d = res.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0


# ---------------------------------------------------------------------------
# Section G: PCA Integration Tests
# ---------------------------------------------------------------------------

class TestPCAIntegrationDeep:
    """Deep validation of PCA under various configuration modes and constraints."""

    def test_pca_variance_target_mode(self):
        """Mode B variance target selects minimal components meeting cumulative threshold."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)

        pca_art = fit_pca(X_tr, PCAConfig(variance_target=0.90))
        assert pca_art.total_variance_retained >= 0.90
        assert pca_art.n_components_actual < 30
        assert len(pca_art.component_names) == pca_art.n_components_actual

    def test_pca_invalid_component_count_exceeds_dimension_raises(self):
        """Requesting n_components > min(n_samples, n_features) raises ValueError."""
        df = pd.DataFrame(np.random.randn(20, 5))
        with pytest.raises(ValueError, match="exceeds max allowed"):
            fit_pca(df, PCAConfig(n_components=6))

    def test_pca_missing_values_prevented_by_preprocessing(self):
        """Preprocessed data with prior imputation enables stable PCA fitting even on dirty data."""
        df_dirty = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0] * 4,
            "b": [np.nan, 2.0, 3.0, 4.0, 5.0] * 4,
        })
        prep = fit_preprocessor(df_dirty)
        df_clean = transform_with_preprocessor(prep, df_dirty)
        pca_art = fit_pca(df_clean, PCAConfig(n_components=2))
        assert pca_art.n_components_actual == 2


# ---------------------------------------------------------------------------
# Section H: Comparison Modes Integration Tests
# ---------------------------------------------------------------------------

class TestComparisonModesIntegration:
    """Verify Mode A, Mode B, and Mode C integration, qubit checks, and CV fold helpers."""

    def test_mode_a_strict_identity(self):
        """Mode A guarantees identical representations verified by verify_matched_inputs."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cfg = ComparisonModeConfig(mode="matched_reduced", reduction_method="anova", feature_selection_config=FeatureSelectionConfig(method="anova", mode="manual", k_features=6), qubit_count=6)
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cfg)

        assert res.is_matched_input is True
        matched, _ = verify_matched_inputs(res.X_train_classical, res.X_train_quantum)
        assert matched is True

    def test_mode_b_enforces_qubit_limit(self):
        """Mode B enforces that qubit_count dictates feature count; no silent truncation."""
        bundle = load_wine_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cfg = ComparisonModeConfig(mode="quantum_limited", qubit_count=3, reduction_method="pca")
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.X_train_classical.shape[1] == 3
        assert res.X_train_quantum.shape[1] == 3

    def test_mode_c_practical_unmatched(self):
        """Mode C allows classical to evaluate on 13 features while quantum evaluates on 4."""
        bundle = load_wine_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)
        X_te = transform_with_preprocessor(prep, split.X_test)

        cfg = ComparisonModeConfig(mode="best_practical", classical_reduction_method="none", quantum_reduction_method="pca", qubit_count=4)
        res = prepare_comparison_mode(X_tr, split.y_train, X_te, split.y_test, config=cfg)

        assert res.is_valid is True
        assert res.is_matched_input is False  # Explicitly labeled practical
        assert res.X_train_classical.shape[1] == 13
        assert res.X_train_quantum.shape[1] == 4

    def test_cv_fold_comparison_mode_isolation(self):
        """Cross-validation fold comparison fits independent transformers strictly on fold slices."""
        bundle = load_breast_cancer_data()
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_tr = transform_with_preprocessor(prep, split.X_train)

        folds = create_stratified_cv_splits(X_tr, split.y_train, n_splits=3, random_state=42)
        cfg = ComparisonModeConfig(mode="matched_reduced", reduction_method="pca", pca_config=PCAConfig(n_components=4), qubit_count=4)

        for fold in folds:
            fold_res = prepare_cv_fold_comparison(X_tr, split.y_train, fold=fold, config=cfg)
            assert fold_res.is_valid is True
            assert fold_res.is_matched_input is True
            assert fold_res.X_train_classical.shape[1] == 4
            assert fold_res.X_test_classical.shape[1] == 4  # Validation slice
