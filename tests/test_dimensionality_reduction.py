"""
tests/test_dimensionality_reduction.py
========================================
Comprehensive tests for Day 10 — PCA dimensionality reduction.

Covers:
  1. Mode A (explicit n_components).
  2. Mode B (variance_target).
  3. Component names (PC1, PC2, ...).
  4. Explained variance ratios.
  5. Cumulative explained variance.
  6. Invalid configs.
  7. Non-finite inputs.
  8. Train/test transformation consistency.
  9. Leakage prevention (transform-only on test, CV fold independence).
  10. Integration with preprocessing pipeline.
"""

import copy
import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine

from backend.data.dimensionality_reduction import (
    fit_pca,
    prepare_cv_fold_pca,
    transform_with_pca,
)
from backend.data.schemas import (
    DatasetBundle,
    PCAConfig,
    PCAArtifact,
)
from backend.data.splitting import create_stratified_cv_splits, create_train_test_split
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def breast_cancer_processed():
    """Preprocessed Breast Cancer split (standardised, no missing values)."""
    bc = load_breast_cancer(as_frame=True)
    X = bc.frame.drop(columns=["target"]).astype(float)
    y = bc.frame["target"]
    bundle = DatasetBundle(
        X=X, y=y,
        feature_names=list(X.columns),
        target_names=["malignant", "benign"],
        dataset_name="breast_cancer", target_name="target"
    )
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    prep = fit_preprocessor(split.X_train)
    X_train_proc = transform_with_preprocessor(prep, split.X_train)
    X_test_proc = transform_with_preprocessor(prep, split.X_test)
    return split, X_train_proc, X_test_proc


# ---------------------------------------------------------------------------
# 1. PCAConfig Validation
# ---------------------------------------------------------------------------

class TestPCAConfig:

    def test_mode_a_valid(self):
        cfg = PCAConfig(n_components=10)
        assert cfg.n_components == 10
        assert cfg.variance_target is None

    def test_mode_b_valid(self):
        cfg = PCAConfig(variance_target=0.95)
        assert cfg.variance_target == 0.95
        assert cfg.n_components is None

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            PCAConfig()

    def test_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            PCAConfig(n_components=5, variance_target=0.9)

    def test_n_components_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            PCAConfig(n_components=0)

    def test_n_components_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            PCAConfig(n_components=-3)

    def test_n_components_non_int_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            PCAConfig(n_components=5.5)

    def test_variance_target_zero_raises(self):
        with pytest.raises(ValueError, match="variance_target"):
            PCAConfig(variance_target=0.0)

    def test_variance_target_too_large_raises(self):
        with pytest.raises(ValueError, match="variance_target"):
            PCAConfig(variance_target=1.5)

    def test_from_dict_mode_a(self):
        cfg = PCAConfig.from_dict({"n_components": 5})
        assert cfg.n_components == 5

    def test_from_dict_mode_b(self):
        cfg = PCAConfig.from_dict({"variance_target": 0.9})
        assert cfg.variance_target == 0.9

    def test_from_dict_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            PCAConfig.from_dict(None)

    def test_from_dict_passthrough(self):
        orig = PCAConfig(n_components=3)
        result = PCAConfig.from_dict(orig)
        assert result is orig

    def test_to_dict(self):
        cfg = PCAConfig(n_components=5, random_state=0)
        d = cfg.to_dict()
        assert d["n_components"] == 5
        assert d["random_state"] == 0


# ---------------------------------------------------------------------------
# 2. Mode A — Explicit Component Count
# ---------------------------------------------------------------------------

class TestModeA:

    def test_output_shape(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        assert artifact.n_components_actual == 10

    def test_component_names(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        assert artifact.component_names == ["PC1", "PC2", "PC3", "PC4", "PC5"]

    def test_transform_shape(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=8))
        X_pca = transform_with_pca(artifact, X_train)
        assert X_pca.shape == (len(X_train), 8)
        assert list(X_pca.columns) == artifact.component_names

    def test_explained_variance_ratio_length(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        assert len(artifact.explained_variance_ratio) == 10

    def test_explained_variance_ratios_sum_leq_one(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        assert 0.0 < artifact.total_variance_retained <= 1.0

    def test_cumulative_evr_monotone(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        cumulative = artifact.cumulative_explained_variance
        assert all(cumulative[i] <= cumulative[i+1] for i in range(len(cumulative)-1))

    def test_cumulative_last_equals_total(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        assert np.isclose(
            artifact.cumulative_explained_variance[-1],
            artifact.total_variance_retained,
            atol=1e-8
        )

    def test_n_exceeds_max_raises(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        max_k = min(X_train.shape)
        with pytest.raises(ValueError, match="exceeds max"):
            fit_pca(X_train, PCAConfig(n_components=max_k + 1))

    def test_original_feature_names_preserved(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        assert artifact.original_feature_names == list(X_train.columns)


# ---------------------------------------------------------------------------
# 3. Mode B — Explained Variance Target
# ---------------------------------------------------------------------------

class TestModeB:

    def test_target_95pct(self, breast_cancer_processed):
        """95% target must select fewer components than the full set."""
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(variance_target=0.95))
        assert artifact.total_variance_retained >= 0.95
        assert artifact.n_components_actual < X_train.shape[1]

    def test_target_99pct(self, breast_cancer_processed):
        """99% target selects more components than 95%."""
        split, X_train, X_test = breast_cancer_processed
        a95 = fit_pca(X_train, PCAConfig(variance_target=0.95))
        a99 = fit_pca(X_train, PCAConfig(variance_target=0.99))
        assert a99.n_components_actual >= a95.n_components_actual

    def test_target_1_selects_all(self, breast_cancer_processed):
        """variance_target=1.0 must select all components (full variance)."""
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(variance_target=1.0))
        max_k = min(X_train.shape)
        assert artifact.n_components_actual == max_k

    def test_component_names_match_actual(self, breast_cancer_processed):
        """Component names must match n_components_actual."""
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(variance_target=0.90))
        assert len(artifact.component_names) == artifact.n_components_actual
        assert artifact.component_names[0] == "PC1"
        assert artifact.component_names[-1] == f"PC{artifact.n_components_actual}"

    def test_cumulative_evr_reaches_target(self, breast_cancer_processed):
        """Cumulative EVR at selected k must meet the target."""
        split, X_train, X_test = breast_cancer_processed
        target = 0.95
        artifact = fit_pca(X_train, PCAConfig(variance_target=target))
        assert artifact.cumulative_explained_variance[-1] >= target


# ---------------------------------------------------------------------------
# 4. Train / Test Transformation Consistency
# ---------------------------------------------------------------------------

class TestTrainTestConsistency:

    def test_test_shape_matches_train(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        X_train_pca = transform_with_pca(artifact, X_train)
        X_test_pca = transform_with_pca(artifact, X_test)
        assert X_train_pca.shape[1] == X_test_pca.shape[1] == 10

    def test_test_column_names_match_train(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        X_train_pca = transform_with_pca(artifact, X_train)
        X_test_pca = transform_with_pca(artifact, X_test)
        assert list(X_train_pca.columns) == list(X_test_pca.columns)

    def test_index_preserved(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        X_pca = transform_with_pca(artifact, X_test)
        assert list(X_pca.index) == list(X_test.index)

    def test_column_mismatch_raises(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        X_bad = X_test.rename(columns={X_test.columns[0]: "wrong"})
        with pytest.raises(ValueError, match="Column mismatch"):
            transform_with_pca(artifact, X_bad)

    def test_no_nans_in_output(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        X_train_pca = transform_with_pca(artifact, X_train)
        X_test_pca = transform_with_pca(artifact, X_test)
        assert not X_train_pca.isnull().any().any()
        assert not X_test_pca.isnull().any().any()


# ---------------------------------------------------------------------------
# 5. Non-Finite Inputs
# ---------------------------------------------------------------------------

class TestNonFiniteInputs:

    def test_nan_in_train_raises(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            fit_pca(df, PCAConfig(n_components=1))

    def test_inf_in_train_raises(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.inf, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        })
        with pytest.raises(ValueError, match="non-finite"):
            fit_pca(df, PCAConfig(n_components=1))


# ---------------------------------------------------------------------------
# 6. Leakage Prevention
# ---------------------------------------------------------------------------

class TestPCALeakagePrevention:

    def test_test_values_do_not_alter_pca_components(self, breast_cancer_processed):
        """PCA components must be unchanged after transforming test data."""
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))

        components_before = copy.deepcopy(artifact.pca.components_)
        evr_before = copy.deepcopy(artifact.explained_variance_ratio)

        _ = transform_with_pca(artifact, X_test)

        np.testing.assert_array_equal(artifact.pca.components_, components_before)
        assert artifact.explained_variance_ratio == evr_before

    def test_cv_fold_pca_artifacts_independent(self, breast_cancer_processed):
        """Each CV fold must produce independently fitted PCA."""
        split, X_train, X_test = breast_cancer_processed
        y_train = split.y_train
        cfg = PCAConfig(n_components=5)

        folds = create_stratified_cv_splits(X_train, y_train, n_splits=3, random_state=42)
        fold_evrs = []
        for fold in folds:
            fold_art, X_tr_pca, X_val_pca = prepare_cv_fold_pca(X_train, fold, cfg)
            assert X_tr_pca.shape[1] == 5
            assert X_val_pca.shape[1] == 5
            assert list(X_tr_pca.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5"]
            fold_evrs.append(fold_art.explained_variance_ratio[:])

        # Different data slices → different explained variance ratios
        assert fold_evrs[0] != fold_evrs[1] or fold_evrs[1] != fold_evrs[2], (
            "CV fold PCA artifacts may not be independent!"
        )

    def test_cv_val_uses_fitted_pca_not_refitted(self, breast_cancer_processed):
        """Val slice must be transformed using the fold training artifact — no refitting."""
        split, X_train, X_test = breast_cancer_processed
        y_train = split.y_train
        cfg = PCAConfig(n_components=3)
        folds = create_stratified_cv_splits(X_train, y_train, n_splits=3, random_state=42)
        fold = folds[0]

        fold_art, X_tr_pca, X_val_pca = prepare_cv_fold_pca(X_train, fold, cfg)

        # Manually transform val using the returned artifact → results must match
        X_fold_val = X_train.loc[fold.val_indices]
        X_val_manual = transform_with_pca(fold_art, X_fold_val)
        pd.testing.assert_frame_equal(X_val_pca, X_val_manual)


# ---------------------------------------------------------------------------
# 7. PCA Metadata
# ---------------------------------------------------------------------------

class TestPCAMetadata:

    def test_component_names_not_original_names(self, breast_cancer_processed):
        """PCA component names must NOT match original feature names."""
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        for name in artifact.component_names:
            assert name not in artifact.original_feature_names
            assert name.startswith("PC")

    def test_to_dict_excludes_pca_object(self, breast_cancer_processed):
        """to_dict() must not include the non-serializable sklearn PCA object."""
        import json
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=5))
        d = artifact.to_dict()
        assert "pca" not in d
        json_str = json.dumps(d)  # Must not raise
        assert len(json_str) > 0

    def test_all_evr_non_negative(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=10))
        assert all(v >= 0.0 for v in artifact.explained_variance_ratio)

    def test_timestamp_set(self, breast_cancer_processed):
        split, X_train, X_test = breast_cancer_processed
        artifact = fit_pca(X_train, PCAConfig(n_components=3))
        assert artifact.fitted_timestamp_utc != ""


# ---------------------------------------------------------------------------
# 8. Integration Tests
# ---------------------------------------------------------------------------

class TestPCAIntegration:

    def test_pca_after_correlation_filter(self, breast_cancer_processed):
        """PCA works correctly on correlation-filtered output."""
        from backend.data.correlation_filter import fit_correlation_filter, transform_with_filter
        from backend.data.schemas import CorrelationFilterConfig

        split, X_train_proc, X_test_proc = breast_cancer_processed

        # Step 1: Correlation filter
        filter_art = fit_correlation_filter(X_train_proc, CorrelationFilterConfig(threshold=0.95))
        X_train_filtered = transform_with_filter(filter_art, X_train_proc)
        X_test_filtered = transform_with_filter(filter_art, X_test_proc)

        n_retained = filter_art.n_retained
        assert n_retained >= 1

        # Step 2: PCA on filtered data
        pca_art = fit_pca(X_train_filtered, PCAConfig(n_components=min(5, n_retained)))
        X_train_pca = transform_with_pca(pca_art, X_train_filtered)
        X_test_pca = transform_with_pca(pca_art, X_test_filtered)

        assert X_train_pca.shape[1] == min(5, n_retained)
        assert X_test_pca.shape[1] == X_train_pca.shape[1]
        assert not X_train_pca.isnull().any().any()
        assert not X_test_pca.isnull().any().any()

        # Component names are PC-prefixed, NOT original feature names
        for col in X_train_pca.columns:
            assert col.startswith("PC")

    def test_pca_after_feature_selection(self, breast_cancer_processed):
        """PCA is compatible with MI-selected features."""
        from backend.data.feature_selection import fit_selector, transform_with_selector
        from backend.data.schemas import FeatureSelectionConfig

        split, X_train_proc, X_test_proc = breast_cancer_processed

        sel_art = fit_selector(
            X_train_proc, split.y_train,
            FeatureSelectionConfig(method="anova", mode="manual", k_features=10)
        )
        X_train_sel = transform_with_selector(sel_art, X_train_proc)
        X_test_sel = transform_with_selector(sel_art, X_test_proc)

        pca_art = fit_pca(X_train_sel, PCAConfig(n_components=5))
        X_train_pca = transform_with_pca(pca_art, X_train_sel)
        X_test_pca = transform_with_pca(pca_art, X_test_sel)

        assert X_train_pca.shape[1] == 5
        assert not X_train_pca.isnull().any().any()
        assert not X_test_pca.isnull().any().any()

    def test_wine_variance_target_mode(self):
        """End-to-end variance-target PCA on Wine dataset."""
        wine = load_wine(as_frame=True)
        X = wine.frame.drop(columns=["target"]).astype(float)
        y = wine.frame["target"]

        bundle = DatasetBundle(
            X=X, y=y,
            feature_names=list(X.columns),
            target_names=["c0", "c1", "c2"],
            dataset_name="wine", target_name="target"
        )
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)
        prep = fit_preprocessor(split.X_train)
        X_train_proc = transform_with_preprocessor(prep, split.X_train)
        X_test_proc = transform_with_preprocessor(prep, split.X_test)

        artifact = fit_pca(X_train_proc, PCAConfig(variance_target=0.95))
        assert artifact.total_variance_retained >= 0.95
        assert artifact.n_components_actual < 13

        X_train_pca = transform_with_pca(artifact, X_train_proc)
        X_test_pca = transform_with_pca(artifact, X_test_proc)
        assert X_train_pca.shape[1] == artifact.n_components_actual
        assert X_test_pca.shape[1] == X_train_pca.shape[1]
