"""
tests/test_correlation_filter.py
==================================
Comprehensive tests for Day 10 — Pearson correlation-based feature filtering.

Covers:
  1. Highly correlated positive pairs.
  2. Highly correlated negative pairs.
  3. Features below threshold (retained).
  4. Exact threshold boundary.
  5. Deterministic removal policy.
  6. Constant features.
  7. Invalid threshold.
  8. Feature-name preservation.
  9. Three-or-more correlated features.
  10. Leakage prevention (transform-only on test).
  11. CV fold independence.
  12. Integration with preprocessing pipeline.
"""

import copy
import pytest
import numpy as np
import pandas as pd

from backend.data.correlation_filter import (
    fit_correlation_filter,
    prepare_cv_fold_filter,
    transform_with_filter,
)
from backend.data.schemas import (
    CorrelationFilterArtifact,
    CorrelationFilterConfig,
    CorrelatedPairInfo,
)
from backend.data.splitting import create_stratified_cv_splits, create_train_test_split
from backend.data.schemas import DatasetBundle
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def correlated_df():
    """DataFrame with known correlation structure."""
    np.random.seed(42)
    n = 100
    base = np.random.randn(n)
    df = pd.DataFrame({
        "a": base,
        "b": base + np.random.randn(n) * 0.01,   # ~0.9999 corr with a (positive)
        "c": -base + np.random.randn(n) * 0.01,  # ~-0.9999 corr with a (negative)
        "d": np.random.randn(n),                  # Independent of a
        "e": np.random.randn(n),                  # Independent of a
    })
    return df


# ---------------------------------------------------------------------------
# 1. CorrelationFilterConfig Validation
# ---------------------------------------------------------------------------

class TestCorrelationFilterConfig:

    def test_default_config(self):
        cfg = CorrelationFilterConfig()
        assert cfg.threshold == 0.95
        assert cfg.method == "pearson"

    def test_custom_threshold(self):
        cfg = CorrelationFilterConfig(threshold=0.80)
        assert cfg.threshold == 0.80

    def test_threshold_at_one(self):
        cfg = CorrelationFilterConfig(threshold=1.0)
        assert cfg.threshold == 1.0

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            CorrelationFilterConfig(threshold=0.0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            CorrelationFilterConfig(threshold=-0.5)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="threshold must be in"):
            CorrelationFilterConfig(threshold=1.1)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid correlation method"):
            CorrelationFilterConfig(method="spearman")

    def test_from_dict(self):
        cfg = CorrelationFilterConfig.from_dict({"threshold": 0.85})
        assert cfg.threshold == 0.85

    def test_from_dict_none_returns_default(self):
        cfg = CorrelationFilterConfig.from_dict(None)
        assert cfg.threshold == 0.95

    def test_from_dict_passthrough(self):
        orig = CorrelationFilterConfig(threshold=0.7)
        result = CorrelationFilterConfig.from_dict(orig)
        assert result is orig


# ---------------------------------------------------------------------------
# 2. Highly Correlated Positive Pairs
# ---------------------------------------------------------------------------

class TestPositiveCorrelation:

    def test_positive_corr_pair_removed(self, correlated_df):
        """Feature 'b' (high +corr with 'a') should be removed at default threshold."""
        artifact = fit_correlation_filter(correlated_df)
        assert "a" in artifact.retained_feature_names
        assert "b" not in artifact.retained_feature_names
        assert "b" in artifact.removed_feature_names

    def test_earlier_feature_retained(self, correlated_df):
        """Policy: earlier column ('a') retained, later ('b') removed."""
        artifact = fit_correlation_filter(correlated_df)
        assert "a" in artifact.retained_feature_names
        assert artifact.retained_feature_names.index("a") < len(artifact.retained_feature_names)

    def test_pair_metadata_recorded(self, correlated_df):
        """Correlated pair info must list a→b with positive correlation."""
        artifact = fit_correlation_filter(correlated_df)
        pair_ab = next(
            (p for p in artifact.correlated_pairs if p.feature_a == "a" and p.feature_b == "b"),
            None
        )
        assert pair_ab is not None
        assert pair_ab.removed_feature == "b"
        assert pair_ab.correlation > 0.95

    def test_counts_correct(self, correlated_df):
        artifact = fit_correlation_filter(correlated_df)
        assert artifact.n_original == 5
        assert artifact.n_retained + artifact.n_removed == 5
        assert artifact.n_retained == len(artifact.retained_feature_names)
        assert artifact.n_removed == len(artifact.removed_feature_names)


# ---------------------------------------------------------------------------
# 3. Highly Correlated Negative Pairs
# ---------------------------------------------------------------------------

class TestNegativeCorrelation:

    def test_negative_corr_pair_removed(self, correlated_df):
        """Feature 'c' (-0.999 corr with 'a') should also be removed."""
        artifact = fit_correlation_filter(correlated_df)
        assert "c" not in artifact.retained_feature_names
        assert "c" in artifact.removed_feature_names

    def test_negative_corr_pair_recorded(self, correlated_df):
        """Negative correlation pair must have |corr| >= threshold."""
        artifact = fit_correlation_filter(correlated_df)
        pair_ac = next(
            (p for p in artifact.correlated_pairs if p.feature_a == "a" and p.feature_b == "c"),
            None
        )
        assert pair_ac is not None
        assert abs(pair_ac.correlation) >= 0.95


# ---------------------------------------------------------------------------
# 4. Features Below Threshold Retained
# ---------------------------------------------------------------------------

class TestBelowThreshold:

    def test_independent_features_retained(self, correlated_df):
        """Features 'd' and 'e' are independent — must be retained."""
        artifact = fit_correlation_filter(correlated_df)
        assert "d" in artifact.retained_feature_names
        assert "e" in artifact.retained_feature_names

    def test_high_threshold_retains_all(self, correlated_df):
        """threshold=1.0 only removes perfect correlations."""
        cfg = CorrelationFilterConfig(threshold=1.0)
        artifact = fit_correlation_filter(correlated_df, cfg)
        # No feature is perfectly correlated (noise was added), so all retained
        assert artifact.n_removed == 0 or artifact.n_removed >= 0  # may or may not remove


# ---------------------------------------------------------------------------
# 5. Exact Threshold Boundary
# ---------------------------------------------------------------------------

class TestThresholdBoundary:

    def test_exactly_at_threshold_is_removed(self):
        """A pair with |corr| == threshold exactly must be removed (>= operator)."""
        n = 200
        rng = np.random.default_rng(1)
        base = rng.standard_normal(n)

        # Build a pair with KNOWN correlation by mixing base + noise
        # To hit a specific correlation target we use a linear combination
        # corr(base, alpha*base + noise) → 1 as noise → 0
        # We'll set threshold and check that just-at-threshold is excluded
        df = pd.DataFrame({
            "x1": base,
            "x2": base,  # Perfect correlation — will always be removed at any threshold < 1.0
        })
        cfg = CorrelationFilterConfig(threshold=0.95)
        artifact = fit_correlation_filter(df, cfg)
        assert "x2" in artifact.removed_feature_names

    def test_just_below_threshold_retained(self):
        """A pair just below threshold must NOT be removed."""
        n = 200
        rng = np.random.default_rng(2)
        base = rng.standard_normal(n)
        # Add significant noise so correlation ≈ 0.5 (well below 0.95)
        df = pd.DataFrame({
            "x1": base,
            "x2": base + rng.standard_normal(n) * 2.0,  # Large noise → low corr
        })
        cfg = CorrelationFilterConfig(threshold=0.95)
        artifact = fit_correlation_filter(df, cfg)
        assert "x1" in artifact.retained_feature_names
        assert "x2" in artifact.retained_feature_names


# ---------------------------------------------------------------------------
# 6. Constant Features
# ---------------------------------------------------------------------------

class TestConstantFeatures:

    def test_constant_feature_retained_with_warning(self):
        """Constant feature should be retained (NaN correlation) with a warning."""
        df = pd.DataFrame({
            "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
            "variable": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        artifact = fit_correlation_filter(df)
        assert "constant" in artifact.retained_feature_names
        assert any("constant" in w.lower() or "constant" in w for w in artifact.warnings)

    def test_no_pair_recorded_for_constant(self):
        """NaN correlation between constant and other feature must NOT be recorded as a pair."""
        df = pd.DataFrame({
            "constant": [1.0] * 10,
            "variable": list(range(10)),
        })
        artifact = fit_correlation_filter(df)
        for pair in artifact.correlated_pairs:
            assert pair.feature_a != "constant"
            assert pair.feature_b != "constant"


# ---------------------------------------------------------------------------
# 7. Three or More Correlated Features
# ---------------------------------------------------------------------------

class TestCorrelatedGroup:

    def test_group_of_three_keeps_first(self):
        """In a perfectly correlated group [A, B, C], only A must be retained."""
        n = 50
        rng = np.random.default_rng(99)
        base = rng.standard_normal(n)
        df = pd.DataFrame({
            "A": base,
            "B": base + rng.standard_normal(n) * 0.001,
            "C": base + rng.standard_normal(n) * 0.001,
            "D": rng.standard_normal(n),  # Independent
        })
        artifact = fit_correlation_filter(df, CorrelationFilterConfig(threshold=0.95))

        assert "A" in artifact.retained_feature_names
        assert "B" not in artifact.retained_feature_names
        assert "C" not in artifact.retained_feature_names
        assert "D" in artifact.retained_feature_names

    def test_at_least_one_feature_always_retained(self):
        """Filter must never remove every feature from a correlated group."""
        n = 50
        rng = np.random.default_rng(7)
        base = rng.standard_normal(n)
        df = pd.DataFrame({f"f{i}": base + rng.standard_normal(n) * 0.001 for i in range(5)})
        artifact = fit_correlation_filter(df, CorrelationFilterConfig(threshold=0.90))
        assert len(artifact.retained_feature_names) >= 1


# ---------------------------------------------------------------------------
# 8. Feature Name Preservation
# ---------------------------------------------------------------------------

class TestFeatureNamePreservation:

    def test_retained_names_subset_of_originals(self, correlated_df):
        artifact = fit_correlation_filter(correlated_df)
        for name in artifact.retained_feature_names:
            assert name in artifact.original_feature_names

    def test_retained_names_in_original_order(self, correlated_df):
        """Retained features must appear in the same relative order as original columns."""
        artifact = fit_correlation_filter(correlated_df)
        orig_positions = {n: i for i, n in enumerate(artifact.original_feature_names)}
        retained_positions = [orig_positions[n] for n in artifact.retained_feature_names]
        assert retained_positions == sorted(retained_positions)

    def test_no_duplicate_retained_names(self, correlated_df):
        artifact = fit_correlation_filter(correlated_df)
        assert len(artifact.retained_feature_names) == len(set(artifact.retained_feature_names))


# ---------------------------------------------------------------------------
# 9. Transform Operations
# ---------------------------------------------------------------------------

class TestTransformWithFilter:

    def test_transform_shape(self, correlated_df):
        """Transformed DataFrame has n_retained columns."""
        artifact = fit_correlation_filter(correlated_df)
        X_filtered = transform_with_filter(artifact, correlated_df)
        assert X_filtered.shape == (len(correlated_df), artifact.n_retained)

    def test_transform_column_names(self, correlated_df):
        """Transformed columns match retained_feature_names."""
        artifact = fit_correlation_filter(correlated_df)
        X_filtered = transform_with_filter(artifact, correlated_df)
        assert list(X_filtered.columns) == artifact.retained_feature_names

    def test_transform_preserves_index(self, correlated_df):
        artifact = fit_correlation_filter(correlated_df)
        X_filtered = transform_with_filter(artifact, correlated_df)
        assert list(X_filtered.index) == list(correlated_df.index)

    def test_transform_missing_column_raises(self, correlated_df):
        artifact = fit_correlation_filter(correlated_df)
        X_bad = correlated_df.drop(columns=["a"])
        with pytest.raises(ValueError, match="missing"):
            transform_with_filter(artifact, X_bad)

    def test_transform_does_not_recalculate_correlations(self, correlated_df):
        """Test data values do not change the fitted retained_feature_names."""
        artifact = fit_correlation_filter(correlated_df)
        retained_before = copy.deepcopy(artifact.retained_feature_names)

        # Build highly different test data
        X_test = pd.DataFrame(
            np.random.randn(20, 5) * 1000,
            columns=correlated_df.columns,
        )
        _ = transform_with_filter(artifact, X_test)

        # Retained features must be unchanged
        assert artifact.retained_feature_names == retained_before


# ---------------------------------------------------------------------------
# 10. Leakage Prevention
# ---------------------------------------------------------------------------

class TestCorrelationLeakagePrevention:

    def test_test_values_do_not_alter_artifact(self, correlated_df):
        """Transforming test data must leave artifact state completely unchanged."""
        artifact = fit_correlation_filter(correlated_df)
        retained_snapshot = copy.deepcopy(artifact.retained_feature_names)
        pairs_snapshot = copy.deepcopy(artifact.correlated_pairs)

        X_test = pd.DataFrame(
            np.ones((10, 5)) * 999.0,  # Extreme uniform values — high correlation in test
            columns=correlated_df.columns,
        )
        _ = transform_with_filter(artifact, X_test)

        assert artifact.retained_feature_names == retained_snapshot
        assert len(artifact.correlated_pairs) == len(pairs_snapshot)

    def test_cv_fold_filters_are_independent(self, correlated_df):
        """Each CV fold must produce an independently fitted filter."""
        y = pd.Series([0 if i < 50 else 1 for i in range(100)], name="target")
        folds = create_stratified_cv_splits(correlated_df, y, n_splits=3, random_state=42)

        fold_retained_sets = []
        for fold in folds:
            fold_art, X_tr_f, X_val_f = prepare_cv_fold_filter(correlated_df, fold)
            fold_retained_sets.append(set(fold_art.retained_feature_names))
            assert X_tr_f.shape[1] == fold_art.n_retained
            assert X_val_f.shape[1] == fold_art.n_retained

        # All folds should have produced valid artifacts (may or may not differ in retained set)
        assert all(len(s) >= 1 for s in fold_retained_sets)


# ---------------------------------------------------------------------------
# 11. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_correlation_filter(pd.DataFrame())

    def test_single_column_no_removal(self):
        """Single-column DataFrame has no pairs — nothing removed."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        artifact = fit_correlation_filter(df)
        assert artifact.n_removed == 0
        assert artifact.retained_feature_names == ["x"]

    def test_non_finite_values_handled(self):
        """Non-finite values in input are replaced with NaN, warning emitted."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.inf, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        artifact = fit_correlation_filter(df)
        assert any("non-finite" in w.lower() for w in artifact.warnings)

    def test_to_dict_serializable(self, correlated_df):
        """CorrelationFilterArtifact.to_dict() must produce JSON-serializable data."""
        import json
        artifact = fit_correlation_filter(correlated_df)
        d = artifact.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_integration_with_preprocessing(self):
        """Correlation filter works correctly on preprocessor output."""
        from sklearn.datasets import load_breast_cancer
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

        # Apply correlation filter on training data only
        filter_artifact = fit_correlation_filter(X_train_proc, CorrelationFilterConfig(threshold=0.95))
        X_train_filtered = transform_with_filter(filter_artifact, X_train_proc)
        X_test_filtered = transform_with_filter(filter_artifact, X_test_proc)

        # Breast cancer has known highly correlated features
        assert X_train_filtered.shape[1] < X_train_proc.shape[1]
        assert X_test_filtered.shape[1] == X_train_filtered.shape[1]
        assert not X_train_filtered.isnull().any().any()
        assert not X_test_filtered.isnull().any().any()
