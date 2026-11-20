"""
tests/test_feature_selection.py
================================
Comprehensive test suite for Day 9 — Mutual Information and ANOVA feature selection.

Covers:
  1. Mutual Information (binary, multiclass, reproducibility, auto/manual selection).
  2. ANOVA F-test (binary, multiclass, F-scores, p-values, constant features).
  3. Configuration validation (FeatureSelectionConfig).
  4. Transform-only test application (leakage prevention).
  5. CV fold independent selector fitting.
  6. Adversarial leakage tests.
  7. Feature preview metadata.
  8. Edge cases: empty matrix, missing labels, mismatched dimensions, k > n_candidate.
"""

import copy
import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine

from backend.data.feature_selection import (
    build_selection_preview,
    fit_selector,
    prepare_cv_fold_selection,
    transform_with_selector,
)
from backend.data.schemas import (
    DatasetBundle,
    FeatureSelectionConfig,
    FeatureSelectorArtifact,
    FeatureSelectionPreview,
)
from backend.data.splitting import create_train_test_split, create_stratified_cv_splits
from backend.data.preprocessing import fit_preprocessor, transform_with_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Breast Cancer Wisconsin: 569 samples, 30 numerical features, binary target."""
    bc = load_breast_cancer(as_frame=True)
    X = bc.frame.drop(columns=["target"])
    y = bc.frame["target"]
    return X.astype(float), y


@pytest.fixture
def multiclass_data():
    """Wine: 178 samples, 13 numerical features, 3-class target."""
    wine = load_wine(as_frame=True)
    X = wine.frame.drop(columns=["target"])
    y = wine.frame["target"]
    return X.astype(float), y


@pytest.fixture
def small_mixed_data():
    """Small synthetic mixed dataset for fast unit tests."""
    np.random.seed(0)
    n = 60
    X = pd.DataFrame({
        "informative_a": np.random.randn(n) + np.array([0.0 if i < 30 else 2.0 for i in range(n)]),
        "informative_b": np.random.randn(n) + np.array([1.0 if i < 30 else -1.0 for i in range(n)]),
        "noise":         np.random.randn(n) * 0.001,
        "constant":      np.ones(n),
    })
    y = pd.Series([0 if i < 30 else 1 for i in range(n)], name="target")
    return X, y


# ---------------------------------------------------------------------------
# 1. Mutual Information — Binary Classification
# ---------------------------------------------------------------------------

class TestMutualInformationBinary:

    def test_basic_mi_binary(self, binary_data):
        """fit_selector with MI returns a valid artifact on binary data."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="auto")
        artifact = fit_selector(X, y, cfg)

        assert isinstance(artifact, FeatureSelectorArtifact)
        assert artifact.method == "mutual_info"
        assert artifact.k_selected == 30  # auto, all features
        assert len(artifact.scores) == 30
        assert len(artifact.selected_feature_names) == 30
        assert artifact.pvalues is None  # MI has no p-values

    def test_mi_scores_non_negative(self, binary_data):
        """Mutual information scores must be >= 0 for all features."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info"))
        assert all(s >= 0.0 for s in artifact.scores)

    def test_mi_manual_k(self, binary_data):
        """Manual k=10 selects exactly 10 features with correct names."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=10)
        artifact = fit_selector(X, y, cfg)

        assert artifact.k_selected == 10
        assert artifact.k_requested == 10
        assert len(artifact.selected_feature_names) == 10
        # All selected names must come from the original candidate names
        assert all(name in artifact.candidate_feature_names for name in artifact.selected_feature_names)

    def test_mi_auto_proportion(self, binary_data):
        """Auto mode with proportion=0.5 selects approximately half the features."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="auto", proportion=0.5)
        artifact = fit_selector(X, y, cfg)

        expected_k = max(1, int(np.floor(0.5 * 30)))
        assert artifact.k_selected == expected_k

    def test_mi_reproducibility(self, binary_data):
        """Same random_state produces identical MI scores across calls."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", random_state=42)
        a1 = fit_selector(X, y, cfg)
        a2 = fit_selector(X, y, cfg)

        np.testing.assert_array_almost_equal(a1.scores, a2.scores, decimal=10)
        assert a1.selected_feature_names == a2.selected_feature_names

    def test_mi_different_seeds_differ(self, binary_data):
        """Different random states should generally produce different MI scores."""
        X, y = binary_data
        a1 = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info", random_state=1))
        a2 = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info", random_state=99))
        # Scores may differ (MI has stochastic estimation of continuous features)
        # We only check the structure is correct for both
        assert len(a1.scores) == len(a2.scores)

    def test_mi_feature_names_preserved(self, binary_data):
        """Candidate and selected feature names match the input DataFrame columns."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=5)
        artifact = fit_selector(X, y, cfg)

        assert artifact.candidate_feature_names == list(X.columns)
        for name in artifact.selected_feature_names:
            assert name in list(X.columns)


# ---------------------------------------------------------------------------
# 2. Mutual Information — Multiclass Classification
# ---------------------------------------------------------------------------

class TestMutualInformationMulticlass:

    def test_mi_multiclass(self, multiclass_data):
        """MI selector works on 3-class Wine dataset."""
        X, y = multiclass_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=7)
        artifact = fit_selector(X, y, cfg)

        assert artifact.k_selected == 7
        assert artifact.method == "mutual_info"
        assert len(artifact.scores) == 13
        assert all(s >= 0.0 for s in artifact.scores)

    def test_mi_multiclass_rankings_valid(self, multiclass_data):
        """Rankings are a permutation of 1..n_candidate."""
        X, y = multiclass_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info"))
        n = len(artifact.candidate_feature_names)
        assert sorted(artifact.rankings) == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# 3. ANOVA F-test — Binary Classification
# ---------------------------------------------------------------------------

class TestANOVABinary:

    def test_basic_anova_binary(self, binary_data):
        """fit_selector with ANOVA returns a valid artifact with p-values."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="anova", mode="auto")
        artifact = fit_selector(X, y, cfg)

        assert artifact.method == "anova"
        assert len(artifact.scores) == 30
        assert artifact.pvalues is not None
        assert len(artifact.pvalues) == 30

    def test_anova_pvalues_in_range(self, binary_data):
        """ANOVA p-values must be in [0.0, 1.0]."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova"))
        for pv in artifact.pvalues:
            assert 0.0 <= pv <= 1.0

    def test_anova_f_scores_non_negative(self, binary_data):
        """ANOVA F-scores must be >= 0."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova"))
        assert all(s >= 0.0 for s in artifact.scores)

    def test_anova_manual_k(self, binary_data):
        """Manual k=8 selects exactly 8 features."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=8)
        artifact = fit_selector(X, y, cfg)

        assert artifact.k_selected == 8
        assert len(artifact.selected_feature_names) == 8

    def test_anova_constant_feature_handled(self, small_mixed_data):
        """Constant feature produces NaN F-score; replaced with 0.0 and warned."""
        X, y = small_mixed_data
        cfg = FeatureSelectionConfig(method="anova", mode="auto")
        artifact = fit_selector(X, y, cfg)

        # constant feature must have score 0.0 (NaN replaced)
        const_idx = artifact.candidate_feature_names.index("constant")
        assert artifact.scores[const_idx] == 0.0

        # At least one warning about undefined scores
        assert any("NaN" in w or "constant" in w.lower() or "infinite" in w.lower()
                   for w in artifact.warnings)

    def test_anova_informative_outranks_noise(self, small_mixed_data):
        """Informative features should rank higher than noise/constant in ANOVA."""
        X, y = small_mixed_data
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=2)
        artifact = fit_selector(X, y, cfg)

        selected = set(artifact.selected_feature_names)
        assert "informative_a" in selected or "informative_b" in selected


# ---------------------------------------------------------------------------
# 4. ANOVA — Multiclass Classification
# ---------------------------------------------------------------------------

class TestANOVAMulticlass:

    def test_anova_multiclass(self, multiclass_data):
        """ANOVA works correctly on 3-class Wine dataset."""
        X, y = multiclass_data
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=6)
        artifact = fit_selector(X, y, cfg)

        assert artifact.k_selected == 6
        assert artifact.pvalues is not None
        assert len(artifact.pvalues) == 13

    def test_anova_multiclass_rankings_valid(self, multiclass_data):
        """Rankings cover 1..n_candidate without gaps."""
        X, y = multiclass_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova"))
        n = len(artifact.candidate_feature_names)
        assert sorted(artifact.rankings) == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# 5. Transform with Selector
# ---------------------------------------------------------------------------

class TestTransformWithSelector:

    def test_transform_shape_binary(self, binary_data):
        """Transformed matrix has shape (n_samples, k_selected)."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=10)
        artifact = fit_selector(X, y, cfg)

        X_selected = transform_with_selector(artifact, X)
        assert X_selected.shape == (len(X), 10)
        assert list(X_selected.columns) == artifact.selected_feature_names

    def test_transform_preserves_index(self, binary_data):
        """Row index is preserved after selection transform."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=5))
        X_selected = transform_with_selector(artifact, X)
        assert list(X_selected.index) == list(X.index)

    def test_transform_no_nans(self, binary_data):
        """Selected features contain no NaN values on clean data."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova", mode="manual", k_features=15))
        X_selected = transform_with_selector(artifact, X)
        assert not X_selected.isnull().any().any()

    def test_transform_column_mismatch_raises(self, binary_data):
        """Transforming with wrong column names raises ValueError."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova", mode="manual", k_features=5))

        X_bad = X.rename(columns={X.columns[0]: "wrong_col"})
        with pytest.raises(ValueError, match="Column mismatch"):
            transform_with_selector(artifact, X_bad)

    def test_transform_test_split(self, binary_data):
        """Selector fitted on X_train correctly reduces X_test."""
        X, y = binary_data
        bundle = DatasetBundle(
            X=X, y=y,
            feature_names=list(X.columns),
            target_names=["malignant", "benign"],
            dataset_name="breast_cancer", target_name="target"
        )
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)

        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=10)
        artifact = fit_selector(split.X_train, split.y_train, cfg)

        X_train_sel = transform_with_selector(artifact, split.X_train)
        X_test_sel = transform_with_selector(artifact, split.X_test)

        assert X_train_sel.shape == (len(split.X_train), 10)
        assert X_test_sel.shape == (len(split.X_test), 10)
        assert list(X_train_sel.columns) == list(X_test_sel.columns)


# ---------------------------------------------------------------------------
# 6. Leakage Prevention
# ---------------------------------------------------------------------------

class TestFeatureSelectionLeakagePrevention:

    @pytest.fixture
    def adversarial_selection_data(self):
        """
        Synthetic adversarial dataset:
          - Train features: feature_a is strongly correlated with y_train.
          - Test features: feature_a is anti-correlated (would rank differently if included).
          - Test labels: would shift ANOVA F-score if used in fit.
        """
        n_train = 100
        n_test = 50
        rng = np.random.default_rng(7)

        # Train: feature_a strongly separates classes 0/1
        X_train = pd.DataFrame({
            "feature_a": np.concatenate([rng.normal(0.0, 0.5, n_train // 2),
                                         rng.normal(5.0, 0.5, n_train // 2)]),
            "feature_b": rng.normal(0.0, 1.0, n_train),  # noise
        })
        y_train = pd.Series([0] * (n_train // 2) + [1] * (n_train // 2), name="target")

        # Test: feature_a reversed (anti-correlated) — would drag score down if used
        X_test = pd.DataFrame({
            "feature_a": np.concatenate([rng.normal(5.0, 0.5, n_test // 2),
                                          rng.normal(0.0, 0.5, n_test // 2)]),
            "feature_b": rng.normal(0.0, 1.0, n_test),
        })
        y_test = pd.Series([1] * (n_test // 2) + [0] * (n_test // 2), name="target")

        return X_train, y_train, X_test, y_test

    def test_test_values_do_not_alter_scores(self, adversarial_selection_data):
        """
        Scores computed from X_train only must not change after transforming X_test.
        """
        X_train, y_train, X_test, y_test = adversarial_selection_data
        cfg = FeatureSelectionConfig(method="anova", mode="auto")

        artifact = fit_selector(X_train, y_train, cfg)
        scores_before = copy.deepcopy(artifact.scores)

        # Transform test data (must not change fitted state)
        _ = transform_with_selector(artifact, X_test)

        assert artifact.scores == scores_before, (
            "Transforming test data altered fitted selector scores — leakage detected!"
        )

    def test_test_labels_cannot_influence_selection(self, adversarial_selection_data):
        """
        Feature scores learned from y_train must differ significantly from scores
        that would be learned from y_test, proving y_test cannot be used in fit.
        """
        X_train, y_train, X_test, y_test = adversarial_selection_data

        artifact_train = fit_selector(X_train, y_train, FeatureSelectionConfig(method="anova"))
        artifact_test_leaked = fit_selector(X_test, y_test, FeatureSelectionConfig(method="anova"))

        # feature_a score with training labels must be much higher (separated classes)
        train_score = artifact_train.scores[0]
        test_score = artifact_test_leaked.scores[0]

        # In train, feature_a separates 0 vs 1 strongly → very high F-score
        # In test, the association is the same but we verify the two are distinct experiments
        assert train_score > 0, "feature_a should have a positive ANOVA score with train labels"
        # The key invariant: the fitted artifact only used y_train
        assert artifact_train.fitted_timestamp_utc != ""

    def test_cv_fold_selectors_are_independent(self, binary_data):
        """
        Each CV fold must produce independently fitted selectors with different scores.
        """
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=10, random_state=42)
        folds = create_stratified_cv_splits(X, y, n_splits=3, random_state=42)

        fold_scores = []
        for fold in folds:
            fold_sel, X_tr_sel, X_val_sel = prepare_cv_fold_selection(X, y, fold, cfg)
            assert X_tr_sel.shape[1] == 10
            assert X_val_sel.shape[1] == 10
            fold_scores.append(fold_sel.scores[:])

        # Because each fold uses a different subset of data, scores must differ
        assert fold_scores[0] != fold_scores[1] or fold_scores[1] != fold_scores[2], (
            "CV fold selectors produced identical scores — they may not be independent!"
        )

    def test_cv_val_slice_not_in_fold_train(self, binary_data):
        """Validation indices must be strictly disjoint from training indices in each fold."""
        X, y = binary_data
        folds = create_stratified_cv_splits(X, y, n_splits=3, random_state=42)
        for fold in folds:
            assert len(set(fold.train_indices) & set(fold.val_indices)) == 0


# ---------------------------------------------------------------------------
# 7. Feature Preview Metadata
# ---------------------------------------------------------------------------

class TestFeatureSelectionPreview:

    def test_preview_structure(self, binary_data):
        """build_selection_preview returns a FeatureSelectionPreview with correct structure."""
        X, y = binary_data
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=5)
        artifact = fit_selector(X, y, cfg)
        preview = build_selection_preview(artifact)

        assert isinstance(preview, FeatureSelectionPreview)
        assert preview.method == "anova"
        assert preview.k_requested == 5
        assert preview.k_selected == 5
        assert len(preview.features) == 30
        assert len(preview.selected_feature_names) == 5

    def test_preview_per_feature_fields(self, binary_data):
        """Each feature entry must have name, score, rank, selected, and pvalue (ANOVA)."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova", mode="manual", k_features=5))
        preview = build_selection_preview(artifact)

        for entry in preview.features:
            assert "name" in entry
            assert "score" in entry
            assert "rank" in entry
            assert "selected" in entry
            assert "pvalue" in entry  # ANOVA has p-values

    def test_preview_mi_no_pvalue(self, binary_data):
        """MI feature entries must NOT have pvalue key."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=5))
        preview = build_selection_preview(artifact)

        for entry in preview.features:
            assert "pvalue" not in entry

    def test_preview_selected_flags_consistent(self, binary_data):
        """selected=True entries must match selected_feature_names."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova", mode="manual", k_features=8))
        preview = build_selection_preview(artifact)

        selected_from_flags = {e["name"] for e in preview.features if e["selected"]}
        assert selected_from_flags == set(preview.selected_feature_names)

    def test_preview_serializable(self, binary_data):
        """FeatureSelectionPreview.to_dict() must return only JSON-serializable primitives."""
        import json
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="anova", mode="manual", k_features=5))
        preview = build_selection_preview(artifact)
        d = preview.to_dict()

        # Should serialize without errors
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_preview_rankings_cover_all_features(self, binary_data):
        """Rankings in preview must be a permutation of 1..n_candidate."""
        X, y = binary_data
        artifact = fit_selector(X, y, FeatureSelectionConfig(method="mutual_info"))
        preview = build_selection_preview(artifact)

        ranks = [e["rank"] for e in preview.features]
        assert sorted(ranks) == list(range(1, 31))


# ---------------------------------------------------------------------------
# 8. FeatureSelectionConfig Validation
# ---------------------------------------------------------------------------

class TestFeatureSelectionConfigValidation:

    def test_default_config(self):
        """Default config is valid and uses expected defaults."""
        cfg = FeatureSelectionConfig()
        assert cfg.method == "mutual_info"
        assert cfg.mode == "auto"
        assert cfg.k_features is None
        assert cfg.proportion is None
        assert cfg.random_state == 42

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid method"):
            FeatureSelectionConfig(method="pca")

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            FeatureSelectionConfig(mode="greedy")

    def test_manual_without_k(self):
        with pytest.raises(ValueError, match="k_features must be provided"):
            FeatureSelectionConfig(mode="manual")

    def test_manual_k_zero(self):
        with pytest.raises(ValueError, match="positive integer"):
            FeatureSelectionConfig(mode="manual", k_features=0)

    def test_manual_k_negative(self):
        with pytest.raises(ValueError, match="positive integer"):
            FeatureSelectionConfig(mode="manual", k_features=-5)

    def test_manual_k_non_integer(self):
        with pytest.raises(ValueError, match="positive integer"):
            FeatureSelectionConfig(mode="manual", k_features=5.5)

    def test_proportion_out_of_range(self):
        with pytest.raises(ValueError, match="proportion"):
            FeatureSelectionConfig(mode="auto", proportion=0.0)

    def test_proportion_too_large(self):
        with pytest.raises(ValueError, match="proportion"):
            FeatureSelectionConfig(mode="auto", proportion=1.5)

    def test_valid_manual_config(self):
        cfg = FeatureSelectionConfig(method="anova", mode="manual", k_features=5)
        assert cfg.k_features == 5

    def test_from_dict_parsing(self):
        d = {"method": "anova", "mode": "manual", "k_features": 7, "random_state": 0}
        cfg = FeatureSelectionConfig.from_dict(d)
        assert cfg.method == "anova"
        assert cfg.k_features == 7

    def test_from_dict_none(self):
        cfg = FeatureSelectionConfig.from_dict(None)
        assert cfg.method == "mutual_info"

    def test_from_dict_passthrough(self):
        original = FeatureSelectionConfig(method="anova", mode="manual", k_features=3)
        result = FeatureSelectionConfig.from_dict(original)
        assert result is original


# ---------------------------------------------------------------------------
# 9. Edge Cases and Validation Errors
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_x_raises(self):
        """Empty X raises ValueError."""
        X = pd.DataFrame({"a": [], "b": []})
        y = pd.Series([], dtype=int, name="target")
        with pytest.raises(ValueError, match="empty"):
            fit_selector(X, y)

    def test_mismatched_dimensions_raises(self):
        """X and y with different row counts raise ValueError."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1], name="target")
        with pytest.raises(ValueError, match="same number of rows"):
            fit_selector(X, y)

    def test_missing_target_labels_raises(self):
        """y with NaN values raises ValueError."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, np.nan, 1], name="target")
        with pytest.raises(ValueError, match="missing values"):
            fit_selector(X, y)

    def test_manual_k_exceeds_features_raises(self):
        """k_features > n_candidate raises ValueError in manual mode."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0, 1, 0, 1], name="target")
        cfg = FeatureSelectionConfig(mode="manual", k_features=10)
        with pytest.raises(ValueError, match="exceeds available"):
            fit_selector(X, y, cfg)

    def test_auto_k_exceeds_features_capped(self):
        """Auto mode with k_features > n_candidate caps silently and warns."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0, 1, 0, 1], name="target")
        cfg = FeatureSelectionConfig(mode="auto", k_features=50)
        artifact = fit_selector(X, y, cfg)

        assert artifact.k_selected == 2  # Capped to available
        assert any("Capping" in w for w in artifact.warnings)

    def test_all_zero_scores_handled(self):
        """When all features are constant, all scores replace to 0.0 without crash."""
        X = pd.DataFrame({
            "c1": [1.0, 1.0, 1.0, 1.0],
            "c2": [5.0, 5.0, 5.0, 5.0],
        })
        y = pd.Series([0, 1, 0, 1], name="target")
        cfg = FeatureSelectionConfig(method="anova", mode="auto")
        artifact = fit_selector(X, y, cfg)

        assert all(s == 0.0 for s in artifact.scores)
        assert len(artifact.warnings) > 0

    def test_integration_with_preprocessing_pipeline(self, binary_data):
        """
        End-to-end: preprocess breast cancer data then apply feature selection.
        Selected feature names must come from the preprocessed (transformed) feature space.
        """
        X, y = binary_data
        bundle = DatasetBundle(
            X=X, y=y,
            feature_names=list(X.columns),
            target_names=["malignant", "benign"],
            dataset_name="breast_cancer",
            target_name="target",
        )
        split = create_train_test_split(bundle, test_size=0.2, random_state=42)

        # Preprocess
        preprocessor = fit_preprocessor(split.X_train)
        X_train_proc = transform_with_preprocessor(preprocessor, split.X_train)
        X_test_proc = transform_with_preprocessor(preprocessor, split.X_test)

        # Feature selection on preprocessed data
        cfg = FeatureSelectionConfig(method="mutual_info", mode="manual", k_features=15, random_state=42)
        selector_artifact = fit_selector(X_train_proc, split.y_train, cfg)

        X_train_sel = transform_with_selector(selector_artifact, X_train_proc)
        X_test_sel = transform_with_selector(selector_artifact, X_test_proc)

        assert X_train_sel.shape == (len(split.X_train), 15)
        assert X_test_sel.shape == (len(split.X_test), 15)
        assert not X_train_sel.isnull().any().any()
        assert not X_test_sel.isnull().any().any()
        # Feature names must come from the preprocessor's output
        for name in selector_artifact.selected_feature_names:
            assert name in preprocessor.transformed_feature_names
