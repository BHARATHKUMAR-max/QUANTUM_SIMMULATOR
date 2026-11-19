"""
tests/test_breast_cancer_profile.py
===================================
Automated test suite for Day 4: Breast Cancer Wisconsin dataset loading,
profiling, synthetic edge-case verification, and artifact export.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.data.dataset_loader import load_breast_cancer_data, load_dataset
from backend.data.dataset_profiler import (
    export_profile,
    generate_visualizations,
    profile_dataset,
)
from backend.data.schemas import DatasetBundle, DatasetProfile


# ---------------------------------------------------------------------------
# Breast Cancer Wisconsin Dataset Loader Tests
# ---------------------------------------------------------------------------

def test_load_breast_cancer_shape_and_features():
    """Verify loading produces exactly 569 samples and 30 numerical features."""
    bundle = load_breast_cancer_data()
    assert isinstance(bundle, DatasetBundle)
    assert bundle.X.shape == (569, 30)
    assert len(bundle.y) == 569
    assert len(bundle.feature_names) == 30
    assert list(bundle.X.columns) == bundle.feature_names
    assert bundle.dataset_name == "breast_cancer"


def test_target_mapping_and_classes():
    """Verify target classes are 'malignant' (0) and 'benign' (1) with correct counts."""
    bundle = load_breast_cancer_data()
    assert bundle.target_names == ["malignant", "benign"]
    
    # Check sample counts per class
    counts = bundle.y.value_counts()
    assert counts[0] == 212  # malignant
    assert counts[1] == 357  # benign
    assert set(bundle.y.unique()) == {0, 1}


def test_no_target_leakage():
    """Verify target column and diagnostic labels are completely absent from feature matrix X."""
    bundle = load_breast_cancer_data()
    forbidden_terms = {"target", "diagnosis", "class", "malignant", "benign"}
    for col in bundle.X.columns:
        assert col.lower() not in forbidden_terms
        assert not col.startswith("target_")
    # Verify no exact identity between any X column and y
    for col in bundle.X.columns:
        assert not bundle.X[col].equals(bundle.y)


def test_generic_loader_interface():
    """Verify load_dataset dispatcher works for breast_cancer and wine, and rejects others."""
    bundle = load_dataset("breast_cancer")
    assert bundle.dataset_name == "breast_cancer"
    assert bundle.X.shape == (569, 30)
    
    # Wine is supported as of Day 5
    wine_bundle = load_dataset("wine")
    assert wine_bundle.dataset_name == "wine"
    assert wine_bundle.X.shape == (178, 13)
        
    # Unknown dataset raises ValueError
    with pytest.raises(ValueError, match="Unknown dataset identifier"):
        load_dataset("unknown_dataset_xyz")


# ---------------------------------------------------------------------------
# Profiler Engine Tests on Breast Cancer
# ---------------------------------------------------------------------------

def test_profile_completeness_and_metrics():
    """Verify all statistical profiling sections are populated with valid values."""
    bundle = load_breast_cancer_data()
    profile = profile_dataset(bundle)
    
    assert isinstance(profile, DatasetProfile)
    assert profile.n_samples == 569
    assert profile.n_features == 30
    assert profile.n_classes == 2
    
    # Class distribution checks
    c_dist = profile.class_distribution
    assert c_dist.counts["benign"] == 357
    assert c_dist.counts["malignant"] == 212
    assert c_dist.percentages["benign"] == pytest.approx(62.74, abs=0.1)
    assert c_dist.percentages["malignant"] == pytest.approx(37.26, abs=0.1)
    assert c_dist.majority_class == "benign"
    assert c_dist.minority_class == "malignant"
    assert c_dist.imbalance_ratio == pytest.approx(1.684, abs=0.01)
    
    # Missing values and duplicates checks
    assert profile.missing_values.total_missing == 0
    assert len(profile.missing_values.columns_with_missing) == 0
    assert profile.duplicates.total_duplicate_rows == 0
    assert not profile.duplicates.has_duplicate_features
    
    # Descriptive statistics checks
    stats = profile.descriptive_statistics.features
    assert len(stats) == 30
    assert "mean radius" in stats
    mean_radius_stat = stats["mean radius"]
    assert mean_radius_stat.count == 569
    assert mean_radius_stat.mean == pytest.approx(14.127, abs=0.01)
    assert mean_radius_stat.min < mean_radius_stat.median < mean_radius_stat.max
    assert mean_radius_stat.range == pytest.approx(mean_radius_stat.max - mean_radius_stat.min)


def test_correlation_matrix_and_collinear_detection():
    """Verify Pearson correlation matrix dimensions and identification of highly correlated pairs."""
    bundle = load_breast_cancer_data()
    profile = profile_dataset(bundle, correlation_threshold=0.95)
    
    corr = profile.correlation
    assert len(corr.highly_correlated_pairs) == 15
    
    # Verify top pair is mean radius & mean perimeter (r > 0.99)
    top_pair = corr.highly_correlated_pairs[0]
    assert {top_pair["feature_1"], top_pair["feature_2"]} == {"mean radius", "mean perimeter"}
    assert top_pair["pearson_correlation"] > 0.99


# ---------------------------------------------------------------------------
# Synthetic Edge-Case Tests (Testing Profiler Robustness)
# ---------------------------------------------------------------------------

def test_synthetic_profiler_edge_cases():
    """
    Test generic profiler functions against synthetic data containing:
      - missing values
      - duplicate rows
      - constant column
      - near-constant column
      - duplicate columns
      - known perfect collinearity
    """
    # Create controlled synthetic DataFrame
    np.random.seed(42)
    n = 50
    feat_a = np.random.normal(10, 2, size=n)
    feat_b = feat_a * 3.0  # Perfect linear correlation (r = 1.0)
    feat_c = np.ones(n) * 7.5  # Constant column (var = 0)
    feat_d = np.random.normal(0, 0.001, size=n)  # Near-constant column (var ~ 1e-6)
    feat_e = feat_a.copy()  # Duplicate column identical to feat_a
    
    df_x = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "feat_d": feat_d,
        "feat_e": feat_e,
    })
    
    # Introduce missing values in feat_a
    df_x.loc[0:2, "feat_a"] = np.nan  # 3 missing values
    
    # Ensure feat_e is an exact duplicate of feat_a (including NaNs)
    df_x["feat_e"] = df_x["feat_a"].copy()
    
    # Introduce duplicate rows (duplicate row 5 into row 6)
    df_x.iloc[6] = df_x.iloc[5]
    
    y = pd.Series([0 if i % 2 == 0 else 1 for i in range(n)], name="target")
    
    synthetic_bundle = DatasetBundle(
        X=df_x,
        y=y,
        feature_names=list(df_x.columns),
        target_names=["class_0", "class_1"],
        dataset_name="synthetic_test",
    )
    
    profile = profile_dataset(synthetic_bundle, low_variance_threshold=1e-4, correlation_threshold=0.95)
    
    # 1. Missing values
    assert profile.missing_values.total_missing == 6  # 3 in feat_a and 3 in feat_e
    assert profile.missing_values.per_column_counts["feat_a"] == 3
    assert profile.missing_values.per_column_counts["feat_e"] == 3
    assert "feat_a" in profile.missing_values.columns_with_missing
    assert "feat_e" in profile.missing_values.columns_with_missing
    
    # 2. Duplicate rows
    assert profile.duplicates.total_duplicate_rows >= 1
    
    # 3. Duplicate columns (feat_a and feat_e)
    assert profile.duplicates.has_duplicate_features is True
    found_dup_group = any(
        "feat_a" in g and "feat_e" in g for g in profile.duplicates.duplicate_feature_groups
    )
    assert found_dup_group is True
    
    # 4. Constant column (feat_c)
    assert "feat_c" in profile.constants.zero_variance_features
    
    # 5. Near-constant column (feat_d)
    low_var_names = [item["feature"] for item in profile.constants.low_variance_features]
    assert "feat_d" in low_var_names
    
    # 6. High correlation (feat_a and feat_b)
    pairs = profile.correlation.highly_correlated_pairs
    assert any(
        {"feat_a", "feat_b"} == {p["feature_1"], p["feature_2"]} and p["pearson_correlation"] > 0.99
        for p in pairs
    )


# ---------------------------------------------------------------------------
# Serialization and Artifact Export Tests
# ---------------------------------------------------------------------------

def test_profile_json_serialization():
    """Verify DatasetProfile serializes into valid JSON according to schema requirements."""
    bundle = load_breast_cancer_data()
    profile = profile_dataset(bundle)
    profile_dict = profile.to_dict()
    
    # Test JSON dump and reload
    json_str = json.dumps(profile_dict, indent=2)
    assert json_str is not None
    reloaded = json.loads(json_str)
    
    assert reloaded["dataset_name"] == "breast_cancer"
    assert reloaded["n_samples"] == 569
    assert reloaded["n_features"] == 30
    assert "class_distribution" in reloaded
    assert "correlation" in reloaded
    assert "descriptive_statistics" in reloaded


def test_export_profile_artifacts(tmp_path):
    """Verify all report artifacts (CSV, JSON, Markdown, PNGs) are written and non-empty."""
    bundle = load_breast_cancer_data()
    profile = profile_dataset(bundle)
    
    exported = export_profile(profile, bundle, tmp_path)
    
    expected_files = [
        "profile.json",
        "descriptive_statistics.csv",
        "class_distribution.csv",
        "correlation_matrix.csv",
        "findings.md",
        "class_distribution.png",
        "feature_distributions.png",
        "correlation_heatmap.png",
        "feature_boxplots.png",
        "feature_variance.png",
    ]
    
    for filename in expected_files:
        assert filename in exported, f"Missing {filename} in export dict."
        filepath = Path(exported[filename])
        assert filepath.exists(), f"File {filepath} was not created."
        assert filepath.stat().st_size > 0, f"File {filepath} is empty."
        
    # Verify CSV content
    stats_df = pd.read_csv(exported["descriptive_statistics.csv"])
    assert len(stats_df) == 30
    assert "mean" in stats_df.columns
    assert "variance" in stats_df.columns
    
    cls_df = pd.read_csv(exported["class_distribution.csv"])
    assert len(cls_df) == 2
    assert "sample_count" in cls_df.columns
    
    corr_df = pd.read_csv(exported["correlation_matrix.csv"], index_col=0)
    assert corr_df.shape == (30, 30)
