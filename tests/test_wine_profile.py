"""
tests/test_wine_profile.py
==========================
Automated test suite for Day 5: Wine dataset loading, multi-class profiling,
correlation analysis, and artifact export.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.data.dataset_loader import load_wine_data, load_dataset
from backend.data.dataset_profiler import (
    export_profile,
    generate_visualizations,
    profile_dataset,
)
from backend.data.schemas import DatasetBundle, DatasetProfile


# ---------------------------------------------------------------------------
# Wine Dataset Loader Tests
# ---------------------------------------------------------------------------

def test_load_wine_shape_and_features():
    """Verify loading produces exactly 178 samples and 13 numerical features."""
    bundle = load_wine_data()
    assert isinstance(bundle, DatasetBundle)
    assert bundle.X.shape == (178, 13)
    assert len(bundle.y) == 178
    assert len(bundle.feature_names) == 13
    assert list(bundle.X.columns) == bundle.feature_names
    assert bundle.dataset_name == "wine"


def test_wine_target_mapping_and_classes():
    """Verify target classes are 'class_0', 'class_1', 'class_2' with correct sample counts."""
    bundle = load_wine_data()
    assert bundle.target_names == ["class_0", "class_1", "class_2"]
    
    # Check sample counts per class: 59, 71, 48
    counts = bundle.y.value_counts().sort_index()
    assert counts[0] == 59
    assert counts[1] == 71
    assert counts[2] == 48
    assert set(bundle.y.unique()) == {0, 1, 2}


def test_wine_no_target_leakage():
    """Verify target column and diagnostic labels are completely absent from feature matrix X."""
    bundle = load_wine_data()
    forbidden_terms = {"target", "class", "cultivar"}
    for col in bundle.X.columns:
        assert col.lower() not in forbidden_terms
        assert not col.startswith("target_")
    for col in bundle.X.columns:
        assert not bundle.X[col].equals(bundle.y)


def test_wine_generic_loader_interface():
    """Verify generic load_dataset('wine') entry point works identically."""
    bundle = load_dataset("wine")
    assert bundle.dataset_name == "wine"
    assert bundle.X.shape == (178, 13)
    assert len(bundle.y) == 178


# ---------------------------------------------------------------------------
# Profiler Engine Tests on Wine (Multiclass Target)
# ---------------------------------------------------------------------------

def test_wine_profile_completeness_and_metrics():
    """Verify all statistical profiling sections for the Wine dataset."""
    bundle = load_wine_data()
    profile = profile_dataset(bundle)
    
    assert isinstance(profile, DatasetProfile)
    assert profile.n_samples == 178
    assert profile.n_features == 13
    assert profile.n_classes == 3
    
    # Class distribution checks
    c_dist = profile.class_distribution
    assert c_dist.counts["class_0"] == 59
    assert c_dist.counts["class_1"] == 71
    assert c_dist.counts["class_2"] == 48
    assert c_dist.percentages["class_0"] == pytest.approx(33.15, abs=0.1)
    assert c_dist.percentages["class_1"] == pytest.approx(39.89, abs=0.1)
    assert c_dist.percentages["class_2"] == pytest.approx(26.97, abs=0.1)
    assert c_dist.majority_class == "class_1"
    assert c_dist.minority_class == "class_2"
    assert c_dist.imbalance_ratio == pytest.approx(1.479, abs=0.01)
    
    # Missing values and duplicates checks
    assert profile.missing_values.total_missing == 0
    assert len(profile.missing_values.columns_with_missing) == 0
    assert profile.duplicates.total_duplicate_rows == 0
    assert not profile.duplicates.has_duplicate_features
    
    # Descriptive statistics checks
    stats = profile.descriptive_statistics.features
    assert len(stats) == 13
    assert "alcohol" in stats
    assert "proline" in stats
    
    # Check variance extremes
    assert profile.descriptive_statistics.largest_variability_features[0]["feature"] == "proline"
    assert profile.descriptive_statistics.smallest_variability_features[0]["feature"] == "nonflavanoid_phenols"


def test_wine_correlation_matrix_and_collinear_detection():
    """Verify Pearson correlation matrix dimensions and that no pairs exceed 0.95."""
    bundle = load_wine_data()
    profile = profile_dataset(bundle, correlation_threshold=0.95)
    
    corr = profile.correlation
    # Wine has no pairs with |r| >= 0.95
    assert len(corr.highly_correlated_pairs) == 0
    
    # Verify strongest positive correlation is total_phenols & flavanoids
    top_pos = corr.strongest_positive_correlations[0]
    assert {top_pos["feature_1"], top_pos["feature_2"]} == {"total_phenols", "flavanoids"}
    assert top_pos["pearson_correlation"] == pytest.approx(0.8646, abs=0.01)


def test_wine_profile_json_serialization():
    """Verify Wine DatasetProfile serializes into valid JSON."""
    bundle = load_wine_data()
    profile = profile_dataset(bundle)
    profile_dict = profile.to_dict()
    
    json_str = json.dumps(profile_dict, indent=2)
    assert json_str is not None
    reloaded = json.loads(json_str)
    
    assert reloaded["dataset_name"] == "wine"
    assert reloaded["n_samples"] == 178
    assert reloaded["n_features"] == 13
    assert reloaded["n_classes"] == 3
    assert "class_distribution" in reloaded
    assert "correlation" in reloaded
    assert "descriptive_statistics" in reloaded


def test_export_wine_profile_artifacts(tmp_path):
    """Verify all report artifacts (CSV, JSON, Markdown, PNGs) are generated for Wine."""
    bundle = load_wine_data()
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
        
    # Verify CSV contents
    stats_df = pd.read_csv(exported["descriptive_statistics.csv"])
    assert len(stats_df) == 13
    assert "mean" in stats_df.columns
    assert "variance" in stats_df.columns
    
    cls_df = pd.read_csv(exported["class_distribution.csv"])
    assert len(cls_df) == 3
    assert "sample_count" in cls_df.columns
    
    corr_df = pd.read_csv(exported["correlation_matrix.csv"], index_col=0)
    assert corr_df.shape == (13, 13)
