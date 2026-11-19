"""
tests/test_generic_profiler.py
==============================
Automated test suite verifying the generic profiling and export engine.

Verifies:
  1. Built-in Breast Cancer dataset profiling.
  2. Built-in Wine dataset profiling.
  3. User CSV binary classification profiling.
  4. User CSV multiclass classification profiling.
  5. Mixed numerical and categorical feature profiling.
  6. Robustness against categorical columns in correlation and variance analyses.
  7. Strict JSON-serializability of DatasetProfile objects.
  8. Export artifacts completeness across built-in and synthetic CSV profiles.
"""

import json
from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd

from backend.data.dataset_loader import (
    load_breast_cancer_data,
    load_csv_dataset,
    load_wine_data,
)
from backend.data.dataset_profiler import (
    export_profile,
    profile_dataset,
)
from backend.data.schemas import DatasetBundle, DatasetProfile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for synthetic datasets and test exports."""
    with tempfile.TemporaryDirectory() as tmp_path:
        yield Path(tmp_path)


def test_builtin_breast_cancer_profiling():
    """Verify generic profiling engine on the Breast Cancer benchmark."""
    bundle = load_breast_cancer_data()
    profile = profile_dataset(bundle)
    
    assert isinstance(profile, DatasetProfile)
    assert profile.dataset_name == "breast_cancer"
    assert profile.n_samples == 569
    assert profile.n_features == 30
    assert profile.n_classes == 2
    assert profile.class_distribution.majority_class == "benign"
    assert profile.missing_values.total_missing == 0
    assert len(profile.descriptive_statistics.features) == 30
    assert len(profile.correlation.highly_correlated_pairs) > 0


def test_builtin_wine_profiling():
    """Verify generic profiling engine on the Wine multiclass benchmark."""
    bundle = load_wine_data()
    profile = profile_dataset(bundle)
    
    assert isinstance(profile, DatasetProfile)
    assert profile.dataset_name == "wine"
    assert profile.n_samples == 178
    assert profile.n_features == 13
    assert profile.n_classes == 3
    assert profile.class_distribution.imbalance_ratio < 2.0
    assert profile.missing_values.total_missing == 0
    assert len(profile.descriptive_statistics.features) == 13


def test_csv_binary_profiling(temp_dir):
    """Verify generic profiling on a synthetic CSV binary classification bundle."""
    csv_file = temp_dir / "synthetic_binary.csv"
    n_rows = 24
    df = pd.DataFrame({
        "var_1": np.linspace(1.0, 10.0, n_rows),
        "var_2": np.linspace(50.0, 100.0, n_rows),
        "label": ["positive" if i % 2 == 0 else "negative" for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False)
    
    bundle = load_csv_dataset(csv_file, target_column="label")
    profile = profile_dataset(bundle)
    
    assert profile.n_samples == 24
    assert profile.n_features == 2
    assert profile.n_classes == 2
    assert "positive" in profile.class_distribution.counts
    assert "negative" in profile.class_distribution.counts
    assert len(profile.descriptive_statistics.features) == 2


def test_csv_multiclass_profiling(temp_dir):
    """Verify generic profiling on a synthetic CSV multiclass classification bundle."""
    csv_file = temp_dir / "synthetic_multiclass.csv"
    n_rows = 30
    df = pd.DataFrame({
        "x1": np.linspace(0.0, 5.0, n_rows),
        "x2": np.linspace(10.0, 20.0, n_rows),
        "x3": np.random.default_rng(123).standard_normal(n_rows),
        "tier": [f"tier_{i % 3}" for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False)
    
    bundle = load_csv_dataset(csv_file, target_column="tier")
    profile = profile_dataset(bundle)
    
    assert profile.n_samples == 30
    assert profile.n_features == 3
    assert profile.n_classes == 3
    assert len(profile.target_names) == 3


def test_mixed_features_profiling_and_categorical_handling(temp_dir):
    """
    Verify that mixed categorical and numerical features are profiled without
    attempting numerical calculations on categorical text columns.
    """
    csv_file = temp_dir / "mixed_profile.csv"
    n_rows = 20
    df = pd.DataFrame({
        "num_measure": np.linspace(1.0, 100.0, n_rows),
        "category_col": ["alpha", "beta", "gamma", "alpha"] * 5,
        "bool_flag": [True, False] * 10,
        "outcome": [0, 1] * 10,
    })
    df.to_csv(csv_file, index=False)
    
    bundle = load_csv_dataset(csv_file, target_column="outcome")
    profile = profile_dataset(bundle)
    
    # Numerical features should only include 'num_measure'
    assert "num_measure" in profile.descriptive_statistics.features
    assert "category_col" not in profile.descriptive_statistics.features
    assert "bool_flag" not in profile.descriptive_statistics.features
    
    # Categorical features summary should include 'category_col' and 'bool_flag'
    assert "category_col" in profile.descriptive_statistics.categorical_features
    cat_stat = profile.descriptive_statistics.categorical_features["category_col"]
    assert cat_stat.unique_count == 3
    assert cat_stat.top_category == "alpha"
    assert cat_stat.top_frequency == 10
    
    # Correlation analysis must exclude categorical columns
    # With only 1 numerical feature ('num_measure'), pairwise correlation is empty
    assert len(profile.correlation.highly_correlated_pairs) == 0


def test_json_serializability(temp_dir):
    """Verify that any generated DatasetProfile converts cleanly to valid JSON."""
    csv_file = temp_dir / "json_test.csv"
    n_rows = 18
    df = pd.DataFrame({
        "score_a": [float(i) for i in range(n_rows)],
        "score_b": [float(i * 2) for i in range(n_rows)],
        "status": ["pass" if i % 2 == 0 else "fail" for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False)
    
    bundle = load_csv_dataset(csv_file, target_column="status")
    profile = profile_dataset(bundle)
    
    profile_dict = profile.to_dict()
    # Test strict JSON serialization
    serialized_str = json.dumps(profile_dict, indent=2)
    assert len(serialized_str) > 0
    deserialized = json.loads(serialized_str)
    assert deserialized["dataset_name"] == "json_test"
    assert deserialized["n_samples"] == 18


def test_export_profile_artifacts(temp_dir):
    """Verify that export_profile writes all required tables, plots, markdown, and JSON files."""
    csv_file = temp_dir / "export_test.csv"
    n_rows = 20
    df = pd.DataFrame({
        "f1": np.linspace(1.0, 10.0, n_rows),
        "f2": np.linspace(20.0, 30.0, n_rows),
        "cat1": ["north", "south"] * 10,
        "target": [0, 1] * 10,
    })
    df.to_csv(csv_file, index=False)
    
    bundle = load_csv_dataset(csv_file, target_column="target")
    profile = profile_dataset(bundle)
    
    export_out = temp_dir / "export_artifacts"
    files = export_profile(profile, bundle, export_out)
    
    # Check required deliverables
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
        "categorical_statistics.csv",
    ]
    for exp_file in expected_files:
        assert (export_out / exp_file).exists(), f"Missing exported artifact: {exp_file}"
        assert (export_out / exp_file).stat().st_size > 0
