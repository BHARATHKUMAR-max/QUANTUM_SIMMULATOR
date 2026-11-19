"""
tests/test_csv_dataset_loading.py
=================================
Automated unit and integration tests for CSV dataset ingestion.

Verifies:
  1. Numerical binary classification CSV loading.
  2. Numerical multiclass classification CSV loading.
  3. Mixed numerical, categorical, and boolean feature CSV loading.
  4. Standard UTF-8 and UTF-8 with BOM encodings.
  5. Target isolation from feature matrix (zero target leakage).
  6. Preservation of user-provided original string and numeric target labels.
  7. Semantic feature type inference and metadata completeness.
  8. Generic load_dataset() interface with CSV parameters.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd

from backend.data.dataset_loader import load_csv_dataset, load_dataset
from backend.data.schemas import DatasetBundle


@pytest.fixture
def temp_dir():
    """Create a temporary directory for synthetic test CSV files."""
    with tempfile.TemporaryDirectory() as tmp_path:
        yield Path(tmp_path)


def test_load_numerical_binary_csv(temp_dir):
    """Verify loading of a pure numerical binary classification CSV."""
    csv_file = temp_dir / "binary_data.csv"
    
    n_rows = 25
    df = pd.DataFrame({
        "feature_a": np.linspace(1.0, 10.0, n_rows),
        "feature_b": np.linspace(20.0, 50.0, n_rows),
        "feature_c": np.random.default_rng(42).standard_normal(n_rows),
        "diagnosis": ["malignant" if i % 2 == 0 else "benign" for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False, encoding="utf-8")
    
    bundle = load_csv_dataset(file_path=csv_file, target_column="diagnosis")
    
    # 1. Structural dimensions
    assert isinstance(bundle, DatasetBundle)
    assert bundle.X.shape == (25, 3)
    assert len(bundle.y) == 25
    assert bundle.dataset_name == "binary_data"
    assert bundle.source_type == "csv"
    assert bundle.target_name == "diagnosis"
    
    # 2. Target isolation (no leakage)
    assert "diagnosis" not in bundle.X.columns
    assert "diagnosis" not in bundle.feature_names
    assert bundle.feature_names == ["feature_a", "feature_b", "feature_c"]
    
    # 3. Target label preservation
    assert sorted(bundle.target_classes) == ["benign", "malignant"]
    assert set(bundle.y.unique()) == {"malignant", "benign"}
    
    # 4. Feature typing
    assert bundle.feature_types["feature_a"] == "numerical"
    assert bundle.feature_types["feature_b"] == "numerical"
    assert bundle.feature_types["feature_c"] == "numerical"
    
    # 5. Metadata verification
    assert bundle.metadata["n_samples"] == 25
    assert bundle.metadata["n_features"] == 3
    assert bundle.metadata["n_classes"] == 2
    assert bundle.metadata["target_column"] == "diagnosis"


def test_load_numerical_multiclass_csv(temp_dir):
    """Verify loading of a multiclass numerical CSV."""
    csv_file = temp_dir / "multiclass_data.csv"
    
    n_rows = 30
    classes = ["cultivar_A", "cultivar_B", "cultivar_C"]
    df = pd.DataFrame({
        "flavanoids": np.linspace(0.5, 4.0, n_rows),
        "alcohol": np.linspace(11.0, 15.0, n_rows),
        "magnesium": np.linspace(70.0, 140.0, n_rows),
        "cultivar": [classes[i % 3] for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False, encoding="utf-8")
    
    bundle = load_csv_dataset(file_path=csv_file, target_column="cultivar", dataset_name="custom_wine_multiclass")
    
    assert bundle.dataset_name == "custom_wine_multiclass"
    assert bundle.X.shape == (30, 3)
    assert len(bundle.target_classes) == 3
    assert sorted(bundle.target_classes) == ["cultivar_A", "cultivar_B", "cultivar_C"]
    assert "cultivar" not in bundle.X.columns
    assert bundle.metadata["n_classes"] == 3


def test_load_mixed_features_csv(temp_dir):
    """Verify loading with mixed numerical, categorical, and boolean features."""
    csv_file = temp_dir / "mixed_data.csv"
    
    n_rows = 20
    df = pd.DataFrame({
        "age": np.linspace(18, 65, n_rows),
        "income": np.linspace(30000, 120000, n_rows),
        "education": ["bachelors", "masters", "phd", "high_school"] * 5,
        "is_active_member": [True, False] * 10,
        "churn": [0, 1] * 10,
    })
    df.to_csv(csv_file, index=False, encoding="utf-8")
    
    bundle = load_csv_dataset(file_path=csv_file, target_column="churn")
    
    assert bundle.X.shape == (20, 4)
    assert bundle.feature_types["age"] == "numerical"
    assert bundle.feature_types["income"] == "numerical"
    assert bundle.feature_types["education"] == "categorical"
    assert bundle.feature_types["is_active_member"] == "boolean"
    
    # Text categories preserved as strings
    assert pd.api.types.is_string_dtype(bundle.X["education"]) or bundle.X["education"].dtype == object
    assert bundle.X["education"].iloc[0] == "bachelors"
    assert "churn" not in bundle.X.columns


def test_load_utf8_bom_csv(temp_dir):
    """Verify that UTF-8 with BOM (Byte Order Mark) is parsed cleanly without header corruption."""
    csv_file = temp_dir / "bom_encoded.csv"
    
    n_rows = 18
    df = pd.DataFrame({
        "first_metric": np.linspace(1.0, 5.0, n_rows),
        "second_metric": np.linspace(10.0, 20.0, n_rows),
        "label": ["active" if i % 2 == 0 else "inactive" for i in range(n_rows)],
    })
    # Save explicitly with UTF-8 BOM
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    bundle = load_csv_dataset(file_path=csv_file, target_column="label")
    
    # Verify BOM (\ufeff) did not corrupt the first column header
    assert "first_metric" in bundle.feature_names
    assert bundle.feature_names[0] == "first_metric"
    assert not bundle.feature_names[0].startswith("\ufeff")
    assert bundle.X.shape == (18, 2)


def test_generic_load_dataset_csv_interface(temp_dir):
    """Verify that the unified load_dataset('csv', ...) dispatch operates properly."""
    csv_file = temp_dir / "generic_dispatch.csv"
    
    df = pd.DataFrame({
        "feature_1": [float(i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    # Mode 1: name="csv"
    bundle1 = load_dataset(dataset_name="csv", file_path=csv_file, target_column="target")
    assert isinstance(bundle1, DatasetBundle)
    assert bundle1.X.shape == (20, 1)
    
    # Mode 2: custom name with file_path
    bundle2 = load_dataset(dataset_name="my_custom_experiment", file_path=csv_file, target_column="target")
    assert isinstance(bundle2, DatasetBundle)
    assert bundle2.dataset_name == "my_custom_experiment"


def test_load_csv_dataset_validation_failure(temp_dir):
    """Verify that load_csv_dataset raises ValueError on invalid CSV datasets."""
    csv_file = temp_dir / "fewer_than_15_rows.csv"
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "target": [0, 1, 0],
    })
    df.to_csv(csv_file, index=False)
    
    # Should reject due to fewer than 15 rows (FILE_ROW_COUNT_LOW)
    with pytest.raises(ValueError, match="FILE_ROW_COUNT_LOW"):
        load_csv_dataset(file_path=csv_file, target_column="target")
