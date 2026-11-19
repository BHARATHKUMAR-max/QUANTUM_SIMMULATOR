"""
tests/test_csv_validation.py
============================
Automated validation test suite for user-uploaded CSV datasets.

Tests all error codes, warning conditions, and edge cases specified in
docs/CSV_VALIDATION_RULES.md.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import pandas as pd

from backend.data.csv_validator import validate_csv_dataset
from backend.data.schemas import CSVValidationResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for synthetic test CSV files."""
    with tempfile.TemporaryDirectory() as tmp_path:
        yield Path(tmp_path)


def test_file_unsupported_extension(temp_dir):
    """Verify FILE_UNSUPPORTED_EXT for non-.csv filenames."""
    bad_file = temp_dir / "data.txt"
    bad_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    
    res = validate_csv_dataset(file_path=bad_file, target_column="c")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FILE_UNSUPPORTED_EXT" for e in res.errors)


def test_file_empty(temp_dir):
    """Verify FILE_EMPTY for 0-byte or whitespace-only files."""
    empty_file = temp_dir / "empty.csv"
    empty_file.write_bytes(b"")
    
    res = validate_csv_dataset(file_path=empty_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FILE_EMPTY" for e in res.errors)

    # Whitespace-only file
    ws_file = temp_dir / "whitespace.csv"
    ws_file.write_text("   \n\n  \t  \n", encoding="utf-8")
    res_ws = validate_csv_dataset(file_path=ws_file, target_column="target")
    assert res_ws.validation_status == "INVALID"
    assert any(e.code == "FILE_EMPTY" for e in res_ws.errors)


def test_file_encoding_error(temp_dir):
    """Verify FILE_ENCODING_ERROR for invalid UTF-8 byte sequences."""
    bad_enc = temp_dir / "bad_encoding.csv"
    # Write invalid UTF-8 sequence (e.g. arbitrary binary / Latin-1 bytes with invalid lead bytes)
    bad_enc.write_bytes(b"col1,col2\n\xff\xfe\xfa\xfb,123\n")
    
    res = validate_csv_dataset(file_path=bad_enc, target_column="col2")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FILE_ENCODING_ERROR" for e in res.errors)


def test_file_parse_error_malformed_csv(temp_dir):
    """Verify FILE_PARSE_ERROR for syntactically malformed CSV content."""
    bad_csv = temp_dir / "malformed.csv"
    bad_csv.write_text('col_a,col_b,target\n1,"unclosed quote,3\n4,5,6\n', encoding="utf-8")
    
    res = validate_csv_dataset(file_path=bad_csv, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FILE_PARSE_ERROR" for e in res.errors)


def test_duplicate_column_headers(temp_dir):
    """Verify COL_DUPLICATE_NAMES when two headers share the same label."""
    dup_file = temp_dir / "duplicate_headers.csv"
    # Write duplicate header names
    dup_file.write_text("feat_1,feat_1,target\n1,2,0\n3,4,1\n" + "1,2,0\n" * 15, encoding="utf-8")
    
    res = validate_csv_dataset(file_path=dup_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "COL_DUPLICATE_NAMES" for e in res.errors)


def test_row_count_low(temp_dir):
    """Verify FILE_ROW_COUNT_LOW when row count < 15."""
    small_file = temp_dir / "small.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(10)],
        "target": [i % 2 for i in range(10)],
    })
    df.to_csv(small_file, index=False)
    
    res = validate_csv_dataset(file_path=small_file, target_column="target", min_rows=15)
    assert res.validation_status == "INVALID"
    assert any(e.code == "FILE_ROW_COUNT_LOW" for e in res.errors)


def test_target_not_specified(temp_dir):
    """Verify TARGET_NOT_SPECIFIED when target_column is None."""
    csv_file = temp_dir / "no_target.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column=None)
    assert res.validation_status == "INVALID"
    assert any(e.code == "TARGET_NOT_SPECIFIED" for e in res.errors)


def test_target_not_found(temp_dir):
    """Verify TARGET_NOT_FOUND when designated target does not exist in CSV."""
    csv_file = temp_dir / "missing_target_col.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="nonexistent_col")
    assert res.validation_status == "INVALID"
    assert any(e.code == "TARGET_NOT_FOUND" for e in res.errors)


def test_target_has_nulls(temp_dir):
    """Verify TARGET_HAS_NULLS when target column contains missing values."""
    csv_file = temp_dir / "target_nulls.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(20)],
        "target": [0 if i % 2 == 0 else np.nan for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "TARGET_HAS_NULLS" for e in res.errors)


def test_target_single_class(temp_dir):
    """Verify TARGET_SINGLE_CLASS when target contains only one unique category."""
    csv_file = temp_dir / "single_class.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(20)],
        "target": ["homogeneous_class"] * 20,
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "TARGET_SINGLE_CLASS" for e in res.errors)


def test_target_high_cardinality(temp_dir):
    """Verify TARGET_HIGH_CARDINALITY when target has continuous values or unique IDs (> 50% unique)."""
    csv_file = temp_dir / "target_high_card.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(30)],
        "target": [f"unique_id_{i}" for i in range(30)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "TARGET_HIGH_CARDINALITY" for e in res.errors)


def test_target_class_imbalance_warning(temp_dir):
    """Verify TARGET_CLASS_IMBALANCE warning when minority class < 20%."""
    csv_file = temp_dir / "imbalanced.csv"
    # 25 samples: 23 class 0 (92%), 2 class 1 (8%)
    df = pd.DataFrame({
        "feat_1": [float(i) for i in range(25)],
        "target": [0] * 23 + [1] * 2,
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "TARGET_CLASS_IMBALANCE" for w in res.warnings)


def test_feat_all_null_error(temp_dir):
    """Verify FEAT_ALL_NULL error when a feature column contains 100% missing values."""
    csv_file = temp_dir / "all_null_feature.csv"
    df = pd.DataFrame({
        "feat_valid": [float(i) for i in range(20)],
        "feat_empty": [np.nan] * 20,
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FEAT_ALL_NULL" for e in res.errors)


def test_feat_has_nulls_warning(temp_dir):
    """Verify FEAT_HAS_NULLS warning when some feature values are missing (< 100%)."""
    csv_file = temp_dir / "partial_nulls.csv"
    df = pd.DataFrame({
        "feat_1": [float(i) if i > 2 else np.nan for i in range(20)],
        "feat_2": [float(i * 2) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "FEAT_HAS_NULLS" for w in res.warnings)
    assert res.missing_value_summary["total_missing_values"] == 3


def test_feat_infinite_values_error(temp_dir):
    """Verify FEAT_INFINITE_VALUES error when numeric column contains inf or -inf."""
    csv_file = temp_dir / "infinite_feature.csv"
    vals = [float(i) for i in range(20)]
    vals[5] = np.inf
    df = pd.DataFrame({
        "feat_inf": vals,
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "INVALID"
    assert any(e.code == "FEAT_INFINITE_VALUES" for e in res.errors)


def test_feat_constant_and_near_constant(temp_dir):
    """Verify FEAT_CONSTANT and FEAT_NEAR_CONSTANT warnings."""
    csv_file = temp_dir / "constant_features.csv"
    df = pd.DataFrame({
        "feat_constant": [42.0] * 20,
        "feat_near_const": [1.0 + (i * 1e-6) for i in range(20)],
        "feat_normal": [float(i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "FEAT_CONSTANT" for w in res.warnings)
    assert any(w.code == "FEAT_NEAR_CONSTANT" for w in res.warnings)


def test_duplicate_rows_warning(temp_dir):
    """Verify DUP_ROWS_DETECTED warning when exact duplicate rows exist."""
    csv_file = temp_dir / "duplicate_rows.csv"
    # Create 16 rows, repeat row 0 five times
    df = pd.DataFrame({
        "feat_1": [1.0] * 5 + [float(i) for i in range(5, 20)],
        "feat_2": [2.0] * 5 + [float(i * 3) for i in range(5, 20)],
        "target": [0] * 5 + [i % 2 for i in range(5, 20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "DUP_ROWS_DETECTED" for w in res.warnings)
    assert res.duplicate_summary["duplicate_row_count"] >= 4


def test_duplicate_cols_warning(temp_dir):
    """Verify DUP_COLS_DETECTED warning when two feature columns have identical values."""
    csv_file = temp_dir / "duplicate_cols.csv"
    vals = [float(i) for i in range(20)]
    df = pd.DataFrame({
        "feat_a": vals,
        "feat_b": vals,  # Exact duplicate of feat_a
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "DUP_COLS_DETECTED" for w in res.warnings)


def test_feat_high_cardinality_categorical(temp_dir):
    """Verify FEAT_HIGH_CARDINALITY warning for categorical columns with high unique count."""
    csv_file = temp_dir / "high_cardinality.csv"
    n_rows = 60
    df = pd.DataFrame({
        "user_guid": [f"user_{i}" for i in range(n_rows)],  # 100% unique strings
        "age": [float(20 + (i % 30)) for i in range(n_rows)],
        "target": [i % 2 for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "FEAT_HIGH_CARDINALITY" for w in res.warnings)


def test_feat_datetime_detected_warning(temp_dir):
    """Verify FEAT_DATETIME_DETECTED warning for ISO datetime format strings."""
    csv_file = temp_dir / "datetime_feature.csv"
    df = pd.DataFrame({
        "signup_date": [f"2026-01-{i:02d}" for i in range(1, 21)],
        "metric": [float(i) for i in range(20)],
        "target": [i % 2 for i in range(20)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID_WITH_WARNINGS"
    assert any(w.code == "FEAT_DATETIME_DETECTED" for w in res.warnings)


def test_clean_valid_csv(temp_dir):
    """Verify that a compliant, clean dataset returns VALID status with no errors or warnings."""
    csv_file = temp_dir / "clean_valid.csv"
    n_rows = 30
    df = pd.DataFrame({
        "x1": np.linspace(10.0, 50.0, n_rows),
        "x2": np.linspace(-5.0, 5.0, n_rows),
        "target": [0 if i % 2 == 0 else 1 for i in range(n_rows)],
    })
    df.to_csv(csv_file, index=False)
    
    res = validate_csv_dataset(file_path=csv_file, target_column="target")
    assert res.validation_status == "VALID"
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert len(res.warnings) == 0
    assert res.row_count == 30
    assert res.column_count == 3
