"""
backend/data/csv_validator.py
=============================
Rigorous validation engine for user-provided CSV datasets in QML Benchmark Studio.

Implements all file-level, target-level, feature-level, and duplication audit rules
strictly conforming to docs/CSV_VALIDATION_RULES.md.
"""

import csv
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from backend.data.schemas import (
    CSVValidationResult,
    ValidationIssue,
    infer_feature_types,
)


def validate_csv_dataset(
    file_path: Union[str, Path],
    target_column: Optional[str] = None,
    min_rows: int = 15,
) -> CSVValidationResult:
    """
    Audit an uploaded or local CSV dataset against QML Benchmark Studio data health rules.
    
    Checks conform to docs/CSV_VALIDATION_RULES.md:
      - File-level: Extension, existence, non-emptiness, UTF-8/BOM encoding, parseability,
        unique headers, and minimum row count (>= 15).
      - Target-level: Target specification, existence, completeness (no nulls), multi-class
        requirement (>= 2 classes), extreme cardinality, and class imbalance.
      - Feature-level: Total nulls (FEAT_ALL_NULL), missing values (FEAT_HAS_NULLS),
        infinite values (FEAT_INFINITE_VALUES), zero-variance (FEAT_CONSTANT),
        near-constant (FEAT_NEAR_CONSTANT), high-cardinality categorical (FEAT_HIGH_CARDINALITY),
        and datetime formats (FEAT_DATETIME_DETECTED).
      - Duplication: Duplicate rows (DUP_ROWS_DETECTED), duplicate feature values (DUP_COLS_DETECTED),
        and duplicate header names (COL_DUPLICATE_NAMES).
        
    Args:
        file_path: Path to the CSV file.
        target_column: Name of the designated ground-truth target column.
        min_rows: Minimum required data rows (default 15).
        
    Returns:
        CSVValidationResult: Structured validation outcome (VALID, VALID_WITH_WARNINGS, INVALID)
                             with itemized errors, warnings, and summary statistics.
    """
    path = Path(file_path)
    filename = path.name
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    
    # Defaults for summary fields
    row_count = 0
    column_count = 0
    feature_columns: List[str] = []
    numerical_columns: List[str] = []
    categorical_columns: List[str] = []
    target_class_distribution: Dict[str, Any] = {}
    missing_value_summary: Dict[str, Any] = {"total_missing_values": 0, "columns_with_missing": []}
    duplicate_summary: Dict[str, Any] = {"duplicate_row_count": 0, "duplicate_feature_groups": []}
    constant_feature_summary: Dict[str, Any] = {"zero_variance_features": [], "low_variance_features": []}
    file_size_bytes = 0

    def _build_result() -> CSVValidationResult:
        if errors:
            status = "INVALID"
        elif warnings:
            status = "VALID_WITH_WARNINGS"
        else:
            status = "VALID"
        return CSVValidationResult(
            validation_status=status,
            filename=filename,
            row_count=row_count,
            column_count=column_count,
            selected_target=target_column,
            feature_columns=feature_columns,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target_class_distribution=target_class_distribution,
            missing_value_summary=missing_value_summary,
            duplicate_summary=duplicate_summary,
            constant_feature_summary=constant_feature_summary,
            warnings=warnings,
            errors=errors,
            file_size_bytes=file_size_bytes,
        )

    # -------------------------------------------------------------------------
    # 1. FILE-01: Extension check
    # -------------------------------------------------------------------------
    if path.suffix.lower() != ".csv":
        errors.append(
            ValidationIssue(
                code="FILE_UNSUPPORTED_EXT",
                severity="ERROR",
                message="Unsupported file format. Only standard comma-separated (.csv) files are supported.",
                suggested_action="Export or save your table as a .csv file and re-upload.",
            )
        )
        return _build_result()

    # -------------------------------------------------------------------------
    # 2. FILE-02: File existence and empty file check
    # -------------------------------------------------------------------------
    if not path.exists():
        errors.append(
            ValidationIssue(
                code="FILE_NOT_FOUND",
                severity="ERROR",
                message=f"File not found: '{file_path}'.",
                suggested_action="Verify the file path and ensure the file exists on disk.",
            )
        )
        return _build_result()

    try:
        file_size_bytes = os.path.getsize(path)
    except OSError as e:
        errors.append(
            ValidationIssue(
                code="FILE_READ_ERROR",
                severity="ERROR",
                message=f"Cannot access file: {str(e)}",
                suggested_action="Check file system permissions.",
            )
        )
        return _build_result()

    if file_size_bytes == 0:
        errors.append(
            ValidationIssue(
                code="FILE_EMPTY",
                severity="ERROR",
                message="The uploaded file is empty. Please select a CSV file containing data.",
                suggested_action="Check the selected file on your computer and select a valid non-empty file.",
            )
        )
        return _build_result()

    # -------------------------------------------------------------------------
    # 3. FILE-03: Character encoding check (UTF-8 / UTF-8 BOM)
    # -------------------------------------------------------------------------
    raw_bytes: bytes = b""
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        errors.append(
            ValidationIssue(
                code="FILE_READ_ERROR",
                severity="ERROR",
                message=f"Failed to read file bytes: {str(e)}",
                suggested_action="Verify that the file is not locked by another process.",
            )
        )
        return _build_result()

    try:
        content_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        errors.append(
            ValidationIssue(
                code="FILE_ENCODING_ERROR",
                severity="ERROR",
                message="Unable to read file text encoding. Please ensure the CSV is saved using standard UTF-8.",
                suggested_action="Re-save the file in your spreadsheet editor selecting 'CSV UTF-8 (Comma delimited)'.",
            )
        )
        return _build_result()

    # Check for whitespace-only content
    if not content_text.strip():
        errors.append(
            ValidationIssue(
                code="FILE_EMPTY",
                severity="ERROR",
                message="The uploaded file is empty. Please select a CSV file containing data.",
                suggested_action="Check the selected file on your computer and select a valid non-empty file.",
            )
        )
        return _build_result()

    # -------------------------------------------------------------------------
    # 4. Header validation: duplicate headers (COL_DUPLICATE_NAMES) and format
    # -------------------------------------------------------------------------
    non_empty_lines = [line for line in content_text.splitlines() if line.strip()]
    if not non_empty_lines:
        errors.append(
            ValidationIssue(
                code="FILE_EMPTY",
                severity="ERROR",
                message="The CSV file contains no readable content rows.",
                suggested_action="Ensure the CSV contains a header row and data records.",
            )
        )
        return _build_result()

    try:
        first_row_reader = csv.reader(io.StringIO(non_empty_lines[0]))
        raw_headers = next(first_row_reader)
    except Exception as e:
        errors.append(
            ValidationIssue(
                code="FILE_PARSE_ERROR",
                severity="ERROR",
                message=f"Malformed CSV header structure detected: {str(e)}",
                suggested_action="Inspect the header row for unescaped quotes or delimiter errors.",
            )
        )
        return _build_result()

    clean_headers = [h.strip() for h in raw_headers]
    if not clean_headers or all(h == "" for h in clean_headers):
        errors.append(
            ValidationIssue(
                code="FILE_PARSE_ERROR",
                severity="ERROR",
                message="The CSV must contain a valid header row defining column names.",
                suggested_action="Ensure the first row of your CSV contains column names.",
            )
        )
        return _build_result()

    # Check duplicate column names (COL_DUPLICATE_NAMES)
    seen_headers: Set[str] = set()
    dup_headers: List[str] = []
    for h in clean_headers:
        if h in seen_headers:
            dup_headers.append(h)
        seen_headers.add(h)

    if dup_headers:
        errors.append(
            ValidationIssue(
                code="COL_DUPLICATE_NAMES",
                severity="ERROR",
                message=f"Duplicate column headers detected: {list(set(dup_headers))}. Column names must be unique to avoid ambiguous feature selection.",
                affected_column=dup_headers[0],
                suggested_action="Rename the duplicated column headers in your CSV file.",
            )
        )

    # -------------------------------------------------------------------------
    # 5. Parse full DataFrame with pandas
    # -------------------------------------------------------------------------
    try:
        # We read with header=0. If duplicate headers existed, pandas renames them,
        # but we already flagged COL_DUPLICATE_NAMES above.
        df = pd.read_csv(io.StringIO(content_text), encoding="utf-8")
    except pd.errors.EmptyDataError:
        errors.append(
            ValidationIssue(
                code="FILE_EMPTY",
                severity="ERROR",
                message="The uploaded CSV contains no data rows.",
                suggested_action="Ensure the CSV contains at least 15 observations.",
            )
        )
        return _build_result()
    except pd.errors.ParserError as e:
        errors.append(
            ValidationIssue(
                code="FILE_PARSE_ERROR",
                severity="ERROR",
                message=f"Malformed CSV structure detected: {str(e)}",
                suggested_action="Inspect the file around the flagged line for unclosed quotes or stray delimiter characters.",
            )
        )
        return _build_result()
    except Exception as e:
        errors.append(
            ValidationIssue(
                code="FILE_PARSE_ERROR",
                severity="ERROR",
                message=f"Unable to parse CSV structure: {str(e)}",
                suggested_action="Verify that rows and quoted fields are properly formatted.",
            )
        )
        return _build_result()

    row_count = len(df)
    column_count = df.shape[1]

    # -------------------------------------------------------------------------
    # 6. FILE-07: Minimum row count check
    # -------------------------------------------------------------------------
    if row_count < min_rows:
        errors.append(
            ValidationIssue(
                code="FILE_ROW_COUNT_LOW",
                severity="ERROR",
                message=f"Insufficient data rows. The dataset contains {row_count} rows, fewer than the minimum of {min_rows} required for benchmark splitting.",
                suggested_action=f"Ensure the dataset contains at least {min_rows} observations.",
            )
        )

    # -------------------------------------------------------------------------
    # 7. Target Column Validation
    # -------------------------------------------------------------------------
    if target_column is None or str(target_column).strip() == "":
        errors.append(
            ValidationIssue(
                code="TARGET_NOT_SPECIFIED",
                severity="ERROR",
                message="No target column selected. Please designate which column represents the prediction target.",
                suggested_action="Select the target column name from the available columns dropdown.",
            )
        )
        target_valid = False
    elif target_column not in df.columns:
        errors.append(
            ValidationIssue(
                code="TARGET_NOT_FOUND",
                severity="ERROR",
                message=f"Selected target column '{target_column}' was not found in the uploaded file.",
                affected_column=str(target_column),
                suggested_action="Select a column that exists in the header list.",
            )
        )
        target_valid = False
    else:
        target_valid = True

    if target_valid and target_column is not None:
        target_series = df[target_column]

        # TGT-04: Missing target values (TARGET_HAS_NULLS)
        target_nulls = int(target_series.isnull().sum())
        if target_nulls > 0:
            errors.append(
                ValidationIssue(
                    code="TARGET_HAS_NULLS",
                    severity="ERROR",
                    message=f"The target column '{target_column}' contains {target_nulls} missing values. Supervised classification requires complete target labels.",
                    affected_column=target_column,
                    suggested_action="Provide labels for all rows or remove unlabelled rows from the CSV file.",
                )
            )

        # Non-null target analysis
        clean_target = target_series.dropna()
        unique_targets = clean_target.unique()
        n_classes = len(unique_targets)

        # TGT-05: Target class count >= 2 (TARGET_SINGLE_CLASS)
        if n_classes < 2:
            errors.append(
                ValidationIssue(
                    code="TARGET_SINGLE_CLASS",
                    severity="ERROR",
                    message=f"This dataset contains only one target class ({unique_targets.tolist()}). Classification requires at least two distinct classes.",
                    affected_column=target_column,
                    suggested_action="Ensure the target column includes samples from at least two different classes.",
                )
            )
        # TGT-08: Extreme high cardinality (TARGET_HIGH_CARDINALITY)
        elif n_classes > 20 or (row_count > 0 and (n_classes / row_count) > 0.50):
            errors.append(
                ValidationIssue(
                    code="TARGET_HIGH_CARDINALITY",
                    severity="ERROR",
                    message=f"The target column '{target_column}' has {n_classes} unique classes ({round((n_classes/row_count)*100, 1)}% of rows). Supervised classification benchmarks support up to 20 discrete classes.",
                    affected_column=target_column,
                    suggested_action="Confirm that the chosen target column represents discrete classification categories rather than continuous values or row identifiers.",
                )
            )

        # Target class distribution computation
        if n_classes >= 1 and len(clean_target) > 0:
            counts_series = clean_target.value_counts()
            class_counts = {str(k): int(v) for k, v in counts_series.items()}
            class_percentages = {
                str(k): round((int(v) / len(clean_target)) * 100.0, 2)
                for k, v in counts_series.items()
            }
            majority_cls = str(counts_series.index[0])
            minority_cls = str(counts_series.index[-1])
            maj_count = int(counts_series.iloc[0])
            min_count = int(counts_series.iloc[-1])
            imbalance_ratio = round(maj_count / min_count, 3) if min_count > 0 else 0.0

            target_class_distribution = {
                "target_name": target_column,
                "num_classes": n_classes,
                "class_counts": class_counts,
                "class_percentages": class_percentages,
                "majority_class": majority_cls,
                "minority_class": minority_cls,
                "imbalance_ratio": imbalance_ratio,
            }

            # TGT-07: Severe class imbalance warning (TARGET_CLASS_IMBALANCE)
            minority_proportion = min_count / len(clean_target)
            if n_classes >= 2 and minority_proportion < 0.20:
                warnings.append(
                    ValidationIssue(
                        code="TARGET_CLASS_IMBALANCE",
                        severity="WARNING",
                        message=f"Class imbalance detected: The minority class '{minority_cls}' represents {minority_proportion * 100:.1f}% of samples (< 20%).",
                        affected_column=target_column,
                        suggested_action="Consider enabling stratified splitting to preserve class proportions.",
                    )
                )

        # Isolate candidate feature columns (strictly excluding target)
        feature_columns = [col for col in df.columns if col != target_column]
    else:
        feature_columns = list(df.columns)

    if not feature_columns:
        errors.append(
            ValidationIssue(
                code="FEAT_ALL_NULL",
                severity="ERROR",
                message="No feature columns remain after target selection.",
                suggested_action="Provide a CSV containing at least one feature column alongside the target.",
            )
        )
        return _build_result()

    # -------------------------------------------------------------------------
    # 8. Feature Typing & Column Validation
    # -------------------------------------------------------------------------
    df_features = df[feature_columns]
    inferred_types = infer_feature_types(df_features)

    for col in feature_columns:
        col_type = inferred_types.get(col, "categorical")
        if col_type == "numerical":
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)

    # 8.1 Missing values analysis on features
    cols_with_missing_summary = []
    total_feature_missing = 0

    for col in feature_columns:
        series = df[col]
        n_missing = int(series.isnull().sum())
        total_feature_missing += n_missing

        if n_missing > 0:
            pct_missing = round((n_missing / row_count) * 100.0, 2)
            cols_with_missing_summary.append({
                "column_name": col,
                "missing_count": n_missing,
                "missing_percentage": pct_missing,
            })

            # Check if column is 100% missing (FEAT_ALL_NULL)
            if n_missing == row_count:
                errors.append(
                    ValidationIssue(
                        code="FEAT_ALL_NULL",
                        severity="ERROR",
                        message=f"Column '{col}' contains entirely missing values and cannot be used as a feature.",
                        affected_column=col,
                        suggested_action=f"Remove or populate the empty column '{col}'.",
                    )
                )
            else:
                warnings.append(
                    ValidationIssue(
                        code="FEAT_HAS_NULLS",
                        severity="WARNING",
                        message=f"Missing values detected in feature column '{col}' ({n_missing} missing values, {pct_missing}%).",
                        affected_column=col,
                        suggested_action="Missing values will be handled by the configured preprocessor (imputation).",
                    )
                )

    missing_value_summary = {
        "total_missing_values": total_feature_missing,
        "columns_with_missing": cols_with_missing_summary,
    }

    # 8.2 Numerical features checks
    zero_var_features: List[str] = []
    low_var_features: List[Dict[str, Any]] = []

    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        # Check for infinite values (FEAT_INFINITE_VALUES)
        num_array = series.to_numpy()
        if np.isinf(num_array).any():
            errors.append(
                ValidationIssue(
                    code="FEAT_INFINITE_VALUES",
                    severity="ERROR",
                    message=f"Infinite values detected in numeric feature '{col}'. Mathematical models cannot process infinite inputs.",
                    affected_column=col,
                    suggested_action=f"Replace infinite entries in '{col}' with finite numbers or null indicators.",
                )
            )

        # Variance checks (FEAT_CONSTANT, FEAT_NEAR_CONSTANT)
        if len(series) > 1 and not np.isinf(num_array).any():
            variance = float(series.var(ddof=1))
            if variance == 0.0 or series.nunique() <= 1:
                zero_var_features.append(col)
                warnings.append(
                    ValidationIssue(
                        code="FEAT_CONSTANT",
                        severity="WARNING",
                        message=f"Zero-variance constant feature detected: '{col}'. This feature contains identical values across all rows.",
                        affected_column=col,
                        suggested_action="Constant features provide no predictive signal and will likely be eliminated by feature selection.",
                    )
                )
            elif 0.0 < variance < 1e-4:
                low_var_features.append({"feature": col, "variance": round(variance, 8)})
                warnings.append(
                    ValidationIssue(
                        code="FEAT_NEAR_CONSTANT",
                        severity="WARNING",
                        message=f"Near-constant feature detected: '{col}' (variance {variance:.8f}).",
                        affected_column=col,
                        suggested_action="Informational note; feature selection will evaluate whether this feature is informative.",
                    )
                )

    constant_feature_summary = {
        "zero_variance_features": zero_var_features,
        "low_variance_features": low_var_features,
    }

    # 8.3 Categorical & Datetime feature checks
    for col in categorical_columns:
        col_type = inferred_types.get(col, "categorical")
        series = df[col].dropna()
        if series.empty:
            continue

        # Datetime detection (FEAT_DATETIME_DETECTED)
        if col_type == "datetime_like":
            warnings.append(
                ValidationIssue(
                    code="FEAT_DATETIME_DETECTED",
                    severity="WARNING",
                    message=f"Date or timestamp format detected in column '{col}'. Datetime columns are not automatically converted into numerical features.",
                    affected_column=col,
                    suggested_action="Extract numerical components (such as day of week, hour, or elapsed seconds) if you wish to use time signals.",
                )
            )

        # High cardinality in categorical features (FEAT_HIGH_CARDINALITY)
        nunique = series.nunique()
        if nunique > 50 or (row_count > 0 and (nunique / row_count) > 0.80):
            warnings.append(
                ValidationIssue(
                    code="FEAT_HIGH_CARDINALITY",
                    severity="WARNING",
                    message=f"High-cardinality categorical feature detected in '{col}' ({nunique} unique categories). This column may be an identifier (e.g. ID, name).",
                    affected_column=col,
                    suggested_action=f"Verify whether '{col}' is a unique identifier that should be excluded from training.",
                )
            )

    # -------------------------------------------------------------------------
    # 9. Duplication Analysis
    # -------------------------------------------------------------------------
    # 9.1 Duplicate rows (DUP_ROWS_DETECTED)
    dup_row_count = int(df.duplicated().sum())
    if dup_row_count > 0:
        warnings.append(
            ValidationIssue(
                code="DUP_ROWS_DETECTED",
                severity="WARNING",
                message=f"Duplicate rows detected ({dup_row_count} identical records). Duplicate records are retained to preserve original data distribution.",
                suggested_action="Review whether repeated rows represent genuine repeated measurements or accidental duplicates.",
            )
        )

    # 9.2 Duplicate feature columns (DUP_COLS_DETECTED)
    dup_col_groups: List[List[str]] = []
    checked_features: Set[str] = set()
    for i in range(len(feature_columns)):
        col_a = feature_columns[i]
        if col_a in checked_features:
            continue
        group = [col_a]
        for j in range(i + 1, len(feature_columns)):
            col_b = feature_columns[j]
            if df[col_a].equals(df[col_b]):
                group.append(col_b)
                checked_features.add(col_b)
        if len(group) > 1:
            dup_col_groups.append(group)
            warnings.append(
                ValidationIssue(
                    code="DUP_COLS_DETECTED",
                    severity="WARNING",
                    message=f"Duplicate feature values detected across columns: {group}. These columns provide identical information.",
                    affected_column=col_a,
                    suggested_action="Redundant columns can be eliminated during feature selection.",
                )
            )

    duplicate_summary = {
        "duplicate_row_count": dup_row_count,
        "duplicate_feature_groups": dup_col_groups,
    }

    return _build_result()
