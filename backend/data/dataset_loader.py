"""
backend/data/dataset_loader.py
==============================
Dataset ingestion module for QML Benchmark Studio.

Responsible for loading built-in datasets (Breast Cancer Wisconsin, Wine) and
user-provided CSV datasets, returning standardized DatasetBundle containers
with preserved metadata, semantic feature typing, and original target labels.
"""

from dataclasses import asdict
from pathlib import Path
import re
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

from backend.data.schemas import DatasetBundle, infer_feature_types
from backend.data.csv_validator import validate_csv_dataset


def load_breast_cancer_data() -> DatasetBundle:
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
    
    Verifies dataset integrity:
      - 569 instances
      - 30 real-valued numerical features
      - 2 target classes: 0 = 'malignant', 1 = 'benign'
      - Feature matrix X contains no ground-truth target columns
      
    Returns:
        DatasetBundle: Standardized container encapsulating X, y, feature types,
                       target metadata, and provenance.
    """
    raw_data = load_breast_cancer(as_frame=True)
    
    X: pd.DataFrame = raw_data.data.copy()
    y: pd.Series = raw_data.target.copy()
    y.name = "target"
    
    # Ensure all feature names are standard Python strings
    feature_names = [str(col) for col in raw_data.feature_names]
    X.columns = feature_names
    
    # Map target names: sklearn convention is index 0 -> malignant, index 1 -> benign
    target_names = [str(name) for name in raw_data.target_names]
    
    # Explicit validation checks
    if X.shape != (569, 30):
        raise ValueError(f"Unexpected Breast Cancer X shape: {X.shape}. Expected (569, 30).")
    
    if len(y) != 569:
        raise ValueError(f"Unexpected Breast Cancer target length: {len(y)}. Expected 569.")
    
    if len(target_names) != 2:
        raise ValueError(f"Expected 2 target classes, got {len(target_names)}: {target_names}")
    
    # Verify no accidental target leakage inside X
    for forbidden_col in ["target", "class", "diagnosis", "malignant", "benign"]:
        if forbidden_col in X.columns:
            raise ValueError(f"Data leakage detected: Target column '{forbidden_col}' found in feature matrix X.")
            
    # Compile provenance metadata
    metadata = {
        "dataset_name": "breast_cancer",
        "scikit_learn_identifier": "load_breast_cancer",
        "n_samples": 569,
        "n_features": 30,
        "n_classes": 2,
        "target_mapping": {0: target_names[0], 1: target_names[1]},
        "citation": "WDBC Dataset, Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)",
    }
    
    bundle = DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dataset_name="breast_cancer",
        target_name="target",
        target_classes=target_names,
        feature_types={col: "numerical" for col in feature_names},
        metadata=metadata,
        source_type="builtin",
        description=str(raw_data.DESCR) if hasattr(raw_data, "DESCR") else "",
    )
    
    bundle.validate_integrity()
    return bundle


def load_wine_data() -> DatasetBundle:
    """
    Load the Wine recognition dataset from scikit-learn.
    
    Verifies dataset integrity:
      - 178 instances
      - 13 continuous chemical/physical attributes
      - 3 target classes representing distinct cultivars: 'class_0', 'class_1', 'class_2'
      - Feature matrix X contains no ground-truth target columns
      
    Returns:
        DatasetBundle: Standardized container encapsulating X, y, feature types,
                       target metadata, and provenance.
    """
    raw_data = load_wine(as_frame=True)
    
    X: pd.DataFrame = raw_data.data.copy()
    y: pd.Series = raw_data.target.copy()
    y.name = "target"
    
    # Ensure all feature names are standard Python strings
    feature_names = [str(col) for col in raw_data.feature_names]
    X.columns = feature_names
    
    # Target names from scikit-learn metadata: ['class_0', 'class_1', 'class_2']
    target_names = [str(name) for name in raw_data.target_names]
    
    # Validation checks
    if X.shape != (178, 13):
        raise ValueError(f"Unexpected Wine X shape: {X.shape}. Expected (178, 13).")
        
    if len(y) != 178:
        raise ValueError(f"Unexpected Wine target length: {len(y)}. Expected 178.")
        
    if len(target_names) != 3:
        raise ValueError(f"Expected 3 target classes, got {len(target_names)}: {target_names}")
        
    # Verify no accidental target leakage inside X
    for forbidden_col in ["target", "class", "cultivar"]:
        if forbidden_col in X.columns:
            raise ValueError(f"Data leakage detected: Target column '{forbidden_col}' found in feature matrix X.")
            
    metadata = {
        "dataset_name": "wine",
        "scikit_learn_identifier": "load_wine",
        "n_samples": 178,
        "n_features": 13,
        "n_classes": 3,
        "target_mapping": {i: target_names[i] for i in range(len(target_names))},
        "citation": "Wine Recognition Dataset, Forina, M. et al., Institute of Pharmaceutical and Food Analysis and Technologies, Genoa (1991)",
    }
    
    bundle = DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dataset_name="wine",
        target_name="target",
        target_classes=target_names,
        feature_types={col: "numerical" for col in feature_names},
        metadata=metadata,
        source_type="builtin",
        description=str(raw_data.DESCR) if hasattr(raw_data, "DESCR") else "",
    )
    
    bundle.validate_integrity()
    return bundle


def load_csv_dataset(
    file_path: Union[str, Path],
    target_column: str,
    dataset_name: Optional[str] = None,
    min_rows: int = 15,
) -> DatasetBundle:
    """
    Load and parse a user-provided CSV dataset into a standardized DatasetBundle.
    
    Performs full data health validation before loading. Original target labels
    and feature names are strictly preserved.
    
    Args:
        file_path: Path to the CSV file.
        target_column: Ground-truth target column name (mandatory).
        dataset_name: Optional identifier; defaults to sanitized file stem.
        min_rows: Minimum required data rows (default 15).
        
    Returns:
        DatasetBundle: Container holding feature matrix X, target vector y,
                       metadata, and inferred feature types.
                       
    Raises:
        ValueError: If validation fails with any fatal ERROR conditions.
    """
    path = Path(file_path)
    if target_column is None or str(target_column).strip() == "":
        raise ValueError("Target column must be explicitly specified when loading a CSV dataset.")

    # 1. Run validation engine
    validation_result = validate_csv_dataset(path, target_column=target_column, min_rows=min_rows)
    if not validation_result.is_valid:
        error_details = "\n".join([f"  - [{e.code}] {e.message}" for e in validation_result.errors])
        raise ValueError(
            f"CSV dataset '{path.name}' failed validation with {len(validation_result.errors)} errors:\n{error_details}"
        )

    # 2. Parse file with UTF-8 / UTF-8-BOM support
    df = pd.read_csv(path, encoding="utf-8-sig")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV headers: {list(df.columns)}")

    # 3. Separate features and target
    y = df[target_column].copy()
    y.name = str(target_column)
    X = df.drop(columns=[target_column]).copy()
    feature_names = [str(col) for col in X.columns]

    # Preserve original target class labels
    unique_classes = sorted(y.dropna().unique().tolist(), key=lambda x: str(x))
    target_names = [str(cls) for cls in unique_classes]

    # Inferred semantic types
    feature_types = infer_feature_types(X)

    # Sanitize dataset name
    if not dataset_name:
        raw_stem = path.stem.strip().lower()
        dataset_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem)

    metadata = {
        "dataset_name": dataset_name,
        "source_type": "csv",
        "file_path": str(path.resolve()),
        "file_name": path.name,
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_classes": len(unique_classes),
        "target_column": target_column,
        "target_classes": unique_classes,
        "feature_types": feature_types,
        "validation_status": validation_result.validation_status,
        "validation_warnings": [asdict(w) for w in validation_result.warnings],
    }

    bundle = DatasetBundle(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dataset_name=dataset_name,
        target_name=target_column,
        target_classes=unique_classes,
        feature_types=feature_types,
        metadata=metadata,
        source_type="csv",
        description=f"User-uploaded CSV dataset '{path.name}' with target '{target_column}'.",
    )

    bundle.validate_integrity()
    return bundle


def load_dataset(
    dataset_name: str,
    file_path: Optional[Union[str, Path]] = None,
    target_column: Optional[str] = None,
) -> DatasetBundle:
    """
    Unified entry point for loading benchmark datasets by identifier or CSV path.
    
    Args:
        dataset_name: Canonical dataset key ('breast_cancer', 'wine', 'csv') or custom name.
        file_path: Optional file path required when loading a CSV dataset.
        target_column: Optional target column required when loading a CSV dataset.
        
    Returns:
        DatasetBundle corresponding to the requested dataset.
    """
    normalized_name = dataset_name.strip().lower()
    if normalized_name in ["breast_cancer", "breast_cancer_wisconsin", "wdbc"]:
        return load_breast_cancer_data()
    elif normalized_name in ["wine", "wine_recognition"]:
        return load_wine_data()
    elif normalized_name == "csv" or file_path is not None:
        if file_path is None:
            raise ValueError("Parameter 'file_path' is required when loading a CSV dataset.")
        if target_column is None:
            raise ValueError("Parameter 'target_column' is required when loading a CSV dataset.")
        custom_name = None if normalized_name == "csv" else dataset_name
        return load_csv_dataset(file_path=file_path, target_column=target_column, dataset_name=custom_name)
    else:
        raise ValueError(
            f"Unknown dataset identifier: '{dataset_name}'. Available: ['breast_cancer', 'wine', 'csv']."
        )
