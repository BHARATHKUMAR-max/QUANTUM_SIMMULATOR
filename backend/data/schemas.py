"""
backend/data/schemas.py
=======================
Data models and profiling schema definitions for QML Benchmark Studio.

These schemas provide consistent, reusable structures for dataset loading,
validation, profiling, and serialization across built-in and custom datasets.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
import re
import numpy as np
import pandas as pd


def infer_feature_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify DataFrame columns into semantic types:
    'numerical', 'categorical', 'boolean', 'datetime_like'.
    
    Args:
        df: Input DataFrame containing feature columns.
        
    Returns:
        Mapping of column names to semantic type labels.
    """
    types: Dict[str, str] = {}
    iso_date_pattern = re.compile(
        r"^\d{4}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])"
    )

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            types[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime_like"
        else:
            # Check string representation for date/time patterns
            non_null_samples = series.dropna().astype(str).head(20)
            if len(non_null_samples) > 0 and all(
                bool(iso_date_pattern.match(s.strip())) for s in non_null_samples
            ):
                types[col] = "datetime_like"
            else:
                types[col] = "categorical"
    return types


@dataclass
class ValidationIssue:
    """
    Structured validation error or warning.
    
    Attributes:
        code: Stable diagnostic identifier (e.g. 'FILE_EMPTY', 'FEAT_HAS_NULLS').
        severity: Severity tier ('ERROR' or 'WARNING').
        message: Clear, human-readable description of the issue.
        affected_column: Name of the impacted column if applicable.
        suggested_action: Concrete remediation guidance for the user.
    """
    code: str
    severity: str
    message: str
    affected_column: Optional[str] = None
    suggested_action: str = ""


@dataclass
class CSVValidationResult:
    """
    Structured audit report produced by the CSV validation subsystem.
    
    Conforms to the validation specification documented in docs/CSV_VALIDATION_RULES.md.
    """
    validation_status: str  # "VALID", "VALID_WITH_WARNINGS", "INVALID"
    filename: str
    row_count: int
    column_count: int
    selected_target: Optional[str]
    feature_columns: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_class_distribution: Dict[str, Any]
    missing_value_summary: Dict[str, Any]
    duplicate_summary: Dict[str, Any]
    constant_feature_summary: Dict[str, Any]
    warnings: List[ValidationIssue]
    errors: List[ValidationIssue]
    file_size_bytes: int = 0

    @property
    def is_valid(self) -> bool:
        """Indicate whether the dataset can proceed to experiment setup."""
        return self.validation_status in ("VALID", "VALID_WITH_WARNINGS")

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result into a serializable nested dictionary."""
        return asdict(self)


@dataclass
class DatasetBundle:
    """
    Standard container representing a loaded dataset partition or full collection.
    
    Attributes:
        X: Feature matrix as a Pandas DataFrame.
        y: Target label vector as a Pandas Series.
        feature_names: List of all column feature names.
        target_names: Human-readable names of the target classes.
        dataset_name: Canonical identifier for the dataset.
        target_name: Explicit name of the target column.
        target_classes: Unique original class labels preserved from the source.
        feature_types: Dictionary mapping feature names to inferred semantic types.
        metadata: Additional dataset attributes (source, citation, timestamp, etc.).
        source_type: Dataset origin identifier ('builtin' or 'csv').
        description: Freeform descriptive text or provenance documentation.
    """
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    target_names: List[str]
    dataset_name: str
    target_name: str = "target"
    target_classes: List[Any] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "builtin"
    description: str = ""

    def __post_init__(self) -> None:
        """Ensure default attributes are populated consistently."""
        if (not self.target_name or self.target_name == "target") and self.y is not None:
            if hasattr(self.y, "name") and self.y.name:
                self.target_name = str(self.y.name)
        if not self.target_classes:
            if self.target_names:
                self.target_classes = list(self.target_names)
            elif self.y is not None:
                self.target_classes = [str(c) for c in self.y.dropna().unique().tolist()]
        if not self.feature_types and self.X is not None and not self.X.empty:
            self.feature_types = infer_feature_types(self.X)

    def validate_integrity(self) -> None:
        """Verify structural consistency between X and y."""
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Row mismatch: X has {len(self.X)} rows but y has {len(self.y)} entries."
            )
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"Feature count mismatch: {len(self.feature_names)} names provided for {self.X.shape[1]} columns."
            )
        # Verify target is strictly isolated from feature matrix
        if self.target_name in self.X.columns:
            raise ValueError(
                f"Data leakage detected: Target column '{self.target_name}' found in feature matrix X."
            )


@dataclass
class ClassDistributionProfile:
    """Statistical summary of target classes and imbalance ratios."""
    counts: Dict[str, int]
    raw_counts: Dict[Any, int]
    percentages: Dict[str, float]
    imbalance_ratio: float
    majority_class: str
    minority_class: str
    total_samples: int


@dataclass
class FeatureStatistics:
    """Descriptive statistics for a single numerical feature."""
    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    variance: float
    range: float


@dataclass
class CategoricalFeatureStatistics:
    """Descriptive statistics for a single categorical or boolean feature."""
    count: int
    unique_count: int
    top_category: Optional[str]
    top_frequency: int
    top_percentage: float
    category_counts: Dict[str, int]


@dataclass
class DescriptiveSummary:
    """Aggregated descriptive statistics across numerical and categorical features."""
    features: Dict[str, FeatureStatistics]
    largest_variability_features: List[Dict[str, Any]]
    smallest_variability_features: List[Dict[str, Any]]
    categorical_features: Dict[str, CategoricalFeatureStatistics] = field(default_factory=dict)


@dataclass
class MissingValuesProfile:
    """Missing value analysis per column and globally."""
    per_column_counts: Dict[str, int]
    per_column_percentages: Dict[str, float]
    total_missing: int
    columns_with_missing: List[str]


@dataclass
class DuplicateProfile:
    """Duplicate record and feature diagnostics."""
    total_duplicate_rows: int
    unique_rows: int
    has_duplicate_features: bool
    duplicate_feature_groups: List[List[str]]


@dataclass
class ConstantFeaturesProfile:
    """Identification of zero-variance and near-constant features."""
    zero_variance_features: List[str]
    low_variance_features: List[Dict[str, Any]]


@dataclass
class CorrelationProfile:
    """Pairwise correlation findings and collinearity indicators."""
    highly_correlated_pairs: List[Dict[str, Any]]
    highest_avg_abs_correlations: List[Dict[str, Any]]
    strongest_positive_correlations: List[Dict[str, Any]]
    strongest_negative_correlations: List[Dict[str, Any]]


@dataclass
class DatasetProfile:
    """
    Complete structured profile of a dataset.
    
    This schema is JSON-serializable and reusable for Breast Cancer, Wine,
    and user-uploaded CSV files.
    """
    dataset_name: str
    source: str
    n_samples: int
    n_features: int
    n_classes: int
    feature_names: List[str]
    dtypes: Dict[str, str]
    memory_usage_bytes: int
    memory_usage_readable: str
    target_names: List[str]
    class_distribution: ClassDistributionProfile
    missing_values: MissingValuesProfile
    duplicates: DuplicateProfile
    constants: ConstantFeaturesProfile
    descriptive_statistics: DescriptiveSummary
    correlation: CorrelationProfile
    timestamp_utc: str
    library_versions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire profile to a JSON-compatible nested dictionary."""
        return asdict(self)
