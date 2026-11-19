"""
backend/data/__init__.py
========================
Data ingestion, validation, and profiling package for QML Benchmark Studio.
"""

from backend.data.schemas import (
    CategoricalFeatureStatistics,
    ClassDistributionProfile,
    ConstantFeaturesProfile,
    CorrelationProfile,
    CSVValidationResult,
    DatasetBundle,
    DatasetProfile,
    DescriptiveSummary,
    DuplicateProfile,
    FeatureStatistics,
    MissingValuesProfile,
    ValidationIssue,
    infer_feature_types,
)
from backend.data.csv_validator import validate_csv_dataset
from backend.data.dataset_loader import (
    load_breast_cancer_data,
    load_csv_dataset,
    load_dataset,
    load_wine_data,
)
from backend.data.dataset_profiler import (
    export_profile,
    generate_visualizations,
    profile_dataset,
)

__all__ = [
    "DatasetBundle",
    "DatasetProfile",
    "ValidationIssue",
    "CSVValidationResult",
    "ClassDistributionProfile",
    "FeatureStatistics",
    "CategoricalFeatureStatistics",
    "DescriptiveSummary",
    "MissingValuesProfile",
    "DuplicateProfile",
    "ConstantFeaturesProfile",
    "CorrelationProfile",
    "infer_feature_types",
    "validate_csv_dataset",
    "load_breast_cancer_data",
    "load_wine_data",
    "load_csv_dataset",
    "load_dataset",
    "profile_dataset",
    "generate_visualizations",
    "export_profile",
]
