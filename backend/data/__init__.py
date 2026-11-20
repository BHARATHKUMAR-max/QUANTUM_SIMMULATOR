"""
backend/data/__init__.py
========================
Data ingestion, validation, splitting, preprocessing, and feature selection package
for QML Benchmark Studio.
"""

from backend.data.schemas import (
    CategoricalFeatureStatistics,
    ClassDistributionProfile,
    ComparisonModeConfig,
    ComparisonModeResult,
    ConstantFeaturesProfile,
    CorrelatedPairInfo,
    CorrelationFilterArtifact,
    CorrelationFilterConfig,
    CorrelationProfile,
    CSVValidationResult,
    CVFoldResult,
    DatasetBundle,
    DatasetProfile,
    DEFAULT_SPLIT_SEEDS,
    DescriptiveSummary,
    DuplicateProfile,
    FeatureSelectionConfig,
    FeatureSelectorArtifact,
    FeatureSelectionPreview,
    FeatureStatistics,
    MAX_QUBITS,
    MIN_QUBITS,
    MissingValuesProfile,
    ModelInputMetadata,
    PCAArtifact,
    PCAConfig,
    PreprocessingConfig,
    PreprocessorArtifact,
    ProcessedDatasetResult,
    QubitCompatibilityResult,
    TrainTestSplitResult,
    ValidationIssue,
    VALID_COMPARISON_MODES,
    VALID_ENCODING_STRATEGIES,
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
from backend.data.splitting import (
    create_stratified_cv_splits,
    create_train_test_split,
)
from backend.data.preprocessing import (
    fit_preprocessor,
    prepare_cv_fold_data,
    prepare_split_and_preprocessed_data,
    transform_with_preprocessor,
)
from backend.data.feature_selection import (
    fit_selector,
    transform_with_selector,
    build_selection_preview,
    prepare_cv_fold_selection,
)
from backend.data.correlation_filter import (
    fit_correlation_filter,
    transform_with_filter,
    prepare_cv_fold_filter,
)
from backend.data.dimensionality_reduction import (
    fit_pca,
    transform_with_pca,
    prepare_cv_fold_pca,
)
from backend.data.comparison_modes import (
    prepare_best_practical_configuration,
    prepare_comparison_mode,
    prepare_cv_fold_comparison,
    prepare_matched_reduced_features,
    prepare_quantum_limited_features,
    validate_qubit_compatibility,
    verify_matched_inputs,
)

__all__ = [
    # Schemas
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
    "CorrelatedPairInfo",
    "CorrelationFilterConfig",
    "CorrelationFilterArtifact",
    "PCAConfig",
    "PCAArtifact",
    "infer_feature_types",
    "DEFAULT_SPLIT_SEEDS",
    "TrainTestSplitResult",
    "CVFoldResult",
    "PreprocessingConfig",
    "PreprocessorArtifact",
    "ProcessedDatasetResult",
    "FeatureSelectionConfig",
    "FeatureSelectorArtifact",
    "FeatureSelectionPreview",
    "MIN_QUBITS",
    "MAX_QUBITS",
    "VALID_COMPARISON_MODES",
    "VALID_ENCODING_STRATEGIES",
    "QubitCompatibilityResult",
    "ModelInputMetadata",
    "ComparisonModeConfig",
    "ComparisonModeResult",
    # CSV validation
    "validate_csv_dataset",
    # Dataset loading
    "load_breast_cancer_data",
    "load_wine_data",
    "load_csv_dataset",
    "load_dataset",
    # Profiling
    "profile_dataset",
    "generate_visualizations",
    "export_profile",
    # Splitting
    "create_train_test_split",
    "create_stratified_cv_splits",
    # Preprocessing
    "fit_preprocessor",
    "transform_with_preprocessor",
    "prepare_split_and_preprocessed_data",
    "prepare_cv_fold_data",
    # Feature selection
    "fit_selector",
    "transform_with_selector",
    "build_selection_preview",
    "prepare_cv_fold_selection",
    # Correlation filtering
    "fit_correlation_filter",
    "transform_with_filter",
    "prepare_cv_fold_filter",
    # PCA
    "fit_pca",
    "transform_with_pca",
    "prepare_cv_fold_pca",
    # Comparison modes
    "validate_qubit_compatibility",
    "verify_matched_inputs",
    "prepare_matched_reduced_features",
    "prepare_quantum_limited_features",
    "prepare_best_practical_configuration",
    "prepare_comparison_mode",
    "prepare_cv_fold_comparison",
]
