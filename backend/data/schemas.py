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


# Default seeds specified for benchmark experiments
DEFAULT_SPLIT_SEEDS = [42, 101, 202, 303, 404]


@dataclass
class TrainTestSplitResult:
    """
    Standard container encapsulating a partitioned dataset.
    
    Attributes:
        X_train: Feature matrix for training partition.
        X_test: Feature matrix for held-out test partition.
        y_train: Target vector for training partition.
        y_test: Target vector for held-out test partition.
        train_indices: Preserved original dataset row indices for training.
        test_indices: Preserved original dataset row indices for testing.
        random_state: Seed used for reproducible splitting.
        test_size: Proportion of samples allocated to test partition.
        stratified: Boolean indicating whether stratified splitting was applied.
        dataset_name: Canonical identifier of the dataset.
        n_train: Number of training samples.
        n_test: Number of testing samples.
        train_class_distribution: Class sample counts in training partition.
        test_class_distribution: Class sample counts in test partition.
        feature_names: List of input feature names.
        target_name: Name of the target column.
        metadata: Additional provenance and configuration metadata.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: List[Any]
    test_indices: List[Any]
    random_state: int
    test_size: float
    stratified: bool
    dataset_name: str
    n_train: int
    n_test: int
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    feature_names: List[str]
    target_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_integrity(self) -> None:
        """Verify strict non-overlap and partition completeness."""
        train_set = set(self.train_indices)
        test_set = set(self.test_indices)
        overlap = train_set.intersection(test_set)
        if overlap:
            raise ValueError(f"Data leakage detected: train and test indices overlap ({len(overlap)} samples).")
        if self.target_name in self.X_train.columns:
            raise ValueError(f"Data leakage detected: Target '{self.target_name}' found in X_train.")
        if self.target_name in self.X_test.columns:
            raise ValueError(f"Data leakage detected: Target '{self.target_name}' found in X_test.")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("Row mismatch between X_train and y_train.")
        if len(self.X_test) != len(self.y_test):
            raise ValueError("Row mismatch between X_test and y_test.")


@dataclass
class CVFoldResult:
    """
    Container representing train/val partitions for a single cross-validation fold.
    """
    fold_index: int
    train_indices: List[Any]
    val_indices: List[Any]
    n_train: int
    n_val: int
    train_class_distribution: Dict[str, int]
    val_class_distribution: Dict[str, int]


@dataclass
class PreprocessingConfig:
    """
    Configuration container for leakage-safe preprocessing pipelines.
    
    Attributes:
        numerical_imputation: Strategy for numerical missing values ('median' or 'mean').
        numerical_scaling: Scaling method for numerical attributes ('standard', 'minmax', or 'none').
        categorical_imputation: Strategy for categorical missing values ('most_frequent').
        categorical_encoding: Encoding scheme for categorical features ('onehot' or 'ordinal').
        boolean_handling: Strategy for boolean columns ('numeric').
        preserve_empty_features: Whether to retain features with 100% missing values in training.
    """
    numerical_imputation: str = "median"
    numerical_scaling: str = "standard"
    categorical_imputation: str = "most_frequent"
    categorical_encoding: str = "onehot"
    boolean_handling: str = "numeric"
    preserve_empty_features: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values strictly."""
        valid_num_imp = {"median", "mean"}
        valid_num_scale = {"standard", "minmax", "none"}
        valid_cat_imp = {"most_frequent"}
        valid_cat_enc = {"onehot", "ordinal"}
        valid_bool_hand = {"numeric"}

        if self.numerical_imputation not in valid_num_imp:
            raise ValueError(
                f"Invalid numerical_imputation: '{self.numerical_imputation}'. Allowed: {sorted(valid_num_imp)}"
            )
        if self.numerical_scaling not in valid_num_scale:
            raise ValueError(
                f"Invalid numerical_scaling: '{self.numerical_scaling}'. Allowed: {sorted(valid_num_scale)}"
            )
        if self.categorical_imputation not in valid_cat_imp:
            raise ValueError(
                f"Invalid categorical_imputation: '{self.categorical_imputation}'. Allowed: {sorted(valid_cat_imp)}"
            )
        if self.categorical_encoding not in valid_cat_enc:
            raise ValueError(
                f"Invalid categorical_encoding: '{self.categorical_encoding}'. Allowed: {sorted(valid_cat_enc)}"
            )
        if self.boolean_handling not in valid_bool_hand:
            raise ValueError(
                f"Invalid boolean_handling: '{self.boolean_handling}'. Allowed: {sorted(valid_bool_hand)}"
            )
        if not isinstance(self.preserve_empty_features, bool):
            raise ValueError(
                f"preserve_empty_features must be a boolean, got: {type(self.preserve_empty_features)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration into a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "PreprocessingConfig"]]) -> "PreprocessingConfig":
        """Construct a validated PreprocessingConfig from a dict or existing instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        normalized = dict(data)
        # Backward compatibility aliases and convenience shorthands
        if "numerical_imputer_strategy" in normalized and "numerical_imputation" not in normalized:
            normalized["numerical_imputation"] = normalized.pop("numerical_imputer_strategy")
        if "imputation" in normalized and "numerical_imputation" not in normalized:
            normalized["numerical_imputation"] = normalized.pop("imputation")
        if "scale_features" in normalized and "numerical_scaling" not in normalized:
            scale_bool = normalized.pop("scale_features")
            normalized["numerical_scaling"] = "standard" if scale_bool else "none"
        if "scaling" in normalized and "numerical_scaling" not in normalized:
            normalized["numerical_scaling"] = normalized.pop("scaling")
        if "categorical_imputer_strategy" in normalized and "categorical_imputation" not in normalized:
            normalized["categorical_imputation"] = normalized.pop("categorical_imputer_strategy")
        if "encoding" in normalized and "categorical_encoding" not in normalized:
            normalized["categorical_encoding"] = normalized.pop("encoding")
        if "preserve_empty" in normalized and "preserve_empty_features" not in normalized:
            normalized["preserve_empty_features"] = normalized.pop("preserve_empty")
        return cls(**normalized)


@dataclass
class PreprocessorArtifact:
    """
    Container storing fitted ColumnTransformer, feature names, and metadata.
    """
    transformer: Any
    numerical_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]
    original_feature_names: List[str]
    transformed_feature_names: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    fitted_timestamp_utc: str = ""
    all_missing_training_features: List[str] = field(default_factory=list)


@dataclass
class ProcessedDatasetResult:
    """
    Container combining train/test split and leakage-safe transformed features.
    """
    split: TrainTestSplitResult
    preprocessor: PreprocessorArtifact
    X_train_processed: pd.DataFrame
    X_test_processed: pd.DataFrame
    transformed_feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Day 9 — Feature Selection Schemas
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionConfig:
    """
    Configuration for supervised feature selection (Mutual Information or ANOVA F-test).

    Attributes:
        method: Selection algorithm. One of 'mutual_info' or 'anova'.
        mode: How the feature count is determined. 'auto' uses k_features or proportion;
              'manual' uses an explicit k_features integer.
        k_features: Number of features to select (manual mode) or upper bound (auto mode).
                    Must be a positive integer or None (auto default = all).
        proportion: Fraction of features to select in auto mode (0.0 < proportion <= 1.0).
                    Ignored when k_features is set.
        random_state: Seed for reproducibility (used by mutual_info_classif).
    """
    method: str = "mutual_info"
    mode: str = "auto"
    k_features: Optional[int] = None
    proportion: Optional[float] = None
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate selection configuration."""
        valid_methods = {"mutual_info", "anova"}
        valid_modes = {"auto", "manual"}

        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method: '{self.method}'. Allowed: {sorted(valid_methods)}"
            )
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: '{self.mode}'. Allowed: {sorted(valid_modes)}"
            )
        if self.mode == "manual":
            if self.k_features is None:
                raise ValueError("k_features must be provided when mode='manual'.")
            if not isinstance(self.k_features, int) or isinstance(self.k_features, bool):
                raise ValueError(
                    f"k_features must be a positive integer, got: {type(self.k_features).__name__}"
                )
            if self.k_features <= 0:
                raise ValueError(
                    f"k_features must be a positive integer, got: {self.k_features}"
                )
        if self.proportion is not None:
            if not (0.0 < self.proportion <= 1.0):
                raise ValueError(
                    f"proportion must be in (0.0, 1.0], got: {self.proportion}"
                )
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(
                f"random_state must be an integer, got: {type(self.random_state).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "FeatureSelectionConfig"]]) -> "FeatureSelectionConfig":
        """Construct a validated FeatureSelectionConfig from a dict or existing instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**dict(data))


@dataclass
class FeatureSelectorArtifact:
    """
    Container storing a fitted sklearn SelectKBest selector and selection metadata.

    Attributes:
        selector: Fitted sklearn SelectKBest instance.
        method: Selection method used ('mutual_info' or 'anova').
        mode: Selection mode ('auto' or 'manual').
        candidate_feature_names: Ordered list of feature names input to the selector.
        selected_feature_names: Ordered list of feature names after selection.
        scores: Score for each candidate feature (MI score or ANOVA F-score).
        pvalues: ANOVA p-values per candidate feature; None for mutual_info.
        rankings: Rank of each candidate feature (1 = highest score).
        k_requested: Number of features requested for selection.
        k_selected: Actual number of features selected.
        random_state: Seed used (relevant for mutual_info).
        warnings: List of advisory messages (e.g. constant features, NaN scores).
        config: Original FeatureSelectionConfig as dict.
        fitted_timestamp_utc: ISO 8601 timestamp of when selector was fitted.
    """
    selector: Any
    method: str
    mode: str
    candidate_feature_names: List[str]
    selected_feature_names: List[str]
    scores: List[float]
    pvalues: Optional[List[float]]
    rankings: List[int]
    k_requested: int
    k_selected: int
    random_state: int
    warnings: List[str]
    config: Dict[str, Any]
    fitted_timestamp_utc: str = ""


@dataclass
class FeatureSelectionPreview:
    """
    JSON-serializable summary of feature selection results for frontend display.

    Attributes:
        method: Selection method label.
        mode: Selection mode label.
        k_requested: Number of features requested.
        k_selected: Actual features selected.
        features: Per-feature metadata list (name, score, pvalue, rank, selected).
        selected_feature_names: Ordered names of selected features.
        warnings: Advisory messages.
    """
    method: str
    mode: str
    k_requested: int
    k_selected: int
    features: List[Dict[str, Any]]
    selected_feature_names: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return fully JSON-serializable dict."""
        return asdict(self)



# ---------------------------------------------------------------------------
# Day 10 — Correlation Filter Schemas
# ---------------------------------------------------------------------------

@dataclass
class CorrelationFilterConfig:
    """
    Configuration for Pearson correlation-based feature filtering.

    Attributes:
        threshold: Absolute correlation threshold in (0.0, 1.0].
                   Features with |corr| >= threshold are considered redundant.
                   Default 0.95.
        method: Correlation method. Currently only 'pearson' is supported.
    """
    threshold: float = 0.95
    method: str = "pearson"

    def __post_init__(self) -> None:
        valid_methods = {"pearson"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid correlation method: '{self.method}'. Allowed: {sorted(valid_methods)}"
            )
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be in (0.0, 1.0], got: {self.threshold}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "CorrelationFilterConfig"]]) -> "CorrelationFilterConfig":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**dict(data))


@dataclass
class CorrelatedPairInfo:
    """Metadata for a single correlated feature pair."""
    feature_a: str
    feature_b: str
    correlation: float
    removed_feature: str
    reason: str


@dataclass
class CorrelationFilterArtifact:
    """
    Container storing the result of a fitted correlation filter.

    Attributes:
        retained_feature_names: Features kept after filtering (original order preserved).
        removed_feature_names: Features removed due to high correlation.
        correlated_pairs: List of CorrelatedPairInfo for each redundant pair detected.
        config: Original CorrelationFilterConfig as dict.
        original_feature_names: All input feature names.
        n_original: Total original feature count.
        n_retained: Retained feature count.
        n_removed: Removed feature count.
        warnings: Advisory messages (constant features, non-finite values, etc.).
        fitted_timestamp_utc: ISO 8601 timestamp.
    """
    retained_feature_names: List[str]
    removed_feature_names: List[str]
    correlated_pairs: List[Any]  # List[CorrelatedPairInfo]
    config: Dict[str, Any]
    original_feature_names: List[str]
    n_original: int
    n_retained: int
    n_removed: int
    warnings: List[str]
    fitted_timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Day 10 — PCA Schemas
# ---------------------------------------------------------------------------

@dataclass
class PCAConfig:
    """
    Configuration for PCA dimensionality reduction.

    Exactly one of n_components or variance_target must be provided.

    Attributes:
        n_components: Explicit number of principal components (Mode A).
        variance_target: Minimum cumulative explained variance to retain (Mode B).
                         Float in (0.0, 1.0].
        random_state: Seed for reproducibility.
        whiten: Whether to whiten PCA components.
    """
    n_components: Optional[int] = None
    variance_target: Optional[float] = None
    random_state: int = 42
    whiten: bool = False

    def __post_init__(self) -> None:
        if self.n_components is None and self.variance_target is None:
            raise ValueError(
                "Exactly one of n_components or variance_target must be provided."
            )
        if self.n_components is not None and self.variance_target is not None:
            raise ValueError(
                "Provide either n_components or variance_target, not both."
            )
        if self.n_components is not None:
            if not isinstance(self.n_components, int) or isinstance(self.n_components, bool):
                raise ValueError(
                    f"n_components must be a positive integer, got: {type(self.n_components).__name__}"
                )
            if self.n_components <= 0:
                raise ValueError(f"n_components must be > 0, got: {self.n_components}")
        if self.variance_target is not None:
            if not (0.0 < self.variance_target <= 1.0):
                raise ValueError(
                    f"variance_target must be in (0.0, 1.0], got: {self.variance_target}"
                )
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(
                f"random_state must be an integer, got: {type(self.random_state).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "PCAConfig"]]) -> "PCAConfig":
        if data is None:
            raise ValueError("PCAConfig.from_dict: data cannot be None. Provide n_components or variance_target.")
        if isinstance(data, cls):
            return data
        return cls(**dict(data))


@dataclass
class PCAArtifact:
    """
    Container storing a fitted PCA transformer and transformation metadata.

    Attributes:
        pca: Fitted sklearn PCA instance.
        original_feature_names: Names of input features (before PCA).
        component_names: Names of output components (PC1, PC2, ...).
        n_components_actual: Number of components actually used.
        explained_variance_ratio: Per-component explained variance ratio.
        cumulative_explained_variance: Cumulative explained variance per component.
        total_variance_retained: Sum of all selected component variance ratios.
        config: Original PCAConfig as dict.
        warnings: Advisory messages.
        fitted_timestamp_utc: ISO 8601 timestamp.
    """
    pca: Any
    original_feature_names: List[str]
    component_names: List[str]
    n_components_actual: int
    explained_variance_ratio: List[float]
    cumulative_explained_variance: List[float]
    total_variance_retained: float
    config: Dict[str, Any]
    warnings: List[str]
    fitted_timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pca", None)  # sklearn objects are not JSON-serializable
        return d


# ---------------------------------------------------------------------------
# Day 11 — Comparison Modes & Compatibility Schemas
# ---------------------------------------------------------------------------

MIN_QUBITS: int = 2
MAX_QUBITS: int = 10
VALID_COMPARISON_MODES = {"matched_reduced", "quantum_limited", "best_practical"}
VALID_ENCODING_STRATEGIES = {"angle", "angle_embedding", "zz_feature_map", "zz"}


@dataclass
class QubitCompatibilityResult:
    """
    Structured audit report of feature-to-qubit compatibility.

    Attributes:
        is_compatible: True if the feature representation meets all quantum circuit requirements.
        n_qubits: Target number of qubits evaluated.
        n_features: Number of input features presented to the quantum model.
        encoding_strategy: Evaluated encoding method ('angle' or 'zz_feature_map').
        feature_to_qubit_map: Deterministic mapping of qubit index to feature name.
        errors: Detailed validation failure descriptions (if any).
        warnings: Advisory messages or non-fatal observations.
    """
    is_compatible: bool
    n_qubits: int
    n_features: int
    encoding_strategy: str
    feature_to_qubit_map: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable representation."""
        return asdict(self)


@dataclass
class ModelInputMetadata:
    """
    Provenance and feature specification for a specific model family's input.

    Attributes:
        model_family: Target model family ('classical' or 'quantum').
        feature_names: Ordered list of feature/component names.
        n_features: Total count of features.
        feature_type: Semantic representation type ('original_features' or 'transformed_components').
        transformations_applied: Sequence of transformation step names applied.
        reduction_details: Detailed configuration and statistics from reduction steps.
    """
    model_family: str
    feature_names: List[str]
    n_features: int
    feature_type: str
    transformations_applied: List[str] = field(default_factory=list)
    reduction_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable representation."""
        return asdict(self)


@dataclass
class ComparisonModeConfig:
    """
    Configuration governing dataset preparation for classical vs. quantum comparison.

    Modes:
      - 'matched_reduced' (Mode A, Primary): Both model families receive identical features.
      - 'quantum_limited' (Mode B): Feature count strictly dictated by quantum hardware limits.
      - 'best_practical' (Mode C): Models may receive separate, independently-optimized representations.

    Attributes:
        mode: Comparison mode ('matched_reduced', 'quantum_limited', 'best_practical').
        qubit_count: Target number of qubits for quantum evaluation (required for Mode B, in [2, 10]).
        encoding_strategy: Quantum data encoding method ('angle' or 'zz_feature_map').
        reduction_method: Shared reduction algorithm for Modes A & B ('mutual_info', 'anova', 'correlation', 'pca', 'none').
        feature_selection_config: Optional FeatureSelectionConfig for MI/ANOVA.
        correlation_filter_config: Optional CorrelationFilterConfig.
        pca_config: Optional PCAConfig.
        classical_reduction_method: Independent reduction method for classical models in Mode C.
        classical_feature_selection_config: FeatureSelectionConfig for classical in Mode C.
        classical_correlation_filter_config: CorrelationFilterConfig for classical in Mode C.
        classical_pca_config: PCAConfig for classical in Mode C.
        quantum_reduction_method: Independent reduction method for quantum models in Mode C.
        quantum_feature_selection_config: FeatureSelectionConfig for quantum in Mode C.
        quantum_correlation_filter_config: CorrelationFilterConfig for quantum in Mode C.
        quantum_pca_config: PCAConfig for quantum in Mode C.
    """
    mode: str = "matched_reduced"
    qubit_count: Optional[int] = None
    encoding_strategy: str = "angle"
    reduction_method: Optional[str] = None
    feature_selection_config: Optional[Union[Dict[str, Any], "FeatureSelectionConfig"]] = None
    correlation_filter_config: Optional[Union[Dict[str, Any], "CorrelationFilterConfig"]] = None
    pca_config: Optional[Union[Dict[str, Any], "PCAConfig"]] = None

    # Mode C (Best Practical) independent pipeline overrides
    classical_reduction_method: Optional[str] = None
    classical_feature_selection_config: Optional[Union[Dict[str, Any], "FeatureSelectionConfig"]] = None
    classical_correlation_filter_config: Optional[Union[Dict[str, Any], "CorrelationFilterConfig"]] = None
    classical_pca_config: Optional[Union[Dict[str, Any], "PCAConfig"]] = None

    quantum_reduction_method: Optional[str] = None
    quantum_feature_selection_config: Optional[Union[Dict[str, Any], "FeatureSelectionConfig"]] = None
    quantum_correlation_filter_config: Optional[Union[Dict[str, Any], "CorrelationFilterConfig"]] = None
    quantum_pca_config: Optional[Union[Dict[str, Any], "PCAConfig"]] = None

    def __post_init__(self) -> None:
        """Validate configuration settings strictly."""
        if self.mode not in VALID_COMPARISON_MODES:
            raise ValueError(
                f"Invalid comparison mode: '{self.mode}'. Allowed: {sorted(VALID_COMPARISON_MODES)}"
            )

        # Standardize encoding strategy
        if self.encoding_strategy in ("angle", "angle_embedding"):
            self.encoding_strategy = "angle"
        elif self.encoding_strategy in ("zz", "zz_feature_map"):
            self.encoding_strategy = "zz_feature_map"
        else:
            raise ValueError(
                f"Invalid encoding strategy: '{self.encoding_strategy}'. Allowed: {sorted(VALID_ENCODING_STRATEGIES)}"
            )

        if self.mode == "quantum_limited":
            if self.qubit_count is None:
                raise ValueError("qubit_count is mandatory when mode='quantum_limited'.")

        if self.qubit_count is not None:
            if not isinstance(self.qubit_count, int) or isinstance(self.qubit_count, bool):
                raise ValueError(
                    f"qubit_count must be an integer, got: {type(self.qubit_count).__name__}"
                )
            if not (MIN_QUBITS <= self.qubit_count <= MAX_QUBITS):
                raise ValueError(
                    f"qubit_count={self.qubit_count} is outside the supported range [{MIN_QUBITS}, {MAX_QUBITS}]."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "ComparisonModeConfig"]]) -> "ComparisonModeConfig":
        """Construct from dict or instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**dict(data))


@dataclass
class ComparisonModeResult:
    """
    Complete prepared dataset bundle partitioned for classical and quantum model execution.

    Attributes:
        mode: The executed comparison mode ('matched_reduced', 'quantum_limited', 'best_practical').
        is_matched_input: True strictly when classical and quantum models receive identical representations.
        X_train_classical: Training features for classical algorithms.
        X_test_classical: Test features for classical algorithms.
        X_train_quantum: Training features for quantum circuits.
        X_test_quantum: Test features for quantum circuits.
        y_train: Training labels (shared across both families).
        y_test: Test labels (shared across both families).
        classical_metadata: Feature provenance and configuration for classical input.
        quantum_metadata: Feature provenance and configuration for quantum input.
        qubit_compatibility: Validation report for quantum hardware compatibility.
        validation_errors: Any errors preventing valid benchmark execution.
        validation_warnings: Any non-fatal warnings or advisories.
        fitted_timestamp_utc: ISO 8601 timestamp.
    """
    mode: str
    is_matched_input: bool
    X_train_classical: pd.DataFrame
    X_test_classical: pd.DataFrame
    X_train_quantum: pd.DataFrame
    X_test_quantum: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    classical_metadata: ModelInputMetadata
    quantum_metadata: ModelInputMetadata
    qubit_compatibility: Optional[QubitCompatibilityResult] = None
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    fitted_timestamp_utc: str = ""

    @property
    def is_valid(self) -> bool:
        """Indicate whether the prepared dataset is valid for model benchmarking."""
        return len(self.validation_errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Return structured, JSON-serializable metadata summary without raw DataFrames."""
        return {
            "mode": self.mode,
            "is_matched_input": self.is_matched_input,
            "is_valid": self.is_valid,
            "classical_metadata": self.classical_metadata.to_dict(),
            "quantum_metadata": self.quantum_metadata.to_dict(),
            "qubit_compatibility": (
                self.qubit_compatibility.to_dict() if self.qubit_compatibility else None
            ),
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "fitted_timestamp_utc": self.fitted_timestamp_utc,
            "n_train_samples": len(self.X_train_classical),
            "n_test_samples": len(self.X_test_classical),
        }
