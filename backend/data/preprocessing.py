"""
backend/data/preprocessing.py
=============================
Configurable, leakage-safe preprocessing and feature transformation engine for QML Benchmark Studio.

Constructs scikit-learn ColumnTransformer pipelines enforcing strict training-only
fitting for numerical, categorical, and boolean feature transformations. Supports configurable
imputation (median/mean), scaling (standard/minmax/none), and encoding (onehot/ordinal)
while robustly handling all-missing training features and unseen test observations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from backend.data.schemas import (
    CVFoldResult,
    DatasetBundle,
    PreprocessingConfig,
    PreprocessorArtifact,
    ProcessedDatasetResult,
    TrainTestSplitResult,
    infer_feature_types,
)
from backend.data.splitting import create_train_test_split


def _cast_bool_to_float(arr: Any) -> np.ndarray:
    """Helper to convert boolean or mixed boolean-null data to standard float64."""
    if isinstance(arr, pd.DataFrame):
        return arr.astype(float).to_numpy()
    elif isinstance(arr, pd.Series):
        return arr.astype(float).to_numpy().reshape(-1, 1)
    else:
        arr_obj = np.asarray(arr)
        return arr_obj.astype(float)


def fit_preprocessor(
    X_train: pd.DataFrame,
    feature_types: Optional[Dict[str, str]] = None,
    config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
) -> PreprocessorArtifact:
    """
    Construct and fit a leakage-safe ColumnTransformer exclusively on training features.
    
    Supported Strategies:
      - Numerical Imputation: 'median' (default) or 'mean'.
      - Numerical Scaling: 'standard' (StandardScaler, default), 'minmax' (MinMaxScaler [0,1]),
                           or 'none' (identity passthrough).
      - Categorical Imputation: 'most_frequent' (default).
      - Categorical Encoding: 'onehot' (OneHotEncoder with handle_unknown='ignore', default)
                              or 'ordinal' (OrdinalEncoder with unknown_value=-1).
      - Boolean Handling: 'numeric' (cast to float64 + most_frequent imputation).
      - Empty Features: Detects columns with 100% missing values in X_train and prevents
                        silent drops when preserve_empty_features=True.
                        
    Args:
        X_train: Training feature DataFrame.
        feature_types: Mapping of column names to semantic types ('numerical', 'categorical', 'boolean').
        config: PreprocessingConfig dataclass or dictionary of configuration options.
        
    Returns:
        PreprocessorArtifact: Container storing the fitted ColumnTransformer, effective config,
                              feature names, and any detected all-missing training features.
    """
    # 1. Parse and validate configuration
    prep_config = PreprocessingConfig.from_dict(config)

    # 2. Infer feature types if not explicitly supplied
    if feature_types is None:
        feature_types = infer_feature_types(X_train)

    columns = list(X_train.columns)

    # 3. Detect all-missing features in training partition
    all_missing_cols = [c for c in columns if X_train[c].isnull().all()]

    active_columns = columns
    if not prep_config.preserve_empty_features:
        # Policy: Drop all-missing training features
        active_columns = [c for c in columns if c not in all_missing_cols]

    num_cols = [c for c in active_columns if feature_types.get(c) == "numerical"]
    cat_cols = [c for c in active_columns if feature_types.get(c) in ("categorical", "datetime_like")]
    bool_cols = [c for c in active_columns if feature_types.get(c) == "boolean"]

    transformers = []

    # 4. Numerical Pipeline
    if num_cols:
        num_imputer = SimpleImputer(
            strategy=prep_config.numerical_imputation,
            keep_empty_features=prep_config.preserve_empty_features,
        )

        if prep_config.numerical_scaling == "standard":
            scaler = StandardScaler()
        elif prep_config.numerical_scaling == "minmax":
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif prep_config.numerical_scaling == "none":
            scaler = FunctionTransformer(func=None, feature_names_out="one-to-one")
        else:
            raise ValueError(f"Unsupported numerical scaling: '{prep_config.numerical_scaling}'")

        num_pipeline = Pipeline(steps=[("imputer", num_imputer), ("scaler", scaler)])
        transformers.append(("num", num_pipeline, num_cols))

    # 5. Categorical Pipeline
    if cat_cols:
        cat_imputer = SimpleImputer(
            strategy=prep_config.categorical_imputation,
            keep_empty_features=prep_config.preserve_empty_features,
        )

        if prep_config.categorical_encoding == "onehot":
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif prep_config.categorical_encoding == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            raise ValueError(f"Unsupported categorical encoding: '{prep_config.categorical_encoding}'")

        encoder_step_name = "onehot" if prep_config.categorical_encoding == "onehot" else "encoder"
        cat_pipeline = Pipeline(steps=[("imputer", cat_imputer), (encoder_step_name, encoder)])
        transformers.append(("cat", cat_pipeline, cat_cols))

    # 6. Boolean Pipeline
    if bool_cols:
        bool_pipeline = Pipeline(
            steps=[
                (
                    "to_float",
                    FunctionTransformer(
                        _cast_bool_to_float,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=prep_config.preserve_empty_features,
                    ),
                ),
            ]
        )
        transformers.append(("bool", bool_pipeline, bool_cols))

    # 7. Construct ColumnTransformer
    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Strictly fit on training partition
    ct.fit(X_train)

    transformed_names = [str(name) for name in ct.get_feature_names_out()]

    artifact = PreprocessorArtifact(
        transformer=ct,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        boolean_features=bool_cols,
        original_feature_names=columns,
        transformed_feature_names=transformed_names,
        config=prep_config.to_dict(),
        fitted_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        all_missing_training_features=all_missing_cols,
    )
    return artifact


def transform_with_preprocessor(
    preprocessor: PreprocessorArtifact,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform feature data using an already-fitted PreprocessorArtifact without refitting.
    
    Args:
        preprocessor: PreprocessorArtifact containing the fitted ColumnTransformer.
        X: Feature DataFrame to transform (e.g. held-out X_test or inference observations).
        
    Returns:
        pd.DataFrame: Transformed features with aligned column names and preserved original row indices.
    """
    transformed_arr = preprocessor.transformer.transform(X)
    transformed_df = pd.DataFrame(
        transformed_arr,
        columns=preprocessor.transformed_feature_names,
        index=X.index,
    )
    return transformed_df


def prepare_split_and_preprocessed_data(
    dataset_bundle: DatasetBundle,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    preprocessor_config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
) -> ProcessedDatasetResult:
    """
    High-level orchestration: partition dataset and apply strictly leakage-safe preprocessing.
    
    Pipeline Steps:
      1. Partition DatasetBundle into stratified X_train, X_test, y_train, y_test.
      2. Fit ColumnTransformer exclusively on X_train.
      3. Transform X_train into processed feature DataFrame.
      4. Transform X_test using the fitted preprocessor (zero test leakage).
      5. Compile and return structured ProcessedDatasetResult.
      
    Args:
        dataset_bundle: Input DatasetBundle.
        test_size: Proportion of samples allocated to test partition.
        random_state: Seed for reproducible split.
        stratify: Whether to preserve class proportions through stratification.
        preprocessor_config: PreprocessingConfig instance or dictionary of options.
        
    Returns:
        ProcessedDatasetResult: Container with split partitions, fitted preprocessor,
                                transformed DataFrames, and provenance metadata.
    """
    # 1. Train/Test Split
    split_res = create_train_test_split(
        dataset_bundle=dataset_bundle,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 2. Fit Preprocessor exclusively on X_train
    preprocessor = fit_preprocessor(
        X_train=split_res.X_train,
        feature_types=dataset_bundle.feature_types,
        config=preprocessor_config,
    )

    # 3. Transform X_train
    X_train_processed = transform_with_preprocessor(preprocessor, split_res.X_train)

    # 4. Transform X_test without refitting
    X_test_processed = transform_with_preprocessor(preprocessor, split_res.X_test)

    metadata = {
        "dataset_name": dataset_bundle.dataset_name,
        "n_train_samples": len(X_train_processed),
        "n_test_samples": len(X_test_processed),
        "n_original_features": len(dataset_bundle.feature_names),
        "n_processed_features": len(preprocessor.transformed_feature_names),
        "random_state": random_state,
        "test_size": test_size,
        "stratified": stratify,
        "preprocessing_config": preprocessor.config,
        "all_missing_training_features": preprocessor.all_missing_training_features,
    }

    return ProcessedDatasetResult(
        split=split_res,
        preprocessor=preprocessor,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        transformed_feature_names=preprocessor.transformed_feature_names,
        metadata=metadata,
    )


def prepare_cv_fold_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fold: CVFoldResult,
    feature_types: Optional[Dict[str, str]] = None,
    config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
) -> Tuple[PreprocessorArtifact, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Safely prepare a single cross-validation fold with independent preprocessing.
    
    Critical Leakage Safeguard:
      The ColumnTransformer is fitted exclusively on this fold's training slice (fold.train_indices).
      The validation slice (fold.val_indices) is transformed without refitting.
      No information leaks between folds or from the held-out test partition.
      
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        fold: CVFoldResult containing fold indices.
        feature_types: Optional semantic feature types.
        config: PreprocessingConfig or dictionary of configuration options.
        
    Returns:
        Tuple of:
          - PreprocessorArtifact (fitted exclusively on fold training slice)
          - X_fold_train_processed (pd.DataFrame)
          - X_fold_val_processed (pd.DataFrame)
          - y_fold_train (pd.Series)
          - y_fold_val (pd.Series)
    """
    X_fold_train = X_train.loc[fold.train_indices]
    X_fold_val = X_train.loc[fold.val_indices]
    y_fold_train = y_train.loc[fold.train_indices]
    y_fold_val = y_train.loc[fold.val_indices]

    # Fit preprocessor on fold train slice exclusively
    fold_preprocessor = fit_preprocessor(
        X_train=X_fold_train,
        feature_types=feature_types,
        config=config,
    )

    X_fold_train_proc = transform_with_preprocessor(fold_preprocessor, X_fold_train)
    X_fold_val_proc = transform_with_preprocessor(fold_preprocessor, X_fold_val)

    return fold_preprocessor, X_fold_train_proc, X_fold_val_proc, y_fold_train, y_fold_val
