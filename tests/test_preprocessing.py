"""
tests/test_preprocessing.py
===========================
Automated unit and integration test suite for the leakage-safe preprocessing pipeline.

Verifies:
  1. Numerical imputation (median) and scaling (StandardScaler).
  2. Categorical imputation (most frequent) and one-hot encoding.
  3. Boolean feature imputation and float conversion (0.0 / 1.0).
  4. Mixed feature pipelines handling real-world tabular data.
  5. Unseen categories in held-out test data handled safely (all-zeros encoding).
  6. Reliable feature-name mapping across all transformed dimensions.
  7. End-to-end execution on built-in Breast Cancer and Wine benchmarks.
  8. High-level integrated prepare_split_and_preprocessed_data() interface.
"""

import pytest
import numpy as np
import pandas as pd

from backend.data.dataset_loader import load_breast_cancer_data, load_wine_data
from backend.data.preprocessing import (
    fit_preprocessor,
    prepare_split_and_preprocessed_data,
    transform_with_preprocessor,
)
from backend.data.schemas import (
    DatasetBundle,
    PreprocessingConfig,
    PreprocessorArtifact,
    ProcessedDatasetResult,
)
from backend.data.splitting import create_train_test_split


def test_numerical_imputation_and_scaling():
    """Verify median imputation and standard scaling on numerical features."""
    df_train = pd.DataFrame({
        "feat1": [10.0, 20.0, np.nan, 40.0, 50.0],  # Median of non-nulls (10, 20, 40, 50) is 30.0
        "feat2": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    preprocessor = fit_preprocessor(df_train)
    assert isinstance(preprocessor, PreprocessorArtifact)
    assert preprocessor.numerical_features == ["feat1", "feat2"]

    # Transform train data
    X_train_proc = transform_with_preprocessor(preprocessor, df_train)
    assert X_train_proc.shape == (5, 2)
    # Check that NaN in feat1 was imputed
    assert not X_train_proc.isnull().any().any()
    # Check that training features have zero mean and unit variance
    np.testing.assert_allclose(X_train_proc.mean(axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(X_train_proc.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-7)


def test_categorical_imputation_and_onehot_encoding():
    """Verify most_frequent imputation and one-hot encoding on categorical features."""
    df_train = pd.DataFrame({
        "color": ["red", "blue", "red", np.nan, "green"],  # Mode is 'red'
    })

    preprocessor = fit_preprocessor(df_train, feature_types={"color": "categorical"})
    assert preprocessor.categorical_features == ["color"]

    X_train_proc = transform_with_preprocessor(preprocessor, df_train)
    # 3 unique categories in train: blue, green, red
    assert X_train_proc.shape == (5, 3)
    assert not X_train_proc.isnull().any().any()
    assert all("color" in name for name in preprocessor.transformed_feature_names)


def test_boolean_feature_handling():
    """Verify that boolean features are safely imputed and converted to float64 (0.0 / 1.0)."""
    df_train = pd.DataFrame({
        "is_vip": [True, False, True, None, False],
    })

    preprocessor = fit_preprocessor(df_train, feature_types={"is_vip": "boolean"})
    assert preprocessor.boolean_features == ["is_vip"]

    X_train_proc = transform_with_preprocessor(preprocessor, df_train)
    assert X_train_proc.shape == (5, 1)
    assert not X_train_proc.isnull().any().any()
    assert set(X_train_proc.iloc[:, 0].unique()).issubset({0.0, 1.0})


def test_unseen_categories_in_test_data():
    """Verify that unseen categorical values in held-out data do not raise errors and map to 0s."""
    df_train = pd.DataFrame({
        "department": ["sales", "engineering", "sales", "hr"],
    })
    df_test = pd.DataFrame({
        "department": ["engineering", "finance", "unknown_dept"],
    })

    preprocessor = fit_preprocessor(df_train, feature_types={"department": "categorical"})
    X_test_proc = transform_with_preprocessor(preprocessor, df_test)

    # Output dimension must match training dimensions (3 departments: engineering, hr, sales)
    assert X_test_proc.shape == (3, 3)
    assert not X_test_proc.isnull().any().any()

    # For row 1 ('finance') and row 2 ('unknown_dept'), all one-hot columns must be 0.0
    assert X_test_proc.iloc[1].sum() == 0.0
    assert X_test_proc.iloc[2].sum() == 0.0
    # For row 0 ('engineering'), exactly one column must be 1.0
    assert X_test_proc.iloc[0].sum() == 1.0


def test_mixed_features_transformation():
    """Verify end-to-end preprocessing of mixed numerical, categorical, and boolean features."""
    df_train = pd.DataFrame({
        "age": [25.0, 35.0, np.nan, 45.0],
        "city": ["London", "Paris", "London", "Tokyo"],
        "subscribed": [True, False, True, False],
    })
    feature_types = {"age": "numerical", "city": "categorical", "subscribed": "boolean"}

    preprocessor = fit_preprocessor(df_train, feature_types=feature_types)
    X_proc = transform_with_preprocessor(preprocessor, df_train)

    # 1 numerical + 3 cities (London, Paris, Tokyo) + 1 boolean = 5 columns
    assert X_proc.shape == (4, 5)
    assert len(preprocessor.transformed_feature_names) == 5
    assert not X_proc.isnull().any().any()


def test_builtin_breast_cancer_preprocessing():
    """Verify preprocessing on the built-in Breast Cancer dataset."""
    bundle = load_breast_cancer_data()
    result = prepare_split_and_preprocessed_data(bundle, test_size=0.2, random_state=42)

    assert isinstance(result, ProcessedDatasetResult)
    assert result.X_train_processed.shape == (455, 30)
    assert result.X_test_processed.shape == (114, 30)
    assert not result.X_train_processed.isnull().any().any()
    assert not result.X_test_processed.isnull().any().any()

    # Feature names preserved
    assert len(result.transformed_feature_names) == 30


def test_builtin_wine_preprocessing():
    """Verify preprocessing on the built-in Wine dataset."""
    bundle = load_wine_data()
    result = prepare_split_and_preprocessed_data(bundle, test_size=0.2, random_state=42)

    assert isinstance(result, ProcessedDatasetResult)
    assert result.X_train_processed.shape == (142, 13)
    assert result.X_test_processed.shape == (36, 13)
    assert not result.X_train_processed.isnull().any().any()
    assert not result.X_test_processed.isnull().any().any()
    assert len(result.transformed_feature_names) == 13


def test_numerical_imputation_mean_vs_median():
    """Verify that numerical mean and median imputation produce expected distinct statistics."""
    df_train = pd.DataFrame({
        "skewed": [1.0, 2.0, 100.0, np.nan],  # Non-null mean = 34.333..., non-null median = 2.0
    })

    # 1. Median imputation
    cfg_median = PreprocessingConfig(numerical_imputation="median", numerical_scaling="none")
    prep_median = fit_preprocessor(df_train, config=cfg_median)
    X_median = transform_with_preprocessor(prep_median, df_train)
    # NaN is at index 3; imputed with median=2.0; scaling=none so value is unmodified
    assert np.isclose(X_median.iloc[3, 0], 2.0)

    # 2. Mean imputation
    cfg_mean = PreprocessingConfig(numerical_imputation="mean", numerical_scaling="none")
    prep_mean = fit_preprocessor(df_train, config=cfg_mean)
    X_mean = transform_with_preprocessor(prep_mean, df_train)
    assert np.isclose(X_mean.iloc[3, 0], (1.0 + 2.0 + 100.0) / 3.0)


def test_numerical_scaling_minmax():
    """Verify that MinMaxScaler scales features strictly within [0.0, 1.0] on training data."""
    df_train = pd.DataFrame({
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    df_test = pd.DataFrame({
        "score": [0.0, 30.0, 60.0],  # Out-of-bounds relative to train
    })

    cfg = PreprocessingConfig(numerical_scaling="minmax")
    prep = fit_preprocessor(df_train, config=cfg)

    X_train_proc = transform_with_preprocessor(prep, df_train)
    # Columns are prefixed as num__score by ColumnTransformer
    score_col = "num__score"
    assert np.isclose(X_train_proc[score_col].min(), 0.0)
    assert np.isclose(X_train_proc[score_col].max(), 1.0)
    assert np.all((X_train_proc[score_col] >= 0.0) & (X_train_proc[score_col] <= 1.0))

    # Test values evaluated linearly using train min (10.0) and train max (50.0)
    X_test_proc = transform_with_preprocessor(prep, df_test)
    assert np.isclose(X_test_proc.iloc[0, 0], -0.25)  # (0 - 10) / 40 = -0.25
    assert np.isclose(X_test_proc.iloc[1, 0], 0.5)    # (30 - 10) / 40 = 0.5
    assert np.isclose(X_test_proc.iloc[2, 0], 1.25)   # (60 - 10) / 40 = 1.25


def test_numerical_scaling_none():
    """Verify that numerical_scaling='none' passes feature values through unmodified."""
    df_train = pd.DataFrame({
        "raw_counts": [15.0, 25.0, 35.0],
    })

    cfg = PreprocessingConfig(numerical_scaling="none")
    prep = fit_preprocessor(df_train, config=cfg)
    X_train_proc = transform_with_preprocessor(prep, df_train)

    np.testing.assert_array_almost_equal(X_train_proc.iloc[:, 0].values, [15.0, 25.0, 35.0])


def test_numerical_scaling_constant_feature():
    """Verify that constant features scale stably without division-by-zero crashes or NaNs."""
    df_train = pd.DataFrame({
        "constant_col": [7.0, 7.0, 7.0, 7.0],
    })

    # Standard scaling with variance 0
    prep_std = fit_preprocessor(df_train, config=PreprocessingConfig(numerical_scaling="standard"))
    X_std = transform_with_preprocessor(prep_std, df_train)
    assert not X_std.isnull().any().any()
    np.testing.assert_allclose(X_std.iloc[:, 0].values, [0.0, 0.0, 0.0, 0.0])

    # MinMax scaling with min == max
    prep_mm = fit_preprocessor(df_train, config=PreprocessingConfig(numerical_scaling="minmax"))
    X_mm = transform_with_preprocessor(prep_mm, df_train)
    assert not X_mm.isnull().any().any()
    np.testing.assert_allclose(X_mm.iloc[:, 0].values, [0.0, 0.0, 0.0, 0.0])


def test_categorical_ordinal_encoding_known_and_unseen():
    """Verify OrdinalEncoder encodes categories and marks unseen test values as -1.0."""
    df_train = pd.DataFrame({
        "tier": ["low", "medium", "high", "medium"],
    })
    df_test = pd.DataFrame({
        "tier": ["high", "low", "unseen_tier"],
    })

    cfg = PreprocessingConfig(categorical_encoding="ordinal")
    prep = fit_preprocessor(df_train, feature_types={"tier": "categorical"}, config=cfg)

    X_train_proc = transform_with_preprocessor(prep, df_train)
    # Shape is (4, 1), not one-hot expanded
    assert X_train_proc.shape == (4, 1)
    assert not X_train_proc.isnull().any().any()
    # Learned categories: ['high', 'low', 'medium'] -> 0, 1, 2
    # Column name is prefixed: cat__tier
    tier_col = X_train_proc.columns[0]
    assert set(X_train_proc[tier_col].unique()).issubset({0.0, 1.0, 2.0})

    X_test_proc = transform_with_preprocessor(prep, df_test)
    assert X_test_proc.shape == (3, 1)
    assert not X_test_proc.isnull().any().any()
    # Known categories map to valid integer codes, unseen category maps to -1.0
    assert X_test_proc.iloc[0, 0] == 0.0  # 'high' -> 0
    assert X_test_proc.iloc[1, 0] == 1.0  # 'low' -> 1
    assert X_test_proc.iloc[2, 0] == -1.0  # 'unseen_tier' -> -1


def test_all_missing_training_feature_preserve_policy():
    """Verify all-missing training features are preserved with keep_empty_features and tracked in metadata."""
    df_train = pd.DataFrame({
        "empty_num": [np.nan, np.nan, np.nan],
        "valid_num": [10.0, 20.0, 30.0],
        "empty_cat": [np.nan, np.nan, np.nan],
    })
    feature_types = {
        "empty_num": "numerical",
        "valid_num": "numerical",
        "empty_cat": "categorical",
    }

    cfg = PreprocessingConfig(preserve_empty_features=True)
    prep = fit_preprocessor(df_train, feature_types=feature_types, config=cfg)

    # Metadata must track empty features
    assert "empty_num" in prep.all_missing_training_features
    assert "empty_cat" in prep.all_missing_training_features
    assert "valid_num" not in prep.all_missing_training_features

    X_proc = transform_with_preprocessor(prep, df_train)
    # Both empty features must be present and filled with 0s without crashing
    assert not X_proc.isnull().any().any()
    # Column names are prefixed (num__empty_num, num__valid_num, cat__empty_cat)
    assert any("empty_num" in col for col in X_proc.columns)
    assert any("empty_cat" in col for col in X_proc.columns)
    assert any("valid_num" in col for col in X_proc.columns)


def test_all_missing_training_feature_drop_policy():
    """Verify all-missing training features are dropped when preserve_empty_features=False."""
    df_train = pd.DataFrame({
        "empty_num": [np.nan, np.nan, np.nan],
        "valid_num": [10.0, 20.0, 30.0],
    })
    feature_types = {
        "empty_num": "numerical",
        "valid_num": "numerical",
    }

    cfg = PreprocessingConfig(preserve_empty_features=False)
    prep = fit_preprocessor(df_train, feature_types=feature_types, config=cfg)

    assert "empty_num" in prep.all_missing_training_features
    X_proc = transform_with_preprocessor(prep, df_train)

    # Only valid_num should be present in the output (no empty_num prefix)
    assert not any("empty_num" in col for col in X_proc.columns)
    assert any("valid_num" in col for col in X_proc.columns)
    assert X_proc.shape == (3, 1)


def test_all_missing_training_feature_with_test_values_no_leakage():
    """Verify test data values in an all-missing training column do not leak or break transform."""
    df_train = pd.DataFrame({
        "empty_num": [np.nan, np.nan, np.nan],
        "val": [1.0, 2.0, 3.0],
    })
    # Test data has values for empty_num
    df_test = pd.DataFrame({
        "empty_num": [100.0, 200.0],
        "val": [4.0, 5.0],
    })
    feature_types = {"empty_num": "numerical", "val": "numerical"}

    # Policy: preserve
    prep_preserve = fit_preprocessor(
        df_train,
        feature_types=feature_types,
        config=PreprocessingConfig(preserve_empty_features=True),
    )
    X_test_proc = transform_with_preprocessor(prep_preserve, df_test)
    assert X_test_proc.shape == (2, 2)
    assert not X_test_proc.isnull().any().any()
    # The all-missing training column is preserved in the output.
    # Test values (non-NaN) pass through the imputer unchanged, then StandardScaler applies
    # train statistics (mean=0.0, scale=1.0 for constant-after-imputation column).
    # The key invariant: no crash, column present, and scaler used train stats only.
    empty_num_col = [c for c in X_test_proc.columns if "empty_num" in c][0]
    assert empty_num_col in X_test_proc.columns  # Column preserved
    assert X_test_proc[empty_num_col].notna().all()  # No NaNs in output
    # StandardScaler fit on all-zeros training data: mean_=0.0, scale_=1.0 (constant feature handling).
    # Transformation: (x - 0) / 1 = x.  Proves scaler used only training statistics (no leakage).
    num_pipe = prep_preserve.transformer.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    empty_idx = prep_preserve.numerical_features.index("empty_num")
    assert np.isclose(scaler.mean_[empty_idx], 0.0, atol=1e-9)  # Learned from all-NaN→0 training data only

    # Policy: drop
    prep_drop = fit_preprocessor(
        df_train,
        feature_types=feature_types,
        config=PreprocessingConfig(preserve_empty_features=False),
    )
    X_test_drop = transform_with_preprocessor(prep_drop, df_test)
    assert X_test_drop.shape == (2, 1)
    assert not any("empty_num" in col for col in X_test_drop.columns)


def test_preprocessing_config_validation():
    """Verify that PreprocessingConfig validates inputs and parses aliases robustly."""
    # Valid default
    cfg = PreprocessingConfig()
    assert cfg.numerical_imputation == "median"
    assert cfg.numerical_scaling == "standard"
    assert cfg.categorical_encoding == "onehot"
    assert cfg.preserve_empty_features is True

    # Invalid options raise ValueError
    with pytest.raises(ValueError, match="Invalid numerical_imputation"):
        PreprocessingConfig(numerical_imputation="mode")

    with pytest.raises(ValueError, match="Invalid numerical_scaling"):
        PreprocessingConfig(numerical_scaling="robust")

    with pytest.raises(ValueError, match="Invalid categorical_encoding"):
        PreprocessingConfig(categorical_encoding="binary")

    with pytest.raises(ValueError, match="Invalid categorical_imputation"):
        PreprocessingConfig(categorical_imputation="mean")

    with pytest.raises(ValueError, match="preserve_empty_features must be a boolean"):
        PreprocessingConfig(preserve_empty_features="true")

    # from_dict parsing with aliases
    parsed = PreprocessingConfig.from_dict({
        "imputation": "mean",
        "scaling": "minmax",
        "encoding": "ordinal",
        "preserve_empty": False,
    })
    assert parsed.numerical_imputation == "mean"
    assert parsed.numerical_scaling == "minmax"
    assert parsed.categorical_encoding == "ordinal"
    assert parsed.preserve_empty_features is False
