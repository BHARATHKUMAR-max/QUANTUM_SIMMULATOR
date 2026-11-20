"""
tests/test_leakage_prevention.py
================================
Adversarial test suite designed specifically to detect and expose data leakage.

Verifies:
  1. The numerical imputer learns statistics strictly from training data.
  2. The StandardScaler learns mean and variance strictly from training data.
  3. The OneHotEncoder learns categories strictly from training data.
  4. Transforming test data leaves fitted preprocessor attributes completely unchanged.
  5. Cross-validation fold preprocessors are fitted independently per fold.
  6. Held-out test samples never appear in any cross-validation training or validation fold.
  7. Regression proof: fitting on training data produces strictly different parameters
     than fitting on the full dataset (proving leakage would alter the model).
"""

import copy
import pytest
import numpy as np
import pandas as pd

from backend.data.preprocessing import (
    fit_preprocessor,
    prepare_cv_fold_data,
    prepare_split_and_preprocessed_data,
    transform_with_preprocessor,
)
from backend.data.schemas import DatasetBundle, PreprocessingConfig
from backend.data.splitting import create_stratified_cv_splits, create_train_test_split


@pytest.fixture
def adversarial_dataset():
    """
    Construct a synthetic dataset specifically crafted to expose data leakage:
      - Training values have mean ~10.0, std ~2.0.
      - Test values have mean ~1000.0, std ~50.0.
      - Training categories: ['A', 'B'].
      - Test categories: ['B', 'UNSEEN_C'].
      - Missing values present in both train and test.
    """
    n_train = 20
    n_test = 20
    total = n_train + n_test

    # Deliberate distribution shift between partitions
    train_vals = np.linspace(8.0, 12.0, n_train)  # mean = 10.0, median = 10.0
    test_vals = np.linspace(900.0, 1100.0, n_test)  # mean = 1000.0, median = 1000.0
    num_col = np.concatenate([train_vals, test_vals])

    # Introduce missing value in train
    num_col[0] = np.nan
    # Introduce missing value in test
    num_col[-1] = np.nan

    cat_col = ["A", "B"] * (n_train // 2) + ["B", "UNSEEN_C"] * (n_test // 2)
    labels = [0, 1] * (total // 2)

    df_x = pd.DataFrame({
        "metric": num_col,
        "category": cat_col,
    })
    s_y = pd.Series(labels, name="target")

    bundle = DatasetBundle(
        X=df_x,
        y=s_y,
        feature_names=["metric", "category"],
        target_names=["class_0", "class_1"],
        dataset_name="adversarial_leakage_benchmark",
        target_name="target",
    )
    return bundle, n_train, n_test


def test_imputer_and_scaler_learned_from_train_only(adversarial_dataset):
    """
    Verify that the imputer and scaler statistics match X_train exactly
    and do NOT incorporate any information from X_test.
    """
    bundle, n_train, n_test = adversarial_dataset
    # Split with first n_train as train and remaining n_test as test (unstratified, un-shuffled for exact partition isolation)
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    preprocessor = fit_preprocessor(split.X_train, feature_types={"metric": "numerical", "category": "categorical"})

    # Extract internal numerical pipeline
    ct = preprocessor.transformer
    num_pipe = ct.named_transformers_["num"]
    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]

    # Expected train statistics (ignoring NaN)
    expected_train_median = split.X_train["metric"].dropna().median()
    expected_train_mean = split.X_train["metric"].dropna().mean()

    # The imputer's learned median must equal training median exactly
    learned_median = imputer.statistics_[0]
    assert np.isclose(learned_median, expected_train_median, atol=1e-5)

    # Combined dataset median would be drastically different due to test values (~1000.0)
    full_median = bundle.X["metric"].dropna().median()
    assert abs(learned_median - full_median) > 100.0, "Leakage detected: Imputer learned full dataset median!"

    # The scaler's learned mean must match training mean (after imputation)
    imputed_train_metric = split.X_train["metric"].fillna(learned_median)
    assert np.isclose(scaler.mean_[0], imputed_train_metric.mean(), atol=1e-5)

    # If test data had leaked, scaler.mean_ would be significantly larger
    full_mean = bundle.X["metric"].dropna().mean()
    assert abs(scaler.mean_[0] - full_mean) > 100.0, "Leakage detected: Scaler learned full dataset mean!"


def test_categorical_encoder_learned_from_train_only(adversarial_dataset):
    """
    Verify that OneHotEncoder learns categories strictly from training data.
    """
    bundle, _, _ = adversarial_dataset
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    preprocessor = fit_preprocessor(split.X_train, feature_types={"metric": "numerical", "category": "categorical"})

    cat_pipe = preprocessor.transformer.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]

    # Categories learned must be only those present in training data
    learned_categories = list(ohe.categories_[0])
    train_unique_cats = sorted(split.X_train["category"].dropna().unique().tolist())
    assert learned_categories == train_unique_cats

    # 'UNSEEN_C' from test data must NOT be in learned categories
    assert "UNSEEN_C" not in learned_categories


def test_preprocessor_parameters_unchanged_after_test_transform(adversarial_dataset):
    """
    Verify that calling transform() on test data does not modify fitted attributes (pure function of X).
    """
    bundle, _, _ = adversarial_dataset
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    preprocessor = fit_preprocessor(split.X_train, feature_types={"metric": "numerical", "category": "categorical"})

    # Snapshot scaler mean and variance before transforming test
    num_pipe = preprocessor.transformer.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    mean_before = copy.deepcopy(scaler.mean_)
    var_before = copy.deepcopy(scaler.var_)

    # Transform test partition
    _ = transform_with_preprocessor(preprocessor, split.X_test)

    # Assert fitted attributes are unchanged
    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.var_, var_before)


def test_cv_folds_fitted_independently():
    """
    Verify that each cross-validation fold fits its own separate preprocessor
    with parameters derived strictly from that fold's training slice.
    """
    # 30 samples with monotonically increasing feature values
    df_x = pd.DataFrame({"value": [float(i * 10) for i in range(30)]})
    s_y = pd.Series([0 if i % 2 == 0 else 1 for i in range(30)], name="target")

    folds = create_stratified_cv_splits(df_x, s_y, n_splits=3, random_state=42)

    fold_means = []
    for fold in folds:
        fold_preprocessor, x_f_tr, x_f_val, y_f_tr, y_f_val = prepare_cv_fold_data(
            X_train=df_x,
            y_train=s_y,
            fold=fold,
            feature_types={"value": "numerical"},
        )
        ct = fold_preprocessor.transformer
        scaler = ct.named_transformers_["num"].named_steps["scaler"]
        fold_means.append(scaler.mean_[0])

        # Validate that fold scaler mean equals that fold's training slice mean
        expected_fold_mean = df_x.loc[fold.train_indices, "value"].mean()
        assert np.isclose(scaler.mean_[0], expected_fold_mean)

    # Because each fold leaves out a different third, fold training means must differ
    assert len(set(fold_means)) == 3, "CV folds did not fit independent preprocessors!"


def test_held_out_test_set_never_in_cv_folds():
    """
    Verify that test indices are strictly quarantined from all cross-validation folds.
    """
    df_x = pd.DataFrame({"x": [float(i) for i in range(50)]})
    s_y = pd.Series([i % 2 for i in range(50)], name="target")
    bundle = DatasetBundle(
        X=df_x,
        y=s_y,
        feature_names=["x"],
        target_names=["0", "1"],
        dataset_name="cv_quarantine_test",
    )

    split = create_train_test_split(bundle, test_size=0.2, random_state=42)
    test_indices_set = set(split.test_indices)

    folds = create_stratified_cv_splits(split.X_train, split.y_train, n_splits=3, random_state=42)

    for fold in folds:
        # Neither fold train nor fold val may contain any test index
        assert len(test_indices_set.intersection(set(fold.train_indices))) == 0
        assert len(test_indices_set.intersection(set(fold.val_indices))) == 0


def test_leakage_regression_proof(adversarial_dataset):
    """
    Regression Test: Prove that fitting on training data produces strictly
    different parameters than fitting on the full dataset (demonstrating why leakage prevention matters).
    """
    bundle, _, _ = adversarial_dataset
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    # 1. Leakage-safe preprocessor (fitted on X_train only)
    safe_preprocessor = fit_preprocessor(split.X_train, feature_types={"metric": "numerical", "category": "categorical"})
    safe_scaler = safe_preprocessor.transformer.named_transformers_["num"].named_steps["scaler"]

    # 2. Deliberately leaked preprocessor (fitted on full bundle.X)
    leaked_preprocessor = fit_preprocessor(bundle.X, feature_types={"metric": "numerical", "category": "categorical"})
    leaked_scaler = leaked_preprocessor.transformer.named_transformers_["num"].named_steps["scaler"]

    # Difference in learned mean must be massive (e.g. ~10 vs ~200)
    mean_diff = abs(safe_scaler.mean_[0] - leaked_scaler.mean_[0])
    assert mean_diff > 50.0, f"Expected large difference in mean due to leakage, got: {mean_diff}"


def test_minmax_scaler_learned_from_train_only(adversarial_dataset):
    """
    Verify that MinMaxScaler learns min and max statistics strictly from training data
    and does not incorporate extreme test distribution values.
    """
    bundle, _, _ = adversarial_dataset
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    cfg = PreprocessingConfig(numerical_scaling="minmax")
    preprocessor = fit_preprocessor(
        split.X_train,
        feature_types={"metric": "numerical", "category": "categorical"},
        config=cfg,
    )

    num_pipe = preprocessor.transformer.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]

    expected_train_min = split.X_train["metric"].dropna().min()
    expected_train_max = split.X_train["metric"].dropna().max()

    assert np.isclose(scaler.data_min_[0], expected_train_min, atol=1e-5)
    assert np.isclose(scaler.data_max_[0], expected_train_max, atol=1e-5)

    # If test data had leaked, data_max_ would be > 1000.0 instead of ~12.0
    full_max = bundle.X["metric"].dropna().max()
    assert abs(scaler.data_max_[0] - full_max) > 500.0, "Leakage detected: MinMaxScaler learned test values!"


def test_ordinal_encoder_learned_from_train_only(adversarial_dataset):
    """
    Verify that OrdinalEncoder learns category mapping strictly from training data,
    and maps unseen test categories to -1.0 without modifying encoder state.
    """
    bundle, _, _ = adversarial_dataset
    split = create_train_test_split(bundle, test_size=0.5, random_state=42, stratify=False, shuffle=False)

    cfg = PreprocessingConfig(categorical_encoding="ordinal")
    preprocessor = fit_preprocessor(
        split.X_train,
        feature_types={"metric": "numerical", "category": "categorical"},
        config=cfg,
    )

    cat_pipe = preprocessor.transformer.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["encoder"]

    # Learned categories must match training categories only
    learned_cats = list(encoder.categories_[0])
    train_cats = sorted(split.X_train["category"].dropna().unique().tolist())
    assert learned_cats == train_cats
    assert "UNSEEN_C" not in learned_cats

    # Transforming test data must map UNSEEN_C to -1.0
    X_test_proc = transform_with_preprocessor(preprocessor, split.X_test)
    assert not X_test_proc.isnull().any().any()
    # Unseen categories must be -1.0 — column is prefixed cat__category
    cat_col = [c for c in X_test_proc.columns if "category" in c][0]
    unseen_rows = (split.X_test["category"] == "UNSEEN_C").values
    assert (X_test_proc.loc[unseen_rows, cat_col] == -1.0).all()

    # Re-verify that categories_ was not altered
    assert list(encoder.categories_[0]) == learned_cats


def test_cv_folds_with_configurable_preprocessors_independent():
    """
    Verify that cross-validation folds with custom PreprocessingConfig
    (mean imputation, minmax scaling, ordinal encoding) fit completely independent preprocessors.
    """
    df_x = pd.DataFrame({
        "num": [float(i * 10) for i in range(30)],
        "cat": ["tier_a" if i < 15 else "tier_b" for i in range(30)],
    })
    s_y = pd.Series([0 if i % 2 == 0 else 1 for i in range(30)], name="target")

    folds = create_stratified_cv_splits(df_x, s_y, n_splits=3, random_state=42)

    cfg = PreprocessingConfig(
        numerical_imputation="mean",
        numerical_scaling="minmax",
        categorical_encoding="ordinal",
    )

    fold_mins = []
    fold_maxs = []
    for fold in folds:
        fold_preprocessor, x_f_tr, x_f_val, y_f_tr, y_f_val = prepare_cv_fold_data(
            X_train=df_x,
            y_train=s_y,
            fold=fold,
            feature_types={"num": "numerical", "cat": "categorical"},
            config=cfg,
        )
        ct = fold_preprocessor.transformer
        scaler = ct.named_transformers_["num"].named_steps["scaler"]
        fold_mins.append(scaler.data_min_[0])
        fold_maxs.append(scaler.data_max_[0])

        # Validate that fold min/max match fold training slice min/max
        expected_min = df_x.loc[fold.train_indices, "num"].min()
        expected_max = df_x.loc[fold.train_indices, "num"].max()
        assert np.isclose(scaler.data_min_[0], expected_min)
        assert np.isclose(scaler.data_max_[0], expected_max)

        # Validation fold features must be transformed cleanly without NaNs
        assert not x_f_val.isnull().any().any()

    # Since each fold omits a different slice of monotonically increasing numbers,
    # the fold mins/maxs must vary across folds
    assert len(set(fold_mins)) > 1 or len(set(fold_maxs)) > 1, "CV folds did not fit independent scalers!"
