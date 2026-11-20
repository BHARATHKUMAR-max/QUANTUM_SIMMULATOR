"""
tests/test_data_splitting.py
============================
Automated unit and integration test suite for reproducible, leakage-safe dataset splitting.

Verifies:
  1. Breast Cancer Wisconsin stratified train/test partitioning.
  2. Wine multiclass stratified train/test partitioning.
  3. Determinism: identical random seeds produce bitwise-identical indices.
  4. Seed sensitivity: distinct seeds generate distinct partition boundaries.
  5. Mutual exclusivity: training and test index sets have zero intersection.
  6. Completeness: union of training and test index sets exactly recovers original instances.
  7. Target isolation: target column is strictly excluded from X_train and X_test.
  8. Index and label preservation: original DataFrame indices and target values are unmodified.
  9. Input validation: out-of-bounds or non-float test_size parameters are rejected.
  10. Stratification safety: insufficient class instances trigger clear, informative errors.
  11. Stratified cross-validation fold generation on the training set.
"""

import pytest
import numpy as np
import pandas as pd

from backend.data.dataset_loader import load_breast_cancer_data, load_wine_data
from backend.data.schemas import (
    DEFAULT_SPLIT_SEEDS,
    DatasetBundle,
    TrainTestSplitResult,
)
from backend.data.splitting import (
    create_stratified_cv_splits,
    create_train_test_split,
)


def test_breast_cancer_stratified_split():
    """Verify stratified partitioning on the binary Breast Cancer benchmark."""
    bundle = load_breast_cancer_data()
    split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)

    assert isinstance(split, TrainTestSplitResult)
    # 569 samples: 80% = 455, 20% = 114
    assert split.X_train.shape == (455, 30)
    assert split.X_test.shape == (114, 30)
    assert len(split.y_train) == 455
    assert len(split.y_test) == 114

    # Target isolation
    assert bundle.target_name not in split.X_train.columns
    assert bundle.target_name not in split.X_test.columns

    # Class proportion preservation (approx 62.7% benign, 37.3% malignant)
    train_benign_ratio = split.train_class_distribution["1"] / split.n_train
    test_benign_ratio = split.test_class_distribution["1"] / split.n_test
    original_benign_ratio = (bundle.y == 1).mean()

    assert abs(train_benign_ratio - original_benign_ratio) < 0.02
    assert abs(test_benign_ratio - original_benign_ratio) < 0.02


def test_wine_multiclass_stratified_split():
    """Verify stratified partitioning on the 3-class Wine benchmark."""
    bundle = load_wine_data()
    split = create_train_test_split(bundle, test_size=0.2, random_state=42, stratify=True)

    # 178 samples: 80% = 142, 20% = 36
    assert split.X_train.shape == (142, 13)
    assert split.X_test.shape == (36, 13)
    assert len(split.y_train) == 142
    assert len(split.y_test) == 36

    # All 3 classes must be present in both partitions
    assert len(split.train_class_distribution) == 3
    assert len(split.test_class_distribution) == 3
    assert set(split.train_class_distribution.keys()) == {"0", "1", "2"}
    assert set(split.test_class_distribution.keys()) == {"0", "1", "2"}


def test_reproducibility_same_seed():
    """Verify that identical random_state produces identical train and test indices."""
    bundle = load_breast_cancer_data()
    split1 = create_train_test_split(bundle, test_size=0.2, random_state=42)
    split2 = create_train_test_split(bundle, test_size=0.2, random_state=42)

    assert split1.train_indices == split2.train_indices
    assert split1.test_indices == split2.test_indices
    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
    pd.testing.assert_frame_equal(split1.X_test, split2.X_test)
    pd.testing.assert_series_equal(split1.y_train, split2.y_train)
    pd.testing.assert_series_equal(split1.y_test, split2.y_test)


def test_distinct_seeds_produce_different_splits():
    """Verify that multiple benchmark seeds generate distinct, reproducible partitions."""
    bundle = load_breast_cancer_data()
    splits = [create_train_test_split(bundle, test_size=0.2, random_state=seed) for seed in DEFAULT_SPLIT_SEEDS]

    # Verify all 5 benchmark seeds generate different train indices
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert splits[i].train_indices != splits[j].train_indices, f"Seeds {DEFAULT_SPLIT_SEEDS[i]} and {DEFAULT_SPLIT_SEEDS[j]} produced identical splits!"


def test_indices_mutual_exclusivity_and_completeness():
    """Verify that train and test indices are disjoint and sum to all original rows."""
    bundle = load_breast_cancer_data()
    split = create_train_test_split(bundle, test_size=0.25, random_state=42)

    train_set = set(split.train_indices)
    test_set = set(split.test_indices)
    original_set = set(bundle.X.index)

    # Disjoint
    assert len(train_set.intersection(test_set)) == 0
    # Complete
    assert train_set.union(test_set) == original_set
    assert len(split.train_indices) + len(split.test_indices) == len(bundle.X)


def test_original_indices_and_target_labels_preserved():
    """Verify that split retains non-standard custom DataFrame indices."""
    custom_index = [f"sample_{i:04d}" for i in range(50)]
    df_x = pd.DataFrame({
        "feat1": np.linspace(1, 10, 50),
        "feat2": np.linspace(20, 30, 50),
    }, index=custom_index)
    s_y = pd.Series(["active" if i % 2 == 0 else "control" for i in range(50)], index=custom_index, name="treatment")

    bundle = DatasetBundle(
        X=df_x,
        y=s_y,
        feature_names=["feat1", "feat2"],
        target_names=["active", "control"],
        dataset_name="custom_index_dataset",
        target_name="treatment",
    )

    split = create_train_test_split(bundle, test_size=0.2, random_state=42)

    # Verify custom string indices are preserved in X and y
    assert all(isinstance(idx, str) and idx.startswith("sample_") for idx in split.train_indices)
    assert all(isinstance(idx, str) and idx.startswith("sample_") for idx in split.test_indices)
    assert list(split.X_train.index) == split.train_indices
    assert list(split.y_train.index) == split.train_indices
    assert set(split.y_train.unique()) == {"active", "control"}


def test_invalid_test_size_raises_value_error():
    """Verify that test_size <= 0.0 or >= 1.0 or non-numeric types are rejected."""
    bundle = load_breast_cancer_data()

    with pytest.raises(ValueError, match="test_size must be strictly between 0.0 and 1.0"):
        create_train_test_split(bundle, test_size=0.0)

    with pytest.raises(ValueError, match="test_size must be strictly between 0.0 and 1.0"):
        create_train_test_split(bundle, test_size=1.0)

    with pytest.raises(ValueError, match="test_size must be strictly between 0.0 and 1.0"):
        create_train_test_split(bundle, test_size=-0.2)

    with pytest.raises(ValueError, match="Invalid test_size type"):
        create_train_test_split(bundle, test_size="0.2")


def test_insufficient_class_count_stratification_error():
    """Verify that requesting stratified split on a minority class with < 2 samples fails clearly."""
    df_x = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    # 5 samples in class 0, only 1 sample in class 1
    s_y = pd.Series([0, 0, 0, 0, 0, 1], name="target")

    bundle = DatasetBundle(
        X=df_x,
        y=s_y,
        feature_names=["f1"],
        target_names=["class_0", "class_1"],
        dataset_name="imbalanced_extreme",
        target_name="target",
    )

    # Must raise informative error and not silently fall back to unstratified
    with pytest.raises(ValueError, match="minority class '1' has only 1 sample"):
        create_train_test_split(bundle, test_size=0.33, stratify=True)

    # But unstratified split is explicitly allowed
    unstratified_split = create_train_test_split(bundle, test_size=0.33, stratify=False, random_state=42)
    assert isinstance(unstratified_split, TrainTestSplitResult)
    assert unstratified_split.stratified is False


def test_stratified_cv_splits():
    """Verify stratified cross-validation fold generation on the training set."""
    bundle = load_wine_data()
    split = create_train_test_split(bundle, test_size=0.2, random_state=42)

    folds = create_stratified_cv_splits(split.X_train, split.y_train, n_splits=3, random_state=42)

    assert len(folds) == 3
    # Check that folds cover all training instances without overlap in validation
    val_indices_collected = []
    for fold in folds:
        assert len(fold.train_indices) + len(fold.val_indices) == split.n_train
        assert len(set(fold.train_indices).intersection(set(fold.val_indices))) == 0
        # All 3 wine classes present in fold train and validation partitions
        assert len(fold.train_class_distribution) == 3
        assert len(fold.val_class_distribution) == 3
        val_indices_collected.extend(fold.val_indices)

    # Every sample in X_train was in validation partition exactly once
    assert len(val_indices_collected) == split.n_train
    assert set(val_indices_collected) == set(split.train_indices)
