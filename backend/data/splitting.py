"""
backend/data/splitting.py
=========================
Leakage-safe, reproducible dataset partitioning module for QML Benchmark Studio.

Responsible for creating stratified train/test partitions and cross-validation
folds while preserving original index mappings, feature schemas, and class balances.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from backend.data.schemas import (
    CVFoldResult,
    DatasetBundle,
    TrainTestSplitResult,
    DEFAULT_SPLIT_SEEDS,
)


def create_train_test_split(
    dataset_bundle: DatasetBundle,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    shuffle: bool = True,
) -> TrainTestSplitResult:
    """
    Partition a DatasetBundle into training and held-out test subsets.
    
    Guarantees:
      - Zero target leakage: target is strictly isolated from feature matrices.
      - Non-overlapping indices: train and test partitions form an exact disjoint cover.
      - Preserved indices: original DataFrame indices are retained for reproducibility.
      - Stratification by default: class proportions are preserved across partitions.
      - Explicit validation: invalid test sizes or insufficient class counts fail fast.
      
    Args:
        dataset_bundle: Standardized container holding features X and labels y.
        test_size: Proportion of samples allocated to test partition (0.0 < test_size < 1.0).
        random_state: Deterministic random seed for reproducibility.
        stratify: Whether to enforce stratified class sampling.
        shuffle: Whether to shuffle before partitioning (mandatory True if stratify=True).
        
    Returns:
        TrainTestSplitResult: Complete structured partition container with metadata.
        
    Raises:
        ValueError: If test_size is out of bounds, samples are insufficient, or
                    stratification is requested on classes with < 2 instances.
    """
    # 1. Validate dataset bundle integrity
    dataset_bundle.validate_integrity()
    
    # 2. Validate test_size parameter
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
        raise ValueError(f"Invalid test_size type: {type(test_size)}. Expected float between 0.0 and 1.0.")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be strictly between 0.0 and 1.0, got: {test_size}")

    if stratify and not shuffle:
        raise ValueError("Stratified train/test splitting requires shuffle=True.")

    X = dataset_bundle.X
    y = dataset_bundle.y
    n_samples = len(X)
    
    # 3. Stratification validation
    stratify_target: Optional[pd.Series] = None
    if stratify:
        class_counts = y.value_counts(dropna=False)
        if len(class_counts) < 2:
            raise ValueError(
                f"Cannot perform stratified split on dataset '{dataset_bundle.dataset_name}': "
                f"target has only {len(class_counts)} class ({list(class_counts.keys())}). "
                "Classification requires at least 2 distinct classes."
            )
        
        min_class_count = int(class_counts.min())
        minority_class = class_counts.idxmin()
        if min_class_count < 2:
            raise ValueError(
                f"Cannot perform stratified split on dataset '{dataset_bundle.dataset_name}': "
                f"minority class '{minority_class}' has only {min_class_count} sample(s). "
                "Stratification requires at least 2 samples per class to populate both partitions. "
                "Provide more observations or explicitly pass stratify=False."
            )
            
        # Verify test partition can accommodate at least one sample per class
        min_test_samples = int(np.ceil(n_samples * test_size))
        if min_test_samples < len(class_counts):
            raise ValueError(
                f"Cannot perform stratified split: requested test size ({test_size} -> {min_test_samples} samples) "
                f"is smaller than the number of classes ({len(class_counts)})."
            )
            
        stratify_target = y

    # 4. Perform split using scikit-learn
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state if shuffle else None,
        stratify=stratify_target,
        shuffle=shuffle,
    )

    # 5. Extract preserved index lists
    train_indices = list(X_train.index)
    test_indices = list(X_test.index)

    # 6. Compute class distributions for verification
    train_class_dist = {str(k): int(v) for k, v in y_train.value_counts().items()}
    test_class_dist = {str(k): int(v) for k, v in y_test.value_counts().items()}

    metadata = {
        "dataset_name": dataset_bundle.dataset_name,
        "source_type": dataset_bundle.source_type,
        "total_samples": n_samples,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": float(test_size),
        "random_state": random_state,
        "stratified": stratify,
        "feature_types": dataset_bundle.feature_types,
    }

    result = TrainTestSplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        random_state=random_state,
        test_size=float(test_size),
        stratified=stratify,
        dataset_name=dataset_bundle.dataset_name,
        n_train=len(X_train),
        n_test=len(X_test),
        train_class_distribution=train_class_dist,
        test_class_distribution=test_class_dist,
        feature_names=list(dataset_bundle.feature_names),
        target_name=dataset_bundle.target_name,
        metadata=metadata,
    )

    # 7. Validate non-overlap and completeness
    result.validate_integrity()
    return result


def create_stratified_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 3,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[CVFoldResult]:
    """
    Generate stratified cross-validation folds strictly from the training partition.
    
    Guarantees:
      - The held-out test partition is never touched.
      - Class distributions are preserved across all folds.
      - Fold indices reference the original training DataFrame indices.
      
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        n_splits: Number of cross-validation folds (>= 2).
        random_state: Seed for reproducible fold shuffling.
        shuffle: Whether to shuffle before partitioning.
        
    Returns:
        List[CVFoldResult]: List of fold partition metadata containers.
        
    Raises:
        ValueError: If n_splits < 2 or any class has fewer instances than n_splits.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got: {n_splits}")

    class_counts = y_train.value_counts(dropna=False)
    min_count = int(class_counts.min())
    minority_class = class_counts.idxmin()
    if min_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} stratified folds: minority class '{minority_class}' "
            f"has only {min_count} samples in training data. Each fold requires at least 1 sample "
            f"per class. Decrease n_splits to <= {min_count} or acquire more observations."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds: List[CVFoldResult] = []

    for fold_idx, (train_pos, val_pos) in enumerate(skf.split(X_train, y_train)):
        fold_train_df = X_train.iloc[train_pos]
        fold_val_df = X_train.iloc[val_pos]
        fold_y_train = y_train.iloc[train_pos]
        fold_y_val = y_train.iloc[val_pos]

        train_indices = list(fold_train_df.index)
        val_indices = list(fold_val_df.index)

        # Integrity assertion
        overlap = set(train_indices).intersection(set(val_indices))
        if overlap:
            raise ValueError(f"Fold {fold_idx} has overlapping train and validation indices!")

        train_dist = {str(k): int(v) for k, v in fold_y_train.value_counts().items()}
        val_dist = {str(k): int(v) for k, v in fold_y_val.value_counts().items()}

        folds.append(
            CVFoldResult(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                n_train=len(train_indices),
                n_val=len(val_indices),
                train_class_distribution=train_dist,
                val_class_distribution=val_dist,
            )
        )

    return folds
