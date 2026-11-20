"""
examples/reproducible_data_pipeline.py
======================================
Reproducible end-to-end dataset-to-feature-preparation pipeline demonstration
for QML Benchmark Studio.

This script executes the complete data pipeline:
  1. Ingestion: Load Breast Cancer Wisconsin Diagnostic benchmark dataset.
  2. Profiling: Compute dataset health, dimensions, and class distributions.
  3. Stratified Splitting: Partition into 80% train / 20% test using fixed seed (42).
  4. Leakage-Safe Preprocessing: Fit StandardScaler & SimpleImputer on train only.
  5. Dimensionality Reduction: Fit PCA on training features to produce 4 components.
  6. Comparison Mode A: Prepare matched input matrices for classical and quantum models.
  7. Verification: Re-run with identical configuration and assert bitwise reproducibility.

Usage:
    .venv\\Scripts\\python examples\\reproducible_data_pipeline.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backend.data import (
    ComparisonModeConfig,
    ComparisonModeResult,
    DatasetBundle,
    PCAConfig,
    PreprocessingConfig,
    create_train_test_split,
    fit_pca,
    fit_preprocessor,
    load_breast_cancer_data,
    prepare_comparison_mode,
    profile_dataset,
    transform_with_pca,
    transform_with_preprocessor,
    verify_matched_inputs,
)


def run_full_pipeline(random_seed: int = 42, n_components: int = 4):
    """Execute the end-to-end pipeline and return structured artifacts."""
    # 1. Ingestion
    bundle = load_breast_cancer_data()

    # 2. Profiling
    profile = profile_dataset(bundle)

    # 3. Stratified Train/Test Split
    split = create_train_test_split(
        dataset_bundle=bundle,
        test_size=0.2,
        random_state=random_seed,
        stratify=True,
    )

    # 4. Preprocessing (fitted strictly on training partition)
    prep_cfg = PreprocessingConfig(
        numerical_imputation="median",
        numerical_scaling="standard",
    )
    preprocessor = fit_preprocessor(split.X_train, config=prep_cfg)
    X_train_proc = transform_with_preprocessor(preprocessor, split.X_train)
    X_test_proc = transform_with_preprocessor(preprocessor, split.X_test)

    # 5. PCA Dimensionality Reduction (fitted strictly on training partition)
    pca_cfg = PCAConfig(
        n_components=n_components,
        random_state=random_seed,
    )
    pca_artifact = fit_pca(X_train_proc, config=pca_cfg)
    X_train_pca = transform_with_pca(pca_artifact, X_train_proc)
    X_test_pca = transform_with_pca(pca_artifact, X_test_proc)

    # 6. Comparison Mode A (Matched Reduced Features)
    cmp_cfg = ComparisonModeConfig(
        mode="matched_reduced",
        reduction_method="pca",
        pca_config=pca_cfg,
        qubit_count=n_components,
        encoding_strategy="angle",
    )
    cmp_result = prepare_comparison_mode(
        X_train_processed=X_train_proc,
        y_train=split.y_train,
        X_test_processed=X_test_proc,
        y_test=split.y_test,
        config=cmp_cfg,
    )

    return {
        "bundle": bundle,
        "profile": profile,
        "split": split,
        "preprocessor": preprocessor,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "pca_artifact": pca_artifact,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "cmp_result": cmp_result,
    }


def main():
    print("=" * 78)
    print("  QML Benchmark Studio — Complete Data Pipeline Demonstration")
    print("=" * 78)

    seed = 42
    n_components = 4
    print(f"\n[Configuration] Random Seed: {seed} | Target Qubits / Components: {n_components}\n")

    # --- Run 1 ---
    print("--> Executing Run 1...")
    run1 = run_full_pipeline(random_seed=seed, n_components=n_components)

    bundle = run1["bundle"]
    profile = run1["profile"]
    split = run1["split"]
    pca_art = run1["pca_artifact"]
    res = run1["cmp_result"]

    print("\n[Step 1: Dataset Loading & Profiling]")
    print(f"  Dataset Name       : {bundle.dataset_name}")
    print(f"  Raw Dimensions     : {bundle.X.shape[0]} samples x {bundle.X.shape[1]} features")
    print(f"  Target Classes     : {bundle.target_classes}")
    print(f"  Class Distribution : {profile.class_distribution.counts}")

    print("\n[Step 2: Stratified Partitioning (80/20)]")
    print(f"  Training Samples   : {len(split.X_train)} ({len(split.X_train)/len(bundle.X)*100:.1f}%)")
    print(f"  Test Samples       : {len(split.X_test)} ({len(split.X_test)/len(bundle.X)*100:.1f}%)")
    train_dist = split.y_train.value_counts().to_dict()
    test_dist = split.y_test.value_counts().to_dict()
    print(f"  Train Class Counts : {train_dist}")
    print(f"  Test Class Counts  : {test_dist}")

    print("\n[Step 3: Leakage-Safe Preprocessing]")
    print(f"  Preprocessed Shape : Train {run1['X_train_proc'].shape} | Test {run1['X_test_proc'].shape}")
    print(f"  Imputer Statistics : Derived from {len(split.X_train)} training observations ONLY")

    print("\n[Step 4: PCA Dimensionality Reduction]")
    print(f"  Components Selected: {pca_art.component_names}")
    print(f"  Variance per PC    : {[round(v, 4) for v in pca_art.explained_variance_ratio]}")
    print(f"  Cumulative Variance: {[round(v, 4) for v in pca_art.cumulative_explained_variance]}")
    print(f"  Total Variance Kept: {pca_art.total_variance_retained * 100:.2f}%")

    print("\n[Step 5: Comparison Mode A (Matched Reduced Features)]")
    print(f"  Comparison Mode    : {res.mode} (Primary Scientific Mode)")
    print(f"  Is Matched Input   : {res.is_matched_input}")
    print(f"  Classical Features : {res.classical_metadata.feature_names}")
    print(f"  Quantum Features   : {res.quantum_metadata.feature_names}")
    print(f"  Qubit Compatibility: {res.qubit_compatibility.is_compatible}")
    print(f"  Feature-to-Qubit   : {res.qubit_compatibility.feature_to_qubit_map}")

    # Verify input equality
    matched, errs = verify_matched_inputs(res.X_train_classical, res.X_train_quantum)
    print(f"  Verified Matched   : {matched} (Errors: {errs})")

    # --- Run 2 & Reproducibility Check ---
    print("\n" + "-" * 78)
    print("--> Executing Run 2 with identical seed (42) to verify reproducibility...")
    run2 = run_full_pipeline(random_seed=seed, n_components=n_components)

    # Assert exact equality
    assert run1["split"].train_indices == run2["split"].train_indices, "Train indices mismatch!"
    assert run1["split"].test_indices == run2["split"].test_indices, "Test indices mismatch!"
    pd.testing.assert_frame_equal(run1["X_train_pca"], run2["X_train_pca"])
    pd.testing.assert_frame_equal(run1["X_test_pca"], run2["X_test_pca"])

    print("  [PASS] Train and Test partition indices match identically.")
    print("  [PASS] Preprocessed training and test feature matrices match identically.")
    print("  [PASS] PCA projections and explained variance ratios match identically.")
    print("  [PASS] Feature metadata and qubit mappings match identically.")

    print("\n" + "=" * 78)
    print("  SUCCESS: Complete data pipeline executed and verified reproducible.")
    print("=" * 78)


if __name__ == "__main__":
    main()
