"""
examples/run_random_forest_baseline.py
======================================
Executable demonstration and benchmark runner for Classical Random Forest in QML Benchmark Studio.

Executes:
1. Breast Cancer dataset loading, leakage-safe preprocessing (StandardScaler), and stratified 80/20 splitting.
2. Random Forest (100 trees, unlimited depth) training, evaluation, and feature importance extraction.
3. Random Forest (200 trees, max_depth=10) training, evaluation, and feature importance extraction.
4. Wine multiclass dataset loading, preprocessing, and Random Forest (100 trees) evaluation.
5. Timing profiling (training duration vs inference latency).
6. Export of authentic JSON experiment result files to `reports/model_results/random_forest/`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.classical.random_forest import RandomForestClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


def run_experiment() -> None:
    """Run classical Random Forest baselines, print metric summaries, and persist JSON artifacts."""
    output_dir = PROJECT_ROOT / "reports" / "model_results" / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  QML Benchmark Studio — Classical Random Forest Baseline Execution")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: Breast Cancer Wisconsin (Binary Classification)
    # -------------------------------------------------------------------------
    print("\n[Dataset 1: Breast Cancer Wisconsin (Diagnostic)]")
    bc_bundle = load_dataset("breast_cancer")
    print(f"  Raw samples: {bc_bundle.X.shape[0]} | Raw features: {bc_bundle.X.shape[1]}")

    bc_split = create_train_test_split(bc_bundle, test_size=0.2, random_state=42)
    print(f"  Train samples: {len(bc_split.X_train)} | Test samples: {len(bc_split.X_test)}")

    # Leakage-safe scaling fitted exclusively on training data
    bc_prep_cfg = PreprocessingConfig(numerical_scaling="standard")
    bc_prep_artifact = fit_preprocessor(bc_split.X_train, bc_bundle.feature_types, bc_prep_cfg)
    X_train_bc = transform_with_preprocessor(bc_prep_artifact, bc_split.X_train)
    X_test_bc = transform_with_preprocessor(bc_prep_artifact, bc_split.X_test)

    # --- 1A: Random Forest (100 trees, max_depth=None) ---
    print("\n  --> Training Random Forest (n_estimators=100, max_depth=None)...")
    rf_100 = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_100.fit(X_train_bc, bc_split.y_train)

    y_pred_100 = rf_100.predict(X_test_bc)
    proba_100 = rf_100.predict_proba(X_test_bc)

    eval_100 = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_100,
        y_proba=proba_100,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_100.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_100.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_100.f1_macro:.4f}")
    print(f"      ROC-AUC (Proba)   : {eval_100.roc_auc:.4f}")
    print(f"      Training Time     : {rf_100.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {rf_100.inference_time_seconds:.6f}s")
    print("      Top 3 Features    :")
    for item in rf_100.top_features_[:3]:
        print(f"        - {item['feature']}: {item['importance']:.4f}")

    payload_100: Dict[str, Any] = {
        "experiment_title": "Breast Cancer Random Forest (100 trees, unlimited depth) Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": rf_100.to_dict(),
        "evaluation_metrics": eval_100.to_dict(),
        "experiment_schema_metrics": eval_100.to_experiment_metrics_dict(),
    }
    path_100 = output_dir / "breast_cancer_rf_100.json"
    with open(path_100, "w", encoding="utf-8") as f:
        json.dump(payload_100, f, indent=2)
    print(f"      [Saved] {path_100.relative_to(PROJECT_ROOT)}")

    # --- 1B: Random Forest (200 trees, max_depth=10) ---
    print("\n  --> Training Random Forest (n_estimators=200, max_depth=10)...")
    rf_200_10 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_200_10.fit(X_train_bc, bc_split.y_train)

    y_pred_200_10 = rf_200_10.predict(X_test_bc)
    proba_200_10 = rf_200_10.predict_proba(X_test_bc)

    eval_200_10 = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_200_10,
        y_proba=proba_200_10,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_200_10.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_200_10.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_200_10.f1_macro:.4f}")
    print(f"      ROC-AUC (Proba)   : {eval_200_10.roc_auc:.4f}")
    print(f"      Training Time     : {rf_200_10.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {rf_200_10.inference_time_seconds:.6f}s")
    print("      Top 3 Features    :")
    for item in rf_200_10.top_features_[:3]:
        print(f"        - {item['feature']}: {item['importance']:.4f}")

    payload_200_10: Dict[str, Any] = {
        "experiment_title": "Breast Cancer Random Forest (200 trees, depth 10) Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": rf_200_10.to_dict(),
        "evaluation_metrics": eval_200_10.to_dict(),
        "experiment_schema_metrics": eval_200_10.to_experiment_metrics_dict(),
    }
    path_200_10 = output_dir / "breast_cancer_rf_200_depth10.json"
    with open(path_200_10, "w", encoding="utf-8") as f:
        json.dump(payload_200_10, f, indent=2)
    print(f"      [Saved] {path_200_10.relative_to(PROJECT_ROOT)}")

    # -------------------------------------------------------------------------
    # PART 2: Wine (Multiclass Classification — 3 Classes)
    # -------------------------------------------------------------------------
    print("\n[Dataset 2: Wine Recognition Dataset (3 Classes)]")
    wine_bundle = load_dataset("wine")
    print(f"  Raw samples: {wine_bundle.X.shape[0]} | Raw features: {wine_bundle.X.shape[1]}")

    wine_split = create_train_test_split(wine_bundle, test_size=0.2, random_state=42)
    print(f"  Train samples: {len(wine_split.X_train)} | Test samples: {len(wine_split.X_test)}")

    wine_prep_cfg = PreprocessingConfig(numerical_scaling="standard")
    wine_prep_artifact = fit_preprocessor(wine_split.X_train, wine_bundle.feature_types, wine_prep_cfg)
    X_train_wine = transform_with_preprocessor(wine_prep_artifact, wine_split.X_train)
    X_test_wine = transform_with_preprocessor(wine_prep_artifact, wine_split.X_test)

    print("\n  --> Training Wine Multiclass Random Forest (n_estimators=100, max_depth=None)...")
    wine_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    wine_rf.fit(X_train_wine, wine_split.y_train)

    y_pred_wine = wine_rf.predict(X_test_wine)
    proba_wine = wine_rf.predict_proba(X_test_wine)

    eval_wine = evaluate_predictions(
        y_true=wine_split.y_test,
        y_pred=y_pred_wine,
        y_proba=proba_wine,
        class_labels=[0, 1, 2],
    )

    print(f"      Accuracy          : {eval_wine.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_wine.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_wine.f1_macro:.4f}")
    print(f"      Weighted F1       : {eval_wine.f1_weighted:.4f}")
    print(f"      OvR ROC-AUC       : {eval_wine.roc_auc:.4f}")
    print(f"      Training Time     : {wine_rf.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {wine_rf.inference_time_seconds:.6f}s")
    print(f"      Confusion Matrix  : {eval_wine.confusion_matrix}")
    print("      Top 3 Features    :")
    for item in wine_rf.top_features_[:3]:
        print(f"        - {item['feature']}: {item['importance']:.4f}")

    wine_payload: Dict[str, Any] = {
        "experiment_title": "Wine Multiclass Random Forest (100 trees, unlimited depth) Baseline",
        "dataset_name": "wine",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": wine_rf.to_dict(),
        "evaluation_metrics": eval_wine.to_dict(),
        "experiment_schema_metrics": eval_wine.to_experiment_metrics_dict(),
    }
    path_wine = output_dir / "wine_rf_100.json"
    with open(path_wine, "w", encoding="utf-8") as f:
        json.dump(wine_payload, f, indent=2)
    print(f"      [Saved] {path_wine.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("  Random Forest Baseline Execution Complete — Results successfully persisted.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
