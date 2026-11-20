"""
examples/run_mlp_baseline.py
============================
Executable demonstration and benchmark runner for Classical MLP neural networks in QML Benchmark Studio.

Executes:
1. Breast Cancer dataset loading, leakage-safe preprocessing (StandardScaler), and stratified 80/20 splitting.
2. MLP (32,) training, evaluation, and convergence profiling.
3. MLP (64, 32) training, evaluation, and convergence profiling.
4. Wine multiclass dataset loading, preprocessing, and MLP (64, 32) evaluation.
5. Timing profiling (training duration vs inference latency).
6. Export of authentic JSON experiment result files to `reports/model_results/mlp/`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.classical.mlp import MLPClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


def run_experiment() -> None:
    """Run classical MLP baselines, print metric summaries, and persist JSON artifacts."""
    output_dir = PROJECT_ROOT / "reports" / "model_results" / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  QML Benchmark Studio — Classical MLP Neural Network Baseline Execution")
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

    # --- 1A: MLP (32,) ---
    print("\n  --> Training MLP (hidden_layer_sizes=(32,), max_iter=300)...")
    mlp_32 = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
    mlp_32.fit(X_train_bc, bc_split.y_train)

    y_pred_32 = mlp_32.predict(X_test_bc)
    proba_32 = mlp_32.predict_proba(X_test_bc)

    eval_32 = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_32,
        y_proba=proba_32,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_32.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_32.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_32.f1_macro:.4f}")
    print(f"      ROC-AUC (Proba)   : {eval_32.roc_auc:.4f}")
    print(f"      Training Time     : {mlp_32.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {mlp_32.inference_time_seconds:.6f}s")
    print(f"      Iterations        : {mlp_32.n_iter_} / {mlp_32.max_iter}")
    print(f"      Final Loss        : {mlp_32.loss_:.6f}")
    print(f"      Converged         : {mlp_32.converged_}")

    payload_32: Dict[str, Any] = {
        "experiment_title": "Breast Cancer MLP (32,) Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": mlp_32.to_dict(),
        "evaluation_metrics": eval_32.to_dict(),
        "experiment_schema_metrics": eval_32.to_experiment_metrics_dict(),
    }
    path_32 = output_dir / "breast_cancer_mlp_32.json"
    with open(path_32, "w", encoding="utf-8") as f:
        json.dump(payload_32, f, indent=2)
    print(f"      [Saved] {path_32.relative_to(PROJECT_ROOT)}")

    # --- 1B: MLP (64, 32) ---
    print("\n  --> Training MLP (hidden_layer_sizes=(64, 32), max_iter=300)...")
    mlp_64_32 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp_64_32.fit(X_train_bc, bc_split.y_train)

    y_pred_64_32 = mlp_64_32.predict(X_test_bc)
    proba_64_32 = mlp_64_32.predict_proba(X_test_bc)

    eval_64_32 = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_64_32,
        y_proba=proba_64_32,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_64_32.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_64_32.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_64_32.f1_macro:.4f}")
    print(f"      ROC-AUC (Proba)   : {eval_64_32.roc_auc:.4f}")
    print(f"      Training Time     : {mlp_64_32.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {mlp_64_32.inference_time_seconds:.6f}s")
    print(f"      Iterations        : {mlp_64_32.n_iter_} / {mlp_64_32.max_iter}")
    print(f"      Final Loss        : {mlp_64_32.loss_:.6f}")
    print(f"      Converged         : {mlp_64_32.converged_}")

    payload_64_32: Dict[str, Any] = {
        "experiment_title": "Breast Cancer MLP (64, 32) Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": mlp_64_32.to_dict(),
        "evaluation_metrics": eval_64_32.to_dict(),
        "experiment_schema_metrics": eval_64_32.to_experiment_metrics_dict(),
    }
    path_64_32 = output_dir / "breast_cancer_mlp_64_32.json"
    with open(path_64_32, "w", encoding="utf-8") as f:
        json.dump(payload_64_32, f, indent=2)
    print(f"      [Saved] {path_64_32.relative_to(PROJECT_ROOT)}")

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

    print("\n  --> Training Wine Multiclass MLP (hidden_layer_sizes=(64, 32), max_iter=400)...")
    wine_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
    wine_mlp.fit(X_train_wine, wine_split.y_train)

    y_pred_wine = wine_mlp.predict(X_test_wine)
    proba_wine = wine_mlp.predict_proba(X_test_wine)

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
    print(f"      Training Time     : {wine_mlp.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {wine_mlp.inference_time_seconds:.6f}s")
    print(f"      Iterations        : {wine_mlp.n_iter_} / {wine_mlp.max_iter}")
    print(f"      Final Loss        : {wine_mlp.loss_:.6f}")
    print(f"      Converged         : {wine_mlp.converged_}")
    print(f"      Confusion Matrix  : {eval_wine.confusion_matrix}")

    wine_payload: Dict[str, Any] = {
        "experiment_title": "Wine Multiclass MLP (64, 32) Baseline",
        "dataset_name": "wine",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": wine_mlp.to_dict(),
        "evaluation_metrics": eval_wine.to_dict(),
        "experiment_schema_metrics": eval_wine.to_experiment_metrics_dict(),
    }
    path_wine = output_dir / "wine_mlp_64_32.json"
    with open(path_wine, "w", encoding="utf-8") as f:
        json.dump(wine_payload, f, indent=2)
    print(f"      [Saved] {path_wine.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("  MLP Baseline Execution Complete — Results successfully persisted.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
