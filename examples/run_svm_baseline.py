"""
examples/run_svm_baseline.py
============================
Executable demonstration and benchmark runner for Classical SVM models in QML Benchmark Studio.

Executes:
1. Breast Cancer dataset loading, leakage-safe preprocessing (StandardScaler), and stratified splitting.
2. Linear SVM training and evaluation with decision scores.
3. RBF SVM training and evaluation with calibrated probabilities.
4. Wine multiclass dataset loading, preprocessing, and RBF SVM evaluation.
5. Timing profiling (training duration vs inference latency).
6. Export of authentic JSON experiment result files to `reports/model_results/svm/`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.classical.svm import SVMClassifier
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import PreprocessingConfig, fit_preprocessor, transform_with_preprocessor
from backend.data.splitting import create_train_test_split
from backend.evaluation.metrics import evaluate_predictions


def run_experiment() -> None:
    """Run classical SVM baselines, print metric summaries, and persist JSON artifacts."""
    output_dir = PROJECT_ROOT / "reports" / "model_results" / "svm"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  QML Benchmark Studio — Classical SVM Baseline Execution")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: Breast Cancer (Binary Classification)
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

    # --- 1A: Linear SVM ---
    print("\n  --> Training Linear SVM (C=1.0, probability=False)...")
    linear_svm = SVMClassifier(kernel="linear", C=1.0, probability=False, random_state=42)
    linear_svm.fit(X_train_bc, bc_split.y_train)

    y_pred_linear = linear_svm.predict(X_test_bc)
    scores_linear = linear_svm.decision_function(X_test_bc)

    eval_linear = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_linear,
        y_score=scores_linear,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_linear.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_linear.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_linear.f1_macro:.4f}")
    print(f"      ROC-AUC (Scores)  : {eval_linear.roc_auc:.4f}")
    print(f"      Training Time     : {linear_svm.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {linear_svm.inference_time_seconds:.6f}s")
    print(f"      Support Vectors   : {linear_svm.support_vectors_count_}")

    linear_payload: Dict[str, Any] = {
        "experiment_title": "Breast Cancer Linear SVM Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": linear_svm.to_dict(),
        "evaluation_metrics": eval_linear.to_dict(),
        "experiment_schema_metrics": eval_linear.to_experiment_metrics_dict(),
    }
    linear_path = output_dir / "breast_cancer_linear_svm.json"
    with open(linear_path, "w", encoding="utf-8") as f:
        json.dump(linear_payload, f, indent=2)
    print(f"      [Saved] {linear_path.relative_to(PROJECT_ROOT)}")

    # --- 1B: RBF SVM ---
    print("\n  --> Training RBF SVM (C=1.0, gamma='scale', probability=True)...")
    rbf_svm = SVMClassifier(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    rbf_svm.fit(X_train_bc, bc_split.y_train)

    y_pred_rbf = rbf_svm.predict(X_test_bc)
    proba_rbf = rbf_svm.predict_proba(X_test_bc)

    eval_rbf = evaluate_predictions(
        y_true=bc_split.y_test,
        y_pred=y_pred_rbf,
        y_proba=proba_rbf,
        class_labels=[0, 1],
        positive_label=1,
    )

    print(f"      Accuracy          : {eval_rbf.accuracy:.4f}")
    print(f"      Balanced Accuracy : {eval_rbf.balanced_accuracy:.4f}")
    print(f"      Macro F1          : {eval_rbf.f1_macro:.4f}")
    print(f"      ROC-AUC (Proba)   : {eval_rbf.roc_auc:.4f}")
    print(f"      Training Time     : {rbf_svm.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {rbf_svm.inference_time_seconds:.6f}s")
    print(f"      Support Vectors   : {rbf_svm.support_vectors_count_}")

    rbf_payload: Dict[str, Any] = {
        "experiment_title": "Breast Cancer RBF SVM Baseline",
        "dataset_name": "breast_cancer",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": rbf_svm.to_dict(),
        "evaluation_metrics": eval_rbf.to_dict(),
        "experiment_schema_metrics": eval_rbf.to_experiment_metrics_dict(),
    }
    rbf_path = output_dir / "breast_cancer_rbf_svm.json"
    with open(rbf_path, "w", encoding="utf-8") as f:
        json.dump(rbf_payload, f, indent=2)
    print(f"      [Saved] {rbf_path.relative_to(PROJECT_ROOT)}")

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

    print("\n  --> Training Wine Multiclass RBF SVM (C=1.0, probability=True)...")
    wine_svm = SVMClassifier(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    wine_svm.fit(X_train_wine, wine_split.y_train)

    y_pred_wine = wine_svm.predict(X_test_wine)
    proba_wine = wine_svm.predict_proba(X_test_wine)

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
    print(f"      Training Time     : {wine_svm.training_time_seconds:.6f}s")
    print(f"      Inference Time    : {wine_svm.inference_time_seconds:.6f}s")
    print(f"      Support Vectors   : {wine_svm.support_vectors_count_}")
    print(f"      Confusion Matrix  : {eval_wine.confusion_matrix}")

    wine_payload: Dict[str, Any] = {
        "experiment_title": "Wine Multiclass RBF SVM Baseline",
        "dataset_name": "wine",
        "split_seed": 42,
        "test_size": 0.2,
        "model_config": wine_svm.to_dict(),
        "evaluation_metrics": eval_wine.to_dict(),
        "experiment_schema_metrics": eval_wine.to_experiment_metrics_dict(),
    }
    wine_path = output_dir / "wine_rbf_svm.json"
    with open(wine_path, "w", encoding="utf-8") as f:
        json.dump(wine_payload, f, indent=2)
    print(f"      [Saved] {wine_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("  SVM Baseline Execution Complete — Results successfully persisted.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
