"""
examples/run_classical_benchmark.py
===================================
Automated, reproducible classical machine-learning benchmark execution script
for QML Benchmark Studio (Day 20).

Evaluates all four classical models:
  1. Dummy Classifier (most_frequent)
  2. Support Vector Machine (RBF kernel, C=1.0, gamma=scale)
  3. Multi-Layer Perceptron (hidden_layer_sizes=(64,32), max_iter=300)
  4. Random Forest (n_estimators=100, max_depth=None)

On both reference datasets:
  - Breast Cancer Wisconsin Diagnostic (569 samples, 30 features, binary)
  - Wine Recognition (178 samples, 13 features, 3-class)

Protocol:
  - 5 independent stratified 80/20 train/test splits
  - Seeds: [42, 101, 202, 303, 404]
  - Synchronized partition indices across all models
  - Sample standard deviation (ddof=1)
  - Preprocessing (StandardScaler) fitted strictly on training slices
  - Generates publication-quality charts and structured JSON/CSV reports
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.data.dataset_loader import load_breast_cancer_data, load_wine_data
from backend.training.repeated_runs import DEFAULT_SEEDS, run_repeated_comparison
from backend.training.schemas import MultiModelComparisonResult


BENCHMARK_MODELS = ["dummy", "svm", "mlp", "random_forest"]

MODEL_CONFIGS = {
    "dummy": {"strategy": "most_frequent"},
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": False},
    "mlp": {"hidden_layer_sizes": (64, 32), "max_iter": 300},
    "random_forest": {"n_estimators": 100, "max_depth": None},
}

MODEL_DISPLAY_NAMES = {
    "dummy": "Dummy (Most Frequent)",
    "svm": "Support Vector Machine (RBF)",
    "mlp": "Multi-Layer Perceptron (64, 32)",
    "random_forest": "Random Forest (100 Trees)",
}

MODEL_COLORS = {
    "dummy": "#9E9E9E",
    "svm": "#1E88E5",
    "mlp": "#FB8C00",
    "random_forest": "#43A047",
}


def compute_summed_confusion_matrices(
    comparison_result: MultiModelComparisonResult,
    target_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute element-wise summed and normalized confusion matrices across the 5 repetitions.

    Args:
        comparison_result: MultiModelComparisonResult container.
        target_names: Human-readable class names.

    Returns:
        Mapping of model_name to dictionary with summed_matrix, normalized_matrix, and labels.
    """
    results: Dict[str, Dict[str, Any]] = {}
    num_classes = len(target_names)

    for model_name, model_res in comparison_result.model_results.items():
        summed = np.zeros((num_classes, num_classes), dtype=int)
        for run in model_res.runs:
            if run.confusion_matrix is not None:
                cm_arr = np.array(run.confusion_matrix, dtype=int)
                if cm_arr.shape == (num_classes, num_classes):
                    summed += cm_arr

        # Row-normalized matrix (percentages of actual classes)
        row_sums = summed.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(summed, dtype=float)
        valid_rows = (row_sums.flatten() > 0)
        normalized[valid_rows] = (summed[valid_rows] / row_sums[valid_rows]) * 100.0

        results[model_name] = {
            "summed_matrix": summed.tolist(),
            "normalized_matrix": [[round(float(c), 2) for c in row] for row in normalized],
            "labels": target_names,
            "total_test_samples_evaluated": int(summed.sum()),
        }

    return results


def extract_mlp_convergence_analysis(
    comparison_result: MultiModelComparisonResult,
) -> Dict[str, Any]:
    """
    Extract detailed MLP convergence diagnostics across all 5 runs.

    Args:
        comparison_result: MultiModelComparisonResult container.

    Returns:
        Dictionary summarizing iteration counts, loss values, convergence flags, and warnings.
    """
    mlp_res = comparison_result.model_results.get("mlp")
    if not mlp_res:
        return {}

    run_diagnostics = []
    converged_count = 0
    warning_count = 0
    loss_curves = []

    for run in mlp_res.runs:
        info = run.convergence_info or {}
        converged = bool(info.get("converged", False))
        if converged:
            converged_count += 1
        
        warn = info.get("convergence_warning")
        if warn:
            warning_count += 1

        curve = info.get("loss_curve")
        if curve:
            loss_curves.append(curve)

        run_diagnostics.append({
            "run_number": run.run_number,
            "random_state": run.random_state,
            "n_iter": info.get("n_iter"),
            "max_iter": info.get("max_iter", 300),
            "final_loss": round(float(info["loss"]), 6) if info.get("loss") is not None else None,
            "converged": converged,
            "convergence_warning": warn,
            "train_time_seconds": round(run.train_time_seconds, 4),
            "early_stopping": info.get("early_stopping", False),
        })

    return {
        "runs": run_diagnostics,
        "total_runs": len(mlp_res.runs),
        "converged_runs": converged_count,
        "warning_runs": warning_count,
        "all_converged": (converged_count == len(mlp_res.runs)),
        "loss_curves": loss_curves,
    }


def generate_benchmark_summary_csv(
    comparison_result: MultiModelComparisonResult,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate and save formatted summary CSV table.

    Args:
        comparison_result: MultiModelComparisonResult instance.
        output_path: Target CSV file path.

    Returns:
        Summary DataFrame.
    """
    rows = []
    for model_name in BENCHMARK_MODELS:
        res = comparison_result.model_results[model_name]
        aggs = res.aggregates

        def _fmt(metric_key: str) -> str:
            agg = aggs.get(metric_key)
            if not agg or agg.mean is None:
                return "N/A"
            if agg.std is None:
                return f"{agg.mean:.4f}"
            return f"{agg.mean:.4f} ± {agg.std:.4f}"

        def _val(metric_key: str, stat: str = "mean") -> Optional[float]:
            agg = aggs.get(metric_key)
            if not agg:
                return None
            return getattr(agg, stat)

        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "Model Identifier": model_name,
            "Accuracy (Mean ± SD)": _fmt("accuracy"),
            "Macro-F1 (Mean ± SD)": _fmt("f1_macro"),
            "Balanced Accuracy (Mean ± SD)": _fmt("balanced_accuracy"),
            "Weighted F1 (Mean ± SD)": _fmt("f1_weighted"),
            "ROC-AUC (Mean ± SD)": _fmt("roc_auc"),
            "Train Time [s] (Mean ± SD)": _fmt("train_time_seconds"),
            "Inference Time [ms]": f"{(_val('inference_time_seconds') or 0.0) * 1000.0:.3f}",
            "Accuracy Mean": _val("accuracy", "mean"),
            "Accuracy Std": _val("accuracy", "std"),
            "Macro-F1 Mean": _val("f1_macro", "mean"),
            "Macro-F1 Std": _val("f1_macro", "std"),
            "Train Time Mean": _val("train_time_seconds", "mean"),
            "Train Time Std": _val("train_time_seconds", "std"),
            "Inference Time Mean": _val("inference_time_seconds", "mean"),
            "Inference Time Std": _val("inference_time_seconds", "std"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def plot_metric_bar_chart(
    models: List[str],
    means: List[float],
    stds: List[Optional[float]],
    metric_title: str,
    ylabel: str,
    output_path: Path,
    dataset_title: str,
    y_limits: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.3f}",
) -> None:
    """Create a publication-quality bar chart with error bars."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x_pos = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#1976D2") for m in models]
    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    y_err = [s if s is not None else 0.0 for s in stds]

    bars = ax.bar(
        x_pos,
        means,
        yerr=y_err,
        align="center",
        alpha=0.88,
        ecolor="#263238",
        capsize=6,
        color=colors,
        edgecolor="#37474F",
        linewidth=1.2,
        width=0.55,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{dataset_title} — {metric_title} (5 Stratified Runs)", fontsize=13, fontweight="bold", pad=12)

    if y_limits:
        ax.set_ylim(y_limits)
    else:
        ax.set_ylim(0, max(means) * 1.25 if max(means) > 0 else 1.0)

    # Annotate value on top of each bar
    for bar, mean_val, std_val in zip(bars, means, stds):
        y_val = bar.get_height()
        err_val = std_val if std_val is not None else 0.0
        txt = value_fmt.format(mean_val)
        if std_val is not None and std_val > 0.0:
            txt += f"\n±{std_val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_val + err_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#212121",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrices_grid(
    cm_dict: Dict[str, Dict[str, Any]],
    dataset_title: str,
    output_path: Path,
) -> None:
    """Create a 2x2 publication-quality grid of summed and normalized confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(BENCHMARK_MODELS):
        ax = axes_flat[idx]
        data = cm_dict[model_name]
        summed = np.array(data["summed_matrix"])
        normalized = np.array(data["normalized_matrix"])
        labels = data["labels"]

        cax = ax.imshow(normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
        ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name), fontsize=11, fontweight="bold", pad=8)

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Print cell labels (count and percentage)
        thresh = 50.0
        for i in range(summed.shape[0]):
            for j in range(summed.shape[1]):
                cnt = summed[i, j]
                pct = normalized[i, j]
                text_color = "white" if pct > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{cnt}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_ylabel("True Class", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight="semibold")
        ax.grid(False)

    fig.suptitle(
        f"{dataset_title} — Summed Confusion Matrices (5 Runs Combined)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mlp_convergence_summary(
    mlp_diagnostics: Dict[str, Any],
    dataset_title: str,
    output_path: Path,
) -> None:
    """Plot MLP training loss curves across iterations for all 5 repetitions."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    loss_curves = mlp_diagnostics.get("loss_curves", [])

    if loss_curves:
        for idx, curve in enumerate(loss_curves):
            seed = DEFAULT_SEEDS[idx] if idx < len(DEFAULT_SEEDS) else idx
            ax.plot(
                range(1, len(curve) + 1),
                curve,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"Repetition {idx + 1} (Seed {seed})",
            )
        ax.set_xlabel("Optimization Epoch / Iteration", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
        ax.set_title(f"{dataset_title} — MLP Loss Trajectory (All 5 Runs)", fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
    else:
        # Fallback iteration bar chart
        runs = mlp_diagnostics.get("runs", [])
        seeds = [r["random_state"] for r in runs]
        iters = [r["n_iter"] for r in runs]
        x_pos = np.arange(len(runs))
        ax.bar(x_pos, iters, color="#FB8C00", width=0.5, edgecolor="#E65100")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Run {i+1} (Seed {s})" for i, s in enumerate(seeds)], rotation=15)
        ax.set_ylabel("Iterations to Convergence", fontsize=11, fontweight="bold")
        ax.set_title(f"{dataset_title} — MLP Convergence Iterations", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_single_dataset_benchmark(
    dataset_name: str,
    output_dir: Path,
) -> Tuple[MultiModelComparisonResult, Dict[str, Any], Dict[str, Any]]:
    """
    Execute full 4-model 5-run benchmark on a single dataset, generate tables and plots.

    Args:
        dataset_name: 'breast_cancer' or 'wine'.
        output_dir: Directory where plots and reports are written.

    Returns:
        Tuple of (MultiModelComparisonResult, confusion_matrices, mlp_convergence).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "breast_cancer":
        bundle = load_breast_cancer_data()
        dataset_title = "Breast Cancer Wisconsin Diagnostic"
    elif dataset_name == "wine":
        bundle = load_wine_data()
        dataset_title = "Wine Recognition"
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    print(f"\n=======================================================")
    print(f"Executing Classical Benchmark: {dataset_title}")
    print(f"Features: {bundle.X.shape[1]} | Samples: {bundle.X.shape[0]} | Classes: {len(bundle.target_names)}")
    print(f"=======================================================")

    # Run multi-model comparison across 5 stratified splits
    comp_result = run_repeated_comparison(
        X=bundle.X,
        y=bundle.y,
        models=BENCHMARK_MODELS,
        model_configs=MODEL_CONFIGS,
        n_repeats=5,
        test_size=0.2,
        stratify=True,
        dataset_name=bundle.dataset_name,
    )

    # 1. Confusion Matrices
    cm_dict = compute_summed_confusion_matrices(comp_result, bundle.target_names)

    # 2. MLP Convergence Diagnostics
    mlp_diagnostics = extract_mlp_convergence_analysis(comp_result)

    # 3. Save Summary CSV Table
    csv_path = output_dir / "benchmark_summary.csv"
    summary_df = generate_benchmark_summary_csv(comp_result, csv_path)

    # 4. Save Full JSON Data Export
    json_path = output_dir / "benchmark_results.json"
    full_export = comp_result.to_dict()
    full_export["dataset_title"] = dataset_title
    full_export["confusion_matrices"] = cm_dict
    full_export["mlp_convergence_diagnostics"] = mlp_diagnostics
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)

    # 5. Generate Visualizations
    models = BENCHMARK_MODELS
    acc_means = [comp_result.model_results[m].aggregates["accuracy"].mean for m in models]
    acc_stds = [comp_result.model_results[m].aggregates["accuracy"].std for m in models]
    f1_means = [comp_result.model_results[m].aggregates["f1_macro"].mean for m in models]
    f1_stds = [comp_result.model_results[m].aggregates["f1_macro"].std for m in models]
    train_times = [comp_result.model_results[m].aggregates["train_time_seconds"].mean for m in models]
    train_stds = [comp_result.model_results[m].aggregates["train_time_seconds"].std for m in models]
    infer_times = [(comp_result.model_results[m].aggregates["inference_time_seconds"].mean or 0.0) * 1000.0 for m in models]
    infer_stds = [((comp_result.model_results[m].aggregates["inference_time_seconds"].std or 0.0) * 1000.0) for m in models]

    # Accuracy Plot
    plot_metric_bar_chart(
        models=models,
        means=acc_means,
        stds=acc_stds,
        metric_title="Accuracy Comparison",
        ylabel="Accuracy Score (0.0 to 1.0)",
        output_path=output_dir / "accuracy_comparison.png",
        dataset_title=dataset_title,
        y_limits=(0.0, 1.15),
        value_fmt="{:.4f}",
    )

    # Macro-F1 Plot
    plot_metric_bar_chart(
        models=models,
        means=f1_means,
        stds=f1_stds,
        metric_title="Macro F1-Score Comparison",
        ylabel="Macro F1-Score (0.0 to 1.0)",
        output_path=output_dir / "macro_f1_comparison.png",
        dataset_title=dataset_title,
        y_limits=(0.0, 1.15),
        value_fmt="{:.4f}",
    )

    # Training Time Plot
    plot_metric_bar_chart(
        models=models,
        means=train_times,
        stds=train_stds,
        metric_title="Training Duration Comparison",
        ylabel="Training Time (seconds)",
        output_path=output_dir / "training_time.png",
        dataset_title=dataset_title,
        value_fmt="{:.4f}s",
    )

    # Inference Time Plot
    plot_metric_bar_chart(
        models=models,
        means=infer_times,
        stds=infer_stds,
        metric_title="Test Inference Latency",
        ylabel="Inference Latency (milliseconds)",
        output_path=output_dir / "inference_time.png",
        dataset_title=dataset_title,
        value_fmt="{:.3f}ms",
    )

    # 4-Model Confusion Matrices Grid
    plot_confusion_matrices_grid(
        cm_dict=cm_dict,
        dataset_title=dataset_title,
        output_path=output_dir / "confusion_matrices.png",
    )

    # MLP Convergence Plot
    plot_mlp_convergence_summary(
        mlp_diagnostics=mlp_diagnostics,
        dataset_title=dataset_title,
        output_path=output_dir / "mlp_convergence.png",
    )

    print(f"Generated results, plots, and summaries in: {output_dir}")
    return comp_result, cm_dict, mlp_diagnostics


def run_all_classical_benchmarks(
    base_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run complete classical benchmark suite across Breast Cancer and Wine datasets.

    Args:
        base_output_dir: Root output directory. Defaults to reports/classical_benchmark.

    Returns:
        Structured dictionary containing results for both datasets.
    """
    base_dir = base_output_dir or (REPO_ROOT / "reports" / "classical_benchmark")
    base_dir.mkdir(parents=True, exist_ok=True)

    bc_res, bc_cm, bc_mlp = run_single_dataset_benchmark("breast_cancer", base_dir / "breast_cancer")
    wine_res, wine_cm, wine_mlp = run_single_dataset_benchmark("wine", base_dir / "wine")

    return {
        "breast_cancer": {
            "comparison": bc_res,
            "confusion_matrices": bc_cm,
            "mlp_diagnostics": bc_mlp,
        },
        "wine": {
            "comparison": wine_res,
            "confusion_matrices": wine_cm,
            "mlp_diagnostics": wine_mlp,
        },
    }


if __name__ == "__main__":
    start = time.time()
    run_all_classical_benchmarks()
    print(f"\nAll classical benchmarks completed in {time.time() - start:.2f} seconds.")
