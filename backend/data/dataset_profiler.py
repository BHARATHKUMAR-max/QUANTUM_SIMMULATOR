"""
backend/data/dataset_profiler.py
================================
Analytical profiling and visualization engine for QML Benchmark Studio.

Performs comprehensive descriptive analysis, missingness, duplicate, variance,
and correlation audits on datasets without applying any preprocessing or model fitting.
Supports built-in benchmarks (Breast Cancer, Wine) and arbitrary user-provided CSV datasets.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure safe Matplotlib configuration in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.data.schemas import (
    CategoricalFeatureStatistics,
    ClassDistributionProfile,
    ConstantFeaturesProfile,
    CorrelationProfile,
    DatasetBundle,
    DatasetProfile,
    DescriptiveSummary,
    DuplicateProfile,
    FeatureStatistics,
    MissingValuesProfile,
)


def _format_memory(num_bytes: int) -> str:
    """Format bytes into a human-readable memory string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def profile_dataset(
    bundle: DatasetBundle,
    low_variance_threshold: float = 1e-4,
    correlation_threshold: float = 0.95,
) -> DatasetProfile:
    """
    Perform a complete exploratory audit on any standardized DatasetBundle.
    
    Supports numerical, categorical, and mixed feature representations.
    Descriptive correlations are computed exclusively on suitable numerical columns.
    
    Args:
        bundle: Input DatasetBundle containing feature matrix X and target y.
        low_variance_threshold: Variance floor below which a feature is flagged as near-constant.
        correlation_threshold: Pearson |r| threshold for flagging highly correlated feature pairs.
        
    Returns:
        DatasetProfile: Comprehensive structured profiling results.
    """
    X = bundle.X
    y = bundle.y
    feature_names = bundle.feature_names
    target_names = bundle.target_names
    
    # ---------------------------------------------------------
    # 1. Basic Metadata
    # ---------------------------------------------------------
    n_samples, n_features = X.shape
    memory_bytes = int(X.memory_usage(deep=True).sum() + y.memory_usage(deep=True))
    dtypes_dict = {col: str(dtype) for col, dtype in X.dtypes.items()}
    
    # ---------------------------------------------------------
    # 2. Class Distribution (Supports integer and string target labels)
    # ---------------------------------------------------------
    raw_counts_series = y.value_counts(dropna=False)
    raw_counts: Dict[Any, int] = {}
    counts: Dict[str, int] = {}
    percentages: Dict[str, float] = {}
    
    for raw_cls, cnt in raw_counts_series.items():
        cnt_int = int(cnt)
        # Convert raw_cls safely to primitive key
        try:
            raw_key: Union[int, str] = int(raw_cls)
        except (ValueError, TypeError):
            raw_key = str(raw_cls)
        raw_counts[raw_key] = cnt_int
        
        # Resolve human-readable class name
        if isinstance(raw_cls, (int, np.integer)) and 0 <= int(raw_cls) < len(target_names):
            name = str(target_names[int(raw_cls)])
        else:
            name = str(raw_cls)
        counts[name] = cnt_int
        percentages[name] = round((cnt_int / n_samples) * 100.0, 2)
        
    sorted_by_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    majority_class, maj_count = sorted_by_count[0]
    minority_class, min_count = sorted_by_count[-1]
    imbalance_ratio = round(maj_count / min_count, 3) if min_count > 0 else 0.0
    
    class_profile = ClassDistributionProfile(
        counts=counts,
        raw_counts=raw_counts,
        percentages=percentages,
        imbalance_ratio=imbalance_ratio,
        majority_class=majority_class,
        minority_class=minority_class,
        total_samples=n_samples,
    )
    
    # ---------------------------------------------------------
    # 3. Descriptive Statistics (Partitioning Numerical vs Categorical)
    # ---------------------------------------------------------
    features_stats: Dict[str, FeatureStatistics] = {}
    categorical_stats: Dict[str, CategoricalFeatureStatistics] = {}
    variances: Dict[str, float] = {}
    
    # Determine feature categories
    feature_types = bundle.feature_types
    
    for col in feature_names:
        series = X[col]
        sem_type = feature_types.get(col, "numerical" if pd.api.types.is_numeric_dtype(series) else "categorical")
        
        if sem_type == "numerical" and pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # Numerical feature statistics
            non_null = series.dropna()
            var = float(non_null.var(ddof=1)) if len(non_null) > 1 else 0.0
            variances[col] = var
            
            stat = FeatureStatistics(
                count=int(non_null.count()),
                mean=round(float(non_null.mean()), 5) if len(non_null) > 0 else 0.0,
                std=round(float(non_null.std(ddof=1)), 5) if len(non_null) > 1 else 0.0,
                min=round(float(non_null.min()), 5) if len(non_null) > 0 else 0.0,
                q25=round(float(non_null.quantile(0.25)), 5) if len(non_null) > 0 else 0.0,
                median=round(float(non_null.median()), 5) if len(non_null) > 0 else 0.0,
                q75=round(float(non_null.quantile(0.75)), 5) if len(non_null) > 0 else 0.0,
                max=round(float(non_null.max()), 5) if len(non_null) > 0 else 0.0,
                variance=round(var, 5),
                range=round(float(non_null.max() - non_null.min()), 5) if len(non_null) > 0 else 0.0,
            )
            features_stats[col] = stat
        else:
            # Categorical / Boolean / Datetime feature statistics
            non_null = series.dropna()
            c_cnt = int(non_null.count())
            u_cnt = int(non_null.nunique())
            vc = non_null.value_counts()
            top_cat = str(vc.index[0]) if len(vc) > 0 else None
            top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0
            top_pct = round((top_freq / c_cnt) * 100.0, 2) if c_cnt > 0 else 0.0
            cat_counts = {str(k): int(v) for k, v in vc.items()}
            
            cat_stat = CategoricalFeatureStatistics(
                count=c_cnt,
                unique_count=u_cnt,
                top_category=top_cat,
                top_frequency=top_freq,
                top_percentage=top_pct,
                category_counts=cat_counts,
            )
            categorical_stats[col] = cat_stat

    sorted_variances = sorted(variances.items(), key=lambda item: item[1], reverse=True)
    largest_variability = [
        {"feature": feat, "variance": round(var, 5), "std": round(np.sqrt(var), 5)}
        for feat, var in sorted_variances[:5]
    ]
    sorted_variances_asc = sorted(variances.items(), key=lambda item: item[1])
    smallest_variability = [
        {"feature": feat, "variance": round(var, 8), "std": round(np.sqrt(var), 8)}
        for feat, var in sorted_variances_asc[:5]
    ]
    
    desc_summary = DescriptiveSummary(
        features=features_stats,
        largest_variability_features=largest_variability,
        smallest_variability_features=smallest_variability,
        categorical_features=categorical_stats,
    )
    
    # ---------------------------------------------------------
    # 4. Missing Values Analysis
    # ---------------------------------------------------------
    missing_counts = X.isnull().sum()
    per_column_missing = {col: int(cnt) for col, cnt in missing_counts.items()}
    per_column_pct = {
        col: round((int(cnt) / n_samples) * 100.0, 2)
        for col, cnt in missing_counts.items()
    }
    cols_with_missing = [col for col, cnt in missing_counts.items() if cnt > 0]
    total_missing = int(missing_counts.sum())
    
    missing_profile = MissingValuesProfile(
        per_column_counts=per_column_missing,
        per_column_percentages=per_column_pct,
        total_missing=total_missing,
        columns_with_missing=cols_with_missing,
    )
    
    # ---------------------------------------------------------
    # 5. Duplicates Analysis
    # ---------------------------------------------------------
    total_duplicates = int(X.duplicated().sum())
    unique_rows = n_samples - total_duplicates
    
    dup_cols_groups: List[List[str]] = []
    checked_cols = set()
    for i in range(len(feature_names)):
        col_a = feature_names[i]
        if col_a in checked_cols:
            continue
        group = [col_a]
        for j in range(i + 1, len(feature_names)):
            col_b = feature_names[j]
            if X[col_a].equals(X[col_b]):
                group.append(col_b)
                checked_cols.add(col_b)
        if len(group) > 1:
            dup_cols_groups.append(group)
            
    dup_profile = DuplicateProfile(
        total_duplicate_rows=total_duplicates,
        unique_rows=unique_rows,
        has_duplicate_features=len(dup_cols_groups) > 0,
        duplicate_feature_groups=dup_cols_groups,
    )
    
    # ---------------------------------------------------------
    # 6. Constant & Near-Constant Features
    # ---------------------------------------------------------
    zero_var = [col for col, var in variances.items() if var == 0.0]
    # Also check categorical features with 1 unique value
    for col, c_stat in categorical_stats.items():
        if c_stat.unique_count <= 1:
            zero_var.append(col)
            
    low_var = [
        {"feature": col, "variance": round(var, 8)}
        for col, var in variances.items()
        if 0.0 < var < low_variance_threshold
    ]
    
    const_profile = ConstantFeaturesProfile(
        zero_variance_features=zero_var,
        low_variance_features=low_var,
    )
    
    # ---------------------------------------------------------
    # 7. Correlation Analysis (Numeric Features Exclusively)
    # ---------------------------------------------------------
    numeric_feature_names = list(features_stats.keys())
    high_pairs: List[Dict[str, Any]] = []
    strongest_pos: List[Dict[str, Any]] = []
    strongest_neg: List[Dict[str, Any]] = []
    highest_avg_abs: List[Dict[str, Any]] = []
    
    if len(numeric_feature_names) >= 2:
        corr_matrix = X[numeric_feature_names].corr(method="pearson")
        pairs_all: List[Tuple[str, str, float, float]] = []
        
        for i in range(len(numeric_feature_names)):
            for j in range(i + 1, len(numeric_feature_names)):
                col_i = numeric_feature_names[i]
                col_j = numeric_feature_names[j]
                val = float(corr_matrix.loc[col_i, col_j])
                if np.isnan(val):
                    continue
                abs_val = abs(val)
                pairs_all.append((col_i, col_j, val, abs_val))
                if abs_val >= correlation_threshold:
                    high_pairs.append({
                        "feature_1": col_i,
                        "feature_2": col_j,
                        "pearson_correlation": round(val, 4),
                        "abs_correlation": round(abs_val, 4),
                    })
                    
        high_pairs = sorted(high_pairs, key=lambda x: x["abs_correlation"], reverse=True)
        
        # Strongest positive correlations
        sorted_pos = sorted(pairs_all, key=lambda x: x[2], reverse=True)[:5]
        strongest_pos = [
            {"feature_1": f1, "feature_2": f2, "pearson_correlation": round(v, 4)}
            for f1, f2, v, _ in sorted_pos
        ]
        
        # Strongest negative correlations
        sorted_neg = sorted(pairs_all, key=lambda x: x[2])[:5]
        strongest_neg = [
            {"feature_1": f1, "feature_2": f2, "pearson_correlation": round(v, 4)}
            for f1, f2, v, _ in sorted_neg
        ]
        
        # Average absolute correlation per numerical feature
        for col in numeric_feature_names:
            col_corrs = [
                abs(float(corr_matrix.loc[col, other]))
                for other in numeric_feature_names
                if other != col and not np.isnan(float(corr_matrix.loc[col, other]))
            ]
            if col_corrs:
                highest_avg_abs.append({
                    "feature": col,
                    "avg_abs_correlation": round(float(np.mean(col_corrs)), 4),
                })
        highest_avg_abs = sorted(highest_avg_abs, key=lambda x: x["avg_abs_correlation"], reverse=True)[:5]
        
    corr_profile = CorrelationProfile(
        highly_correlated_pairs=high_pairs,
        highest_avg_abs_correlations=highest_avg_abs,
        strongest_positive_correlations=strongest_pos,
        strongest_negative_correlations=strongest_neg,
    )
    
    # ---------------------------------------------------------
    # 8. Assemble Complete Profile
    # ---------------------------------------------------------
    timestamp_now = datetime.now(timezone.utc).isoformat()
    lib_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    
    return DatasetProfile(
        dataset_name=bundle.dataset_name,
        source=bundle.source_type,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=len(counts),
        feature_names=feature_names,
        dtypes=dtypes_dict,
        memory_usage_bytes=memory_bytes,
        memory_usage_readable=_format_memory(memory_bytes),
        target_names=target_names,
        class_distribution=class_profile,
        missing_values=missing_profile,
        duplicates=dup_profile,
        constants=const_profile,
        descriptive_statistics=desc_summary,
        correlation=corr_profile,
        timestamp_utc=timestamp_now,
        library_versions=lib_versions,
    )


def generate_visualizations(
    bundle: DatasetBundle,
    profile: DatasetProfile,
    output_dir: Union[Path, str],
) -> Dict[str, str]:
    """
    Generate standard exploratory visualizations across any DatasetBundle.
    
    Outputs:
      1. class_distribution.png
      2. feature_distributions.png
      3. correlation_heatmap.png
      4. feature_boxplots.png
      5. feature_variance.png
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_subfolder = out_dir / "visualizations"
    viz_subfolder.mkdir(parents=True, exist_ok=True)
    
    saved_plots: Dict[str, str] = {}
    
    def _save_fig(fig: plt.Figure, filename: str):
        primary_path = out_dir / filename
        sub_path = viz_subfolder / filename
        fig.savefig(primary_path, dpi=300, bbox_inches="tight")
        fig.savefig(sub_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_plots[filename] = str(primary_path)

    base_palette = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3", "#e7298a", "#e6ab02", "#a6761d", "#66a61e"]
    dataset_title = bundle.dataset_name.replace("_", " ").title()

    # ---------------------------------------------------------
    # Plot 1: Target Class Distribution
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = list(profile.class_distribution.counts.keys())
    counts = [profile.class_distribution.counts[c] for c in classes]
    percentages = [profile.class_distribution.percentages[c] for c in classes]
    colors = [base_palette[i % len(base_palette)] for i in range(len(classes))]
    
    bars = ax.bar(classes, counts, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    ax.set_title(f"{dataset_title} — Target Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_xlabel("Class Label", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{count} ({pct}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        
    ax.set_ylim(0, max(counts) * 1.18 if counts else 10)
    _save_fig(fig, "class_distribution.png")

    # ---------------------------------------------------------
    # Plot 2: Feature Distribution Plots (Numerical Features)
    # ---------------------------------------------------------
    num_features = list(profile.descriptive_statistics.features.keys())
    if bundle.dataset_name == "breast_cancer":
        selected_features = [
            "mean radius", "mean texture", "mean area",
            "mean smoothness", "mean concavity", "worst radius",
        ]
    elif bundle.dataset_name == "wine":
        selected_features = [
            "alcohol", "malic_acid", "ash",
            "total_phenols", "flavanoids", "proline",
        ]
    else:
        selected_features = num_features[:6]

    valid_num_features = [f for f in selected_features if f in bundle.X.columns and f in num_features][:6]
    
    if valid_num_features:
        n_cols_plot = min(3, len(valid_num_features))
        n_rows_plot = (len(valid_num_features) + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5 * n_cols_plot, 4.5 * n_rows_plot))
        axes = np.atleast_1d(axes).flatten()
        
        y_target = bundle.y
        for i, col in enumerate(valid_num_features):
            ax = axes[i]
            for cls_idx, cls_name in enumerate(bundle.target_names):
                mask = (y_target == cls_idx) | (y_target == cls_name)
                vals = bundle.X.loc[mask, col].dropna()
                c_color = base_palette[cls_idx % len(base_palette)]
                if len(vals) > 0:
                    ax.hist(
                        vals,
                        bins=min(20, max(5, len(vals) // 2)),
                        alpha=0.55,
                        label=f"{cls_name}",
                        color=c_color,
                        edgecolor="none",
                        density=True,
                    )
            
            ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            if i == 0:
                ax.legend(frameon=True, fontsize=8)
                
        for j in range(len(valid_num_features), len(axes)):
            axes[j].set_visible(False)
                
        fig.suptitle(f"{dataset_title} — Feature Distributions by Class", fontsize=14, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_fig(fig, "feature_distributions.png")
    else:
        # Placeholder figure when no numerical features exist
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numerical features available for distribution histogram.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        _save_fig(fig, "feature_distributions.png")

    # ---------------------------------------------------------
    # Plot 3: Feature Correlation Heatmap (Numerical Features)
    # ---------------------------------------------------------
    if len(num_features) >= 2:
        corr_matrix = bundle.X[num_features].corr(method="pearson").to_numpy()
        heatmap_dim = max(7, min(14, len(num_features) * 0.7))
        fig, ax = plt.subplots(figsize=(heatmap_dim + 1, heatmap_dim))
        
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson Correlation Coefficient (r)", rotation=270, labelpad=15, fontsize=11)
        
        feature_labels = [f.replace(" ", "_") for f in num_features]
        ax.set_xticks(range(len(feature_labels)))
        ax.set_yticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_labels, fontsize=8)
        
        ax.set_title(
            f"{dataset_title} — Pearson Correlation Matrix ({len(num_features)} Numerical Features)",
            fontsize=13, fontweight="bold", pad=12
        )
        fig.tight_layout()
        _save_fig(fig, "correlation_heatmap.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Correlation requires >= 2 numerical features.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        _save_fig(fig, "correlation_heatmap.png")

    # ---------------------------------------------------------
    # Plot 4: Boxplots for Numerical Features
    # ---------------------------------------------------------
    box_features = valid_num_features if valid_num_features else num_features[:6]
    if box_features:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        standardized_box_data = []
        labels = []
        for col in box_features:
            col_data = bundle.X[col].dropna()
            col_std = col_data.std(ddof=1) if len(col_data) > 1 else 0.0
            z_scored = (col_data - col_data.mean()) / col_std if col_std > 0 else col_data - col_data.mean()
            standardized_box_data.append(z_scored.values)
            labels.append(col.replace("_", " ").title())
            
        ax.boxplot(
            standardized_box_data,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            boxprops=dict(facecolor="#c6dbef", color="#08519c"),
            medianprops=dict(color="#d95f02", linewidth=1.5),
            meanprops=dict(color="#006d2c", linewidth=1.5, linestyle="--"),
        )
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        
        ax.set_title(f"{dataset_title} — Standardized Distribution of Key Features (Boxplots)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Z-Score (Standard Deviations)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
        ax.plot([], [], color="#d95f02", linewidth=1.5, label="Median")
        ax.plot([], [], color="#006d2c", linewidth=1.5, linestyle="--", label="Mean")
        ax.legend(loc="upper right", frameon=True)
        
        fig.tight_layout()
        _save_fig(fig, "feature_boxplots.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numerical features available for boxplot analysis.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        _save_fig(fig, "feature_boxplots.png")

    # ---------------------------------------------------------
    # Plot 5: Feature Variance Comparison (Log Scale)
    # ---------------------------------------------------------
    if num_features:
        variances = [profile.descriptive_statistics.features[f].variance for f in num_features]
        sorted_pairs = sorted(zip(num_features, variances), key=lambda x: x[1])
        sorted_names = [p[0] for p in sorted_pairs]
        # Avoid log(0)
        sorted_vars = [max(1e-12, p[1]) for p in sorted_pairs]
        
        var_height = max(5, min(12, len(num_features) * 0.38))
        fig, ax = plt.subplots(figsize=(10, var_height))
        y_pos = np.arange(len(sorted_names))
        
        ax.barh(y_pos, sorted_vars, color="#41b6c4", edgecolor="#225ea8", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xscale("log")
        
        ax.set_xlabel("Sample Variance (Logarithmic Scale)", fontsize=11)
        ax.set_title(f"{dataset_title} — Feature Variance Comparison ({len(num_features)} Features)", fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6, which="both")
        
        fig.tight_layout()
        _save_fig(fig, "feature_variance.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numerical features available for variance analysis.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        _save_fig(fig, "feature_variance.png")
        
    return saved_plots


def export_profile(
    profile: DatasetProfile,
    bundle: DatasetBundle,
    output_dir: Union[Path, str],
) -> Dict[str, str]:
    """
    Save complete profiling reports, tables, markdown findings, and plots.
    
    Exports:
      - profile.json
      - descriptive_statistics.csv
      - class_distribution.csv
      - correlation_matrix.csv
      - findings.md
      - 5 visualization PNG files
      - categorical_statistics.csv (when categorical features exist)
      
    Args:
        profile: Computed DatasetProfile.
        bundle: Source DatasetBundle.
        output_dir: Target directory path for file exports.
        
    Returns:
        Mapping of export filenames to their absolute filesystem paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    generated_files: Dict[str, str] = {}
    
    # 1. Export profile.json (JSON-serializable)
    profile_json_path = out_path / "profile.json"
    with open(profile_json_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    generated_files["profile.json"] = str(profile_json_path)
    
    # 2. Export descriptive_statistics.csv (numerical features)
    stats_rows = []
    for col, stat in profile.descriptive_statistics.features.items():
        row = {"feature": col}
        row.update(stat.__dict__)
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame(columns=["feature"])
    stats_csv_path = out_path / "descriptive_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    generated_files["descriptive_statistics.csv"] = str(stats_csv_path)
    
    # 2b. Export categorical_statistics.csv if categorical features exist
    if profile.descriptive_statistics.categorical_features:
        cat_rows = []
        for col, c_stat in profile.descriptive_statistics.categorical_features.items():
            cat_rows.append({
                "feature": col,
                "count": c_stat.count,
                "unique_count": c_stat.unique_count,
                "top_category": c_stat.top_category,
                "top_frequency": c_stat.top_frequency,
                "top_percentage": c_stat.top_percentage,
            })
        cat_df = pd.DataFrame(cat_rows)
        cat_csv_path = out_path / "categorical_statistics.csv"
        cat_df.to_csv(cat_csv_path, index=False)
        generated_files["categorical_statistics.csv"] = str(cat_csv_path)
    
    # 3. Export class_distribution.csv
    cls_rows = []
    for cls_name, count in profile.class_distribution.counts.items():
        cls_rows.append({
            "class_label": cls_name,
            "sample_count": count,
            "percentage": profile.class_distribution.percentages[cls_name],
        })
    cls_df = pd.DataFrame(cls_rows)
    cls_csv_path = out_path / "class_distribution.csv"
    cls_df.to_csv(cls_csv_path, index=False)
    generated_files["class_distribution.csv"] = str(cls_csv_path)
    
    # 4. Export correlation_matrix.csv (Numerical features only)
    num_features = list(profile.descriptive_statistics.features.keys())
    if len(num_features) >= 2:
        corr_matrix = bundle.X[num_features].corr(method="pearson")
    else:
        corr_matrix = pd.DataFrame(index=num_features, columns=num_features)
    corr_csv_path = out_path / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv_path)
    generated_files["correlation_matrix.csv"] = str(corr_csv_path)
    
    # 5. Generate Visualizations
    plots = generate_visualizations(bundle, profile, out_path)
    generated_files.update(plots)
    
    # 6. Generate findings.md
    findings_md_path = out_path / "findings.md"
    findings_text = _generate_findings_markdown(profile, bundle)
    with open(findings_md_path, "w", encoding="utf-8") as f:
        f.write(findings_text)
    generated_files["findings.md"] = str(findings_md_path)
    
    return generated_files


def _generate_findings_markdown(profile: DatasetProfile, bundle: DatasetBundle) -> str:
    """Compose the findings.md documentation according to dataset specification."""
    c_dist = profile.class_distribution
    m_val = profile.missing_values
    d_val = profile.duplicates
    corr = profile.correlation
    dataset_name = bundle.dataset_name
    
    high_corr_count = len(corr.highly_correlated_pairs)
    if high_corr_count > 0:
        high_corr_table_rows = "\n".join([
            f"| `{p['feature_1']}` | `{p['feature_2']}` | {p['pearson_correlation']:+.4f} |"
            for p in corr.highly_correlated_pairs[:10]
        ])
    elif corr.strongest_positive_correlations:
        high_corr_table_rows = "\n".join([
            f"| `{p['feature_1']}` | `{p['feature_2']}` | {p['pearson_correlation']:+.4f} (Top Correlation) |"
            for p in corr.strongest_positive_correlations[:5]
        ])
    else:
        high_corr_table_rows = "| N/A | N/A | None (< 2 numerical features) |"
    
    largest_var_rows = "\n".join([
        f"| `{item['feature']}` | {item['variance']:,.2f} | {item['std']:,.2f} |"
        for item in profile.descriptive_statistics.largest_variability_features
    ]) if profile.descriptive_statistics.largest_variability_features else "| N/A | N/A | N/A |"
    
    smallest_var_rows = "\n".join([
        f"| `{item['feature']}` | {item['variance']:.8f} | {item['std']:.8f} |"
        for item in profile.descriptive_statistics.smallest_variability_features
    ]) if profile.descriptive_statistics.smallest_variability_features else "| N/A | N/A | N/A |"

    class_table_rows = "\n".join([
        f"| **{cls_name}** | {cls_idx} | {c_dist.counts.get(cls_name, 0)} | {c_dist.percentages.get(cls_name, 0.0)}% |"
        for cls_idx, cls_name in enumerate(bundle.target_names)
    ])

    if dataset_name == "wine":
        return f"""# Exploratory Data Profile: Wine Recognition Dataset

> **Generated:** {profile.timestamp_utc}  
> **Source:** {profile.source}  
> **Module:** `backend/data/dataset_profiler.py`

---

## 1. Dataset Overview

The Wine recognition dataset is a classical multiclass benchmark derived from chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.

* **Total Samples ($N$):** {profile.n_samples}
* **Total Features ($D$):** {profile.n_features} (all continuous, real-valued numerical constituents)
* **Target Classes:** {profile.n_classes} ({', '.join(profile.target_names)})
* **Memory Footprint:** {profile.memory_usage_readable}

---

## 2. Sample and Feature Counts

* **Sample Matrix Shape:** `({profile.n_samples}, {profile.n_features})`
* **Target Vector Shape:** `({profile.n_samples},)`
* **Feature Scope:** Encapsulates 13 chemical and physical measurements: alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, and proline.

---

## 3. Target Class Distribution

| Class Label | Encoded Index | Sample Count | Proportion |
|---|---|---|---|
{class_table_rows}

* **Imbalance Ratio:** {c_dist.imbalance_ratio}:1 ({c_dist.majority_class} : {c_dist.minority_class})
* **Multiclass Assessment:** The dataset is well-balanced across all 3 cultivars with class percentages ranging from 26.97% to 39.89%.

---

## 4. Missing-Value Findings

* **Missing Values Detected:** {m_val.total_missing}
* **Columns with Missing Data:** {len(m_val.columns_with_missing)}
* **Assessment:** The dataset is completely complete with zero missing values across all samples and attributes.

---

## 5. Duplicate Findings

* **Duplicate Rows:** {d_val.total_duplicate_rows}
* **Unique Instances:** {d_val.unique_rows}
* **Identical Columns:** {d_val.has_duplicate_features}
* **Assessment:** All 178 observations are distinct.

---

## 6. Feature Variance Extremes

### Largest Variances
| Feature Name | Variance | Std Dev |
|---|---|---|
{largest_var_rows}

### Smallest Variances
| Feature Name | Variance | Std Dev |
|---|---|---|
{smallest_var_rows}

---

## 7. Correlation Summary

* **Highly Correlated Pairs ($|r| \\ge 0.95$):** {high_corr_count}
* **Strongest Pairwise Correlations:**
| Feature 1 | Feature 2 | Pearson Correlation ($r$) |
|---|---|---|
{high_corr_table_rows}

---

## 8. Limitations & Preprocessing Rules

> **Strict Non-Leakage Rule:**  
> All scalers (`StandardScaler`), feature selection masks, and transformers must be **fitted exclusively on training partitions (`X_train`)**. Correlation and exploratory statistics describe sample relationships and must never be used directly as model parameters.
"""

    elif dataset_name == "breast_cancer":
        return f"""# Exploratory Data Profile: Breast Cancer Wisconsin (Diagnostic) Dataset

> **Generated:** {profile.timestamp_utc}  
> **Source:** {profile.source}  
> **Module:** `backend/data/dataset_profiler.py`

---

## 1. Dataset Overview

The Breast Cancer Wisconsin (Diagnostic) dataset is a binary classification benchmark for predicting whether an excised breast mass FNA biopsy represents a malignant or benign tumour.

* **Total Samples ($N$):** {profile.n_samples}
* **Total Features ($D$):** {profile.n_features} (all continuous, real-valued numerical measurements)
* **Target Classes:** {profile.n_classes} ({', '.join(profile.target_names)})
* **Memory Footprint:** {profile.memory_usage_readable}

---

## 2. Sample and Feature Counts

* **Sample Matrix Shape:** `({profile.n_samples}, {profile.n_features})`
* **Target Vector Shape:** `({profile.n_samples},)`
* **Feature Scope:** Encapsulates 10 fundamental nuclear characteristics evaluated across three statistical summaries: Mean, Standard Error, and Worst.

---

## 3. Target Class Distribution

| Class Label | Encoded Value | Sample Count | Proportion | Imbalance Ratio |
|---|---|---|---|---|
| **{c_dist.majority_class}** (Majority) | 1 | {c_dist.counts.get('benign', c_dist.counts.get(c_dist.majority_class, 0))} | {c_dist.percentages.get('benign', c_dist.percentages.get(c_dist.majority_class, 0.0))}% | **{c_dist.imbalance_ratio}:1** |
| **{c_dist.minority_class}** (Minority) | 0 | {c_dist.counts.get('malignant', c_dist.counts.get(c_dist.minority_class, 0))} | {c_dist.percentages.get('malignant', c_dist.percentages.get(c_dist.minority_class, 0.0))}% | 1.00:1 |

* **Class Balance Status:** Moderate imbalance ({c_dist.percentages.get('benign', 62.74)}% benign vs {c_dist.percentages.get('malignant', 37.26)}% malignant). Stratified partitioning is recommended.

---

## 4. Missing-Value Findings

* **Missing Values Detected:** {m_val.total_missing}
* **Columns with Missing Data:** {len(m_val.columns_with_missing)}
* **Assessment:** The dataset is completely intact with zero null values.

---

## 5. Duplicate-Row Findings

* **Duplicate Rows:** {d_val.total_duplicate_rows}
* **Unique Instances:** {d_val.unique_rows}
* **Identical Columns:** {d_val.has_duplicate_features}
* **Assessment:** Every sample in the dataset represents a distinct observation.

---

## 6. Feature Range and Variance Observations

### Largest Feature Variances
| Feature Name | Sample Variance (\\sigma^2) | Sample Std Dev (\\sigma) |
|---|---|---|
{largest_var_rows}

### Smallest Feature Variances
| Feature Name | Sample Variance (\\sigma^2) | Sample Std Dev (\\sigma) |
|---|---|---|
{smallest_var_rows}

---

## 7. Highly Correlated Feature Pairs (|r| >= 0.95)

Identified **{high_corr_count}** feature pairs exceeding the collinearity threshold of |r| >= 0.95:

| Feature 1 | Feature 2 | Pearson Correlation ($r$) |
|---|---|---|
{high_corr_table_rows}

* **Note on Causation:** These correlation coefficients describe mutual linear association, not biological or physical causation.

---

## 8. Important Observations for Future Preprocessing

1. **Mandatory Scaling:** Features must undergo z-score standardization (`StandardScaler`) prior to model ingestion.
2. **Quantum Feature Dimension Ceiling:** The 30 features far exceed the 10-qubit simulator ceiling. Dimensionality reduction (Mutual Information, ANOVA, or PCA) to k <= 10 features is strictly required.

---

## 9. Limitations of Exploratory Analysis

* **No Generalization Claims:** These findings apply strictly to the 569 instances in the Wisconsin Diagnostic benchmark.
* **Correlation != Predictive Power:** High collinearity does not dictate which feature is more informative for classification.

---

## 10. Reminder: Training-Only Preprocessing

> **Strict Non-Leakage Rule:**  
> All scalers (`StandardScaler`), imputers (`SimpleImputer`), PCA transformations, and feature selection masks (`SelectKBest`) during model training must be **fitted exclusively on training partitions (`X_train`)**.
"""
    else:
        # Generic Markdown profile report for user CSVs and custom datasets
        cat_stats = profile.descriptive_statistics.categorical_features
        cat_section = ""
        if cat_stats:
            cat_rows = "\n".join([
                f"| `{col}` | {stat.unique_count} | `{stat.top_category}` | {stat.top_frequency} ({stat.top_percentage}%) |"
                for col, stat in cat_stats.items()
            ])
            cat_section = f"""
## 5. Categorical Feature Summaries

| Feature Name | Unique Categories | Top Category | Top Category Frequency |
|---|---|---|---|
{cat_rows}
"""

        return f"""# Exploratory Data Profile: {dataset_name.replace('_', ' ').title()}

> **Generated:** {profile.timestamp_utc}  
> **Source:** {profile.source}  
> **Module:** `backend/data/dataset_profiler.py`

---

## 1. Dataset Overview

* **Dataset Identifier:** `{profile.dataset_name}`
* **Total Observations ($N$):** {profile.n_samples}
* **Total Features ($D$):** {profile.n_features}
* **Numerical Features:** {len(profile.descriptive_statistics.features)}
* **Categorical / Other Features:** {len(profile.descriptive_statistics.categorical_features)}
* **Target Classes:** {profile.n_classes} ({', '.join([str(c) for c in profile.target_names])})
* **Memory Footprint:** {profile.memory_usage_readable}

---

## 2. Target Class Distribution

| Class Label | Encoded Index | Sample Count | Proportion |
|---|---|---|---|
{class_table_rows}

* **Majority Class:** `{c_dist.majority_class}`
* **Minority Class:** `{c_dist.minority_class}`
* **Imbalance Ratio:** **{c_dist.imbalance_ratio}:1**

---

## 3. Data Integrity & Missingness

* **Total Missing Values:** {m_val.total_missing} ({len(m_val.columns_with_missing)} columns affected).
* **Duplicate Rows:** {d_val.total_duplicate_rows} ({d_val.unique_rows} unique observations).
* **Duplicate Feature Columns:** {d_val.has_duplicate_features} ({len(d_val.duplicate_feature_groups)} redundant groups).
* **Zero-Variance Constant Features:** {len(profile.constants.zero_variance_features)}
* **Near-Constant Features:** {len(profile.constants.low_variance_features)}

---

## 4. Numerical Feature Variability

### Largest Sample Variances
| Feature Name | Variance | Std Dev |
|---|---|---|
{largest_var_rows}

### Smallest Sample Variances
| Feature Name | Variance | Std Dev |
|---|---|---|
{smallest_var_rows}
{cat_section}
---

## 6. Correlation Analysis (Numerical Features Only)

* **Highly Collinear Feature Pairs ($|r| \\ge 0.95$):** {high_corr_count}
| Feature 1 | Feature 2 | Pearson Correlation ($r$) |
|---|---|---|
{high_corr_table_rows}

> **Note on Correlation & Causality:**  
> Pearson correlation measures linear pairwise association across sample attributes; it does **not** imply causal relationships or directional influence between variables.

---

## 7. Preprocessing & Leakage Boundary Reminder

> **Strict Non-Leakage Rule:**  
> All scalers (`StandardScaler`), imputers (`SimpleImputer`), encoders (`OneHotEncoder`), and feature selectors (`SelectKBest`) must be **fitted exclusively on training splits (`X_train`)**. The exploratory metrics presented in this profile are purely descriptive.
"""
