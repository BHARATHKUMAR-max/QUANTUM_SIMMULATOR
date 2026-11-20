"""
backend/classical/model_recommender.py
======================================
Transparent, rule-based recommendation engine for classical models in QML Benchmark Studio.

Evaluates dataset characteristics (sample size, feature count, class balance, data types)
and compute budget ('low', 'medium', 'high') to suggest suitable classical model candidates
and configurations.

CRITICAL DESIGN PRINCIPLES:
1. Heuristic and Transparent: All rules are deterministic, explainable, and inspectable.
   No black-box machine learning or hidden scoring models are used for recommendations.
2. No Winner Predictions: NEVER predicts which model will achieve the highest accuracy,
   best F1 score, or declare a winner prior to empirical execution.
3. No Fabricated Metrics: Does NOT generate synthetic accuracy, ROC-AUC, or exact runtime figures.
4. Execution Transparency: Scikit-learn classical models execute on CPU multi-threading
   and do not automatically use GPU acceleration.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from backend.classical.model_configs import (
    DummyClassifierConfig,
    MLPConfig,
    RandomForestConfig,
    SVMConfig,
)


@dataclass
class RecommenderConfig:
    """
    Configuration settings for the model recommendation engine.

    Attributes:
        compute_budget: Available compute tier ('low', 'medium', 'high').
        include_nonlinear: Whether non-linear models (RBF SVM, MLP, Random Forest) are eligible.
        max_rbf_svm_samples: Maximum dataset sample size before flagging RBF SVM scaling limits
            under 'medium' and 'high' compute budgets. Heuristic guardrail; not a universal SVM
            limit. Default: 5000.
        max_low_budget_rbf_samples: Maximum dataset sample size for RBF SVM under 'low' compute
            budget. Above this threshold, linear kernel is preferred to avoid O(N^2) Gram matrix
            overhead. Heuristic guardrail; not a universal SVM limit. Default: 1500.
        imbalance_threshold: Imbalance ratio threshold above which class_weight='balanced' is
            suggested. Default: 3.0.
    """

    compute_budget: str = "medium"
    include_nonlinear: bool = True
    max_rbf_svm_samples: int = 5000
    max_low_budget_rbf_samples: int = 1500
    imbalance_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.compute_budget not in ("low", "medium", "high"):
            raise ValueError(
                f"compute_budget must be 'low', 'medium', or 'high', got '{self.compute_budget}'."
            )
        if not isinstance(self.include_nonlinear, bool):
            raise TypeError("include_nonlinear must be a boolean.")
        if self.max_rbf_svm_samples < 100:
            raise ValueError("max_rbf_svm_samples must be at least 100.")
        if self.max_low_budget_rbf_samples < 100:
            raise ValueError("max_low_budget_rbf_samples must be at least 100.")
        if self.imbalance_threshold <= 1.0:
            raise ValueError("imbalance_threshold must be strictly greater than 1.0.")


@dataclass
class RecommendationItem:
    """
    Individual classical model recommendation.

    Attributes:
        model_name: Canonical model identifier ('dummy', 'svm', 'mlp', 'random_forest').
        recommended: Whether the model is recommended for empirical evaluation.
        suggested_config: Suggested configuration dictionary for the model.
        reasons: Transparent explanations justifying the recommendation and configuration.
        warnings: Practical warnings regarding computational complexity, scaling, or optimization.
        compute_estimate: Categorical compute demand tier ('low', 'medium', 'high').
        is_performance_prediction: Always False. Explicit marker that no performance prediction is made.
    """

    model_name: str
    recommended: bool
    suggested_config: Dict[str, Any]
    reasons: List[str]
    warnings: List[str]
    compute_estimate: str
    is_performance_prediction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation item to JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ModelRecommendationReport:
    """
    Structured report containing model recommendations and dataset diagnostic summary.

    Attributes:
        dataset_summary: Extracted summary of dataset characteristics.
        compute_budget: Compute budget tier evaluated.
        recommendations: List of model recommendation items.
        disclaimer: Explicit statement clarifying that recommendations are heuristic experiment design aids.
    """

    dataset_summary: Dict[str, Any]
    compute_budget: str
    recommendations: List[RecommendationItem]
    disclaimer: str = (
        "These recommendations are transparent, rule-based heuristics designed to assist in "
        "experiment setup based on sample size, feature geometry, and compute budget. They DO NOT "
        "predict which model will achieve higher accuracy or declare any winner prior to empirical benchmarking."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation report to JSON-serializable dictionary."""
        return {
            "dataset_summary": self.dataset_summary,
            "compute_budget": self.compute_budget,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "disclaimer": self.disclaimer,
        }


class ModelRecommender:
    """
    Rule-based recommender evaluating classical baseline models against dataset profiles.
    """

    def __init__(self, config: Optional[RecommenderConfig] = None) -> None:
        """
        Initialize the model recommender with optional configuration.

        Args:
            config: RecommenderConfig instance. If None, default configuration is used.
        """
        self.config = config or RecommenderConfig()

    def recommend(
        self,
        dataset: Union[Any, Dict[str, Any]],
        compute_budget: Optional[str] = None,
    ) -> ModelRecommendationReport:
        """
        Generate recommendations for all classical models based on dataset characteristics.

        Args:
            dataset: DatasetProfile instance, dictionary of characteristics, or DatasetBundle.
            compute_budget: Optional override for compute budget ('low', 'medium', 'high').

        Returns:
            ModelRecommendationReport containing structured recommendations.
        """
        budget = compute_budget or self.config.compute_budget
        if budget not in ("low", "medium", "high"):
            raise ValueError(f"compute_budget must be 'low', 'medium', or 'high', got '{budget}'.")

        summary = self._extract_dataset_summary(dataset)

        recommendations = [
            self._recommend_dummy(summary, budget),
            self._recommend_svm(summary, budget),
            self._recommend_mlp(summary, budget),
            self._recommend_random_forest(summary, budget),
        ]

        return ModelRecommendationReport(
            dataset_summary=summary,
            compute_budget=budget,
            recommendations=recommendations,
        )

    def _extract_dataset_summary(self, dataset: Any) -> Dict[str, Any]:
        """Extract standardized summary dictionary from various dataset representations."""
        # 1. DatasetProfile dataclass
        if hasattr(dataset, "n_samples") and hasattr(dataset, "n_features"):
            n_samples = int(dataset.n_samples)
            n_features = int(dataset.n_features)
            n_classes = int(getattr(dataset, "n_classes", 2))

            imbalance_ratio = 1.0
            if hasattr(dataset, "class_distribution") and dataset.class_distribution is not None:
                imbalance_ratio = float(getattr(dataset.class_distribution, "imbalance_ratio", 1.0))

            total_missing = 0
            if hasattr(dataset, "missing_values") and dataset.missing_values is not None:
                total_missing = int(getattr(dataset.missing_values, "total_missing", 0))

            n_numerical = n_features
            n_categorical = 0
            if hasattr(dataset, "descriptive_statistics") and dataset.descriptive_statistics is not None:
                num_feats = getattr(dataset.descriptive_statistics, "numerical_features", [])
                cat_feats = getattr(dataset.descriptive_statistics, "categorical_features", [])
                n_numerical = len(num_feats)
                n_categorical = len(cat_feats)

            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "total_missing": total_missing,
                "n_numerical": n_numerical,
                "n_categorical": n_categorical,
            }

        # 2. Dictionary of characteristics
        if isinstance(dataset, dict):
            n_samples = int(dataset.get("n_samples", 100))
            n_features = int(dataset.get("n_features", 10))
            n_classes = int(dataset.get("n_classes", 2))
            imbalance_ratio = float(dataset.get("imbalance_ratio", 1.0))
            total_missing = int(dataset.get("total_missing", dataset.get("n_missing", 0)))
            n_numerical = int(dataset.get("n_numerical", n_features))
            n_categorical = int(dataset.get("n_categorical", 0))

            return {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "total_missing": total_missing,
                "n_numerical": n_numerical,
                "n_categorical": n_categorical,
            }

        # 3. DataFrame or DatasetBundle fallback
        if hasattr(dataset, "X") and hasattr(dataset, "y"):
            X, y = dataset.X, dataset.y
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y)) if y is not None else 2
            imbalance_ratio = 1.0
            if y is not None:
                counts = pd.Series(y).value_counts().values
                if len(counts) > 1 and counts[-1] > 0:
                    imbalance_ratio = float(counts[0] / counts[-1])

            return {
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_classes": int(n_classes),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "total_missing": int(pd.DataFrame(X).isna().sum().sum()),
                "n_numerical": int(n_features),
                "n_categorical": 0,
            }

        raise TypeError(
            f"Unsupported dataset input type: {type(dataset).__name__}. "
            f"Expected DatasetProfile, dictionary, or DatasetBundle."
        )

    # -------------------------------------------------------------------------
    # Model-Specific Heuristic Evaluators
    # -------------------------------------------------------------------------

    def _recommend_dummy(self, summary: Dict[str, Any], budget: str) -> RecommendationItem:
        """Evaluate Dummy Baseline."""
        reasons = [
            "Canonical zero-knowledge baseline predicting the most frequent training class.",
            "Essential benchmark floor to verify whether trained classical or quantum models extract predictive signal.",
        ]
        warnings = [
            "Dummy classifier does not learn feature relationships and provides no predictive utility beyond the prior.",
        ]

        if summary["imbalance_ratio"] > self.config.imbalance_threshold:
            reasons.append(
                f"Severe class imbalance (ratio {summary['imbalance_ratio']:.2f}) makes dummy accuracy misleadingly high; "
                "evaluating macro-F1 and balanced accuracy against this baseline is critical."
            )

        config = DummyClassifierConfig(strategy="most_frequent", random_state=42).to_dict()

        return RecommendationItem(
            model_name="dummy",
            recommended=True,
            suggested_config=config,
            reasons=reasons,
            warnings=warnings,
            compute_estimate="low",
            is_performance_prediction=False,
        )

    def _recommend_svm(self, summary: Dict[str, Any], budget: str) -> RecommendationItem:
        """Evaluate Support Vector Machine (SVM)."""
        n_samples = summary["n_samples"]
        imbalance = summary["imbalance_ratio"] > self.config.imbalance_threshold
        class_weight = "balanced" if imbalance else None

        reasons = []
        warnings = [
            "Scikit-learn SVC executes on CPU via libsvm (does not utilize GPU acceleration).",
            "Requires standardized numerical features; unscaled features will degrade margin optimization.",
        ]

        # Determine kernel and suitability based on sample size and compute budget
        if not self.config.include_nonlinear:
            kernel = "linear"
            recommended = True
            reasons.append("Linear SVM recommended as non-linear models are disabled by configuration.")
            compute_estimate = "low" if n_samples <= 2000 else "medium"
        elif budget == "low":
            if n_samples > self.config.max_low_budget_rbf_samples:
                kernel = "linear"
                recommended = True
                reasons.append(
                    f"Linear kernel selected because dataset sample size (N={n_samples}) exceeds "
                    f"low-compute budget heuristic RBF threshold ({self.config.max_low_budget_rbf_samples} samples) "
                    "for O(N^2) RBF kernel Gram matrix evaluation."
                )
                warnings.append("RBF kernel was omitted to adhere to low compute budget constraints.")
                compute_estimate = "low"
            else:
                kernel = "rbf"
                recommended = True
                reasons.append("RBF SVM provides effective non-linear margin classification for small sample sizes.")
                compute_estimate = "low"
        elif budget == "medium":
            if n_samples > self.config.max_rbf_svm_samples:
                kernel = "linear"
                recommended = True
                reasons.append(
                    f"Linear kernel recommended: sample size (N={n_samples}) exceeds medium budget RBF threshold "
                    f"({self.config.max_rbf_svm_samples} samples)."
                )
                warnings.append("RBF kernel scaling becomes prohibitive on CPU for datasets of this scale.")
                compute_estimate = "medium"
            else:
                kernel = "rbf"
                recommended = True
                reasons.append("RBF SVM evaluates smooth non-linear boundaries; crucial classical comparator for Quantum Kernels.")
                compute_estimate = "medium" if n_samples <= 2000 else "high"
        else:  # high budget
            kernel = "rbf"
            recommended = True
            reasons.append("RBF SVM evaluates maximum-margin non-linear transformations under generous compute budget.")
            compute_estimate = "medium" if n_samples <= 2000 else "high"

        if imbalance:
            reasons.append(
                f"Class imbalance detected (ratio {summary['imbalance_ratio']:.2f}); balanced class weighting applied."
            )

        if summary["n_classes"] > 2:
            reasons.append(
                f"Multiclass dataset detected ({summary['n_classes']} classes); SVC handles this via One-vs-One decision voting."
            )

        config = SVMConfig(
            kernel=kernel,
            C=1.0,
            gamma="scale",
            probability=False,
            random_state=42,
            class_weight=class_weight,
        ).to_dict()

        return RecommendationItem(
            model_name="svm",
            recommended=recommended,
            suggested_config=config,
            reasons=reasons,
            warnings=warnings,
            compute_estimate=compute_estimate,
            is_performance_prediction=False,
        )

    def _recommend_mlp(self, summary: Dict[str, Any], budget: str) -> RecommendationItem:
        """Evaluate Multi-Layer Perceptron (MLP)."""
        n_samples = summary["n_samples"]
        n_features = summary["n_features"]
        imbalance = summary["imbalance_ratio"] > self.config.imbalance_threshold

        reasons = []
        warnings = [
            "Scikit-learn MLPClassifier executes on multi-threaded CPU without CUDA acceleration.",
            "Requires standardized or min-max scaled numerical features for stable gradient descent.",
        ]

        if not self.config.include_nonlinear:
            return RecommendationItem(
                model_name="mlp",
                recommended=False,
                suggested_config=MLPConfig().to_dict(),
                reasons=["MLP excluded because non-linear models are disabled in configuration."],
                warnings=warnings,
                compute_estimate="low",
                is_performance_prediction=False,
            )

        if n_samples < 50:
            warnings.append(
                f"Very small dataset (N={n_samples} samples); neural networks risk severe overfitting or failure to converge."
            )
            return RecommendationItem(
                model_name="mlp",
                recommended=False,
                suggested_config=MLPConfig(hidden_layer_sizes=(32,), max_iter=100).to_dict(),
                reasons=[
                    f"Sample size (N={n_samples}) is insufficient for reliable gradient-based neural network training."
                ],
                warnings=warnings,
                compute_estimate="low",
                is_performance_prediction=False,
            )

        # Budget-dependent layer architectures and iterations.
        # Note: MLPConfig global default is max_iter=300. The medium-budget override to 200
        # is an intentional conservative guardrail to bound CPU training time for typical
        # classification datasets. The global default is preserved and used for 'high' budget.
        if budget == "low":
            hidden_layer_sizes = (32,)
            max_iter = 150
            compute_estimate = "medium"
            reasons.append("Single compact hidden layer (32 neurons) selected for low compute budget.")
        elif budget == "medium":
            hidden_layer_sizes = (64, 32)
            max_iter = 200  # Conservative override: global MLPConfig default is 300; reduced for medium-budget time constraint.
            compute_estimate = "medium"
            reasons.append(
                "Standard two-layer architecture (64, 32) with max_iter=200 (conservative medium-budget override "
                "of global default 300) balancing representational depth and CPU runtime."
            )
        else:  # high
            hidden_layer_sizes = (128, 64)
            max_iter = 300  # Global MLPConfig default: full epoch budget used for high compute.
            compute_estimate = "high"
            reasons.append("Deeper architecture (128, 64) with 300 epochs (global MLPConfig default) permitted by high compute budget.")

        reasons.append("Evaluates continuous non-linear representations learned via backpropagation.")

        if imbalance:
            warnings.append(
                f"Class imbalance detected (ratio {summary['imbalance_ratio']:.2f}). "
                "MLPClassifier in scikit-learn lacks native class_weight reweighting; inspect macro-F1."
            )

        if summary["n_classes"] > 2:
            reasons.append(
                f"Multiclass dataset detected ({summary['n_classes']} classes); MLP outputs softmax cross-entropy over all classes."
            )

        config = MLPConfig(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=False,
            random_state=42,
        ).to_dict()

        return RecommendationItem(
            model_name="mlp",
            recommended=True,
            suggested_config=config,
            reasons=reasons,
            warnings=warnings,
            compute_estimate=compute_estimate,
            is_performance_prediction=False,
        )

    def _recommend_random_forest(self, summary: Dict[str, Any], budget: str) -> RecommendationItem:
        """Evaluate Random Forest."""
        imbalance = summary["imbalance_ratio"] > self.config.imbalance_threshold
        class_weight = "balanced" if imbalance else None

        reasons = [
            "Robust bagged tree ensemble; captures non-linear interactions and orthogonal boundaries without scaling sensitivity.",
            "Multi-threaded CPU execution scales efficiently across all logical cores via n_jobs=-1.",
        ]
        warnings = [
            "Gini feature importance may exhibit bias toward continuous features with high variance.",
        ]

        if not self.config.include_nonlinear:
            return RecommendationItem(
                model_name="random_forest",
                recommended=False,
                suggested_config=RandomForestConfig().to_dict(),
                reasons=["Random Forest excluded because non-linear models are disabled in configuration."],
                warnings=warnings,
                compute_estimate="low",
                is_performance_prediction=False,
            )

        if budget == "low":
            n_estimators = 50
            max_depth = 8
            compute_estimate = "low"
            reasons.append("Reduced tree count (50 trees, max depth 8) configured for low compute budget.")
        elif budget == "medium":
            n_estimators = 100
            max_depth = None
            compute_estimate = "low"
            reasons.append("Standard forest of 100 trees with unconstrained depth.")
        else:  # high
            n_estimators = 200
            max_depth = None
            compute_estimate = "medium"
            reasons.append("Expanded ensemble of 200 trees for thorough variance reduction.")

        if imbalance:
            reasons.append(
                f"Class imbalance detected (ratio {summary['imbalance_ratio']:.2f}); balanced bootstrap weighting applied."
            )

        if summary["n_classes"] > 2:
            reasons.append(
                f"Multiclass dataset detected ({summary['n_classes']} classes); "
                "decision trees natively support multiclass classification via per-node class distribution statistics."
            )

        config = RandomForestConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        ).to_dict()

        return RecommendationItem(
            model_name="random_forest",
            recommended=True,
            suggested_config=config,
            reasons=reasons,
            warnings=warnings,
            compute_estimate=compute_estimate,
            is_performance_prediction=False,
        )


def recommend_models(
    dataset: Union[Any, Dict[str, Any]],
    compute_budget: str = "medium",
    include_nonlinear: bool = True,
    max_rbf_svm_samples: int = 5000,
    max_low_budget_rbf_samples: int = 1500,
    imbalance_threshold: float = 3.0,
) -> ModelRecommendationReport:
    """
    Convenience function to generate transparent, rule-based classical model recommendations.

    Args:
        dataset: DatasetProfile, dictionary of dataset metrics, or DatasetBundle.
        compute_budget: Available compute tier ('low', 'medium', 'high'). Defaults to 'medium'.
        include_nonlinear: Whether non-linear models are eligible. Defaults to True.
        max_rbf_svm_samples: Heuristic maximum samples for RBF SVM under 'medium'/'high' budget.
            Not a universal SVM limit. Defaults to 5000.
        max_low_budget_rbf_samples: Heuristic maximum samples for RBF SVM under 'low' budget.
            Not a universal SVM limit. Defaults to 1500.
        imbalance_threshold: Ratio above which class imbalance compensation is triggered. Defaults to 3.0.

    Returns:
        Structured ModelRecommendationReport with model recommendations, configurations, and warnings.
    """
    config = RecommenderConfig(
        compute_budget=compute_budget,
        include_nonlinear=include_nonlinear,
        max_rbf_svm_samples=max_rbf_svm_samples,
        max_low_budget_rbf_samples=max_low_budget_rbf_samples,
        imbalance_threshold=imbalance_threshold,
    )
    recommender = ModelRecommender(config=config)
    return recommender.recommend(dataset, compute_budget=compute_budget)
