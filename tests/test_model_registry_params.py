"""
tests/test_model_registry_params.py
=====================================
Targeted parameter-mapping verification tests for Day 17 verification checks:
 1. MLP default max_iter=300 and medium-budget recommender override to 200.
 2. SVM sample thresholds are configurable via RecommenderConfig.
 3. Random Forest multiclass terminology is correct (no 'multi-label' language).
 4. Registry create_model correctly maps every parameter for all four models.
"""

import numpy as np
import pytest

from backend.classical.dummy_baseline import DummyBaseline
from backend.classical.mlp import MLPClassifier
from backend.classical.model_configs import (
    DummyClassifierConfig,
    MLPConfig,
    RandomForestConfig,
    SVMConfig,
)
from backend.classical.model_recommender import (
    RecommenderConfig,
    recommend_models,
)
from backend.classical.model_registry import create_model
from backend.classical.random_forest import RandomForestClassifier
from backend.classical.svm import SVMClassifier


# =============================================================================
# Point 1 — MLP default max_iter=300; recommender medium-budget overrides to 200
# =============================================================================

class TestMLPDefaultVsRecommendedMaxIter:

    def test_mlp_config_global_default_is_300(self) -> None:
        """Global MLPConfig default must remain max_iter=300."""
        cfg = MLPConfig()
        assert cfg.max_iter == 300, (
            f"MLPConfig global default max_iter must be 300, got {cfg.max_iter}."
        )

    def test_mlp_wrapper_default_is_200(self) -> None:
        """MLPClassifier wrapper internal default is 200; this is different from MLPConfig."""
        # The wrapper and the config have different defaults — this is intentional and documented.
        mlp = MLPClassifier()
        assert mlp.max_iter == 200  # Wrapper default; registry passes config value explicitly.

    def test_registry_mlp_uses_config_max_iter(self) -> None:
        """create_model('mlp') forwards max_iter from MLPConfig, not any hardcoded value."""
        model = create_model("mlp")  # Uses MLPConfig default: max_iter=300
        assert model.max_iter == 300, (
            f"Registry should forward MLPConfig default max_iter=300, got {model.max_iter}."
        )

    def test_registry_mlp_respects_override_max_iter(self) -> None:
        """create_model('mlp') forwards explicitly provided max_iter override."""
        model = create_model("mlp", {"max_iter": 500})
        assert model.max_iter == 500

    def test_recommender_medium_budget_mlp_overrides_to_200(self) -> None:
        """Medium-budget recommender suggests max_iter=200, explicitly less than global default 300."""
        summary = {"n_samples": 500, "n_features": 10, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary, compute_budget="medium")
        mlp = next(r for r in report.recommendations if r.model_name == "mlp")

        # Medium budget override: 200 < global default 300
        assert mlp.suggested_config["max_iter"] == 200, (
            f"Medium-budget recommender should override max_iter to 200, got {mlp.suggested_config['max_iter']}."
        )
        # Verify reason mentions the override intent
        combined_reasons = " ".join(mlp.reasons)
        assert "200" in combined_reasons, "Reason should document the medium-budget max_iter=200 override."

    def test_recommender_high_budget_mlp_uses_300(self) -> None:
        """High-budget recommender should use max_iter=300, matching global MLPConfig default."""
        summary = {"n_samples": 500, "n_features": 10, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary, compute_budget="high")
        mlp = next(r for r in report.recommendations if r.model_name == "mlp")
        assert mlp.suggested_config["max_iter"] == 300, (
            f"High-budget recommender should use max_iter=300 (global default), got {mlp.suggested_config['max_iter']}."
        )

    def test_recommender_low_budget_mlp_uses_150(self) -> None:
        """Low-budget recommender should use max_iter=150."""
        summary = {"n_samples": 500, "n_features": 10, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary, compute_budget="low")
        mlp = next(r for r in report.recommendations if r.model_name == "mlp")
        assert mlp.suggested_config["max_iter"] == 150


# =============================================================================
# Point 2 — SVM sample thresholds are configurable via RecommenderConfig
# =============================================================================

class TestSVMSampleThresholdsConfigurable:

    def test_recommender_config_exposes_both_thresholds(self) -> None:
        """RecommenderConfig must expose both max_rbf_svm_samples and max_low_budget_rbf_samples."""
        cfg = RecommenderConfig()
        assert hasattr(cfg, "max_rbf_svm_samples"), "max_rbf_svm_samples must be on RecommenderConfig."
        assert hasattr(cfg, "max_low_budget_rbf_samples"), "max_low_budget_rbf_samples must be on RecommenderConfig."
        assert cfg.max_rbf_svm_samples == 5000
        assert cfg.max_low_budget_rbf_samples == 1500

    def test_custom_max_rbf_svm_samples_applied_on_medium_budget(self) -> None:
        """Custom max_rbf_svm_samples should control medium-budget RBF-to-linear crossover."""
        from backend.classical.model_recommender import ModelRecommender

        # With threshold=200, N=300 should trigger linear
        cfg = RecommenderConfig(compute_budget="medium", max_rbf_svm_samples=200)
        report = ModelRecommender(config=cfg).recommend(
            {"n_samples": 300, "n_features": 5, "n_classes": 2, "imbalance_ratio": 1.0}
        )
        svm = next(r for r in report.recommendations if r.model_name == "svm")
        assert svm.suggested_config["kernel"] == "linear", (
            "Custom max_rbf_svm_samples=200 should force linear kernel for N=300."
        )

        # With threshold=5000 (default), N=300 should use RBF
        cfg2 = RecommenderConfig(compute_budget="medium", max_rbf_svm_samples=5000)
        report2 = ModelRecommender(config=cfg2).recommend(
            {"n_samples": 300, "n_features": 5, "n_classes": 2, "imbalance_ratio": 1.0}
        )
        svm2 = next(r for r in report2.recommendations if r.model_name == "svm")
        assert svm2.suggested_config["kernel"] == "rbf"

    def test_custom_max_low_budget_rbf_samples_applied_on_low_budget(self) -> None:
        """Custom max_low_budget_rbf_samples should control low-budget RBF-to-linear crossover."""
        from backend.classical.model_recommender import ModelRecommender

        # Custom threshold=500: N=600 triggers linear on low budget
        cfg = RecommenderConfig(compute_budget="low", max_low_budget_rbf_samples=500)
        report = ModelRecommender(config=cfg).recommend(
            {"n_samples": 600, "n_features": 5, "n_classes": 2, "imbalance_ratio": 1.0},
            compute_budget="low",
        )
        svm = next(r for r in report.recommendations if r.model_name == "svm")
        assert svm.suggested_config["kernel"] == "linear"

        # Custom threshold=5000: N=600 uses RBF on low budget
        cfg2 = RecommenderConfig(compute_budget="low", max_low_budget_rbf_samples=5000)
        report2 = ModelRecommender(config=cfg2).recommend(
            {"n_samples": 600, "n_features": 5, "n_classes": 2, "imbalance_ratio": 1.0},
            compute_budget="low",
        )
        svm2 = next(r for r in report2.recommendations if r.model_name == "svm")
        assert svm2.suggested_config["kernel"] == "rbf"

    def test_max_low_budget_rbf_samples_validation(self) -> None:
        """max_low_budget_rbf_samples below 100 should raise ValueError."""
        with pytest.raises(ValueError, match="max_low_budget_rbf_samples must be at least 100"):
            RecommenderConfig(max_low_budget_rbf_samples=50)

    def test_svm_threshold_documented_as_heuristic_not_universal_limit(self) -> None:
        """Verify reason strings communicate heuristic nature, not absolute SVM limit."""
        from backend.classical.model_recommender import ModelRecommender
        cfg = RecommenderConfig(compute_budget="low", max_low_budget_rbf_samples=100)
        report = ModelRecommender(config=cfg).recommend(
            {"n_samples": 200, "n_features": 5, "n_classes": 2, "imbalance_ratio": 1.0},
            compute_budget="low",
        )
        svm = next(r for r in report.recommendations if r.model_name == "svm")
        all_text = " ".join(svm.reasons + svm.warnings).lower()
        # Should contain heuristic budget language, not "SVM cannot handle"
        assert "budget" in all_text or "heuristic" in all_text or "threshold" in all_text


# =============================================================================
# Point 3 — Random Forest terminology: no "multi-label" in multiclass reasons
# =============================================================================

class TestRandomForestMulticlassTerminology:

    def test_rf_multiclass_reason_does_not_say_multi_label(self) -> None:
        """RF multiclass reason must not use the inaccurate term 'multi-label'."""
        summary = {"n_samples": 300, "n_features": 10, "n_classes": 3, "imbalance_ratio": 1.0}
        report = recommend_models(summary, compute_budget="medium")
        rf = next(r for r in report.recommendations if r.model_name == "random_forest")

        all_text = " ".join(rf.reasons + rf.warnings).lower()
        assert "multi-label" not in all_text, (
            "RF reasons must not use 'multi-label' — decision trees support multiclass, not multi-label."
        )
        # Correct terminology present
        assert "multiclass" in all_text, "RF reasons should contain 'multiclass'."

    def test_rf_multiclass_reason_on_wine_dataset(self) -> None:
        """Confirm correct terminology on the actual Wine dataset profile."""
        from backend.data.dataset_loader import load_wine_data
        from backend.data.dataset_profiler import profile_dataset
        bundle = load_wine_data()
        profile = profile_dataset(bundle)
        report = recommend_models(profile, compute_budget="medium")
        rf = next(r for r in report.recommendations if r.model_name == "random_forest")
        all_text = " ".join(rf.reasons).lower()
        assert "multi-label" not in all_text
        assert "multiclass" in all_text


# =============================================================================
# Point 4 — Registry parameter mapping for all four models
# =============================================================================

class TestRegistryParameterMapping:

    def test_dummy_constant_strategy_forwarded(self) -> None:
        """create_model('dummy') must forward 'constant' when strategy='constant'."""
        model = create_model("dummy", {"strategy": "constant", "constant": 1, "random_state": 7})
        assert isinstance(model, DummyBaseline)
        assert model.strategy == "constant"
        assert model.constant == 1
        assert model.random_state == 7

    def test_dummy_all_params_forwarded(self) -> None:
        model = create_model("dummy", {"strategy": "stratified", "random_state": 99})
        assert model.strategy == "stratified"
        assert model.random_state == 99

    def test_svm_all_params_forwarded(self) -> None:
        model = create_model("svm", {
            "kernel": "linear",
            "C": 3.5,
            "gamma": "auto",
            "probability": False,
            "random_state": 7,
            "class_weight": "balanced",
        })
        assert isinstance(model, SVMClassifier)
        assert model.kernel == "linear"
        assert model.C == 3.5
        assert model.gamma == "auto"
        assert model.probability is False
        assert model.random_state == 7
        assert model.class_weight == "balanced"

    def test_mlp_all_params_forwarded(self) -> None:
        model = create_model("mlp", {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "tanh",
            "solver": "sgd",
            "alpha": 0.001,
            "learning_rate_init": 0.01,
            "max_iter": 400,
            "early_stopping": True,
            "random_state": 13,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
        })
        assert isinstance(model, MLPClassifier)
        assert model.hidden_layer_sizes == (128, 64, 32)
        assert model.activation == "tanh"
        assert model.solver == "sgd"
        assert model.alpha == 0.001
        assert model.learning_rate_init == 0.01
        assert model.max_iter == 400
        assert model.early_stopping is True
        assert model.random_state == 13
        assert model.validation_fraction == 0.15
        assert model.n_iter_no_change == 20

    def test_random_forest_all_params_forwarded(self) -> None:
        model = create_model("random_forest", {
            "n_estimators": 75,
            "max_depth": 6,
            "min_samples_split": 5,
            "max_features": "log2",
            "class_weight": "balanced_subsample",
            "random_state": 21,
            "n_jobs": 2,
            "compute_feature_importances": False,
        })
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 75
        assert model.max_depth == 6
        assert model.min_samples_split == 5
        assert model.max_features == "log2"
        assert model.class_weight == "balanced_subsample"
        assert model.random_state == 21
        assert model.n_jobs == 2
        assert model.compute_feature_importances is False

    def test_mlp_hidden_layer_as_list_forwarded_as_tuple(self) -> None:
        """MLPConfig normalises list → tuple; registry should forward the resulting tuple."""
        model = create_model("mlp", {"hidden_layer_sizes": [64, 32]})
        assert model.hidden_layer_sizes == (64, 32)

    def test_rf_min_samples_split_float_forwarded(self) -> None:
        """Float min_samples_split should propagate correctly through registry."""
        model = create_model("random_forest", {"min_samples_split": 0.1})
        assert model.min_samples_split == 0.1

    def test_rf_max_features_int_forwarded(self) -> None:
        """Integer max_features should propagate correctly through registry."""
        model = create_model("random_forest", {"max_features": 5})
        assert model.max_features == 5

    def test_rf_max_features_none_forwarded(self) -> None:
        """max_features=None should propagate correctly through registry."""
        model = create_model("random_forest", {"max_features": None})
        assert model.max_features is None

    def test_svm_numeric_gamma_forwarded(self) -> None:
        """Numeric gamma value should propagate correctly through registry."""
        model = create_model("svm", {"gamma": 0.01})
        assert model.gamma == 0.01

    def test_mlp_zero_alpha_accepted(self) -> None:
        """alpha=0 (no regularization) should be valid and forwarded correctly."""
        model = create_model("mlp", {"alpha": 0.0, "max_iter": 50})
        assert model.alpha == 0.0

    def test_all_models_fit_predict_with_nondefault_params(self) -> None:
        """End-to-end: models with non-default configs should fit and predict without error."""
        np.random.seed(0)
        X = np.random.randn(60, 6)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        configs = {
            "dummy": {"strategy": "stratified", "random_state": 1},
            "svm": {"kernel": "linear", "C": 2.0, "random_state": 1},
            "mlp": {"hidden_layer_sizes": (32,), "max_iter": 50, "random_state": 1},
            "random_forest": {"n_estimators": 20, "max_depth": 3, "random_state": 1},
        }
        for model_name, cfg in configs.items():
            model = create_model(model_name, cfg)
            model.fit(X, y)
            assert model.is_fitted
            preds = model.predict(X)
            assert len(preds) == 60
