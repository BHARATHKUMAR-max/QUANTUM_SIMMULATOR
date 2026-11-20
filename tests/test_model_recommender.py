"""
tests/test_model_recommender.py
===============================
Unit and integration tests for transparent, rule-based ModelRecommender:
- Recommendations on Breast Cancer (binary) and Wine (multiclass) datasets
- Handling of class imbalance and compute budget tiers (low, medium, high)
- Dataset scale thresholds (small sample guardrails, large sample scaling warnings)
- Determinism and explanation completeness
- Strict adherence to no-winner-prediction and no-fabricated-metrics policies
"""

import json
import pytest

from backend.classical.model_recommender import (
    ModelRecommendationReport,
    ModelRecommender,
    RecommendationItem,
    RecommenderConfig,
    recommend_models,
)
from backend.data.dataset_loader import load_breast_cancer_data, load_wine_data
from backend.data.dataset_profiler import profile_dataset


class TestModelRecommenderDatasets:
    """Tests evaluating recommender on canonical benchmarks."""

    def test_recommendations_for_breast_cancer(self) -> None:
        bundle = load_breast_cancer_data()
        profile = profile_dataset(bundle)

        report = recommend_models(profile, compute_budget="medium")

        assert isinstance(report, ModelRecommendationReport)
        assert report.dataset_summary["n_samples"] == 569
        assert report.dataset_summary["n_features"] == 30
        assert report.dataset_summary["n_classes"] == 2

        # 4 models recommended
        rec_map = {r.model_name: r for r in report.recommendations}
        assert set(rec_map.keys()) == {"dummy", "svm", "mlp", "random_forest"}

        # Dummy baseline is always recommended
        assert rec_map["dummy"].recommended is True
        assert rec_map["dummy"].suggested_config["strategy"] == "most_frequent"

        # SVM recommended (RBF on medium budget for 569 samples)
        assert rec_map["svm"].recommended is True
        assert rec_map["svm"].suggested_config["kernel"] == "rbf"

        # MLP recommended
        assert rec_map["mlp"].recommended is True
        assert rec_map["mlp"].suggested_config["hidden_layer_sizes"] == [64, 32]

        # Random Forest recommended
        assert rec_map["random_forest"].recommended is True
        assert rec_map["random_forest"].suggested_config["n_estimators"] == 100

    def test_recommendations_for_wine_multiclass(self) -> None:
        bundle = load_wine_data()
        profile = profile_dataset(bundle)

        report = recommend_models(profile, compute_budget="medium")

        assert report.dataset_summary["n_samples"] == 178
        assert report.dataset_summary["n_classes"] == 3

        rec_map = {r.model_name: r for r in report.recommendations}

        # Multiclass mentioned in reasons for non-dummy models
        svm_reasons = " ".join(rec_map["svm"].reasons).lower()
        mlp_reasons = " ".join(rec_map["mlp"].reasons).lower()
        rf_reasons = " ".join(rec_map["random_forest"].reasons).lower()

        assert "multiclass" in svm_reasons
        assert "multiclass" in mlp_reasons
        assert "multiclass" in rf_reasons


class TestModelRecommenderRulesAndBudgets:
    """Tests for compute budget tiers and rule-based adaptations."""

    def test_compute_budget_variations(self) -> None:
        summary = {
            "n_samples": 2000,
            "n_features": 20,
            "n_classes": 2,
            "imbalance_ratio": 1.1,
        }

        # Low budget
        report_low = recommend_models(summary, compute_budget="low")
        rf_low = next(r for r in report_low.recommendations if r.model_name == "random_forest")
        mlp_low = next(r for r in report_low.recommendations if r.model_name == "mlp")
        svm_low = next(r for r in report_low.recommendations if r.model_name == "svm")

        assert rf_low.suggested_config["n_estimators"] == 50
        assert rf_low.suggested_config["max_depth"] == 8
        assert mlp_low.suggested_config["hidden_layer_sizes"] == [32]
        assert mlp_low.suggested_config["max_iter"] == 150
        assert svm_low.suggested_config["kernel"] == "linear"  # N=2000 > 1500 in low budget

        # High budget
        report_high = recommend_models(summary, compute_budget="high")
        rf_high = next(r for r in report_high.recommendations if r.model_name == "random_forest")
        mlp_high = next(r for r in report_high.recommendations if r.model_name == "mlp")

        assert rf_high.suggested_config["n_estimators"] == 200
        assert mlp_high.suggested_config["hidden_layer_sizes"] == [128, 64]
        assert mlp_high.suggested_config["max_iter"] == 300

    def test_class_imbalance_handling(self) -> None:
        imbalanced_summary = {
            "n_samples": 1000,
            "n_features": 15,
            "n_classes": 2,
            "imbalance_ratio": 8.5,  # High imbalance
        }

        report = recommend_models(imbalanced_summary)
        rec_map = {r.model_name: r for r in report.recommendations}

        # SVM should suggest balanced class weights
        assert rec_map["svm"].suggested_config["class_weight"] == "balanced"
        assert any("imbalance" in r.lower() for r in rec_map["svm"].reasons)

        # Random Forest should suggest balanced class weights
        assert rec_map["random_forest"].suggested_config["class_weight"] == "balanced"
        assert any("imbalance" in r.lower() for r in rec_map["random_forest"].reasons)

        # MLP should warn about lack of native class_weight
        assert any("imbalance" in w.lower() for w in rec_map["mlp"].warnings)

    def test_small_sample_size_mlp_guardrail(self) -> None:
        tiny_summary = {
            "n_samples": 30,  # Below 50 samples
            "n_features": 5,
            "n_classes": 2,
            "imbalance_ratio": 1.0,
        }

        report = recommend_models(tiny_summary)
        mlp_rec = next(r for r in report.recommendations if r.model_name == "mlp")

        # MLP should NOT be recommended for N < 50
        assert mlp_rec.recommended is False
        assert any("insufficient" in r.lower() or "too small" in r.lower() for r in mlp_rec.reasons)
        assert any("overfitting" in w.lower() for w in mlp_rec.warnings)

    def test_large_dataset_rbf_svm_guardrail(self) -> None:
        large_summary = {
            "n_samples": 12000,
            "n_features": 30,
            "n_classes": 2,
            "imbalance_ratio": 1.2,
        }

        report = recommend_models(large_summary, compute_budget="medium")
        svm_rec = next(r for r in report.recommendations if r.model_name == "svm")

        # Linear SVM recommended over RBF for large N on CPU
        assert svm_rec.suggested_config["kernel"] == "linear"
        assert any("exceeds" in r.lower() or "rbf" in r.lower() for r in svm_rec.reasons)

    def test_include_nonlinear_flag(self) -> None:
        summary = {"n_samples": 500, "n_features": 10, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary, include_nonlinear=False)

        rec_map = {r.model_name: r for r in report.recommendations}
        assert rec_map["dummy"].recommended is True
        assert rec_map["svm"].recommended is True
        assert rec_map["svm"].suggested_config["kernel"] == "linear"
        assert rec_map["mlp"].recommended is False
        assert rec_map["random_forest"].recommended is False


class TestModelRecommenderIntegrity:
    """Tests verifying transparency, determinism, and absence of fabricated predictions."""

    def test_determinism(self) -> None:
        summary = {"n_samples": 600, "n_features": 20, "n_classes": 2, "imbalance_ratio": 2.0}
        report1 = recommend_models(summary, compute_budget="medium")
        report2 = recommend_models(summary, compute_budget="medium")

        assert report1.to_dict() == report2.to_dict()

    def test_no_performance_predictions_or_winner_declaration(self) -> None:
        summary = {"n_samples": 500, "n_features": 10, "n_classes": 2, "imbalance_ratio": 1.5}
        report = recommend_models(summary)

        # Every item must explicitly state is_performance_prediction is False
        for rec in report.recommendations:
            assert rec.is_performance_prediction is False
            d = rec.to_dict()
            # Must not contain fabricated metrics
            assert "accuracy" not in d
            assert "f1_score" not in d
            assert "roc_auc" not in d
            assert "predicted_winner" not in d
            assert "winner" not in d

        # Report must contain prominent disclaimer
        assert "DO NOT predict" in report.disclaimer

    def test_all_items_have_explanations(self) -> None:
        summary = {"n_samples": 800, "n_features": 12, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary)

        for rec in report.recommendations:
            assert len(rec.reasons) > 0, f"Model {rec.model_name} missing reasons."
            assert len(rec.warnings) > 0, f"Model {rec.model_name} missing warnings."
            assert rec.compute_estimate in ("low", "medium", "high")

    def test_json_serializability(self) -> None:
        summary = {"n_samples": 400, "n_features": 8, "n_classes": 2, "imbalance_ratio": 1.0}
        report = recommend_models(summary)
        d = report.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert len(deserialized["recommendations"]) == 4
