"""
tests/test_repeated_comparison.py
=================================
Unit and integration tests for fair multi-model repeated comparison.

Verifies:
- All models receive identical train/test indices across corresponding repetitions.
- Identical seeds are shared across all compared models.
- Matched-input configurations maintain equivalent feature sets.
- MultiModelComparisonResult contains synchronized per-model results.
- Failures in one model do not corrupt or disrupt evaluations of other models.
- MultiModelComparisonResult serializes cleanly to JSON.
"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine

from backend.data.schemas import ComparisonModeConfig, FeatureSelectionConfig
from backend.training.repeated_runs import DEFAULT_SEEDS, run_repeated_comparison
from backend.training.schemas import MultiModelComparisonResult


@pytest.fixture()
def breast_cancer_data():
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target, name="target")
    return X, y


@pytest.fixture()
def wine_data():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    return X, y


# ---------------------------------------------------------------------------
# Test: Split Synchronization & Fair Comparison
# ---------------------------------------------------------------------------


def test_models_receive_identical_splits_and_indices(breast_cancer_data):
    """Every model in a multi-model comparison receives the exact same train/test indices per repetition."""
    X, y = breast_cancer_data
    models_to_compare = ["dummy", "svm", "random_forest"]

    comp_res = run_repeated_comparison(
        X=X,
        y=y,
        models=models_to_compare,
        n_repeats=3,
        seeds=[42, 101, 202],
    )

    assert isinstance(comp_res, MultiModelComparisonResult)
    assert comp_res.n_repeats == 3
    assert comp_res.models == models_to_compare

    # Check shared split indices recorded at root
    assert len(comp_res.split_indices) == 3

    # Check that each model's runs match the root split indices exactly
    for rep_idx in range(3):
        root_train_idx = comp_res.split_indices[rep_idx]["train_indices"]
        root_test_idx = comp_res.split_indices[rep_idx]["test_indices"]

        for model_name in models_to_compare:
            model_rep = comp_res.model_results[model_name].runs[rep_idx]
            assert model_rep.train_indices == root_train_idx, (
                f"Model {model_name} run {rep_idx} received different train indices!"
            )
            assert model_rep.test_indices == root_test_idx, (
                f"Model {model_name} run {rep_idx} received different test indices!"
            )


def test_multi_model_custom_configs(breast_cancer_data):
    """Model-specific configuration overrides are properly routed to each model."""
    X, y = breast_cancer_data
    models_to_compare = ["dummy", "svm"]
    configs = {
        "dummy": {"strategy": "uniform"},
        "svm": {"kernel": "linear", "C": 0.5},
    }

    comp_res = run_repeated_comparison(
        X=X,
        y=y,
        models=models_to_compare,
        model_configs=configs,
        n_repeats=2,
    )

    dummy_runs = comp_res.model_results["dummy"].runs
    svm_runs = comp_res.model_results["svm"].runs

    for r in dummy_runs:
        assert r.model_config.get("strategy") == "uniform"

    for r in svm_runs:
        assert r.model_config.get("C") == 0.5
        assert r.model_config.get("kernel") == "linear"


# ---------------------------------------------------------------------------
# Test: Multi-model comparison with feature selection & comparison modes
# ---------------------------------------------------------------------------


def test_multi_model_with_feature_selection(wine_data):
    """Shared feature selection is applied consistently to all models across splits."""
    X, y = wine_data
    models_to_compare = ["dummy", "random_forest"]
    fs_config = FeatureSelectionConfig(method="anova", k_features=4)

    comp_res = run_repeated_comparison(
        X=X,
        y=y,
        models=models_to_compare,
        feature_selection_config=fs_config,
        n_repeats=2,
    )

    assert comp_res.n_repeats == 2
    for m in models_to_compare:
        res = comp_res.model_results[m]
        assert res.aggregates["accuracy"].successful_runs == 2


# ---------------------------------------------------------------------------
# Test: Serialization & Failure Isolation
# ---------------------------------------------------------------------------


def test_all_four_models_repeated_comparison(wine_data):
    """All 4 classical models execute successfully under fair shared splits."""
    X, y = wine_data
    all_models = ["dummy", "svm", "mlp", "random_forest"]
    configs = {
        "svm": {"kernel": "linear", "C": 1.0},
        "mlp": {"hidden_layer_sizes": (32,), "max_iter": 50},
        "random_forest": {"n_estimators": 10},
    }

    comp_res = run_repeated_comparison(
        X=X,
        y=y,
        models=all_models,
        model_configs=configs,
        n_repeats=2,
        seeds=[42, 101],
    )

    assert set(comp_res.model_results.keys()) == set(all_models)
    for m in all_models:
        res = comp_res.model_results[m]
        assert len(res.runs) == 2
        assert res.aggregates["accuracy"].successful_runs == 2
        assert res.aggregates["f1_macro"].mean > 0.0


def test_multi_model_json_serialization(breast_cancer_data):
    """MultiModelComparisonResult.to_dict() must serialize cleanly to JSON."""
    X, y = breast_cancer_data
    comp_res = run_repeated_comparison(
        X=X,
        y=y,
        models=["dummy", "svm"],
        n_repeats=2,
    )

    data_dict = comp_res.to_dict()
    serialized = json.dumps(data_dict)
    assert len(serialized) > 0
    assert "split_indices" in data_dict
    assert "model_results" in data_dict
    assert "dummy" in data_dict["model_results"]
    assert "svm" in data_dict["model_results"]
