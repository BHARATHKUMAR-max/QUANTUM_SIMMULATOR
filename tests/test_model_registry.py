"""
tests/test_model_registry.py
============================
Unit and integration tests for the classical model registry:
- Discovery: list_available_models
- Retrieval: get_default_config, get_model_class, get_config_class, get_model_metadata
- Validation: validate_model_config
- Instantiation: create_model
- Compatibility with existing model wrappers
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
from backend.classical.model_registry import (
    create_model,
    get_config_class,
    get_default_config,
    get_model_class,
    get_model_metadata,
    list_available_models,
    validate_model_config,
)
from backend.classical.random_forest import RandomForestClassifier
from backend.classical.svm import SVMClassifier


class TestModelRegistryDiscovery:
    """Tests for model discovery and metadata functions."""

    def test_list_available_models(self) -> None:
        models = list_available_models()
        assert isinstance(models, list)
        assert set(models) == {"dummy", "svm", "mlp", "random_forest"}

    @pytest.mark.parametrize(
        "model_name,expected_model_cls,expected_config_cls",
        [
            ("dummy", DummyBaseline, DummyClassifierConfig),
            ("svm", SVMClassifier, SVMConfig),
            ("mlp", MLPClassifier, MLPConfig),
            ("random_forest", RandomForestClassifier, RandomForestConfig),
        ],
    )
    def test_get_classes(
        self,
        model_name: str,
        expected_model_cls: type,
        expected_config_cls: type,
    ) -> None:
        assert get_model_class(model_name) is expected_model_cls
        assert get_config_class(model_name) is expected_config_cls

    def test_get_default_config(self) -> None:
        dummy_cfg = get_default_config("dummy")
        assert isinstance(dummy_cfg, DummyClassifierConfig)
        assert dummy_cfg.strategy == "most_frequent"

        svm_cfg = get_default_config("svm")
        assert isinstance(svm_cfg, SVMConfig)
        assert svm_cfg.kernel == "rbf"

        mlp_cfg = get_default_config("mlp")
        assert isinstance(mlp_cfg, MLPConfig)
        assert mlp_cfg.activation == "relu"

        rf_cfg = get_default_config("random_forest")
        assert isinstance(rf_cfg, RandomForestConfig)
        assert rf_cfg.n_estimators == 100

    def test_get_model_metadata(self) -> None:
        meta = get_model_metadata("svm")
        assert meta["model_name"] == "svm"
        assert meta["execution_device"] == "CPU"
        assert "kernel" in meta["default_config"]
        assert meta["model_class"] == "SVMClassifier"

    def test_case_insensitivity_and_aliases(self) -> None:
        assert get_model_class("SVM") is SVMClassifier
        assert get_model_class("svc") is SVMClassifier
        assert get_model_class("RF") is RandomForestClassifier
        assert get_model_class("randomforest") is RandomForestClassifier
        assert get_model_class("dummy_baseline") is DummyBaseline

    def test_unknown_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown classical model 'quantum_vqc'"):
            get_model_class("quantum_vqc")


class TestModelRegistryValidation:
    """Tests for validate_model_config."""

    def test_none_config_returns_default(self) -> None:
        cfg = validate_model_config("svm", None)
        assert isinstance(cfg, SVMConfig)
        assert cfg.kernel == "rbf"

    def test_dict_config_overrides(self) -> None:
        cfg = validate_model_config("svm", {"kernel": "linear", "C": 5.0})
        assert isinstance(cfg, SVMConfig)
        assert cfg.kernel == "linear"
        assert cfg.C == 5.0
        assert cfg.gamma == "scale"  # Default preserved

    def test_config_instance_validated(self) -> None:
        cfg_in = MLPConfig(hidden_layer_sizes=(32,))
        cfg_out = validate_model_config("mlp", cfg_in)
        assert cfg_out is cfg_in

    def test_incompatible_config_class_raises_type_error(self) -> None:
        rf_cfg = RandomForestConfig()
        with pytest.raises(TypeError, match="Configuration for model 'svm' must be a dict or SVMConfig"):
            validate_model_config("svm", rf_cfg)  # type: ignore

    def test_unknown_parameter_in_dict_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter.*learning_rate"):
            validate_model_config("svm", {"learning_rate": 0.01})


class TestModelRegistryInstantiation:
    """Tests for create_model."""

    def test_create_dummy_default(self) -> None:
        model = create_model("dummy")
        assert isinstance(model, DummyBaseline)
        assert model.strategy == "most_frequent"
        assert not model.is_fitted

    def test_create_svm_with_overrides(self) -> None:
        model = create_model("svm", {"kernel": "linear", "C": 2.0})
        assert isinstance(model, SVMClassifier)
        assert model.kernel == "linear"
        assert model.C == 2.0

    def test_create_mlp_with_overrides(self) -> None:
        model = create_model("mlp", {"hidden_layer_sizes": (32, 16), "max_iter": 100})
        assert isinstance(model, MLPClassifier)
        assert model.hidden_layer_sizes == (32, 16)
        assert model.max_iter == 100

    def test_create_random_forest_with_overrides(self) -> None:
        model = create_model("random_forest", {"n_estimators": 50, "max_depth": 5})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5

    def test_model_lifecycle_compatibility(self) -> None:
        """Verify models created via registry fit and predict correctly on synthetic data."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = np.random.choice([0, 1], size=30)

        for model_name in list_available_models():
            model = create_model(model_name)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            model.fit(X, y)
            assert model.is_fitted
            preds = model.predict(X)
            assert len(preds) == len(y)
