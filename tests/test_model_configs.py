"""
tests/test_model_configs.py
===========================
Comprehensive unit test suite for centralized classical model configuration schemas:
- DummyClassifierConfig
- SVMConfig
- MLPConfig
- RandomForestConfig
"""

import json
import pytest

from backend.classical.model_configs import (
    BaseModelConfig,
    DummyClassifierConfig,
    MLPConfig,
    RandomForestConfig,
    SVMConfig,
)


class TestDummyClassifierConfig:
    """Test suite for DummyClassifierConfig."""

    def test_default_config(self) -> None:
        cfg = DummyClassifierConfig()
        assert cfg.strategy == "most_frequent"
        assert cfg.random_state == 42
        assert cfg.constant is None

    def test_valid_override(self) -> None:
        cfg = DummyClassifierConfig.from_dict({"strategy": "stratified", "random_state": 123})
        assert cfg.strategy == "stratified"
        assert cfg.random_state == 123

    def test_unspecified_parameters_keep_defaults(self) -> None:
        cfg = DummyClassifierConfig.from_dict({"strategy": "uniform"})
        assert cfg.strategy == "uniform"
        assert cfg.random_state == 42  # default preserved

    def test_invalid_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dummy strategy"):
            DummyClassifierConfig(strategy="invalid_strat")

    def test_constant_strategy_requires_constant_value(self) -> None:
        with pytest.raises(ValueError, match="strategy='constant' requires"):
            DummyClassifierConfig(strategy="constant", constant=None)

        cfg = DummyClassifierConfig(strategy="constant", constant=1)
        assert cfg.constant == 1

    def test_invalid_random_state_type(self) -> None:
        with pytest.raises(TypeError, match="random_state must be an integer"):
            DummyClassifierConfig(random_state="seed")  # type: ignore

    def test_unknown_parameter_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter.*invalid_param"):
            DummyClassifierConfig.from_dict({"invalid_param": 99})

    def test_json_serializability(self) -> None:
        cfg = DummyClassifierConfig()
        d = cfg.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["strategy"] == "most_frequent"


class TestSVMConfig:
    """Test suite for SVMConfig."""

    def test_default_config(self) -> None:
        cfg = SVMConfig()
        assert cfg.kernel == "rbf"
        assert cfg.C == 1.0
        assert cfg.gamma == "scale"
        assert cfg.probability is False
        assert cfg.random_state == 42
        assert cfg.class_weight is None

    def test_valid_linear_kernel_override(self) -> None:
        cfg = SVMConfig.from_dict({"kernel": "linear", "C": 2.5, "class_weight": "balanced"})
        assert cfg.kernel == "linear"
        assert cfg.C == 2.5
        assert cfg.class_weight == "balanced"

    def test_numeric_gamma_accepted(self) -> None:
        cfg = SVMConfig(gamma=0.05)
        assert cfg.gamma == 0.05

    def test_invalid_kernel_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported kernel"):
            SVMConfig(kernel="sigmoid")

    def test_negative_or_zero_c_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Regularization parameter C must be strictly positive"):
            SVMConfig(C=0.0)
        with pytest.raises(ValueError, match="Regularization parameter C must be strictly positive"):
            SVMConfig(C=-1.0)

    def test_invalid_gamma_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="String gamma must be 'scale' or 'auto'"):
            SVMConfig(gamma="custom")
        with pytest.raises(ValueError, match="Numeric gamma must be strictly positive"):
            SVMConfig(gamma=-0.5)

    def test_invalid_probability_type(self) -> None:
        with pytest.raises(TypeError, match="probability must be a boolean"):
            SVMConfig(probability="true")  # type: ignore

    def test_invalid_class_weight(self) -> None:
        with pytest.raises(ValueError, match="String class_weight must be 'balanced'"):
            SVMConfig(class_weight="inverse")
        with pytest.raises(ValueError, match="class_weight dictionary cannot be empty"):
            SVMConfig(class_weight={})
        with pytest.raises(ValueError, match="must be positive"):
            SVMConfig(class_weight={0: -1.0})

    def test_unknown_parameter_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter.*cuda_enabled"):
            SVMConfig.from_dict({"cuda_enabled": True})

    def test_json_serializability(self) -> None:
        cfg = SVMConfig(kernel="linear", C=0.5)
        d = cfg.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["kernel"] == "linear"
        assert deserialized["C"] == 0.5


class TestMLPConfig:
    """Test suite for MLPConfig."""

    def test_default_config(self) -> None:
        cfg = MLPConfig()
        assert cfg.hidden_layer_sizes == (64, 32)
        assert cfg.activation == "relu"
        assert cfg.solver == "adam"
        assert cfg.alpha == 0.0001
        assert cfg.learning_rate_init == 0.001
        assert cfg.max_iter == 300
        assert cfg.early_stopping is False
        assert cfg.random_state == 42

    def test_single_int_hidden_layer_converted_to_tuple(self) -> None:
        cfg = MLPConfig(hidden_layer_sizes=64)  # type: ignore
        assert cfg.hidden_layer_sizes == (64,)

    def test_list_hidden_layer_converted_to_tuple(self) -> None:
        cfg = MLPConfig(hidden_layer_sizes=[128, 64, 32])  # type: ignore
        assert cfg.hidden_layer_sizes == (128, 64, 32)

    def test_empty_or_invalid_hidden_layers_rejected(self) -> None:
        with pytest.raises(ValueError, match="hidden_layer_sizes cannot be empty"):
            MLPConfig(hidden_layer_sizes=())
        with pytest.raises(ValueError, match="must be strictly positive"):
            MLPConfig(hidden_layer_sizes=(64, 0))
        with pytest.raises(ValueError, match="must be strictly positive"):
            MLPConfig(hidden_layer_sizes=(64, -10))

    def test_invalid_activation_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPConfig(activation="swish")

    def test_invalid_solver_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported solver"):
            MLPConfig(solver="rmsprop")

    def test_negative_alpha_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            MLPConfig(alpha=-0.01)

    def test_negative_or_zero_learning_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="learning_rate_init must be strictly positive"):
            MLPConfig(learning_rate_init=0.0)

    def test_negative_or_zero_max_iter_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be strictly positive"):
            MLPConfig(max_iter=0)

    def test_invalid_validation_fraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="validation_fraction must be in range"):
            MLPConfig(validation_fraction=1.5)

    def test_unknown_parameter_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter.*batch_size"):
            MLPConfig.from_dict({"batch_size": 32})

    def test_json_serializability(self) -> None:
        cfg = MLPConfig(hidden_layer_sizes=(100, 50))
        d = cfg.to_dict()
        assert isinstance(d["hidden_layer_sizes"], list)
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["hidden_layer_sizes"] == [100, 50]


class TestRandomForestConfig:
    """Test suite for RandomForestConfig."""

    def test_default_config(self) -> None:
        cfg = RandomForestConfig()
        assert cfg.n_estimators == 100
        assert cfg.max_depth is None
        assert cfg.min_samples_split == 2
        assert cfg.max_features == "sqrt"
        assert cfg.class_weight is None
        assert cfg.random_state == 42
        assert cfg.n_jobs == -1
        assert cfg.compute_feature_importances is True

    def test_valid_overrides(self) -> None:
        cfg = RandomForestConfig.from_dict({
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_split": 5,
            "max_features": "log2",
            "class_weight": "balanced_subsample",
        })
        assert cfg.n_estimators == 250
        assert cfg.max_depth == 12
        assert cfg.min_samples_split == 5
        assert cfg.max_features == "log2"
        assert cfg.class_weight == "balanced_subsample"

    def test_zero_or_negative_estimators_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_estimators must be strictly positive"):
            RandomForestConfig(n_estimators=0)

    def test_zero_or_negative_max_depth_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_depth must be strictly positive"):
            RandomForestConfig(max_depth=0)

    def test_invalid_min_samples_split_rejected(self) -> None:
        with pytest.raises(ValueError, match="Integer min_samples_split must be >= 2"):
            RandomForestConfig(min_samples_split=1)
        with pytest.raises(ValueError, match="Float min_samples_split must be in"):
            RandomForestConfig(min_samples_split=0.0)

    def test_invalid_max_features_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported string max_features"):
            RandomForestConfig(max_features="all")

    def test_invalid_class_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported string class_weight"):
            RandomForestConfig(class_weight="inverse")

    def test_unknown_parameter_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown parameter.*criterion"):
            RandomForestConfig.from_dict({"criterion": "entropy"})

    def test_json_serializability(self) -> None:
        cfg = RandomForestConfig(n_estimators=50, max_depth=6)
        d = cfg.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["n_estimators"] == 50
        assert deserialized["max_depth"] == 6
