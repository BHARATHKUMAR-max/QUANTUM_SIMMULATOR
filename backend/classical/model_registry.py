"""
backend/classical/model_registry.py
===================================
Centralized model registry for classical machine learning baselines in QML Benchmark Studio.

Provides unified discovery, configuration retrieval, validation, and model instantiation
for:
- dummy: DummyBaseline
- svm: SVMClassifier
- mlp: MLPClassifier
- random_forest: RandomForestClassifier
"""

from typing import Any, Dict, List, Optional, Type, Union

from backend.classical.dummy_baseline import DummyBaseline
from backend.classical.mlp import MLPClassifier
from backend.classical.model_configs import (
    BaseModelConfig,
    DummyClassifierConfig,
    MLPConfig,
    RandomForestConfig,
    SVMConfig,
)
from backend.classical.random_forest import RandomForestClassifier
from backend.classical.svm import SVMClassifier


# Internal registry mapping canonical names to implementation details
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dummy": {
        "display_name": "Dummy Baseline Classifier",
        "model_class": DummyBaseline,
        "config_class": DummyClassifierConfig,
        "execution_device": "CPU",
        "description": "Zero-knowledge heuristic baseline predicting the most frequent training class.",
        "aliases": ["dummy_baseline", "dummy_classifier"],
    },
    "svm": {
        "display_name": "Support Vector Machine Classifier",
        "model_class": SVMClassifier,
        "config_class": SVMConfig,
        "execution_device": "CPU",
        "description": "Maximum-margin classifier supporting linear and radial basis function (RBF) kernels.",
        "aliases": ["svc", "support_vector_machine"],
    },
    "mlp": {
        "display_name": "Multi-Layer Perceptron Classifier",
        "model_class": MLPClassifier,
        "config_class": MLPConfig,
        "execution_device": "CPU",
        "description": "Feed-forward neural network with backpropagation and configurable hidden layers.",
        "aliases": ["neural_network", "mlp_classifier"],
    },
    "random_forest": {
        "display_name": "Random Forest Classifier",
        "model_class": RandomForestClassifier,
        "config_class": RandomForestConfig,
        "execution_device": "CPU",
        "description": "Bagged decision tree ensemble with Gini feature importance and multi-threaded CPU execution.",
        "aliases": ["rf", "randomforest"],
    },
}

# Alias resolution mapping
_ALIAS_MAP: Dict[str, str] = {}
for canonical_name, entry in _MODEL_REGISTRY.items():
    _ALIAS_MAP[canonical_name] = canonical_name
    for alias in entry["aliases"]:
        _ALIAS_MAP[alias] = canonical_name


def _resolve_model_name(model_name: str) -> str:
    """Resolve aliases or case variations to canonical model name."""
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}.")

    normalized = model_name.strip().lower()
    if normalized not in _ALIAS_MAP:
        raise ValueError(
            f"Unknown classical model '{model_name}'. "
            f"Available models are: {list_available_models()}."
        )
    return _ALIAS_MAP[normalized]


def list_available_models() -> List[str]:
    """
    List all registered classical model identifiers.

    Returns:
        List of canonical model identifiers ['dummy', 'svm', 'mlp', 'random_forest'].
    """
    return list(_MODEL_REGISTRY.keys())


def get_model_class(model_name: str) -> Type[Any]:
    """
    Get the underlying model class for a model identifier.

    Args:
        model_name: Identifier of the classical model.

    Returns:
        The Python model class.
    """
    canonical = _resolve_model_name(model_name)
    return _MODEL_REGISTRY[canonical]["model_class"]


def get_config_class(model_name: str) -> Type[BaseModelConfig]:
    """
    Get the configuration dataclass for a model identifier.

    Args:
        model_name: Identifier of the classical model.

    Returns:
        The configuration dataclass type.
    """
    canonical = _resolve_model_name(model_name)
    return _MODEL_REGISTRY[canonical]["config_class"]


def get_default_config(model_name: str) -> BaseModelConfig:
    """
    Get the default configuration instance for a model identifier.

    Args:
        model_name: Identifier of the classical model.

    Returns:
        Default validated BaseModelConfig instance.
    """
    config_cls = get_config_class(model_name)
    return config_cls()


def validate_model_config(
    model_name: str,
    config: Optional[Union[BaseModelConfig, Dict[str, Any]]] = None,
) -> BaseModelConfig:
    """
    Validate and resolve configuration parameters for a specified model.

    If config is None, returns the default configuration.
    If config is a dictionary, unknown parameters are rejected and defaults are populated.
    If config is a BaseModelConfig instance, its types and constraints are validated.

    Args:
        model_name: Identifier of the classical model.
        config: Configuration instance or dictionary of overrides.

    Returns:
        Resolved and validated BaseModelConfig instance.

    Raises:
        ValueError: If model_name is unknown or configuration validation fails.
        TypeError: If configuration object type is incompatible.
    """
    canonical = _resolve_model_name(model_name)
    config_cls = _MODEL_REGISTRY[canonical]["config_class"]

    if config is None:
        return config_cls()

    if isinstance(config, dict):
        return config_cls.from_dict(config)

    if isinstance(config, config_cls):
        config.validate()
        return config

    raise TypeError(
        f"Configuration for model '{canonical}' must be a dict or {config_cls.__name__}, "
        f"got {type(config).__name__}."
    )


def create_model(
    model_name: str,
    config: Optional[Union[BaseModelConfig, Dict[str, Any]]] = None,
) -> Any:
    """
    Instantiate a classical model using a validated configuration.

    Args:
        model_name: Identifier of the classical model.
        config: Optional configuration instance or dictionary of overrides.
            If None, default configuration is used.

    Returns:
        Instantiated classical model wrapper (DummyBaseline, SVMClassifier, MLPClassifier, or RandomForestClassifier).
    """
    canonical = _resolve_model_name(model_name)
    validated_config = validate_model_config(canonical, config)
    model_cls = _MODEL_REGISTRY[canonical]["model_class"]

    cfg_dict = validated_config.to_dict()

    # Pass relevant kwargs based on model type
    if canonical == "dummy":
        # DummyBaseline accepts strategy, random_state, and constant (required when strategy='constant')
        return model_cls(
            strategy=cfg_dict.get("strategy", "most_frequent"),
            random_state=cfg_dict.get("random_state", 42),
            constant=cfg_dict.get("constant", None),
        )
    elif canonical == "svm":
        return model_cls(
            kernel=cfg_dict.get("kernel", "rbf"),
            C=cfg_dict.get("C", 1.0),
            gamma=cfg_dict.get("gamma", "scale"),
            probability=cfg_dict.get("probability", False),
            random_state=cfg_dict.get("random_state", 42),
            class_weight=cfg_dict.get("class_weight", None),
        )
    elif canonical == "mlp":
        return model_cls(
            hidden_layer_sizes=tuple(cfg_dict.get("hidden_layer_sizes", (64, 32))),
            activation=cfg_dict.get("activation", "relu"),
            solver=cfg_dict.get("solver", "adam"),
            alpha=cfg_dict.get("alpha", 0.0001),
            learning_rate_init=cfg_dict.get("learning_rate_init", 0.001),
            max_iter=cfg_dict.get("max_iter", 300),
            random_state=cfg_dict.get("random_state", 42),
            early_stopping=cfg_dict.get("early_stopping", False),
            validation_fraction=cfg_dict.get("validation_fraction", 0.1),
            n_iter_no_change=cfg_dict.get("n_iter_no_change", 10),
        )
    elif canonical == "random_forest":
        return model_cls(
            n_estimators=cfg_dict.get("n_estimators", 100),
            max_depth=cfg_dict.get("max_depth", None),
            min_samples_split=cfg_dict.get("min_samples_split", 2),
            random_state=cfg_dict.get("random_state", 42),
            class_weight=cfg_dict.get("class_weight", None),
            max_features=cfg_dict.get("max_features", "sqrt"),
            n_jobs=cfg_dict.get("n_jobs", -1),
            compute_feature_importances=cfg_dict.get("compute_feature_importances", True),
        )

    # Fallback instantiation
    return model_cls(**cfg_dict)


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Get descriptive metadata and configuration schema information for a model.

    Args:
        model_name: Identifier of the classical model.

    Returns:
        Dictionary containing metadata, defaults, and parameter documentation.
    """
    canonical = _resolve_model_name(model_name)
    entry = _MODEL_REGISTRY[canonical]
    config_cls = entry["config_class"]
    default_cfg = config_cls()

    return {
        "model_name": canonical,
        "display_name": entry["display_name"],
        "model_class": entry["model_class"].__name__,
        "config_class": config_cls.__name__,
        "execution_device": entry["execution_device"],
        "description": entry["description"],
        "default_config": default_cfg.to_dict(),
    }
