"""
backend/classical/__init__.py
=============================
Classical machine learning baseline and benchmark models for QML Benchmark Studio.

Exposes:
- Model implementations: DummyBaseline, SVMClassifier, MLPClassifier, RandomForestClassifier
- Typed configuration schemas: DummyClassifierConfig, SVMConfig, MLPConfig, RandomForestConfig
- Model registry interface: list_available_models, get_default_config, validate_model_config, create_model
- Recommendation engine: ModelRecommender, recommend_models, RecommenderConfig
"""

from backend.classical.dummy_baseline import DummyBaseline
from backend.classical.mlp import MLPClassifier
from backend.classical.model_configs import (
    BaseModelConfig,
    DummyClassifierConfig,
    MLPConfig,
    RandomForestConfig,
    SVMConfig,
)
from backend.classical.model_recommender import (
    ModelRecommendationReport,
    ModelRecommender,
    RecommendationItem,
    RecommenderConfig,
    recommend_models,
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

__all__ = [
    # Models
    "DummyBaseline",
    "MLPClassifier",
    "RandomForestClassifier",
    "SVMClassifier",
    # Configs
    "BaseModelConfig",
    "DummyClassifierConfig",
    "SVMConfig",
    "MLPConfig",
    "RandomForestConfig",
    # Registry
    "create_model",
    "get_config_class",
    "get_default_config",
    "get_model_class",
    "get_model_metadata",
    "list_available_models",
    "validate_model_config",
    # Recommender
    "ModelRecommendationReport",
    "ModelRecommender",
    "RecommendationItem",
    "RecommenderConfig",
    "recommend_models",
]
