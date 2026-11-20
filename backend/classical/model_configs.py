"""
backend/classical/model_configs.py
==================================
Centralized, typed configuration schemas and validation for classical models.

Defines typed dataclass configurations for:
- DummyClassifierConfig
- SVMConfig
- MLPConfig
- RandomForestConfig

Provides validation rules, default configurations, override resolution,
and JSON serialization.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class BaseModelConfig(ABC):
    """Abstract base class for classical model configuration dataclasses."""

    @abstractmethod
    def validate(self) -> None:
        """Validate hyperparameter types, constraints, and supported options."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as a JSON-serializable dictionary.

        Returns:
            Dictionary containing all configuration fields.
        """
        d = asdict(self)
        # Ensure collections are JSON-serializable
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "BaseModelConfig":
        """
        Instantiate configuration from dictionary overrides with validation.

        Args:
            data: Dictionary of hyperparameter overrides. If None or empty, returns default config.

        Returns:
            Validated configuration instance.

        Raises:
            ValueError: If unknown parameters are provided or validation fails.
            TypeError: If parameter types are invalid.
        """
        if data is None:
            return cls()

        if not isinstance(data, dict):
            raise TypeError(f"Configuration overrides must be a dict, got {type(data).__name__}.")

        valid_fields = {f.name for f in fields(cls)}
        unknown_keys = sorted(list(set(data.keys()) - valid_fields))
        if unknown_keys:
            sorted_valid = sorted(list(valid_fields))
            raise ValueError(
                f"Unknown parameter(s) for {cls.__name__}: {unknown_keys}. "
                f"Allowed parameters are: {sorted_valid}"
            )

        return cls(**data)


# =============================================================================
# 1. Dummy Baseline Configuration
# =============================================================================

@dataclass
class DummyClassifierConfig(BaseModelConfig):
    """
    Configuration schema for DummyBaseline classifier.

    Attributes:
        strategy: Prediction strategy. Supported: 'most_frequent', 'prior', 'stratified', 'uniform', 'constant'.
        random_state: Seed for pseudo-random number generator.
        constant: Constant target value when strategy='constant'.
    """

    SUPPORTED_STRATEGIES: Tuple[str, ...] = (
        "most_frequent",
        "prior",
        "stratified",
        "uniform",
        "constant",
    )

    strategy: str = "most_frequent"
    random_state: Optional[int] = 42
    constant: Optional[Union[int, str]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate Dummy baseline parameters."""
        if not isinstance(self.strategy, str):
            raise TypeError(f"strategy must be a string, got {type(self.strategy).__name__}.")

        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported dummy strategy '{self.strategy}'. "
                f"Supported strategies are: {list(self.SUPPORTED_STRATEGIES)}."
            )

        if self.strategy == "constant" and self.constant is None:
            raise ValueError("strategy='constant' requires providing a non-None 'constant' value.")

        if self.random_state is not None:
            if not isinstance(self.random_state, int):
                raise TypeError(f"random_state must be an integer or None, got {type(self.random_state).__name__}.")


# =============================================================================
# 2. Support Vector Machine Configuration
# =============================================================================

@dataclass
class SVMConfig(BaseModelConfig):
    """
    Configuration schema for SVMClassifier (sklearn SVC).

    Attributes:
        kernel: Kernel type ('linear' or 'rbf').
        C: Regularization parameter (must be > 0).
        gamma: Kernel coefficient for RBF ('scale', 'auto', or float > 0).
        probability: Whether to enable Platt scaling probability calibration.
        random_state: Seed for pseudo-random number generator.
        class_weight: Class weighting mode (None, 'balanced', or custom dict).
    """

    SUPPORTED_KERNELS: Tuple[str, ...] = ("linear", "rbf")

    kernel: str = "rbf"
    C: float = 1.0
    gamma: Union[str, float] = "scale"
    probability: bool = False
    random_state: Optional[int] = 42
    class_weight: Optional[Union[str, Dict[Any, float]]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate Support Vector Classifier parameters."""
        if not isinstance(self.kernel, str):
            raise TypeError(f"kernel must be a string, got {type(self.kernel).__name__}.")

        if self.kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{self.kernel}'. "
                f"Supported kernels are: {list(self.SUPPORTED_KERNELS)}."
            )

        if not isinstance(self.C, (int, float)):
            raise TypeError(f"C must be a positive float or int, got {type(self.C).__name__}.")

        if self.C <= 0:
            raise ValueError(f"Regularization parameter C must be strictly positive (> 0), got {self.C}.")

        if isinstance(self.gamma, str):
            if self.gamma not in ("scale", "auto"):
                raise ValueError(
                    f"String gamma must be 'scale' or 'auto', got '{self.gamma}'."
                )
        elif isinstance(self.gamma, (int, float)):
            if self.gamma <= 0:
                raise ValueError(f"Numeric gamma must be strictly positive (> 0), got {self.gamma}.")
        else:
            raise TypeError(f"gamma must be 'scale', 'auto', or a positive float, got {type(self.gamma).__name__}.")

        if not isinstance(self.probability, bool):
            raise TypeError(f"probability must be a boolean, got {type(self.probability).__name__}.")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(f"random_state must be an integer or None, got {type(self.random_state).__name__}.")

        if self.class_weight is not None:
            if isinstance(self.class_weight, str):
                if self.class_weight != "balanced":
                    raise ValueError(f"String class_weight must be 'balanced', got '{self.class_weight}'.")
            elif isinstance(self.class_weight, dict):
                if not self.class_weight:
                    raise ValueError("class_weight dictionary cannot be empty.")
                for k, w in self.class_weight.items():
                    if not isinstance(w, (int, float)) or w <= 0:
                        raise ValueError(f"class_weight for class {k} must be positive (> 0), got {w}.")
            else:
                raise TypeError(
                    f"class_weight must be None, 'balanced', or a dict of weights, got {type(self.class_weight).__name__}."
                )


# =============================================================================
# 3. Multi-Layer Perceptron Configuration
# =============================================================================

@dataclass
class MLPConfig(BaseModelConfig):
    """
    Configuration schema for MLPClassifier.

    Attributes:
        hidden_layer_sizes: Structure of hidden layers as tuple of positive ints.
        activation: Activation function ('relu', 'tanh', 'logistic', 'identity').
        solver: Optimization algorithm ('adam', 'sgd', 'lbfgs').
        alpha: L2 regularization penalty parameter (alpha >= 0).
        learning_rate_init: Initial learning rate (> 0).
        max_iter: Maximum optimization iterations / epochs (> 0).
        early_stopping: Whether to use early stopping on validation split.
        random_state: Seed for pseudo-random number generator.
        validation_fraction: Validation proportion reserved for early stopping.
        n_iter_no_change: Epoch patience for early stopping.
    """

    SUPPORTED_ACTIVATIONS: Tuple[str, ...] = ("relu", "tanh", "logistic", "identity")
    SUPPORTED_SOLVERS: Tuple[str, ...] = ("adam", "sgd", "lbfgs")

    hidden_layer_sizes: Union[Tuple[int, ...], List[int], int] = (64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    max_iter: int = 300
    early_stopping: bool = False
    random_state: Optional[int] = 42
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10

    def __post_init__(self) -> None:
        # Normalize hidden_layer_sizes to tuple
        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)
        elif isinstance(self.hidden_layer_sizes, list):
            self.hidden_layer_sizes = tuple(self.hidden_layer_sizes)

        self.validate()

    def validate(self) -> None:
        """Validate Multi-Layer Perceptron parameters."""
        if not isinstance(self.hidden_layer_sizes, tuple):
            raise TypeError(
                f"hidden_layer_sizes must be a tuple, list, or int, got {type(self.hidden_layer_sizes).__name__}."
            )

        if len(self.hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes cannot be empty.")

        for idx, size in enumerate(self.hidden_layer_sizes):
            if not isinstance(size, int):
                raise TypeError(f"hidden_layer_sizes[{idx}] must be an integer, got {type(size).__name__}.")
            if size <= 0:
                raise ValueError(f"hidden_layer_sizes[{idx}] must be strictly positive (> 0), got {size}.")

        if not isinstance(self.activation, str):
            raise TypeError(f"activation must be a string, got {type(self.activation).__name__}.")

        if self.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Supported activations are: {list(self.SUPPORTED_ACTIVATIONS)}."
            )

        if not isinstance(self.solver, str):
            raise TypeError(f"solver must be a string, got {type(self.solver).__name__}.")

        if self.solver not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{self.solver}'. "
                f"Supported solvers are: {list(self.SUPPORTED_SOLVERS)}."
            )

        if not isinstance(self.alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(self.alpha).__name__}.")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative (>= 0), got {self.alpha}.")

        if not isinstance(self.learning_rate_init, (int, float)):
            raise TypeError(f"learning_rate_init must be numeric, got {type(self.learning_rate_init).__name__}.")
        if self.learning_rate_init <= 0:
            raise ValueError(f"learning_rate_init must be strictly positive (> 0), got {self.learning_rate_init}.")

        if not isinstance(self.max_iter, int):
            raise TypeError(f"max_iter must be an integer, got {type(self.max_iter).__name__}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be strictly positive (> 0), got {self.max_iter}.")

        if not isinstance(self.early_stopping, bool):
            raise TypeError(f"early_stopping must be a boolean, got {type(self.early_stopping).__name__}.")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(f"random_state must be an integer or None, got {type(self.random_state).__name__}.")

        if not isinstance(self.validation_fraction, (int, float)):
            raise TypeError(f"validation_fraction must be a float, got {type(self.validation_fraction).__name__}.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(f"validation_fraction must be in range (0, 1), got {self.validation_fraction}.")

        if not isinstance(self.n_iter_no_change, int):
            raise TypeError(f"n_iter_no_change must be an integer, got {type(self.n_iter_no_change).__name__}.")
        if self.n_iter_no_change <= 0:
            raise ValueError(f"n_iter_no_change must be strictly positive (> 0), got {self.n_iter_no_change}.")


# =============================================================================
# 4. Random Forest Configuration
# =============================================================================

@dataclass
class RandomForestConfig(BaseModelConfig):
    """
    Configuration schema for RandomForestClassifier.

    Attributes:
        n_estimators: Number of decision trees in forest (must be >= 1).
        max_depth: Maximum tree depth (None for unlimited, or int >= 1).
        min_samples_split: Minimum samples required to split internal node (int >= 2 or float in (0, 1]).
        max_features: Feature subset strategy per split ('sqrt', 'log2', float in (0, 1], int >= 1, or None).
        class_weight: Class weighting mode (None, 'balanced', 'balanced_subsample', or dict).
        random_state: Seed for pseudo-random number generator.
        n_jobs: Number of CPU worker threads (-1 for all cores).
        compute_feature_importances: Whether to compute Gini feature importances.
    """

    SUPPORTED_MAX_FEATURES: Tuple[str, ...] = ("sqrt", "log2")
    SUPPORTED_CLASS_WEIGHTS: Tuple[str, ...] = ("balanced", "balanced_subsample")

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    max_features: Optional[Union[str, int, float]] = "sqrt"
    class_weight: Optional[Union[str, Dict[Any, float]]] = None
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = -1
    compute_feature_importances: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate Random Forest parameters."""
        if not isinstance(self.n_estimators, int):
            raise TypeError(f"n_estimators must be an integer, got {type(self.n_estimators).__name__}.")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be strictly positive (>= 1), got {self.n_estimators}.")

        if self.max_depth is not None:
            if not isinstance(self.max_depth, int):
                raise TypeError(f"max_depth must be an integer or None, got {type(self.max_depth).__name__}.")
            if self.max_depth < 1:
                raise ValueError(f"max_depth must be strictly positive (>= 1), got {self.max_depth}.")

        if isinstance(self.min_samples_split, int):
            if self.min_samples_split < 2:
                raise ValueError(f"Integer min_samples_split must be >= 2, got {self.min_samples_split}.")
        elif isinstance(self.min_samples_split, float):
            if not (0.0 < self.min_samples_split <= 1.0):
                raise ValueError(f"Float min_samples_split must be in (0.0, 1.0], got {self.min_samples_split}.")
        else:
            raise TypeError(
                f"min_samples_split must be int >= 2 or float in (0.0, 1.0], got {type(self.min_samples_split).__name__}."
            )

        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features not in self.SUPPORTED_MAX_FEATURES:
                    raise ValueError(
                        f"Unsupported string max_features '{self.max_features}'. "
                        f"Supported options: {list(self.SUPPORTED_MAX_FEATURES)}."
                    )
            elif isinstance(self.max_features, int):
                if self.max_features < 1:
                    raise ValueError(f"Integer max_features must be >= 1, got {self.max_features}.")
            elif isinstance(self.max_features, float):
                if not (0.0 < self.max_features <= 1.0):
                    raise ValueError(f"Float max_features must be in (0.0, 1.0], got {self.max_features}.")
            else:
                raise TypeError(
                    f"max_features must be 'sqrt', 'log2', int >= 1, float in (0, 1], or None, "
                    f"got {type(self.max_features).__name__}."
                )

        if self.class_weight is not None:
            if isinstance(self.class_weight, str):
                if self.class_weight not in self.SUPPORTED_CLASS_WEIGHTS:
                    raise ValueError(
                        f"Unsupported string class_weight '{self.class_weight}'. "
                        f"Supported options: {list(self.SUPPORTED_CLASS_WEIGHTS)}."
                    )
            elif isinstance(self.class_weight, dict):
                if not self.class_weight:
                    raise ValueError("class_weight dictionary cannot be empty.")
                for k, w in self.class_weight.items():
                    if not isinstance(w, (int, float)) or w <= 0:
                        raise ValueError(f"class_weight for class {k} must be positive (> 0), got {w}.")
            else:
                raise TypeError(
                    f"class_weight must be None, 'balanced', 'balanced_subsample', or a dict, "
                    f"got {type(self.class_weight).__name__}."
                )

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(f"random_state must be an integer or None, got {type(self.random_state).__name__}.")

        if self.n_jobs is not None and not isinstance(self.n_jobs, int):
            raise TypeError(f"n_jobs must be an integer or None, got {type(self.n_jobs).__name__}.")

        if not isinstance(self.compute_feature_importances, bool):
            raise TypeError(
                f"compute_feature_importances must be a boolean, got {type(self.compute_feature_importances).__name__}."
            )
