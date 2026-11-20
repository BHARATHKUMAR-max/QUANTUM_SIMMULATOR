"""
backend/classical/mlp.py
========================
Configurable Multi-Layer Perceptron (MLP) neural-network classifier for QML Benchmark Studio.

Wraps scikit-learn's `sklearn.neural_network.MLPClassifier` under the uniform
classical model interface. Supports configurable hidden layers, activation functions,
solvers, learning rates, iteration limits, early stopping, and convergence diagnostics.

Key Invariants:
- Fitted strictly on training data; held-out test data is never seen during fit.
- Preserves original target class labels (integers, strings, booleans).
- Measures precise wall-clock training and inference time via `time.perf_counter()`.
- Captures and records ConvergenceWarning diagnostics without silently suppressing them.
- Predicts discrete class labels and calibrated class probability distributions.
- Execution device is CPU (standard scikit-learn implementation).
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.neural_network import MLPClassifier as SklearnMLP


class MLPClassifier:
    """
    Classical Multi-Layer Perceptron (MLP) feed-forward neural-network classifier.

    Wraps `sklearn.neural_network.MLPClassifier` with hyperparameter validation,
    training/inference timing, convergence diagnostics, and model metadata export.

    Attributes:
        hidden_layer_sizes: Tuple of positive integers defining neurons per hidden layer.
        activation: Activation function for the hidden layers ('relu', 'tanh', 'logistic').
        solver: Optimization algorithm ('adam', 'sgd', 'lbfgs').
        alpha: L2 penalty (regularization term) parameter (alpha >= 0).
        learning_rate_init: Initial learning rate for gradient-based optimization (> 0).
        max_iter: Maximum number of iterations / epochs (> 0).
        random_state: Random seed for weight initialization and batch shuffling.
        early_stopping: Whether to use early stopping based on validation score.
        validation_fraction: Proportion of training data set aside as validation for early stopping.
        n_iter_no_change: Maximum number of epochs with no improvement before stopping.
        execution_device: Execution hardware platform (always 'CPU' for scikit-learn).
    """

    SUPPORTED_ACTIVATIONS = ("relu", "tanh", "logistic", "identity")
    SUPPORTED_SOLVERS = ("adam", "sgd", "lbfgs")

    def __init__(
        self,
        hidden_layer_sizes: Union[Tuple[int, ...], List[int], int] = (64, 32),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = 42,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None:
        """
        Initialize the MLP Classifier.

        Args:
            hidden_layer_sizes: Structure of hidden layers. Defaults to (64, 32).
            activation: Activation function ('relu', 'tanh', 'logistic'). Defaults to 'relu'.
            solver: Optimizer solver ('adam', 'sgd', 'lbfgs'). Defaults to 'adam'.
            alpha: L2 penalty parameter (alpha >= 0). Defaults to 0.0001.
            learning_rate_init: Initial learning rate (> 0). Defaults to 0.001.
            max_iter: Maximum iterations (> 0). Defaults to 200.
            random_state: Random seed. Defaults to 42.
            early_stopping: Whether to use early stopping. Defaults to False.
            validation_fraction: Validation fraction for early stopping (0.0 to 1.0). Defaults to 0.1.
            n_iter_no_change: Epochs with no improvement before stopping. Defaults to 10.

        Raises:
            ValueError: If hyperparameter values are outside valid ranges or unsupported.
        """
        # Validate hidden_layer_sizes
        if isinstance(hidden_layer_sizes, int):
            if hidden_layer_sizes <= 0:
                raise ValueError(
                    f"Hidden layer neuron count must be positive, got {hidden_layer_sizes}."
                )
            self.hidden_layer_sizes: Tuple[int, ...] = (hidden_layer_sizes,)
        elif isinstance(hidden_layer_sizes, (tuple, list)):
            if len(hidden_layer_sizes) == 0:
                raise ValueError("hidden_layer_sizes must contain at least one hidden layer.")
            for idx, size in enumerate(hidden_layer_sizes):
                if not isinstance(size, (int, np.integer)) or size <= 0:
                    raise ValueError(
                        f"Hidden layer size at index {idx} must be a positive integer, got {size}."
                    )
            self.hidden_layer_sizes = tuple(int(s) for s in hidden_layer_sizes)
        else:
            raise ValueError(
                f"hidden_layer_sizes must be an int or tuple/list of ints, got {type(hidden_layer_sizes).__name__}."
            )

        # Validate activation
        if not isinstance(activation, str) or activation.lower() not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. Supported activations are: {list(self.SUPPORTED_ACTIVATIONS)}."
            )
        self.activation = activation.lower()

        # Validate solver
        if not isinstance(solver, str) or solver.lower() not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver}'. Supported solvers are: {list(self.SUPPORTED_SOLVERS)}."
            )
        self.solver = solver.lower()

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError(f"Regularization parameter alpha must be non-negative, got {alpha}.")
        self.alpha = float(alpha)

        # Validate learning_rate_init
        if not isinstance(learning_rate_init, (int, float)) or learning_rate_init <= 0:
            raise ValueError(
                f"Initial learning rate learning_rate_init must be positive, got {learning_rate_init}."
            )
        self.learning_rate_init = float(learning_rate_init)

        # Validate max_iter
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError(f"Maximum iterations max_iter must be a positive integer, got {max_iter}.")
        self.max_iter = int(max_iter)

        # Validate early_stopping parameters
        if not isinstance(early_stopping, bool):
            raise ValueError(f"early_stopping must be a boolean, got {type(early_stopping).__name__}.")
        self.early_stopping = early_stopping

        if not isinstance(validation_fraction, (int, float)) or not (0.0 < validation_fraction < 1.0):
            raise ValueError(
                f"validation_fraction must be in (0.0, 1.0), got {validation_fraction}."
            )
        self.validation_fraction = float(validation_fraction)

        if not isinstance(n_iter_no_change, (int, np.integer)) or n_iter_no_change <= 0:
            raise ValueError(
                f"n_iter_no_change must be a positive integer, got {n_iter_no_change}."
            )
        self.n_iter_no_change = int(n_iter_no_change)

        self.random_state = random_state
        self.execution_device = "CPU"

        # Internal estimator and state variables
        self._clf: Optional[SklearnMLP] = None
        self._is_fitted: bool = False
        self._n_features_in: Optional[int] = None
        self._n_samples_in: Optional[int] = None
        self._feature_names_in: Optional[List[str]] = None
        self._training_time_seconds: Optional[float] = None
        self._inference_time_seconds: Optional[float] = None

        # Convergence diagnostics
        self._converged: Optional[bool] = None
        self._convergence_warning: Optional[str] = None
        self._captured_warnings: List[str] = []
        self._n_iter: Optional[int] = None
        self._loss: Optional[float] = None
        self._best_loss: Optional[float] = None
        self._n_layers: Optional[int] = None
        self._n_outputs: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the MLP estimator has been fitted."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """
        Unique class labels identified during training.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._clf.classes_

    @property
    def n_features_in_(self) -> int:
        """
        Number of features seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_features_in is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._n_features_in

    @property
    def n_samples_in_(self) -> int:
        """
        Number of training samples seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_samples_in is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._n_samples_in

    @property
    def training_time_seconds(self) -> Optional[float]:
        """Elapsed training wall-clock time in seconds."""
        return self._training_time_seconds

    @property
    def inference_time_seconds(self) -> Optional[float]:
        """Elapsed inference wall-clock time in seconds for the most recent predict call."""
        return self._inference_time_seconds

    @property
    def n_iter_(self) -> int:
        """
        The number of iterations the solver ran.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_iter is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._n_iter

    @property
    def loss_(self) -> float:
        """
        The minimum loss computed with the loss function for the given data.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._loss is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._loss

    @property
    def best_loss_(self) -> Optional[float]:
        """The minimum loss reached by the solver throughout fitting."""
        return self._best_loss

    @property
    def converged_(self) -> bool:
        """
        Whether optimization converged without triggering a ConvergenceWarning.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._converged is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._converged

    @property
    def convergence_warning_(self) -> Optional[str]:
        """Convergence warning message if convergence failed, else None."""
        return self._convergence_warning

    @property
    def n_layers_(self) -> int:
        """
        Number of layers in the neural network (input + hidden + output).

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_layers is None:
            raise NotFittedError("This MLPClassifier instance is not fitted yet.")
        return self._n_layers

    @property
    def total_hidden_neurons_(self) -> int:
        """Total count of hidden neurons across all hidden layers."""
        return sum(self.hidden_layer_sizes)

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray, List],
        y_train: Union[pd.Series, np.ndarray, List],
    ) -> "MLPClassifier":
        """
        Fit the MLP classifier strictly on training data.

        Measures wall-clock training duration with `time.perf_counter()`.
        Captures ConvergenceWarning diagnostics and re-emits warnings for the caller.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Target labels (n_samples,).

        Returns:
            self: The fitted MLPClassifier instance.

        Raises:
            ValueError: If inputs are None, empty, or have mismatched dimensions.
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        # Convert to numpy/pandas structures for validation
        if isinstance(X_train, pd.DataFrame):
            self._feature_names_in = list(X_train.columns)
            X_arr = X_train.to_numpy()
        else:
            X_arr = np.asarray(X_train)
            self._feature_names_in = None

        if isinstance(y_train, (pd.Series, pd.Index)):
            y_arr = y_train.to_numpy()
        else:
            y_arr = np.asarray(y_train)

        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("X_train and y_train must not be empty.")

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Sample count mismatch: X_train has {len(X_arr)} samples, "
                f"but y_train has {len(y_arr)} labels."
            )

        # Build scikit-learn MLPClassifier
        self._clf = SklearnMLP(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
        )

        # Fit model and measure execution time while capturing convergence warnings
        self._captured_warnings = []
        self._convergence_warning = None
        has_convergence_warning = False

        start_time = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            self._clf.fit(X_arr, y_arr)

        self._training_time_seconds = round(time.perf_counter() - start_time, 6)

        # Process captured warnings and re-emit so they are not swallowed
        for w in captured:
            msg = str(w.message)
            self._captured_warnings.append(msg)
            if issubclass(w.category, ConvergenceWarning):
                has_convergence_warning = True
                self._convergence_warning = msg
            # Re-emit warning
            warnings.warn(w.message, w.category)

        self._converged = not has_convergence_warning
        self._n_iter = int(self._clf.n_iter_)
        self._loss = round(float(self._clf.loss_), 6)
        self._best_loss = (
            round(float(self._clf.best_loss_), 6)
            if getattr(self._clf, "best_loss_", None) is not None
            else None
        )
        self._n_layers = int(self._clf.n_layers_)
        self._n_outputs = int(self._clf.n_outputs_)

        self._is_fitted = True
        self._n_samples_in = len(X_arr)
        self._n_features_in = X_arr.shape[1]

        return self

    def _validate_prediction_input(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """Validate input matrix prior to prediction."""
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This MLPClassifier instance is not fitted yet. Call 'fit' before predicting."
            )

        if X is None:
            raise ValueError("Input feature matrix X cannot be None.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)

        if X_arr.size == 0:
            raise ValueError("Input feature matrix X must not be empty.")

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if X_arr.shape[1] != self._n_features_in:
            raise ValueError(
                f"Feature count mismatch: fitted with {self._n_features_in} features, "
                f"but received input with {X_arr.shape[1]} features."
            )

        return X_arr

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """
        Predict discrete class labels for input observations X.

        Measures wall-clock inference duration with `time.perf_counter()`.

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            1D numpy array of predicted labels matching target class dtype.

        Raises:
            NotFittedError: If model is not fitted.
            ValueError: If X is empty or feature count is mismatched.
        """
        X_arr = self._validate_prediction_input(X)

        start_time = time.perf_counter()
        y_pred = self._clf.predict(X_arr)
        self._inference_time_seconds = round(time.perf_counter() - start_time, 6)

        return y_pred

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """
        Predict class probability distributions for input observations X.

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            2D numpy array of shape (n_samples, n_classes).

        Raises:
            NotFittedError: If model is not fitted.
            ValueError: If X is empty or feature count is mismatched.
        """
        X_arr = self._validate_prediction_input(X)
        return self._clf.predict_proba(X_arr)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export serializable model metadata and configuration dictionary.

        Returns:
            Dictionary containing model hyperparameters, convergence diagnostics,
            timings, and fitted state.
        """
        data: Dict[str, Any] = {
            "model_name": "mlp",
            "model_family": "classical",
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "total_hidden_neurons": self.total_hidden_neurons_,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "execution_device": self.execution_device,
            "is_fitted": self._is_fitted,
            "training_time_seconds": self._training_time_seconds,
            "inference_time_seconds": self._inference_time_seconds,
        }
        if self._is_fitted and self._clf is not None:
            data.update({
                "classes": [
                    int(c) if isinstance(c, (np.integer, int)) else str(c)
                    for c in self.classes_
                ],
                "n_features_in": self._n_features_in,
                "n_samples_in": self._n_samples_in,
                "feature_names_in": self._feature_names_in,
                "n_iter": self._n_iter,
                "loss": self._loss,
                "best_loss": self._best_loss,
                "converged": self._converged,
                "convergence_warning": self._convergence_warning,
                "n_layers": self._n_layers,
                "warnings": self._captured_warnings,
            })
        return data
