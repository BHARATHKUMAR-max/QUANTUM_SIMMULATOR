"""
backend/classical/svm.py
========================
Configurable Support Vector Machine (SVC) classifier for QML Benchmark Studio.

Wraps scikit-learn's `sklearn.svm.SVC` under the standard classical model interface.
Supports Linear and RBF kernels, configurable regularization (C) and kernel width (gamma),
optional probability estimates, decision functions, and execution timing.

Key Invariants:
- Fitted strictly on training data; held-out test data is never seen during fit.
- Preserves original target class labels (integers, strings, booleans).
- Measures precise wall-clock training and inference time via `time.perf_counter()`.
- Probability estimation is disabled by default to avoid internal cross-validation overhead.
- When probability=False, exposes decision_function for margin-based evaluation.
- Execution device is CPU (standard scikit-learn libsvm implementation).
"""

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC


class SVMClassifier:
    """
    Classical Support Vector Machine (SVC) classifier.

    Wraps `sklearn.svm.SVC` with configurable kernels, hyperparameter validation,
    training/inference timing, decision score extraction, and model metadata export.

    Attributes:
        kernel: Kernel type ('linear' or 'rbf').
        C: Regularization parameter (must be > 0).
        gamma: Kernel coefficient for RBF ('scale', 'auto', or float > 0).
        probability: Whether to enable probability calibration via Platt scaling.
        random_state: Seed for pseudo-random number generator (used when probability=True).
        class_weight: Class weighting mode (None, 'balanced', or dict).
        execution_device: Execution hardware platform (always 'CPU' for scikit-learn).
    """

    SUPPORTED_KERNELS = ("linear", "rbf")

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        probability: bool = False,
        random_state: Optional[int] = 42,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        """
        Initialize the Support Vector Classifier.

        Args:
            kernel: Kernel specification ('linear' or 'rbf'). Defaults to 'rbf'.
            C: Regularization parameter. Must be positive. Defaults to 1.0.
            gamma: Kernel coefficient for 'rbf' ('scale', 'auto', or float > 0). Defaults to 'scale'.
            probability: Whether to enable probability estimation. Defaults to False.
            random_state: Deterministic random seed for probability estimation. Defaults to 42.
            class_weight: Optional class weights ('balanced' or mapping). Defaults to None.

        Raises:
            ValueError: If hyperparameter values are outside valid ranges or unsupported.
        """
        # Validate kernel
        if not isinstance(kernel, str) or kernel.lower() not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{kernel}'. Supported kernels are: {list(self.SUPPORTED_KERNELS)}."
            )
        self.kernel = kernel.lower()

        # Validate C
        if not isinstance(C, (int, float)) or C <= 0:
            raise ValueError(f"Regularization parameter C must be a positive number, got {C}.")
        self.C = float(C)

        # Validate gamma
        if self.kernel == "rbf":
            if isinstance(gamma, str):
                if gamma.lower() not in ("scale", "auto"):
                    raise ValueError(
                        f"String gamma for RBF kernel must be 'scale' or 'auto', got '{gamma}'."
                    )
                self.gamma: Union[str, float] = gamma.lower()
            elif isinstance(gamma, (int, float)):
                if gamma <= 0:
                    raise ValueError(f"Numerical gamma for RBF kernel must be positive, got {gamma}.")
                self.gamma = float(gamma)
            else:
                raise ValueError(
                    f"gamma must be 'scale', 'auto', or a positive float, got {type(gamma).__name__}."
                )
        else:
            # For linear kernel, gamma is not an active hyperparameter
            self.gamma = gamma

        # Validate probability
        if not isinstance(probability, bool):
            raise ValueError(f"probability must be a boolean, got {type(probability).__name__}.")
        self.probability = probability

        # Validate class_weight
        if class_weight is not None and class_weight != "balanced" and not isinstance(class_weight, dict):
            raise ValueError(
                f"class_weight must be None, 'balanced', or a dictionary, got {class_weight}."
            )
        self.class_weight = class_weight

        self.random_state = random_state
        self.execution_device = "CPU"

        # Internal estimator and state variables
        self._clf: Optional[SVC] = None
        self._is_fitted: bool = False
        self._n_features_in: Optional[int] = None
        self._n_samples_in: Optional[int] = None
        self._feature_names_in: Optional[List[str]] = None
        self._training_time_seconds: Optional[float] = None
        self._inference_time_seconds: Optional[float] = None

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the SVM estimator has been fitted."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """
        Unique class labels identified during training.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This SVMClassifier instance is not fitted yet.")
        return self._clf.classes_

    @property
    def n_features_in_(self) -> int:
        """
        Number of features seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_features_in is None:
            raise NotFittedError("This SVMClassifier instance is not fitted yet.")
        return self._n_features_in

    @property
    def n_samples_in_(self) -> int:
        """
        Number of training samples seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_samples_in is None:
            raise NotFittedError("This SVMClassifier instance is not fitted yet.")
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
    def support_vectors_count_(self) -> int:
        """
        Total number of support vectors learned across all classes.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This SVMClassifier instance is not fitted yet.")
        return int(np.sum(self._clf.n_support_))

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray, List],
        y_train: Union[pd.Series, np.ndarray, List],
    ) -> "SVMClassifier":
        """
        Fit the SVM classifier strictly on training data.

        Measures wall-clock training duration with `time.perf_counter()`.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Target labels (n_samples,).

        Returns:
            self: The fitted SVMClassifier instance.

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

        # Build scikit-learn SVC
        svc_kwargs: Dict[str, Any] = {
            "kernel": self.kernel,
            "C": self.C,
            "probability": self.probability,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
        }
        if self.kernel == "rbf":
            svc_kwargs["gamma"] = self.gamma

        self._clf = SVC(**svc_kwargs)

        # Fit model and measure training execution duration
        start_time = time.perf_counter()
        self._clf.fit(X_arr, y_arr)
        self._training_time_seconds = round(time.perf_counter() - start_time, 6)

        self._is_fitted = True
        self._n_samples_in = len(X_arr)
        self._n_features_in = X_arr.shape[1]

        return self

    def _validate_prediction_input(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """Validate input matrix prior to prediction or decision evaluation."""
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This SVMClassifier instance is not fitted yet. Call 'fit' before predicting."
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
        Predict class probabilities for input observations X.

        Requires probability=True to have been configured during initialization.

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            2D numpy array of shape (n_samples, n_classes).

        Raises:
            AttributeError: If probability estimation was not enabled.
            NotFittedError: If model is not fitted.
            ValueError: If X is empty or feature count is mismatched.
        """
        if not self.probability:
            raise AttributeError(
                "predict_proba is not available when probability=False. "
                "Initialize with probability=True or use decision_function for continuous scores."
            )

        X_arr = self._validate_prediction_input(X)
        return self._clf.predict_proba(X_arr)

    def decision_function(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """
        Compute decision function values (signed distances to separating hyperplane).

        For binary classification, returns a 1D array of shape (n_samples,).
        For multiclass classification, returns a 2D array of shape (n_samples, n_classes).

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            Continuous decision scores.

        Raises:
            NotFittedError: If model is not fitted.
            ValueError: If X is empty or feature count is mismatched.
        """
        X_arr = self._validate_prediction_input(X)
        return self._clf.decision_function(X_arr)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export serializable model metadata and configuration dictionary.

        Returns:
            Dictionary containing model hyperparameters, timings, and fitted state.
        """
        data: Dict[str, Any] = {
            "model_name": "svm",
            "model_family": "classical",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma if self.kernel == "rbf" else None,
            "probability": self.probability,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
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
                "support_vectors_count": self.support_vectors_count_,
            })
        return data
