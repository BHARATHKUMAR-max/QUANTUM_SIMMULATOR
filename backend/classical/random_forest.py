"""
backend/classical/random_forest.py
==================================
Configurable Random Forest classifier for QML Benchmark Studio.

Wraps scikit-learn's `sklearn.ensemble.RandomForestClassifier` under the uniform
classical model interface. Supports configurable number of estimators, tree depth,
sample splitting thresholds, feature selection heuristics, feature importance
extraction, execution timing, and JSON serialization.

Key Invariants:
- Fitted strictly on training data; held-out test data is never seen during fit.
- Preserves original target class labels (integers, strings, booleans).
- Measures precise wall-clock training and inference time via `time.perf_counter()`.
- Supports optional feature importance extraction with original feature name alignment.
- Predicts discrete class labels and calibrated class probability distributions.
- Execution device is CPU (standard scikit-learn multi-threaded implementation).
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.exceptions import NotFittedError


class RandomForestClassifier:
    """
    Classical Random Forest ensemble classifier.

    Wraps `sklearn.ensemble.RandomForestClassifier` with hyperparameter validation,
    training/inference timing, feature importance extraction, and model metadata export.

    Attributes:
        n_estimators: Number of trees in the forest (> 0).
        max_depth: Maximum depth of each decision tree (None or > 0).
        min_samples_split: Minimum number of samples required to split an internal node.
        random_state: Seed for pseudo-random number generator and bootstrap sampling.
        class_weight: Weights associated with classes ('balanced', 'balanced_subsample', or dict).
        max_features: Number of features to consider when looking for the best split.
        n_jobs: Number of jobs to run in parallel (-1 for all processors).
        compute_feature_importances: Whether to compute and map feature importances.
        execution_device: Execution hardware platform (always 'CPU' for scikit-learn).
    """

    SUPPORTED_MAX_FEATURES = ("sqrt", "log2")
    SUPPORTED_CLASS_WEIGHTS = ("balanced", "balanced_subsample")

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        random_state: Optional[int] = 42,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        n_jobs: Optional[int] = -1,
        compute_feature_importances: bool = True,
    ) -> None:
        """
        Initialize the Random Forest Classifier.

        Args:
            n_estimators: Number of trees in the forest. Must be positive. Defaults to 100.
            max_depth: Maximum tree depth (None for unlimited). Defaults to None.
            min_samples_split: Minimum samples to split an internal node (int >= 2 or float in (0, 1]). Defaults to 2.
            random_state: Random seed for bootstrap sampling. Defaults to 42.
            class_weight: Class weighting mode (None, 'balanced', 'balanced_subsample', or dict). Defaults to None.
            max_features: Features to consider per split ('sqrt', 'log2', int > 0, float in (0, 1], or None). Defaults to 'sqrt'.
            n_jobs: Parallel worker count (-1 for all cores). Defaults to -1.
            compute_feature_importances: Whether to extract feature importance scores. Defaults to True.

        Raises:
            ValueError: If hyperparameter values are outside valid ranges or unsupported.
        """
        # Validate n_estimators
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators <= 0:
            raise ValueError(f"n_estimators must be a positive integer, got {n_estimators}.")
        self.n_estimators = int(n_estimators)

        # Validate max_depth
        if max_depth is not None:
            if not isinstance(max_depth, (int, np.integer)) or max_depth <= 0:
                raise ValueError(f"max_depth must be None or a positive integer, got {max_depth}.")
            self.max_depth: Optional[int] = int(max_depth)
        else:
            self.max_depth = None

        # Validate min_samples_split
        if isinstance(min_samples_split, (int, np.integer)):
            if min_samples_split < 2:
                raise ValueError(
                    f"Integer min_samples_split must be at least 2, got {min_samples_split}."
                )
            self.min_samples_split: Union[int, float] = int(min_samples_split)
        elif isinstance(min_samples_split, (float, np.floating)):
            if not (0.0 < min_samples_split <= 1.0):
                raise ValueError(
                    f"Float min_samples_split must be in the interval (0.0, 1.0], got {min_samples_split}."
                )
            self.min_samples_split = float(min_samples_split)
        else:
            raise ValueError(
                f"min_samples_split must be an integer >= 2 or float in (0.0, 1.0], got {type(min_samples_split).__name__}."
            )

        # Validate max_features
        if isinstance(max_features, str):
            if max_features.lower() not in self.SUPPORTED_MAX_FEATURES:
                raise ValueError(
                    f"Unsupported string max_features '{max_features}'. "
                    f"Supported options: {list(self.SUPPORTED_MAX_FEATURES)} or float/int/None."
                )
            self.max_features: Optional[Union[str, int, float]] = max_features.lower()
        elif isinstance(max_features, (int, np.integer)):
            if max_features <= 0:
                raise ValueError(f"Integer max_features must be positive, got {max_features}.")
            self.max_features = int(max_features)
        elif isinstance(max_features, (float, np.floating)):
            if not (0.0 < max_features <= 1.0):
                raise ValueError(f"Float max_features must be in (0.0, 1.0], got {max_features}.")
            self.max_features = float(max_features)
        elif max_features is None:
            self.max_features = None
        else:
            raise ValueError(
                f"max_features must be 'sqrt', 'log2', positive int, float in (0, 1], or None, "
                f"got {type(max_features).__name__}."
            )

        # Validate class_weight
        if class_weight is not None:
            if isinstance(class_weight, str):
                if class_weight.lower() not in self.SUPPORTED_CLASS_WEIGHTS:
                    raise ValueError(
                        f"Unsupported class_weight '{class_weight}'. "
                        f"Supported options: {list(self.SUPPORTED_CLASS_WEIGHTS)} or dict."
                    )
                self.class_weight: Optional[Union[str, Dict[Any, float]]] = class_weight.lower()
            elif isinstance(class_weight, dict):
                self.class_weight = class_weight
            else:
                raise ValueError(
                    f"class_weight must be None, 'balanced', 'balanced_subsample', or a dictionary, "
                    f"got {type(class_weight).__name__}."
                )
        else:
            self.class_weight = None

        # Validate compute_feature_importances
        if not isinstance(compute_feature_importances, bool):
            raise ValueError(
                f"compute_feature_importances must be a boolean, got {type(compute_feature_importances).__name__}."
            )
        self.compute_feature_importances = compute_feature_importances

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.execution_device = "CPU"

        # Internal estimator and state variables
        self._clf: Optional[SklearnRF] = None
        self._is_fitted: bool = False
        self._n_features_in: Optional[int] = None
        self._n_samples_in: Optional[int] = None
        self._feature_names_in: Optional[List[str]] = None
        self._training_time_seconds: Optional[float] = None
        self._inference_time_seconds: Optional[float] = None
        self._feature_importances_dict: Optional[Dict[str, float]] = None
        self._top_features_list: Optional[List[Dict[str, Any]]] = None

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the Random Forest estimator has been fitted."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """
        Unique class labels identified during training.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return self._clf.classes_

    @property
    def n_features_in_(self) -> int:
        """
        Number of features seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_features_in is None:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return self._n_features_in

    @property
    def n_samples_in_(self) -> int:
        """
        Number of training samples seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._n_samples_in is None:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return self._n_samples_in

    @property
    def feature_names_in_(self) -> List[str]:
        """
        Feature names seen during fitting.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._feature_names_in is None:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return list(self._feature_names_in)

    @property
    def training_time_seconds(self) -> Optional[float]:
        """Elapsed training wall-clock time in seconds."""
        return self._training_time_seconds

    @property
    def inference_time_seconds(self) -> Optional[float]:
        """Elapsed inference wall-clock time in seconds for the most recent predict call."""
        return self._inference_time_seconds

    @property
    def feature_importances_(self) -> Optional[Dict[str, float]]:
        """
        Mapping of feature names to Gini impurity feature importance scores in original feature order.

        Returns None if compute_feature_importances=False.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return self._feature_importances_dict

    @property
    def feature_importances_raw_(self) -> Optional[np.ndarray]:
        """
        Raw NumPy array of feature importances from scikit-learn.

        Returns None if compute_feature_importances=False.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        if not self.compute_feature_importances:
            return None
        return self._clf.feature_importances_

    @property
    def top_features_(self) -> Optional[List[Dict[str, Any]]]:
        """
        List of features sorted by descending importance score.

        Returns None if compute_feature_importances=False.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise NotFittedError("This RandomForestClassifier instance is not fitted yet.")
        return self._top_features_list

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray, List],
        y_train: Union[pd.Series, np.ndarray, List],
        feature_names: Optional[List[str]] = None,
    ) -> "RandomForestClassifier":
        """
        Fit the Random Forest classifier strictly on training data.

        Measures wall-clock training duration with `time.perf_counter()`.
        Optionally computes and structures Gini impurity feature importance scores.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Target labels (n_samples,).
            feature_names: Optional explicit list of feature names for array inputs.

        Returns:
            self: The fitted RandomForestClassifier instance.

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
            if feature_names is not None:
                if len(feature_names) != X_arr.shape[1] if X_arr.ndim > 1 else 1:
                    raise ValueError(
                        f"feature_names length ({len(feature_names)}) does not match "
                        f"input column count ({X_arr.shape[1] if X_arr.ndim > 1 else 1})."
                    )
                self._feature_names_in = list(feature_names)
            else:
                n_cols = X_arr.shape[1] if X_arr.ndim > 1 else 1
                self._feature_names_in = [f"feature_{i}" for i in range(n_cols)]

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

        # Build scikit-learn RandomForestClassifier
        self._clf = SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # Fit model and measure execution duration
        start_time = time.perf_counter()
        self._clf.fit(X_arr, y_arr)
        self._training_time_seconds = round(time.perf_counter() - start_time, 6)

        self._is_fitted = True
        self._n_samples_in = len(X_arr)
        self._n_features_in = X_arr.shape[1]

        # Extract feature importances if requested
        if self.compute_feature_importances:
            raw_importances = self._clf.feature_importances_
            # Preserve original feature order
            self._feature_importances_dict = {
                name: round(float(imp), 6)
                for name, imp in zip(self._feature_names_in, raw_importances)
            }
            # Sorted presentation list by descending importance
            sorted_indices = np.argsort(raw_importances)[::-1]
            self._top_features_list = [
                {
                    "feature": self._feature_names_in[i],
                    "importance": round(float(raw_importances[i]), 6),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(sorted_indices)
            ]
        else:
            self._feature_importances_dict = None
            self._top_features_list = None

        return self

    def _validate_prediction_input(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """Validate input matrix prior to prediction."""
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This RandomForestClassifier instance is not fitted yet. Call 'fit' before predicting."
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
            Dictionary containing model hyperparameters, feature importances,
            timings, and fitted state.
        """
        data: Dict[str, Any] = {
            "model_name": "random_forest",
            "model_family": "classical",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "compute_feature_importances": self.compute_feature_importances,
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
                "feature_importances": self.feature_importances_,
                "top_features": self.top_features_[:10] if self.top_features_ else None,
            })
        return data
