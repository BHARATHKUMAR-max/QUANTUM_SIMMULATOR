"""
backend/classical/dummy_baseline.py
===================================
Lightweight wrapper around scikit-learn's DummyClassifier for QML Benchmark Studio.

Provides the minimal baseline model (strategy="most_frequent") against which all
classical ML models (SVM, Random Forest, MLP) and quantum ML models (VQC, QK-SVM)
are compared.

Key Invariants:
- Fitted EXCLUSIVELY on training data (X_train, y_train); held-out test data is never seen during fit.
- Predicts the most frequent training class label for all input observations.
- Preserves original target class labels (integers, strings, booleans).
- Exposes standard scikit-learn estimator attributes: classes_, predict, predict_proba.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError


class DummyBaseline:
    """
    Classical baseline estimator wrapping sklearn.dummy.DummyClassifier.

    Predicts the most frequent class observed in the training distribution.
    Serves as the zero-knowledge baseline for benchmarking classical and quantum classifiers.

    Attributes:
        strategy: Strategy used by DummyClassifier (default: 'most_frequent').
        random_state: Seed for pseudo-random number generator (if strategy requires it).
        is_fitted: Boolean flag indicating whether the model has been fitted.
        n_features_in_: Number of features seen during fitting.
        n_samples_in_: Number of training samples seen during fitting.
    """

    def __init__(
        self,
        strategy: str = "most_frequent",
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the Dummy baseline classifier.

        Args:
            strategy: Prediction strategy. Defaults to 'most_frequent'.
            random_state: Optional random seed for reproducibility.
        """
        self.strategy = strategy
        self.random_state = random_state
        self._clf: Optional[DummyClassifier] = None
        self._is_fitted: bool = False
        self._n_features_in: Optional[int] = None
        self._n_samples_in: Optional[int] = None
        self._feature_names_in: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the baseline estimator has been fitted."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """
        Array of unique class labels seen during fitting.

        Raises:
            NotFittedError: If the estimator is not fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This DummyBaseline instance is not fitted yet. "
                "Call 'fit' with training data before accessing classes_."
            )
        return self._clf.classes_

    @property
    def n_features_in_(self) -> int:
        """
        Number of features seen during fitting.

        Raises:
            NotFittedError: If the estimator is not fitted.
        """
        if not self._is_fitted or self._n_features_in is None:
            raise NotFittedError("This DummyBaseline instance is not fitted yet.")
        return self._n_features_in

    @property
    def n_samples_in_(self) -> int:
        """
        Number of training samples seen during fitting.

        Raises:
            NotFittedError: If the estimator is not fitted.
        """
        if not self._is_fitted or self._n_samples_in is None:
            raise NotFittedError("This DummyBaseline instance is not fitted yet.")
        return self._n_samples_in

    @property
    def most_frequent_class_(self) -> Any:
        """
        The most frequent class label identified during fitting.

        Raises:
            NotFittedError: If the estimator is not fitted.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError("This DummyBaseline instance is not fitted yet.")
        # In DummyClassifier with most_frequent, class_prior_ has 1.0 for the most frequent class
        most_freq_idx = int(np.argmax(self._clf.class_prior_))
        return self._clf.classes_[most_freq_idx]

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray, List],
        y_train: Union[pd.Series, np.ndarray, List],
    ) -> "DummyBaseline":
        """
        Fit the baseline classifier on training features and targets.

        Strictly learns only from X_train and y_train.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            y_train: Training target labels (n_samples,).

        Returns:
            self: The fitted DummyBaseline instance.

        Raises:
            ValueError: If inputs are None, empty, or have mismatched dimensions.
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")

        # Convert to numpy/pandas structures for shape validation
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
            # Reshape 1D array to 2D single-feature column
            X_arr = X_arr.reshape(-1, 1)

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Sample count mismatch: X_train has {len(X_arr)} samples, "
                f"but y_train has {len(y_arr)} labels."
            )

        # Initialize underlying scikit-learn DummyClassifier
        self._clf = DummyClassifier(
            strategy=self.strategy,
            random_state=self.random_state,
        )

        # Fit on training data
        self._clf.fit(X_arr, y_arr)
        self._is_fitted = True
        self._n_samples_in = len(X_arr)
        self._n_features_in = X_arr.shape[1]

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """
        Predict class labels for test observations X.

        For strategy='most_frequent', returns the most frequent training label for all rows.

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            1D numpy array of predicted labels matching target class dtype.

        Raises:
            NotFittedError: If the estimator is not fitted.
            ValueError: If X is empty or has mismatched feature count.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This DummyBaseline instance is not fitted yet. "
                "Call 'fit' before using 'predict'."
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

        return self._clf.predict(X_arr)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, List],
    ) -> np.ndarray:
        """
        Predict class probabilities for test observations X.

        For strategy='most_frequent', returns 1.0 for the most frequent training class
        and 0.0 for all other classes.

        Args:
            X: Input feature matrix (n_samples, n_features).

        Returns:
            2D numpy array of shape (n_samples, n_classes).

        Raises:
            NotFittedError: If the estimator is not fitted.
            ValueError: If X is empty or has mismatched feature count.
        """
        if not self._is_fitted or self._clf is None:
            raise NotFittedError(
                "This DummyBaseline instance is not fitted yet. "
                "Call 'fit' before using 'predict_proba'."
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

        return self._clf.predict_proba(X_arr)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export serializable model metadata.

        Returns:
            Dictionary containing model configuration and fitted state.
        """
        base_dict: Dict[str, Any] = {
            "model_name": "dummy",
            "model_family": "classical",
            "strategy": self.strategy,
            "random_state": self.random_state,
            "is_fitted": self._is_fitted,
        }
        if self._is_fitted and self._clf is not None:
            base_dict.update({
                "classes": [
                    int(c) if isinstance(c, (np.integer, int)) else str(c)
                    for c in self._clf.classes_
                ],
                "most_frequent_class": (
                    int(self.most_frequent_class_)
                    if isinstance(self.most_frequent_class_, (np.integer, int))
                    else str(self.most_frequent_class_)
                ),
                "n_features_in": self._n_features_in,
                "n_samples_in": self._n_samples_in,
                "feature_names_in": self._feature_names_in,
            })
        return base_dict
