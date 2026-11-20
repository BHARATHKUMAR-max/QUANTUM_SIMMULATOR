"""
backend/quantum/vqc_classifier.py
=================================
Binary and Multiclass Variational Quantum Classifier (VQC) prediction interface
for QML Benchmark Studio.

Logical Architecture:
---------------------
1. Binary Classification:
   - Uses quantum Pauli-Z expectation value on a designated `readout_wire` (default: wire 0).
   - Fixed, deterministic probability mapping:
       P(y = 1 | x) = (1 - z) / 2
       P(y = 0 | x) = (1 + z) / 2
     where z = <Z_{readout_wire}> in [-1.0, 1.0].
   - Output probability shape: (n_samples, 2).
   - Positive class (y = 1) is strictly classes_[1]; negative class (y = 0) is classes_[0].
   - Predicts via argmax over probabilities and decodes back to original target labels.

2. Multiclass Classification (3 or more classes):
   - Classical trainable linear head on top of all n_qubits Pauli-Z expectation values:
       h(x) = [<Z_0>, ..., <Z_{n-1}>]
       s(x) = W h(x) + b
       P(y = k | x) = exp(s_k) / sum_j exp(s_j)
   - Numerically stable softmax prevents logit overflow.
   - Head weights: (n_classes, n_qubits); Head bias: (n_classes,).
   - Output probability shape: (n_samples, n_classes).
   - Predicts via argmax over class probabilities and decodes to original target labels.

3. Label Encoding & Ordering:
   - Uses LabelEncoder to map arbitrary labels (strings, integers) to 0..K-1.
   - Preserves deterministic sorted class ordering in `classes_`.
   - Never leaks test labels or invents missing classes.
"""

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS
from backend.quantum.vqc import (
    DEFAULT_ENTANGLEMENT,
    DEFAULT_ROTATION_GATES,
    MAX_DEPTH,
    MIN_DEPTH,
    ParameterizedVQC,
    VQCConfig,
)


# ---------------------------------------------------------------------------
# 1. Mathematical Probability Helpers
# ---------------------------------------------------------------------------

def expectation_to_binary_proba(z: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert Pauli-Z expectation values in [-1.0, 1.0] to binary class probabilities.

    Mathematical formulation:
      P(y=1|x) = (1 - z) / 2
      P(y=0|x) = (1 + z) / 2 = 1 - P(y=1|x)

    When z = -1.0 (state |1>): P(y=1) = 1.0, P(y=0) = 0.0
    When z = +1.0 (state |0>): P(y=1) = 0.0, P(y=0) = 1.0
    When z =  0.0 (equatorial): P(y=1) = 0.5, P(y=0) = 0.5

    Args:
        z: Scalar float or 1D array of Pauli-Z expectation values.

    Returns:
        np.ndarray: Array of shape (n_samples, 2) where column 0 is P(y=0)
                    and column 1 is P(y=1). Probabilities are in [0.0, 1.0] and sum to 1.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    # Clip numerical edge effects to [-1.0, 1.0]
    z_clipped = np.clip(z_arr, -1.0, 1.0)

    p1 = (1.0 - z_clipped) / 2.0
    p0 = 1.0 - p1

    if z_clipped.ndim == 0:
        return np.array([p0, p1], dtype=np.float64)
    elif z_clipped.ndim == 1:
        return np.column_stack([p0, p1])
    else:
        raise ValueError(f"Expected 0D or 1D array for expectation values, got {z_clipped.ndim}D.")


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities along the last dimension.

    Subtracts the maximum logit per sample to prevent numerical overflow:
      P_k = exp(s_k - max(s)) / sum_j exp(s_j - max(s))

    Args:
        logits: Array of shape (n_samples, n_classes) or 1D (n_classes,).

    Returns:
        np.ndarray: Normalized probabilities in [0.0, 1.0] summing to 1.0 per sample.
    """
    s = np.asarray(logits, dtype=np.float64)
    if s.ndim == 1:
        s_shifted = s - np.max(s)
        exp_s = np.exp(s_shifted)
        return exp_s / np.sum(exp_s)
    elif s.ndim == 2:
        max_per_sample = np.max(s, axis=1, keepdims=True)
        s_shifted = s - max_per_sample
        exp_s = np.exp(s_shifted)
        sum_exp = np.sum(exp_s, axis=1, keepdims=True)
        return exp_s / sum_exp
    else:
        raise ValueError(f"Expected 1D or 2D array for logits, got {s.ndim}D.")


# ---------------------------------------------------------------------------
# 2. VQC Classifier Configuration Schema
# ---------------------------------------------------------------------------

@dataclass
class VQCClassifierConfig:
    """
    Configuration specification for the VQC Classifier.

    Attributes:
        n_qubits: Number of quantum circuit wires (2 to 10).
        depth: Number of variational layers (1 to 4).
        entanglement: Entanglement topology ('linear' or 'circular').
        rotation_gates: Trainable single-qubit rotation sequence (default 'RX+RY+RZ').
        device_name: Quantum simulator backend ('default.qubit' or 'lightning.qubit').
        encoding_rotation: Feature encoding rotation gate (default 'RY').
        readout_wire: Wire index used for binary classification readout (default 0).
        shots: Measurement shots (None for analytic statevector).
        seed: Random seed for reproducible initialization.
        head_seed: Random seed for multiclass linear head initialization.
        random_state: Master seed for reproducible operations.
    """
    n_qubits: int = 4
    depth: int = 2
    entanglement: str = DEFAULT_ENTANGLEMENT
    rotation_gates: str = DEFAULT_ROTATION_GATES
    device_name: str = "default.qubit"
    encoding_rotation: str = "RY"
    readout_wire: int = 0
    shots: Optional[int] = None
    seed: Optional[int] = None
    head_seed: Optional[int] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration upon instantiation."""
        # Harmonize seeds
        if self.random_state is not None:
            if self.seed is None:
                self.seed = self.random_state
            if self.head_seed is None:
                self.head_seed = self.random_state

        self.validate()

    def validate(self) -> None:
        """
        Validate all classifier configuration parameters strictly.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values violate architectural guardrails.
        """
        # Validate readout_wire
        if isinstance(self.readout_wire, bool) or not isinstance(self.readout_wire, int):
            raise TypeError(f"readout_wire must be an integer, got {type(self.readout_wire).__name__}.")
        if not (0 <= self.readout_wire < self.n_qubits):
            raise ValueError(
                f"readout_wire={self.readout_wire} is invalid for n_qubits={self.n_qubits}. "
                f"Must be between 0 and {self.n_qubits - 1}."
            )

        # Validate underlying circuit config
        vqc_cfg = self.to_vqc_config()
        vqc_cfg.validate()

    def to_vqc_config(self) -> VQCConfig:
        """Extract underlying VQCConfig instance."""
        return VQCConfig(
            n_qubits=self.n_qubits,
            depth=self.depth,
            entanglement=self.entanglement,
            rotation_gates=self.rotation_gates,
            device_name=self.device_name,
            encoding_rotation=self.encoding_rotation,
            shots=self.shots,
            seed=self.seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Optional[Union[Dict[str, Any], "VQCClassifierConfig"]],
    ) -> "VQCClassifierConfig":
        """Instantiate validated VQCClassifierConfig from dictionary or instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            data.validate()
            return data
        return cls(**data)


# ---------------------------------------------------------------------------
# 3. VQC Classifier Model Class
# ---------------------------------------------------------------------------

class VQCClassifier:
    """
    Variational Quantum Classifier (VQC) with unified binary and multiclass prediction interface.

    Supports:
      - Binary classification via single-wire Pauli-Z expectation probability conversion.
      - Multiclass classification via trainable classical linear head + softmax over all qubit expectations.
      - Full preservation of arbitrary original class labels (strings, integers).
      - Output conventions compatible with scikit-learn estimators and the project evaluation engine.
    """

    def __init__(
        self,
        config: Optional[VQCClassifierConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the VQC Classifier.

        Args:
            config: Optional VQCClassifierConfig instance.
            **kwargs: Configuration keyword arguments.
        """
        if config is None:
            self.config = VQCClassifierConfig(**kwargs)
        else:
            self.config = VQCClassifierConfig.from_dict(config)

        # Initialize underlying ParameterizedVQC
        vqc_cfg = self.config.to_vqc_config()
        self.vqc = ParameterizedVQC(config=vqc_cfg)

        # Label encoding and class state
        self.label_encoder = LabelEncoder()
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0
        self.is_binary_: bool = False

        # Parameters
        self.weights_: np.ndarray = np.array([])
        self.head_weights_: Optional[np.ndarray] = None
        self.head_bias_: Optional[np.ndarray] = None

        # Lifecycle attributes
        self.is_fitted_: bool = False
        self.n_features_in_: int = self.config.n_qubits
        self.n_samples_in_: int = 0
        self.feature_names_in_: Optional[List[str]] = None

        # Timing tracking
        self.training_time_: float = 0.0
        self.inference_time_: float = 0.0

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List[Any]],
    ) -> "VQCClassifier":
        """
        Fit the VQC Classifier on training data.

        Encodes target labels, fits the input AngleEncoder, and initializes circuit
        variational weights and multiclass linear head parameters.

        Args:
            X: Training feature matrix (shape: n_samples, n_features).
            y: Target class labels (shape: n_samples,).

        Returns:
            self: Fitted VQCClassifier instance.

        Raises:
            ValueError: If input dimensions mismatch, labels are invalid, or < 2 classes.
            TypeError: If input types are non-numeric.
        """
        t0 = time.perf_counter()

        # Validate and prepare X
        X_arr, feat_names = self._validate_and_extract_features(X, is_fit=True)
        n_samples, n_features = X_arr.shape

        if n_features != self.config.n_qubits:
            raise ValueError(
                f"Feature dimension mismatch: Configured for {self.config.n_qubits} qubits, "
                f"but received training data with {n_features} features. "
                f"Angle encoding requires exactly 1 feature per qubit."
            )

        # Validate and prepare y
        y_arr = self._validate_target(y, expected_len=n_samples)

        # Fit label encoder
        self.label_encoder.fit(y_arr)
        self.classes_ = np.asarray(self.label_encoder.classes_)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"Target y contains {self.n_classes_} class. "
                f"At least 2 distinct target classes are required for classification."
            )

        self.is_binary_ = (self.n_classes_ == 2)

        # Fit the VQC's internal AngleEncoder on training data ONLY
        self.vqc.encoder.fit(X_arr, feature_names=feat_names)

        # Initialize variational quantum circuit weights
        self.weights_ = self.vqc.init_weights(seed=self.config.seed)

        # If multiclass, initialize classical linear head: W in (K, n_qubits), b in (K,)
        if not self.is_binary_:
            rng_seed = self.config.head_seed if self.config.head_seed is not None else self.config.seed
            rng = np.random.default_rng(rng_seed)
            # Xavier/Glorot-like initialization for linear head
            scale = 1.0 / math.sqrt(self.config.n_qubits)
            self.head_weights_ = rng.normal(0.0, scale, size=(self.n_classes_, self.config.n_qubits)).astype(np.float64)
            self.head_bias_ = np.zeros(self.n_classes_, dtype=np.float64)
        else:
            self.head_weights_ = None
            self.head_bias_ = None

        self.n_features_in_ = n_features
        self.n_samples_in_ = n_samples
        self.feature_names_in_ = feat_names
        self.is_fitted_ = True

        self.training_time_ = time.perf_counter() - t0
        return self

    # -----------------------------------------------------------------------
    # Prediction & Probabilities
    # -----------------------------------------------------------------------

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute class probabilities for input observations.

        Args:
            X: Input feature matrix (shape: n_samples, n_features or single sample 1D).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, n_classes).
                        Each row sums to 1.0; columns correspond to self.classes_.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If feature count mismatches or contains NaN/Inf.
        """
        if not self.is_fitted_:
            raise RuntimeError("VQCClassifier instance is not fitted yet. Call fit() before predict_proba().")

        t0 = time.perf_counter()
        X_arr, _ = self._validate_and_extract_features(X, is_fit=False)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: Model was fitted on {self.n_features_in_} features, "
                f"but received input with {X_arr.shape[1]} features."
            )

        # Evaluate VQC Pauli-Z expectations on all qubits: shape (n_samples, n_qubits)
        z_expectations = self.vqc.evaluate(X_arr, self.weights_, is_encoded_angles=False)

        if self.is_binary_:
            # Extract Pauli-Z expectation from designated readout wire
            readout = z_expectations[:, self.config.readout_wire]
            proba = expectation_to_binary_proba(readout)
        else:
            # Multiclass linear head: s(x) = Z @ W^T + b
            logits = z_expectations @ self.head_weights_.T + self.head_bias_
            proba = softmax(logits)

        self.inference_time_ += (time.perf_counter() - t0)
        return proba

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for input observations.

        Args:
            X: Input feature matrix.

        Returns:
            np.ndarray: Array of predicted labels in the original target domain.
        """
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_indices)
        return pred_labels

    # -----------------------------------------------------------------------
    # Parameter Access & Mutation (For Day 25 Optimizer)
    # -----------------------------------------------------------------------

    def get_vqc_weights(self) -> np.ndarray:
        """Return a copy of the circuit variational parameters."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")
        return np.copy(self.weights_)

    def set_vqc_weights(self, weights: np.ndarray) -> None:
        """Update circuit variational parameters."""
        w_arr = np.asarray(weights, dtype=np.float64)
        expected_shape = self.config.to_vqc_config().get_weight_shape()
        if w_arr.shape != expected_shape:
            raise ValueError(f"Weight shape mismatch: expected {expected_shape}, got {w_arr.shape}.")
        self.weights_ = np.copy(w_arr)

    def get_head_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Return copies of multiclass linear head parameters (or None if binary)."""
        if not self.is_fitted_ or self.is_binary_:
            return None
        return {
            "weights": np.copy(self.head_weights_),
            "bias": np.copy(self.head_bias_),
        }

    def set_head_parameters(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Update multiclass linear head parameters."""
        if self.is_binary_:
            raise ValueError("Cannot set classical head parameters on a binary VQC classifier.")
        w_arr = np.asarray(weights, dtype=np.float64)
        b_arr = np.asarray(bias, dtype=np.float64)

        expected_w_shape = (self.n_classes_, self.config.n_qubits)
        expected_b_shape = (self.n_classes_,)

        if w_arr.shape != expected_w_shape:
            raise ValueError(f"Head weights shape mismatch: expected {expected_w_shape}, got {w_arr.shape}.")
        if b_arr.shape != expected_b_shape:
            raise ValueError(f"Head bias shape mismatch: expected {expected_b_shape}, got {b_arr.shape}.")

        self.head_weights_ = np.copy(w_arr)
        self.head_bias_ = np.copy(b_arr)

    # -----------------------------------------------------------------------
    # Introspection & Timing
    # -----------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serializable dictionary of model configuration and fitted state."""
        base_cfg = self.config.to_dict()
        base_cfg["is_fitted"] = self.is_fitted_
        base_cfg["n_classes"] = self.n_classes_
        base_cfg["is_binary"] = self.is_binary_
        base_cfg["classes"] = self.classes_.tolist() if self.is_fitted_ else []
        return base_cfg

    def get_timing(self) -> Dict[str, float]:
        """Return training and cumulative inference elapsed wall-clock times in seconds."""
        return {
            "training_time_seconds": float(self.training_time_),
            "inference_time_seconds": float(self.inference_time_),
        }

    # -----------------------------------------------------------------------
    # Internal Validation Helpers
    # -----------------------------------------------------------------------

    def _validate_and_extract_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        is_fit: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Validate input feature matrix and extract column names."""
        feat_names: Optional[List[str]] = None

        if isinstance(X, pd.DataFrame):
            feat_names = [str(c) for c in X.columns]
            try:
                X_arr = X.to_numpy(dtype=np.float64)
            except (ValueError, TypeError) as err:
                raise TypeError(f"DataFrame must contain purely numerical features: {err}") from err
        elif isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=np.float64).reshape(1, -1)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_arr = X.reshape(1, -1).astype(np.float64)
            else:
                X_arr = X.astype(np.float64)
        else:
            raise TypeError(f"Expected numpy.ndarray or pandas.DataFrame, got {type(X).__name__}.")

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix (n_samples, n_features), got shape {X_arr.shape}.")

        if X_arr.shape[0] == 0:
            raise ValueError("Input feature matrix contains 0 samples.")

        if X_arr.shape[1] == 0:
            raise ValueError("Input feature matrix contains 0 features.")

        if np.isnan(X_arr).any():
            raise ValueError("Input feature matrix contains NaN values.")

        if np.isinf(X_arr).any():
            raise ValueError("Input feature matrix contains infinite values.")

        if not is_fit and self.feature_names_in_ is not None and feat_names is not None:
            if feat_names != self.feature_names_in_:
                raise ValueError(
                    f"DataFrame columns {feat_names} do not match fitted feature names {self.feature_names_in_}."
                )

        return X_arr, feat_names

    def _validate_target(
        self,
        y: Union[np.ndarray, pd.Series, List[Any]],
        expected_len: int,
    ) -> np.ndarray:
        """Validate target labels array."""
        if isinstance(y, (pd.Series, pd.Index)):
            y_arr = y.to_numpy()
        elif isinstance(y, (list, tuple)):
            y_arr = np.asarray(y)
        elif isinstance(y, np.ndarray):
            y_arr = y
        else:
            raise TypeError(f"Expected target vector y, got {type(y).__name__}.")

        if y_arr.ndim != 1:
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            else:
                raise ValueError(f"Target vector y must be 1-dimensional, got shape {y_arr.shape}.")

        if len(y_arr) != expected_len:
            raise ValueError(
                f"Sample count mismatch: X has {expected_len} samples, but y has {len(y_arr)} labels."
            )

        if len(y_arr) == 0:
            raise ValueError("Target vector y is empty.")

        # Check for null / NaN in target labels
        if pd.isna(y_arr).any():
            raise ValueError("Target vector y contains missing/NaN labels.")

        return y_arr
