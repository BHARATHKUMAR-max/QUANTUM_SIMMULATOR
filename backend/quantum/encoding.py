"""
backend/quantum/encoding.py
===========================
Quantum angle encoding module for QML Benchmark Studio.

Converts continuous tabular dataset features into rotation angles for
parameterized PennyLane quantum circuits (specifically Variational Quantum Classifiers).

Core Principles & Architectural Invariants:
-------------------------------------------
1. Feature-to-Qubit Mapping:
   - Standard mapping is strictly 1-to-1 (one feature per wire).
   - Wire 0 encodes feature 0, wire 1 encodes feature 1, ..., wire k-1 encodes feature k-1.
   - Feature ordering and original feature names (e.g. from Pandas DataFrames or PCA)
     are preserved identically.
   - Silent feature truncation (e.g. slicing X[:, :k]) and silent zero-padding are forbidden.

2. Qubit Constraints:
   - Supported wire counts are strictly bounded to [2, 10] (MIN_QUBITS=2, MAX_QUBITS=10).
   - An explicit ValueError is raised if feature count != qubit count or if qubit count
     falls outside [2, 10].

3. Leakage Safety (Training-Only Scaling):
   - Scaling parameters (x_min, x_max, is_constant) are fitted exclusively on training data.
   - Validation and test partitions are transformed using the already-fitted parameters.
   - Transforming test data NEVER modifies fitted parameters.

4. Linear Scaling Formula:
   angle = angle_min + (x - x_min) * (angle_max - angle_min) / (x_max - x_min)
   Default target angle range: [-π, π].
   Configurable target range: [angle_min, angle_max] with angle_min < angle_max.

5. Edge Cases & Robustness:
   - Constant-valued features (x_max == x_min): mapped to the exact midpoint of the
     target angle range: (angle_min + angle_max) / 2.0 (i.e. 0.0 for [-π, π]).
   - Out-of-range test values (x < x_min or x > x_max): clipped to [angle_min, angle_max]
     when clipping=True (default), preventing uncalibrated rotations.
   - NaN and infinite values: strictly rejected with ValueError.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pennylane as qml

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS

VALID_ROTATIONS = ("RX", "RY", "RZ")
DEFAULT_ROTATION = "RY"
DEFAULT_ANGLE_RANGE: Tuple[float, float] = (-math.pi, math.pi)


# ---------------------------------------------------------------------------
# 1. Configuration Schema
# ---------------------------------------------------------------------------

@dataclass
class AngleEncodingConfig:
    """
    Configuration for quantum angle encoding.

    Attributes:
        rotation: Quantum rotation gate to apply ('RX', 'RY', or 'RZ'). Default 'RY'.
        angle_range: Target angle interval (min_angle, max_angle). Default (-π, π).
        n_qubits: Target number of qubits (wires). Must match feature count and fall in [2, 10].
                  If None, inferred from feature dimensions during fit() and validated in [2, 10].
        clipping: Whether to clip transformed angles to [min_angle, max_angle]. Default True.
        constant_feature_policy: Policy for constant features ('midpoint' or 'zero'). Default 'midpoint'.
    """
    rotation: str = DEFAULT_ROTATION
    angle_range: Tuple[float, float] = DEFAULT_ANGLE_RANGE
    n_qubits: Optional[int] = None
    clipping: bool = True
    constant_feature_policy: str = "midpoint"

    def __post_init__(self) -> None:
        """Validate configuration parameters upon initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all configuration settings strictly.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If values fall outside acceptable domains.
        """
        # Validate rotation
        if not isinstance(self.rotation, str):
            raise TypeError(f"rotation must be a string, got {type(self.rotation).__name__}.")
        norm_rot = self.rotation.strip().upper()
        if norm_rot not in VALID_ROTATIONS:
            raise ValueError(
                f"Unsupported rotation gate '{self.rotation}'. "
                f"Supported rotation gates are: {sorted(VALID_ROTATIONS)}."
            )
        self.rotation = norm_rot

        # Validate angle_range
        if not isinstance(self.angle_range, (tuple, list)) or len(self.angle_range) != 2:
            raise TypeError("angle_range must be a tuple or list of exactly two numbers (min_angle, max_angle).")
        angle_min, angle_max = float(self.angle_range[0]), float(self.angle_range[1])
        if math.isnan(angle_min) or math.isnan(angle_max) or math.isinf(angle_min) or math.isinf(angle_max):
            raise ValueError("angle_range boundaries must be finite numbers.")
        if angle_min >= angle_max:
            raise ValueError(
                f"angle_range lower bound ({angle_min}) must be strictly smaller "
                f"than upper bound ({angle_max})."
            )
        self.angle_range = (angle_min, angle_max)

        # Validate n_qubits
        if self.n_qubits is not None:
            if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
                raise TypeError(f"n_qubits must be an integer or None, got {type(self.n_qubits).__name__}.")
            if not (MIN_QUBITS <= self.n_qubits <= MAX_QUBITS):
                raise ValueError(
                    f"n_qubits={self.n_qubits} is outside supported simulator range [{MIN_QUBITS}, {MAX_QUBITS}]. "
                    f"PennyLane quantum simulation is constrained to 2–10 qubits."
                )

        # Validate clipping
        if not isinstance(self.clipping, bool):
            raise TypeError(f"clipping must be a boolean, got {type(self.clipping).__name__}.")

        # Validate constant_feature_policy
        valid_policies = ("midpoint", "zero")
        if self.constant_feature_policy not in valid_policies:
            raise ValueError(
                f"Unsupported constant_feature_policy '{self.constant_feature_policy}'. "
                f"Supported policies are: {list(valid_policies)}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "AngleEncodingConfig"]]) -> "AngleEncodingConfig":
        """Instantiate validated AngleEncodingConfig from dictionary or instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            data.validate()
            return data
        # Handle angle_range if deserialized as a list
        if "angle_range" in data and isinstance(data["angle_range"], list):
            data = dict(data)
            data["angle_range"] = tuple(data["angle_range"])
        return cls(**data)


# ---------------------------------------------------------------------------
# 2. Angle Encoder Class
# ---------------------------------------------------------------------------

class AngleEncoder:
    """
    Stateless/Stateful quantum feature encoder mapping tabular features to rotation angles.

    Scales tabular features to the target angle range [angle_min, angle_max] and establishes
    a deterministic 1-to-1 mapping from feature columns to quantum circuit wires.
    """

    def __init__(
        self,
        config: Optional[AngleEncodingConfig] = None,
        rotation: str = DEFAULT_ROTATION,
        angle_range: Tuple[float, float] = DEFAULT_ANGLE_RANGE,
        n_qubits: Optional[int] = None,
        clipping: bool = True,
        constant_feature_policy: str = "midpoint",
    ) -> None:
        """
        Initialize the angle encoder.

        Args:
            config: Optional AngleEncodingConfig instance. If provided, overrides individual kwargs.
            rotation: Rotation gate ('RX', 'RY', or 'RZ'). Default 'RY'.
            angle_range: Target angle interval (min_angle, max_angle). Default (-π, π).
            n_qubits: Expected number of qubits (2–10). If None, inferred during fit().
            clipping: Whether to clip values to angle_range. Default True.
            constant_feature_policy: Constant feature handling ('midpoint' or 'zero').
        """
        if config is not None:
            self.config = AngleEncodingConfig.from_dict(config)
        else:
            self.config = AngleEncodingConfig(
                rotation=rotation,
                angle_range=angle_range,
                n_qubits=n_qubits,
                clipping=clipping,
                constant_feature_policy=constant_feature_policy,
            )

        # Fitted state attributes
        self.is_fitted_: bool = False
        self.n_features_in_: int = 0
        self.n_qubits_: int = 0
        self.feature_names_: List[str] = []
        self.x_min_: np.ndarray = np.array([], dtype=np.float64)
        self.x_max_: np.ndarray = np.array([], dtype=np.float64)
        self.is_constant_: np.ndarray = np.array([], dtype=bool)
        self.wire_mapping_: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "AngleEncoder":
        """
        Fit the angle encoder on training data ONLY.

        Computes feature-wise minima, maxima, and constant-feature flags. Establishes
        the deterministic feature-to-wire mapping.

        Args:
            X: Training feature matrix (shape: n_samples, n_features).
            feature_names: Optional explicit list of feature names. If X is a DataFrame,
                           X.columns is used (or validated against feature_names).

        Returns:
            Self (fitted AngleEncoder).

        Raises:
            ValueError: If X has invalid dimensions, contains NaN/Inf, or violates qubit bounds.
            TypeError: If input types are invalid.
        """
        X_arr, extracted_names = self._validate_and_extract_input(X, feature_names=feature_names, is_fit=True)

        n_samples, n_features = X_arr.shape

        # Validate dimensional constraints
        if self.config.n_qubits is not None:
            if n_features != self.config.n_qubits:
                raise ValueError(
                    f"Dimensionality mismatch: Configured n_qubits={self.config.n_qubits}, "
                    f"but training data has {n_features} features. Angle encoding requires exactly "
                    f"1 feature per qubit (n_features == n_qubits). "
                    f"Expected {self.config.n_qubits} features, got {n_features}."
                )
            self.n_qubits_ = self.config.n_qubits
        else:
            if not (MIN_QUBITS <= n_features <= MAX_QUBITS):
                raise ValueError(
                    f"Feature count ({n_features}) is outside the supported qubit simulator range "
                    f"[{MIN_QUBITS}, {MAX_QUBITS}]. "
                    f"Angle encoding requires exactly 1 feature per qubit, and PennyLane simulation "
                    f"in this project is constrained to 2–10 qubits. "
                    f"Please apply feature selection or PCA to reduce features to [2, 10]."
                )
            self.n_qubits_ = n_features

        self.n_features_in_ = n_features
        self.feature_names_ = extracted_names

        # Compute training min and max per feature
        self.x_min_ = np.min(X_arr, axis=0)
        self.x_max_ = np.max(X_arr, axis=0)

        # Detect constant features (atol=1e-12 handles floating point rounding)
        self.is_constant_ = np.isclose(self.x_min_, self.x_max_, atol=1e-12)

        # Build explicit feature-to-wire mapping metadata
        self.wire_mapping_ = [
            {
                "wire": i,
                "feature_name": name,
                "feature_index": i,
                "rotation": self.config.rotation,
                "x_min": float(self.x_min_[i]),
                "x_max": float(self.x_max_[i]),
                "is_constant": bool(self.is_constant_[i]),
            }
            for i, name in enumerate(self.feature_names_)
        ]

        self.is_fitted_ = True
        return self

    # -----------------------------------------------------------------------
    # Transformation
    # -----------------------------------------------------------------------

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform feature matrix to rotation angles in radians.

        Uses the already-fitted training scaling parameters. Never mutates fitted state.

        Args:
            X: Input feature matrix (shape: (n_samples, n_features) or single sample (n_features,)).

        Returns:
            np.ndarray: Encoded angles in radians with identical shape to input.

        Raises:
            RuntimeError: If encoder has not been fitted.
            ValueError: If dimensions mismatch or input contains NaN/Inf.
        """
        if not self.is_fitted_:
            raise RuntimeError("AngleEncoder instance is not fitted yet. Call fit() before transform().")

        is_1d = False
        if isinstance(X, np.ndarray) and X.ndim == 1:
            is_1d = True
            X_eval = X.reshape(1, -1)
        elif isinstance(X, pd.Series):
            is_1d = True
            X_eval = X.to_frame().T
        else:
            X_eval = X

        X_arr, _ = self._validate_and_extract_input(X_eval, feature_names=None, is_fit=False)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: Encoder was fitted on {self.n_features_in_} features, "
                f"but received input with {X_arr.shape[1]} features."
            )

        angle_min, angle_max = self.config.angle_range
        angle_delta = angle_max - angle_min

        # Avoid zero-division on constant features
        range_x = np.where(self.is_constant_, 1.0, self.x_max_ - self.x_min_)

        # Standard linear mapping:
        # angle = angle_min + (x - x_min) * (angle_max - angle_min) / (x_max - x_min)
        angles = angle_min + (X_arr - self.x_min_) * (angle_delta / range_x)

        # Apply constant feature policy
        if np.any(self.is_constant_):
            if self.config.constant_feature_policy == "midpoint":
                midpoint = (angle_min + angle_max) / 2.0
                angles[:, self.is_constant_] = midpoint
            elif self.config.constant_feature_policy == "zero":
                angles[:, self.is_constant_] = 0.0

        # Apply clipping to angle range if configured
        if self.config.clipping:
            angles = np.clip(angles, angle_min, angle_max)

        if is_1d:
            return angles[0]
        return angles

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X: Training feature matrix.
            feature_names: Optional feature names.

        Returns:
            np.ndarray: Encoded angles.
        """
        return self.fit(X, feature_names=feature_names).transform(X)

    # -----------------------------------------------------------------------
    # Metadata & Introspection
    # -----------------------------------------------------------------------

    def get_feature_wire_mapping(self) -> List[Dict[str, Any]]:
        """
        Return the structured feature-to-wire mapping metadata.

        Returns:
            List of dicts, each containing: wire, feature_name, feature_index, rotation, etc.
        """
        if not self.is_fitted_:
            raise RuntimeError("AngleEncoder is not fitted yet. Call fit() first.")
        return [dict(mapping) for mapping in self.wire_mapping_]

    def get_feature_to_wire_dict(self) -> Dict[str, int]:
        """Return dictionary mapping feature_name -> wire index."""
        if not self.is_fitted_:
            raise RuntimeError("AngleEncoder is not fitted yet. Call fit() first.")
        return {item["feature_name"]: item["wire"] for item in self.wire_mapping_}

    def get_wire_to_feature_dict(self) -> Dict[int, str]:
        """Return dictionary mapping wire index -> feature_name."""
        if not self.is_fitted_:
            raise RuntimeError("AngleEncoder is not fitted yet. Call fit() first.")
        return {item["wire"]: item["feature_name"] for item in self.wire_mapping_}

    # -----------------------------------------------------------------------
    # Internal Validation Helper
    # -----------------------------------------------------------------------

    def _validate_and_extract_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
        is_fit: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Validate input types, finite numerical values, and extract feature names."""
        extracted_names: List[str] = []

        if isinstance(X, pd.DataFrame):
            df_cols = [str(col) for col in X.columns]
            if is_fit and feature_names is not None:
                passed_names = list(feature_names)
                if passed_names != df_cols:
                    raise ValueError(
                        f"feature_names passed ({passed_names}) does not match "
                        f"DataFrame column names ({df_cols})."
                    )
            extracted_names = df_cols

            # Verify DataFrame columns are numeric
            try:
                X_arr = X.to_numpy(dtype=np.float64)
            except (ValueError, TypeError) as err:
                raise TypeError(f"DataFrame must contain purely numerical features: {err}") from err

        elif isinstance(X, np.ndarray):
            if X.dtype.kind not in ("f", "i", "u"):
                raise TypeError(f"NumPy array must be of numeric type, got dtype '{X.dtype}'.")
            X_arr = X.astype(np.float64, copy=False)

            if is_fit:
                if feature_names is not None:
                    if len(feature_names) != X_arr.shape[1]:
                        raise ValueError(
                            f"Length of feature_names ({len(feature_names)}) does not match "
                            f"array column count ({X_arr.shape[1]})."
                        )
                    extracted_names = [str(n) for n in feature_names]
                else:
                    extracted_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        else:
            raise TypeError(
                f"Expected numpy.ndarray or pandas.DataFrame, got {type(X).__name__}."
            )

        # Check array dimensions
        if X_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array of shape (n_samples, n_features), got shape {X_arr.shape}."
            )

        if X_arr.shape[0] == 0:
            raise ValueError("Input dataset contains 0 samples. Cannot process empty array.")

        if X_arr.shape[1] == 0:
            raise ValueError("Input dataset contains 0 features. Cannot process empty feature set.")

        # Reject NaN and infinite values
        if np.isnan(X_arr).any():
            raise ValueError("Input data contains NaN values. Angle encoding requires complete numerical data.")

        if np.isinf(X_arr).any():
            raise ValueError("Input data contains infinite values. Angle encoding requires finite numerical data.")

        # If transforming a DataFrame, verify columns match fitted names
        if not is_fit and isinstance(X, pd.DataFrame) and self.is_fitted_:
            if extracted_names != self.feature_names_:
                raise ValueError(
                    f"DataFrame columns {extracted_names} do not match fitted feature names {self.feature_names_}."
                )

        return X_arr, extracted_names


# ---------------------------------------------------------------------------
# 3. Circuit Application & Integration Helpers
# ---------------------------------------------------------------------------

def apply_angle_encoding(
    angles: Union[np.ndarray, Sequence[float]],
    wires: Optional[Sequence[int]] = None,
    rotation: str = DEFAULT_ROTATION,
) -> None:
    """
    Apply single-qubit rotation gates to wires within a PennyLane quantum circuit.

    This function is intended to be called inside a PennyLane QNode function.

    Args:
        angles: 1D array or sequence of numerical rotation angles (one per wire).
        wires: Target wire indices. If None, uses sequential wires 0, 1, ..., len(angles)-1.
        rotation: Rotation gate to apply ('RX', 'RY', or 'RZ'). Default 'RY'.

    Raises:
        ValueError: If rotation gate is unsupported or angles/wires lengths mismatch.
    """
    norm_rot = rotation.strip().upper()
    if norm_rot not in VALID_ROTATIONS:
        raise ValueError(
            f"Unsupported rotation gate '{rotation}'. Supported: {sorted(VALID_ROTATIONS)}."
        )

    n_angles = len(angles)
    if wires is None:
        target_wires = list(range(n_angles))
    else:
        target_wires = list(wires)

    if len(target_wires) != n_angles:
        raise ValueError(
            f"Dimension mismatch: Received {n_angles} angles but {len(target_wires)} wires."
        )

    for angle, wire in zip(angles, target_wires):
        if norm_rot == "RY":
            qml.RY(angle, wires=wire)
        elif norm_rot == "RX":
            qml.RX(angle, wires=wire)
        elif norm_rot == "RZ":
            qml.RZ(angle, wires=wire)


def build_angle_encoding_circuit(
    device: Any,
    n_qubits: int,
    rotation: str = DEFAULT_ROTATION,
    wires: Optional[Sequence[int]] = None,
) -> qml.QNode:
    """
    Construct a PennyLane QNode that encodes angles and returns Pauli-Z expectation values.

    For RY encoding on state |0>:
      <Z_i> = cos(angle_i)

    Args:
        device: PennyLane quantum device (e.g. from create_quantum_device).
        n_qubits: Number of qubits in circuit.
        rotation: Rotation gate ('RX', 'RY', or 'RZ').
        wires: Wire indices. If None, wires 0..n_qubits-1.

    Returns:
        Callable QNode accepting an array of angles of length n_qubits.
    """
    norm_rot = rotation.strip().upper()
    if norm_rot not in VALID_ROTATIONS:
        raise ValueError(f"Unsupported rotation '{rotation}'. Supported: {sorted(VALID_ROTATIONS)}.")

    target_wires = list(wires) if wires is not None else list(range(n_qubits))
    if len(target_wires) != n_qubits:
        raise ValueError(f"Mismatch: n_qubits={n_qubits} but wires has length {len(target_wires)}.")

    @qml.qnode(device)
    def encoding_qnode(angles: Union[np.ndarray, Sequence[float]]) -> Union[float, List[float]]:
        apply_angle_encoding(angles, wires=target_wires, rotation=norm_rot)
        if len(target_wires) == 1:
            return qml.expval(qml.PauliZ(target_wires[0]))
        return [qml.expval(qml.PauliZ(w)) for w in target_wires]

    return encoding_qnode


def build_rz_phase_diagnostic_circuit(
    device: Any,
    wire: int = 0,
    observable: str = "X",
) -> qml.QNode:
    """
    Construct a diagnostic QNode to verify RZ phase observability:
    |0> -> H -> RZ(theta) -> Measure <observable>.

    Mathematical analytical values:
      - observable="X": <X> = cos(theta)
      - observable="Y": <Y> = -sin(theta)
      - observable="Z": <Z> = 0.0

    Args:
        device: PennyLane quantum device.
        wire: Target qubit wire.
        observable: Observable to measure ('X', 'Y', or 'Z').

    Returns:
        Callable QNode accepting a float angle theta.
    """
    norm_obs = observable.strip().upper()
    if norm_obs not in ("X", "Y", "Z"):
        raise ValueError(f"Unsupported observable '{observable}'. Supported: ('X', 'Y', 'Z').")

    @qml.qnode(device)
    def rz_phase_qnode(theta: Union[float, np.float64]) -> float:
        qml.Hadamard(wires=wire)
        qml.RZ(theta, wires=wire)
        if norm_obs == "X":
            return qml.expval(qml.PauliX(wire))
        elif norm_obs == "Y":
            return qml.expval(qml.PauliY(wire))
        else:
            return qml.expval(qml.PauliZ(wire))

    return rz_phase_qnode


def build_rz_noncommuting_diagnostic_circuit(
    device: Any,
    wire: int = 0,
    post_rotation: str = "RY",
    post_angle: float = math.pi / 2,
) -> qml.QNode:
    """
    Construct a diagnostic QNode testing RZ in superposition followed by a noncommuting gate:
    |0> -> H -> RZ(theta) -> post_rotation(post_angle) -> Measure <Z>.

    For post_rotation="RY" and post_angle=pi/2:
      <Z> = -cos(theta)
    For post_rotation="RY" and post_angle=-pi/2:
      <Z> = cos(theta)

    Args:
        device: PennyLane quantum device.
        wire: Target qubit wire.
        post_rotation: Gate name ('RY' or 'RX').
        post_angle: Rotation angle in radians.

    Returns:
        Callable QNode accepting a float angle theta.
    """
    norm_rot = post_rotation.strip().upper()
    if norm_rot not in ("RY", "RX"):
        raise ValueError(f"Unsupported post_rotation '{post_rotation}'. Supported: ('RY', 'RX').")

    @qml.qnode(device)
    def rz_noncommuting_qnode(theta: Union[float, np.float64]) -> float:
        qml.Hadamard(wires=wire)
        qml.RZ(theta, wires=wire)
        if norm_rot == "RY":
            qml.RY(post_angle, wires=wire)
        elif norm_rot == "RX":
            qml.RX(post_angle, wires=wire)
        return qml.expval(qml.PauliZ(wire))

    return rz_noncommuting_qnode

