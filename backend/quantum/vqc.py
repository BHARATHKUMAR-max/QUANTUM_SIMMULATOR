"""
backend/quantum/vqc.py
======================
Parameterized Variational Quantum Circuit (VQC) module for QML Benchmark Studio.

Implements the quantum circuit layer for the Variational Quantum Classifier (VQC):
  1. Step A — Data Encoding: Angle encoding via AngleEncoder from Day 22.
  2. Step B — Trainable Variational Layers: Configurable rotation sequences (RX, RY, RZ).
  3. Step C — Entanglement: Linear and Circular CNOT entangling gates.
  4. Step D — Measurement: Multi-qubit Pauli-Z expectation values.
  5. PennyLane autodiff differentiability and circuit visualization.

Architectural Invariants & Constraints:
---------------------------------------
- Supported Qubits: 2 to 10 (MIN_QUBITS=2, MAX_QUBITS=10).
- Supported Depth: 1 to 4 (MIN_DEPTH=1, MAX_DEPTH=4).
- Entanglement: 'linear' (0->1, 1->2, ...) or 'circular' (linear + (n-1)->0 for n >= 3).
  For n=2 qubits, circular entanglement does NOT duplicate the single CNOT connection.
- Parameter Tensor Shape: (depth, n_qubits, n_rotation_gates).
- Outputs: Exactly n_qubits Pauli-Z expectation values, strictly in [-1.0, 1.0].
"""

from dataclasses import asdict, dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pennylane as qml

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS
from backend.quantum.device import DEFAULT_DEVICE, DeviceConfig, create_quantum_device
from backend.quantum.encoding import (
    DEFAULT_ROTATION,
    VALID_ROTATIONS,
    AngleEncoder,
    AngleEncodingConfig,
    apply_angle_encoding,
)

MIN_DEPTH: int = 1
MAX_DEPTH: int = 4

VALID_ENTANGLEMENTS = ("linear", "circular")
DEFAULT_ENTANGLEMENT = "linear"

SUPPORTED_ROTATION_SEQUENCES = ("RX", "RY", "RZ", "RX+RY", "RX+RY+RZ")
DEFAULT_ROTATION_GATES = "RX+RY+RZ"


# ---------------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------------

def parse_rotation_gates(rotation_gates: Union[str, Sequence[str]]) -> List[str]:
    """
    Parse and validate a rotation gate sequence specification.

    Args:
        rotation_gates: String (e.g. "RX+RY+RZ", "RY", "RX+RY") or list of gate names.

    Returns:
        List of normalized uppercase rotation gate strings (e.g. ["RX", "RY", "RZ"]).

    Raises:
        TypeError: If input is not a string or sequence of strings.
        ValueError: If any specified gate is unsupported.
    """
    if isinstance(rotation_gates, str):
        cleaned = rotation_gates.strip().upper()
        if "+" in cleaned:
            gates = [g.strip() for g in cleaned.split("+") if g.strip()]
        else:
            gates = [cleaned]
    elif isinstance(rotation_gates, (list, tuple)):
        gates = [str(g).strip().upper() for g in rotation_gates]
    else:
        raise TypeError(
            f"rotation_gates must be a string or sequence of strings, got {type(rotation_gates).__name__}."
        )

    if not gates:
        raise ValueError("rotation_gates sequence cannot be empty.")

    for gate in gates:
        if gate not in VALID_ROTATIONS:
            raise ValueError(
                f"Unsupported rotation gate '{gate}'. Supported gates are: {sorted(VALID_ROTATIONS)}."
            )

    return gates


def apply_vqc_entanglement(wires: Sequence[int], entanglement: str = DEFAULT_ENTANGLEMENT) -> None:
    """
    Apply CNOT entangling gates according to configured pattern.

    Patterns:
      - 'linear': CNOT(q_i, q_{i+1}) for i in 0..n-2
      - 'circular': linear connections, plus CNOT(q_{n-1}, q_0) for n >= 3.
        (For n=2 qubits, no duplicate CNOT is applied).

    Args:
        wires: Sequence of qubit wire indices.
        entanglement: Pattern ('linear' or 'circular').

    Raises:
        ValueError: If entanglement pattern is unsupported.
    """
    n = len(wires)
    norm_ent = entanglement.strip().lower()
    if norm_ent not in VALID_ENTANGLEMENTS:
        raise ValueError(
            f"Unsupported entanglement '{entanglement}'. Supported: {sorted(VALID_ENTANGLEMENTS)}."
        )

    # Linear CNOT chain: 0 -> 1, 1 -> 2, ..., n-2 -> n-1
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Circular connection: (n-1) -> 0 only when n >= 3 to avoid duplicate gates for n=2
    if norm_ent == "circular" and n >= 3:
        qml.CNOT(wires=[wires[n - 1], wires[0]])


# ---------------------------------------------------------------------------
# 2. VQC Configuration
# ---------------------------------------------------------------------------

@dataclass
class VQCConfig:
    """
    Configuration specification for a Parameterized Variational Quantum Circuit.

    Attributes:
        n_qubits: Number of simulated qubits (wires), must be in [2, 10].
        depth: Number of variational layers (depth), must be in [1, 4].
        entanglement: CNOT entanglement pattern ('linear' or 'circular').
        rotation_gates: Trainable single-qubit rotation sequence (default 'RX+RY+RZ').
        device_name: Name of simulated quantum backend ('default.qubit' or 'lightning.qubit').
        encoding_rotation: Rotation gate used for input feature encoding (default 'RY').
        shots: Measurement shot count (None for exact analytic statevector simulation).
        seed: Random seed for parameter initialization or stochastic devices.
    """
    n_qubits: int = 4
    depth: int = 2
    entanglement: str = DEFAULT_ENTANGLEMENT
    rotation_gates: str = DEFAULT_ROTATION_GATES
    device_name: str = DEFAULT_DEVICE
    encoding_rotation: str = DEFAULT_ROTATION
    shots: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters upon initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all VQC configuration parameters strictly.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If parameter values violate architecture boundaries.
        """
        # Validate n_qubits
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise TypeError(f"n_qubits must be an integer, got {type(self.n_qubits).__name__}.")
        if not (MIN_QUBITS <= self.n_qubits <= MAX_QUBITS):
            raise ValueError(
                f"n_qubits={self.n_qubits} is outside supported range [{MIN_QUBITS}, {MAX_QUBITS}]. "
                "PennyLane statevector simulations in this project are strictly constrained to 2–10 qubits."
            )

        # Validate depth
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError(f"depth must be an integer, got {type(self.depth).__name__}.")
        if not (MIN_DEPTH <= self.depth <= MAX_DEPTH):
            raise ValueError(
                f"depth={self.depth} is outside supported range [{MIN_DEPTH}, {MAX_DEPTH}]. "
                "VQC circuit depth in this project is constrained to 1–4 layers."
            )

        # Validate entanglement
        if not isinstance(self.entanglement, str):
            raise TypeError(f"entanglement must be a string, got {type(self.entanglement).__name__}.")
        norm_ent = self.entanglement.strip().lower()
        if norm_ent not in VALID_ENTANGLEMENTS:
            raise ValueError(
                f"Unsupported entanglement '{self.entanglement}'. "
                f"Supported patterns are: {sorted(VALID_ENTANGLEMENTS)}."
            )
        self.entanglement = norm_ent

        # Validate rotation_gates
        parsed = parse_rotation_gates(self.rotation_gates)
        self.rotation_gates = "+".join(parsed)

        # Validate device_name
        if not isinstance(self.device_name, str):
            raise TypeError(f"device_name must be a string, got {type(self.device_name).__name__}.")

        # Validate encoding_rotation
        if not isinstance(self.encoding_rotation, str):
            raise TypeError(f"encoding_rotation must be a string, got {type(self.encoding_rotation).__name__}.")
        norm_enc = self.encoding_rotation.strip().upper()
        if norm_enc not in VALID_ROTATIONS:
            raise ValueError(
                f"Unsupported encoding_rotation '{self.encoding_rotation}'. "
                f"Supported gates are: {sorted(VALID_ROTATIONS)}."
            )
        self.encoding_rotation = norm_enc

        # Validate shots
        if self.shots is not None:
            if isinstance(self.shots, bool) or not isinstance(self.shots, int):
                raise TypeError(f"shots must be an integer or None, got {type(self.shots).__name__}.")
            if self.shots <= 0:
                raise ValueError(f"shots must be a positive integer, got {self.shots}.")

        # Validate seed
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError(f"seed must be an integer or None, got {type(self.seed).__name__}.")

    def get_parsed_rotation_gates(self) -> List[str]:
        """Return list of individual rotation gate strings."""
        return parse_rotation_gates(self.rotation_gates)

    def get_weight_shape(self) -> Tuple[int, int, int]:
        """
        Return the expected 3D parameter tensor shape.

        Shape: (depth, n_qubits, n_rotation_gates)
        """
        n_gates = len(self.get_parsed_rotation_gates())
        return (self.depth, self.n_qubits, n_gates)

    def get_num_parameters(self) -> int:
        """Return total number of trainable scalar parameters in circuit."""
        d, q, g = self.get_weight_shape()
        return d * q * g

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "VQCConfig"]]) -> "VQCConfig":
        """Instantiate validated VQCConfig from dictionary or instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            data.validate()
            return data
        return cls(**data)


# ---------------------------------------------------------------------------
# 3. Circuit Builder
# ---------------------------------------------------------------------------

def build_vqc_circuit(
    device: Any,
    config: Optional[VQCConfig] = None,
    **kwargs: Any,
) -> qml.QNode:
    """
    Construct a differentiable PennyLane QNode implementing the VQC circuit.

    Logical Architecture:
      1. Step A: Input Feature Encoding (apply_angle_encoding).
      2. Step B: Variational Layers (trainable single-qubit rotations).
      3. Step C: Entanglement (linear or circular CNOT gates).
      4. Step D: Output Measurement (Pauli-Z expectation on all qubits).

    Args:
        device: PennyLane quantum device (e.g. from create_quantum_device).
        config: Optional VQCConfig instance. If None, constructed from kwargs.
        **kwargs: Configuration keyword arguments.

    Returns:
        Callable QNode accepting:
          - angles (1D array of length n_qubits)
          - weights (3D array of shape (depth, n_qubits, n_gates))
        Returning list of n_qubits Pauli-Z expectation values.
    """
    if config is None:
        cfg = VQCConfig(**kwargs)
    else:
        cfg = VQCConfig.from_dict(config)

    gate_list = cfg.get_parsed_rotation_gates()
    n_gates = len(gate_list)
    wires = list(range(cfg.n_qubits))
    expected_weight_shape = cfg.get_weight_shape()

    @qml.qnode(device)
    def vqc_qnode(angles: Any, weights: Any) -> List[Any]:
        # Validate angles shape
        if len(angles) != cfg.n_qubits:
            raise ValueError(
                f"Input angles dimension mismatch: expected {cfg.n_qubits} angles, got {len(angles)}."
            )

        # Validate weights shape
        w_arr = np.asarray(weights)
        if w_arr.shape != expected_weight_shape:
            # Allow 2D shape (depth, n_qubits) when n_gates == 1
            if n_gates == 1 and w_arr.shape == (cfg.depth, cfg.n_qubits):
                w_eval = weights.reshape(cfg.depth, cfg.n_qubits, 1)
            else:
                raise ValueError(
                    f"Parameter tensor shape mismatch: expected {expected_weight_shape}, got {w_arr.shape}."
                )
        else:
            w_eval = weights

        # Check for NaN / Inf in inputs (unwrapping Autograd Box if being differentiated)
        raw_w = getattr(weights, "_value", weights)
        try:
            raw_w_arr = np.asarray(raw_w, dtype=np.float64)
            if np.isnan(raw_w_arr).any() or np.isinf(raw_w_arr).any():
                raise ValueError("Parameter tensor contains NaN or Inf values.")
        except (TypeError, ValueError) as err:
            if "NaN or Inf" in str(err):
                raise

        # Step A — Input feature encoding
        apply_angle_encoding(angles, wires=wires, rotation=cfg.encoding_rotation)

        # Step B & C — Trainable variational layers + Entanglement
        for layer in range(cfg.depth):
            # Single-qubit trainable rotations
            for q in range(cfg.n_qubits):
                for g_idx, gate in enumerate(gate_list):
                    param = w_eval[layer, q, g_idx]
                    if gate == "RX":
                        qml.RX(param, wires=wires[q])
                    elif gate == "RY":
                        qml.RY(param, wires=wires[q])
                    elif gate == "RZ":
                        qml.RZ(param, wires=wires[q])

            # Entanglement layer
            apply_vqc_entanglement(wires, entanglement=cfg.entanglement)

        # Step D — Pauli-Z expectation values on all qubits
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return vqc_qnode


def get_vqc_expectations(
    qnode: qml.QNode,
    angles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Evaluate QNode on an angle vector and weights, returning a NumPy float array.

    Args:
        qnode: PennyLane QNode constructed by build_vqc_circuit.
        angles: 1D array of angles of length n_qubits.
        weights: 3D parameter tensor.

    Returns:
        np.ndarray: Array of expectation values in [-1.0, 1.0].
    """
    raw = qnode(angles, weights)
    arr = np.asarray(raw, dtype=np.float64)
    return arr


# ---------------------------------------------------------------------------
# 4. ParameterizedVQC Wrapper Class
# ---------------------------------------------------------------------------

class ParameterizedVQC:
    """
    Reusable high-level manager for a Parameterized Variational Quantum Circuit.

    Combines:
      - Device management (DeviceConfig / create_quantum_device).
      - Input feature scaling (AngleEncoder).
      - Circuit execution (PennyLane QNode).
      - Weight initialization and introspection.
      - Circuit drawing and visualization.
    """

    def __init__(self, config: Optional[VQCConfig] = None, **kwargs: Any) -> None:
        """
        Initialize the Parameterized VQC.

        Args:
            config: Optional VQCConfig instance.
            **kwargs: Configuration parameters if config is None.
        """
        if config is None:
            self.config = VQCConfig(**kwargs)
        else:
            self.config = VQCConfig.from_dict(config)

        # Initialize quantum simulator device
        self.device = create_quantum_device(
            device_name=self.config.device_name,
            wires=self.config.n_qubits,
            shots=self.config.shots,
            seed=self.config.seed,
        )

        # Build QNode circuit
        self.qnode = build_vqc_circuit(self.device, self.config)

        # Initialize AngleEncoder
        self.encoder = AngleEncoder(
            rotation=self.config.encoding_rotation,
            n_qubits=self.config.n_qubits,
            clipping=True,
        )

    def init_weights(
        self,
        seed: Optional[int] = None,
        low: float = 0.0,
        high: float = 2 * math.pi,
    ) -> np.ndarray:
        """
        Initialize trainable weights with uniform random values in [low, high].

        Args:
            seed: Optional random seed for deterministic initialization.
            low: Lower bound of uniform initialization (default 0.0).
            high: Upper bound of uniform initialization (default 2*pi).

        Returns:
            np.ndarray: Initial weight tensor of shape (depth, n_qubits, n_gates).
        """
        rng_seed = seed if seed is not None else self.config.seed
        rng = np.random.default_rng(rng_seed)
        shape = self.config.get_weight_shape()
        weights = rng.uniform(low=low, high=high, size=shape).astype(np.float64)
        return weights

    def evaluate(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        is_encoded_angles: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the VQC circuit on input samples.

        Supports single-sample (1D: shape (n_qubits,)) and batch inputs (2D: shape (n_samples, n_qubits)).

        Args:
            X: Input features or encoded angles.
            weights: Trainable weight tensor.
            is_encoded_angles: If True, X is assumed already scaled to angles in radians.
                               If False, X is scaled via self.encoder (which must be fitted).

        Returns:
            np.ndarray: Expectation values array with shape (n_qubits,) for 1D input
                        or (n_samples, n_qubits) for 2D batch input.
        """
        is_1d = (X.ndim == 1)
        if is_1d:
            X_batch = X.reshape(1, -1)
        else:
            X_batch = X

        if X_batch.shape[1] != self.config.n_qubits:
            raise ValueError(
                f"Feature dimension mismatch: VQC is configured for {self.config.n_qubits} qubits, "
                f"but received input with {X_batch.shape[1]} features."
            )

        # Encode features if not already angles
        if not is_encoded_angles:
            if not self.encoder.is_fitted_:
                # Fit encoder on provided batch if not previously fitted
                self.encoder.fit(X_batch)
            angles_batch = self.encoder.transform(X_batch)
        else:
            angles_batch = X_batch

        # Evaluate circuit across samples
        n_samples = angles_batch.shape[0]
        results = np.zeros((n_samples, self.config.n_qubits), dtype=np.float64)

        for i in range(n_samples):
            results[i] = get_vqc_expectations(self.qnode, angles_batch[i], weights)

        if is_1d:
            return results[0]
        return results

    def __call__(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Convenience callable alias for evaluate()."""
        return self.evaluate(x, weights)

    def draw(
        self,
        format: str = "text",
        x: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> str:
        """
        Generate text-based circuit visualization.

        Args:
            format: Visualization format ('text').
            x: Sample input angles (if None, zeros are used).
            weights: Sample weight tensor (if None, zeros are used).

        Returns:
            str: Formatted ASCII/Unicode circuit drawing.
        """
        if x is None:
            angles = np.zeros(self.config.n_qubits, dtype=np.float64)
        else:
            angles = np.asarray(x, dtype=np.float64)

        if weights is None:
            w = np.zeros(self.config.get_weight_shape(), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)

        drawer = qml.draw(self.qnode)
        return drawer(angles, w)

    def draw_mpl(
        self,
        x: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """
        Generate Matplotlib figure visualization of the circuit.

        Args:
            x: Sample input angles (if None, zeros are used).
            weights: Sample weight tensor (if None, zeros are used).
            **kwargs: Extra arguments passed to qml.draw_mpl.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        if x is None:
            angles = np.zeros(self.config.n_qubits, dtype=np.float64)
        else:
            angles = np.asarray(x, dtype=np.float64)

        if weights is None:
            w = np.zeros(self.config.get_weight_shape(), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)

        drawer = qml.draw_mpl(self.qnode, **kwargs)
        fig, ax = drawer(angles, w)
        return fig, ax


# ---------------------------------------------------------------------------
# 5. Circuit Visualization Utilities
# ---------------------------------------------------------------------------

def draw_vqc_circuit(
    config: Optional[VQCConfig] = None,
    x: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    format: str = "text",
    **kwargs: Any,
) -> str:
    """
    Construct and draw a VQC circuit.

    Args:
        config: Optional VQCConfig instance.
        x: Input angles (optional).
        weights: Weights tensor (optional).
        format: Format ('text').
        **kwargs: Configuration parameters if config is None.

    Returns:
        str: Circuit diagram string.
    """
    vqc = ParameterizedVQC(config=config, **kwargs)
    return vqc.draw(format=format, x=x, weights=weights)


def save_circuit_visualization(
    config: VQCConfig,
    filepath: str,
    weights: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
) -> str:
    """
    Generate and save circuit visualization to disk (both .png and .txt).

    Args:
        config: VQCConfig instance.
        filepath: Base file path (without extension or with .png/.txt).
        weights: Optional weights tensor.
        x: Optional input angles.

    Returns:
        str: Path of the saved image file.
    """
    import matplotlib.pyplot as plt

    base_path, ext = os.path.splitext(filepath)
    png_path = f"{base_path}.png"
    txt_path = f"{base_path}.txt"

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)

    vqc = ParameterizedVQC(config=config)

    # Save text visualization
    text_diagram = vqc.draw(x=x, weights=weights)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_diagram)

    # Save Matplotlib visualization
    fig, ax = vqc.draw_mpl(x=x, weights=weights)
    ax.set_title(
        f"Parameterized VQC ({config.n_qubits} Qubits, Depth {config.depth}, {config.entanglement.capitalize()} Entanglement)",
        fontsize=11,
        pad=12,
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path
