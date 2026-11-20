"""
backend/quantum/device.py
=========================
Quantum simulator device configuration and factory for QML Benchmark Studio.

Provides unified creation, configuration, and validation for simulated quantum
devices supported by PennyLane (default.qubit and lightning.qubit).

Hardware & Architectural Invariants:
------------------------------------
1. Supported Devices: 'default.qubit' (Python statevector simulator) and
   'lightning.qubit' (high-performance C++ CPU statevector simulator).
2. Qubit Bounding: Configured wire counts must strictly fall within [2, 10]
   (MIN_QUBITS=2, MAX_QUBITS=10), consistent with docs/MVP_SCOPE.md.
3. CPU-Only Execution: Quantum simulation executes on classical CPU hardware.
   No GPU acceleration (lightning.gpu) or practical quantum speedup is assumed.
4. Determinism: Analytic statevector simulations (shots=None) are naturally
   deterministic. When finite sampling (shots) is used, seeds ensure reproducibility.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union
import pennylane as qml

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS

SUPPORTED_DEVICES = ("default.qubit", "lightning.qubit")
DEFAULT_DEVICE = "default.qubit"


@dataclass
class DeviceConfig:
    """
    Configuration specification for a quantum simulator device.

    Attributes:
        device_name: Name of the PennyLane device ('default.qubit' or 'lightning.qubit').
        wires: Number of simulated qubits / wires (must be between 2 and 10).
        shots: Number of measurement samples (None for exact analytic statevector simulation).
        seed: Deterministic random seed for stochastic sampling (optional).
    """
    device_name: str = DEFAULT_DEVICE
    wires: int = 2
    shots: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate device configuration parameters."""
        self.validate()

    def validate(self) -> None:
        """
        Validate device configuration parameters strictly.

        Raises:
            TypeError: If parameter types are invalid.
            ValueError: If device name is unsupported or wire count is outside [2, 10].
        """
        if not isinstance(self.device_name, str):
            raise TypeError(f"device_name must be a string, got {type(self.device_name).__name__}.")

        norm_name = self.device_name.strip().lower()
        if norm_name not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported quantum device '{self.device_name}'. "
                f"Allowed devices are: {sorted(SUPPORTED_DEVICES)}."
            )
        self.device_name = norm_name

        if isinstance(self.wires, bool) or not isinstance(self.wires, int):
            raise TypeError(f"wires must be an integer, got {type(self.wires).__name__}.")

        if not (MIN_QUBITS <= self.wires <= MAX_QUBITS):
            raise ValueError(
                f"wires={self.wires} is outside the supported range [{MIN_QUBITS}, {MAX_QUBITS}]. "
                "PennyLane statevector simulations in this project are strictly constrained to 2–10 qubits."
            )

        if self.shots is not None:
            if isinstance(self.shots, bool) or not isinstance(self.shots, int):
                raise TypeError(f"shots must be an integer, got {type(self.shots).__name__}.")
            if self.shots <= 0:
                raise ValueError(f"shots must be a positive integer, got {self.shots}.")

        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError(f"seed must be an integer, got {type(self.seed).__name__}.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert device configuration to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Union[Dict[str, Any], "DeviceConfig"]]) -> "DeviceConfig":
        """Construct a validated DeviceConfig from dictionary or instance."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            data.validate()
            return data
        return cls(**dict(data))


def get_available_devices() -> Dict[str, bool]:
    """
    Query which quantum simulator devices can be instantiated in the active environment.

    Returns:
        Dictionary mapping device identifiers to availability booleans.
    """
    devices_status = {
        "default.qubit": False,
        "lightning.qubit": False,
        "lightning.gpu": False,
    }

    # Test default.qubit
    try:
        qml.device("default.qubit", wires=2)
        devices_status["default.qubit"] = True
    except Exception:
        devices_status["default.qubit"] = False

    # Test lightning.qubit
    try:
        qml.device("lightning.qubit", wires=2)
        devices_status["lightning.qubit"] = True
    except Exception:
        devices_status["lightning.qubit"] = False

    # Test lightning.gpu (CUDA check)
    try:
        qml.device("lightning.gpu", wires=2)
        devices_status["lightning.gpu"] = True
    except Exception:
        devices_status["lightning.gpu"] = False

    return devices_status


def get_device_capabilities(device_name: str = DEFAULT_DEVICE) -> Dict[str, Any]:
    """
    Get architectural capability and hardware execution metadata for a simulator device.

    Args:
        device_name: Device identifier ('default.qubit' or 'lightning.qubit').

    Returns:
        Dictionary containing execution device, backend language, and simulation properties.
    """
    norm_name = device_name.strip().lower()
    if norm_name not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unknown quantum device '{device_name}'. Supported: {sorted(SUPPORTED_DEVICES)}."
        )

    if norm_name == "default.qubit":
        return {
            "device_name": "default.qubit",
            "backend": "pure_python",
            "execution_hardware": "CPU",
            "analytic_statevector": True,
            "max_supported_qubits": MAX_QUBITS,
            "min_supported_qubits": MIN_QUBITS,
            "description": "Standard Python statevector simulator with full autodiff integration.",
        }
    elif norm_name == "lightning.qubit":
        return {
            "device_name": "lightning.qubit",
            "backend": "c++_multithreaded",
            "execution_hardware": "CPU",
            "analytic_statevector": True,
            "max_supported_qubits": MAX_QUBITS,
            "min_supported_qubits": MIN_QUBITS,
            "description": "High-performance C++ CPU statevector simulator for accelerated circuit evaluation.",
        }


def create_quantum_device(
    config: Optional[Union[DeviceConfig, Dict[str, Any]]] = None,
    device_name: Optional[str] = None,
    wires: Optional[int] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
) -> Any:
    """
    Instantiate and return a validated PennyLane quantum simulator device.

    Args:
        config: DeviceConfig instance or dictionary of options.
        device_name: Explicit override for device name ('default.qubit', 'lightning.qubit').
        wires: Explicit override for number of simulated wires (2 to 10).
        shots: Explicit override for measurement shots (None for analytic statevector).
        seed: Explicit override for random seed.

    Returns:
        Instantiated PennyLane Device.

    Raises:
        ValueError: If device_name is unsupported or wire count is outside [2, 10].
        TypeError: If parameter types are invalid.
    """
    if config is not None:
        cfg = DeviceConfig.from_dict(config)
    else:
        cfg = DeviceConfig()

    # Apply overrides if provided
    if device_name is not None:
        cfg.device_name = device_name
    if wires is not None:
        cfg.wires = wires
    if shots is not None:
        cfg.shots = shots
    if seed is not None:
        cfg.seed = seed

    # Validate configuration
    cfg.validate()

    # Device creation kwargs
    kwargs: Dict[str, Any] = {"wires": cfg.wires}
    if cfg.shots is not None:
        kwargs["shots"] = cfg.shots
    if cfg.seed is not None:
        kwargs["seed"] = cfg.seed

    try:
        device = qml.device(cfg.device_name, **kwargs)
        return device
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate quantum device '{cfg.device_name}' with {cfg.wires} wires: {exc}"
        ) from exc
