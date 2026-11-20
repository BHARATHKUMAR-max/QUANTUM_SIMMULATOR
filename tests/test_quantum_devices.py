"""
tests/test_quantum_devices.py
=============================
Unit tests for quantum simulator device creation, configuration, and bounds.

Verifies:
- PennyLane import and version compatibility.
- Creation of 'default.qubit' and 'lightning.qubit'.
- Configurable wire counts within the strictly bounded range [2, 10].
- Invalid wire counts (< 2, > 10, non-integer, bool) raise ValueError/TypeError.
- Unsupported device names raise ValueError.
- DeviceConfig dataclass validation and serialization.
- get_available_devices reports correct environment capabilities (CPU vs GPU).
"""

import pytest
import pennylane as qml

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS
from backend.quantum.device import (
    DEFAULT_DEVICE,
    SUPPORTED_DEVICES,
    DeviceConfig,
    create_quantum_device,
    get_available_devices,
    get_device_capabilities,
)


# ---------------------------------------------------------------------------
# Test: PennyLane Import & Version
# ---------------------------------------------------------------------------


def test_pennylane_import_and_version():
    """Verify PennyLane is imported and version is 0.45.x."""
    assert hasattr(qml, "__version__")
    version_str = qml.__version__
    assert version_str.startswith("0.45"), f"Expected PennyLane 0.45.x, got {version_str}"


def test_available_devices_status():
    """Verify default.qubit and lightning.qubit are available on CPU."""
    dev_status = get_available_devices()
    assert dev_status["default.qubit"] is True
    assert dev_status["lightning.qubit"] is True
    # lightning.gpu is expected False on standard CPU installation
    assert "lightning.gpu" in dev_status


# ---------------------------------------------------------------------------
# Test: Device Creation & Defaults
# ---------------------------------------------------------------------------


def test_default_device_creation():
    """Verify create_quantum_device defaults to default.qubit with 2 wires."""
    dev = create_quantum_device()
    assert dev.name.startswith("default.qubit")
    assert len(dev.wires) == 2


@pytest.mark.parametrize("dev_name", ["default.qubit", "lightning.qubit"])
def test_create_supported_devices(dev_name):
    """Verify each supported device instantiates cleanly."""
    dev = create_quantum_device(device_name=dev_name, wires=3)
    assert dev_name in dev.name
    assert len(dev.wires) == 3


@pytest.mark.parametrize("wires", [2, 3, 4, 6, 8, 10])
def test_valid_wire_counts_within_range(wires):
    """Verify wire counts from 2 to 10 are accepted."""
    dev = create_quantum_device(wires=wires)
    assert len(dev.wires) == wires


# ---------------------------------------------------------------------------
# Test: Invalid Configurations & Boundary Enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_wires", [0, 1, -1, 11, 12, 100])
def test_out_of_bound_wire_counts_rejected(invalid_wires):
    """Verify wire counts outside [2, 10] raise ValueError."""
    with pytest.raises(ValueError, match="outside the supported range"):
        create_quantum_device(wires=invalid_wires)


@pytest.mark.parametrize("invalid_type_wires", [True, False, 2.5, "4", None])
def test_non_integer_wire_types_rejected(invalid_type_wires):
    """Verify non-integer or boolean wire arguments raise TypeError."""
    if invalid_type_wires is None:
        # None defaults to 2 wires in factory, so test via DeviceConfig directly
        with pytest.raises((TypeError, ValueError)):
            DeviceConfig(wires=None).validate()
    else:
        with pytest.raises((TypeError, ValueError)):
            create_quantum_device(wires=invalid_type_wires)


@pytest.mark.parametrize("invalid_name", ["lightning.gpu", "qiskit.aer", "statevector", "unknown_device"])
def test_unsupported_device_name_rejected(invalid_name):
    """Verify unsupported simulator device names raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported quantum device"):
        create_quantum_device(device_name=invalid_name)


# ---------------------------------------------------------------------------
# Test: DeviceConfig Dataclass & Serialization
# ---------------------------------------------------------------------------


def test_device_config_roundtrip():
    """Verify DeviceConfig serializes to dict and reconstructs faithfully."""
    cfg = DeviceConfig(device_name="lightning.qubit", wires=4, shots=500, seed=123)
    d = cfg.to_dict()
    assert d["device_name"] == "lightning.qubit"
    assert d["wires"] == 4
    assert d["shots"] == 500
    assert d["seed"] == 123

    cfg_reconstructed = DeviceConfig.from_dict(d)
    assert cfg_reconstructed == cfg


def test_device_config_invalid_shots():
    """Verify invalid shots arguments raise TypeError or ValueError."""
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        DeviceConfig(shots=-50)

    with pytest.raises(TypeError, match="shots must be an integer"):
        DeviceConfig(shots="1000")


# ---------------------------------------------------------------------------
# Test: Device Capabilities Metadata
# ---------------------------------------------------------------------------


def test_get_device_capabilities():
    """Verify device capabilities reflect CPU execution without GPU claims."""
    caps_def = get_device_capabilities("default.qubit")
    assert caps_def["execution_hardware"] == "CPU"
    assert caps_def["backend"] == "pure_python"
    assert caps_def["max_supported_qubits"] == MAX_QUBITS

    caps_light = get_device_capabilities("lightning.qubit")
    assert caps_light["execution_hardware"] == "CPU"
    assert caps_light["backend"] == "c++_multithreaded"

    with pytest.raises(ValueError, match="Unknown quantum device"):
        get_device_capabilities("invalid_device")
