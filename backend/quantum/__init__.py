"""
backend/quantum/__init__.py
===========================
Quantum simulation, encoding, and circuit foundation package for QML Benchmark Studio.

Provides verified quantum simulator devices (default.qubit, lightning.qubit),
quantum feature encoders (AngleEncoder), circuit factories, analytical expectation
value validators, and autodiff gradient utilities.

Public API:
    - Device Management:
        create_quantum_device, DeviceConfig, get_available_devices,
        get_device_capabilities, SUPPORTED_DEVICES, DEFAULT_DEVICE
    - Quantum Feature Encoding:
        AngleEncoder, AngleEncodingConfig, apply_angle_encoding,
        build_angle_encoding_circuit, DEFAULT_ROTATION,
        DEFAULT_ANGLE_RANGE, VALID_ROTATIONS
    - Circuits & Verification:
        build_ry_rotation_circuit, build_single_qubit_rotation_circuit,
        build_two_qubit_entanglement_circuit, build_bell_state_circuit,
        build_parameterized_circuit, compute_circuit_gradient,
        finite_difference_gradient, verify_gradient_correctness,
        verify_circuit_determinism
"""

from backend.quantum.circuits import (
    build_bell_state_circuit,
    build_parameterized_circuit,
    build_ry_rotation_circuit,
    build_single_qubit_rotation_circuit,
    build_two_qubit_entanglement_circuit,
    compute_circuit_gradient,
    finite_difference_gradient,
    verify_circuit_determinism,
    verify_gradient_correctness,
)
from backend.quantum.device import (
    DEFAULT_DEVICE,
    SUPPORTED_DEVICES,
    DeviceConfig,
    create_quantum_device,
    get_available_devices,
    get_device_capabilities,
)
from backend.quantum.encoding import (
    DEFAULT_ANGLE_RANGE,
    DEFAULT_ROTATION,
    VALID_ROTATIONS,
    AngleEncoder,
    AngleEncodingConfig,
    apply_angle_encoding,
    build_angle_encoding_circuit,
    build_rz_noncommuting_diagnostic_circuit,
    build_rz_phase_diagnostic_circuit,
)
from backend.quantum.vqc import (
    DEFAULT_ENTANGLEMENT,
    DEFAULT_ROTATION_GATES,
    MAX_DEPTH,
    MIN_DEPTH,
    SUPPORTED_ROTATION_SEQUENCES,
    VALID_ENTANGLEMENTS,
    ParameterizedVQC,
    VQCConfig,
    build_vqc_circuit,
    draw_vqc_circuit,
    get_vqc_expectations,
    save_circuit_visualization,
)

__all__ = [
    # Devices
    "DeviceConfig",
    "create_quantum_device",
    "get_available_devices",
    "get_device_capabilities",
    "SUPPORTED_DEVICES",
    "DEFAULT_DEVICE",
    # Quantum Feature Encoding
    "AngleEncoder",
    "AngleEncodingConfig",
    "apply_angle_encoding",
    "build_angle_encoding_circuit",
    "build_rz_phase_diagnostic_circuit",
    "build_rz_noncommuting_diagnostic_circuit",
    "DEFAULT_ROTATION",
    "DEFAULT_ANGLE_RANGE",
    "VALID_ROTATIONS",
    # Parameterized VQC
    "VQCConfig",
    "ParameterizedVQC",
    "build_vqc_circuit",
    "get_vqc_expectations",
    "draw_vqc_circuit",
    "save_circuit_visualization",
    "MIN_DEPTH",
    "MAX_DEPTH",
    "VALID_ENTANGLEMENTS",
    "DEFAULT_ENTANGLEMENT",
    "SUPPORTED_ROTATION_SEQUENCES",
    "DEFAULT_ROTATION_GATES",
    "DEFAULT_ROTATION",
    "DEFAULT_ANGLE_RANGE",
    "VALID_ROTATIONS",
    # Circuits & Verification
    "build_ry_rotation_circuit",
    "build_single_qubit_rotation_circuit",
    "build_two_qubit_entanglement_circuit",
    "build_bell_state_circuit",
    "build_parameterized_circuit",
    "compute_circuit_gradient",
    "finite_difference_gradient",
    "verify_gradient_correctness",
    "verify_circuit_determinism",
]
