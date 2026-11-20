"""
tests/test_quantum_circuits.py
==============================
Unit tests for quantum circuit execution, expectation values, autodiff gradients, and determinism.

Verifies:
- Single-qubit rotation circuits (RX, RY, RZ) with analytical Pauli-Z validation.
- Known analytical rotation values: theta=0 -> 1.0, theta=pi/2 -> 0.0, theta=pi -> -1.0.
- Two-qubit entanglement circuits (rotations + CNOT) with output dimension and bound checks.
- Bell state |Phi+> validation (<Z0>=0, <Z1>=0, <Z0*Z1>=1).
- Autodiff gradients using PennyLane autograd.
- Agreement between autodiff gradients and central finite-difference approximations.
- Deterministic circuit execution across multiple consecutive runs.
- Multi-device compatibility (default.qubit and lightning.qubit).
"""

import math
import numpy as np
import pytest
import pennylane as qml
from pennylane import numpy as pnp

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
from backend.quantum.device import create_quantum_device


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["default.qubit", "lightning.qubit"])
def test_device(request):
    """Fixture providing both default.qubit and lightning.qubit simulator devices."""
    return create_quantum_device(device_name=request.param, wires=4)


# ---------------------------------------------------------------------------
# Test: Single-Qubit Rotations & Analytical Expectation Values
# ---------------------------------------------------------------------------


def test_ry_zero_angle_produces_one(test_device):
    """Analytical verification: RY(0)|0> -> <Z> = cos(0) = 1.0."""
    ry_circ = build_ry_rotation_circuit(test_device, wire=0)
    exp_val = ry_circ(0.0)
    assert isinstance(exp_val, (float, np.floating))
    assert math.isclose(float(exp_val), 1.0, abs_tol=1e-7)


def test_ry_pi_over_two_produces_zero(test_device):
    """Analytical verification: RY(pi/2)|0> -> <Z> = cos(pi/2) = 0.0."""
    ry_circ = build_ry_rotation_circuit(test_device, wire=0)
    exp_val = ry_circ(math.pi / 2.0)
    assert math.isclose(float(exp_val), 0.0, abs_tol=1e-7)


def test_ry_pi_produces_minus_one(test_device):
    """Analytical verification: RY(pi)|0> -> <Z> = cos(pi) = -1.0."""
    ry_circ = build_ry_rotation_circuit(test_device, wire=0)
    exp_val = ry_circ(math.pi)
    assert math.isclose(float(exp_val), -1.0, abs_tol=1e-7)


@pytest.mark.parametrize("angle", [0.25, 0.75, 1.25, 2.0, 2.8, -1.5])
def test_ry_analytical_cosine_curve(test_device, angle):
    """Analytical verification: RY(theta)|0> produces cos(theta) for arbitrary angles."""
    ry_circ = build_ry_rotation_circuit(test_device, wire=0)
    exp_val = float(ry_circ(angle))
    expected = math.cos(angle)
    assert math.isclose(exp_val, expected, abs_tol=1e-7)
    assert -1.0 <= exp_val <= 1.0


def test_single_qubit_rotations_bounded_and_finite(test_device):
    """Verify combined RX, RY, RZ rotations output finite scalar bounded in [-1, 1]."""
    circ = build_single_qubit_rotation_circuit(test_device, wire=0)
    for phi, theta, omega in [(0.1, 0.2, 0.3), (-0.5, 1.2, -0.8), (math.pi, math.pi / 4, 0.0)]:
        val = float(circ(phi, theta, omega))
        assert math.isfinite(val)
        assert -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Test: Two-Qubit Entanglement & Bell State
# ---------------------------------------------------------------------------


def test_two_qubit_circuit_dimensions_and_bounds(test_device):
    """Two-qubit circuit must return array of 2 expectation values bounded in [-1, 1]."""
    circ = build_two_qubit_entanglement_circuit(test_device, wires=(0, 1))
    out = circ(0.5, 0.8)
    assert len(out) == 2
    for v in out:
        val = float(v)
        assert math.isfinite(val)
        assert -1.0 <= val <= 1.0


def test_two_qubit_known_states(test_device):
    """Verify known input states in two-qubit rotation and CNOT circuit."""
    circ = build_two_qubit_entanglement_circuit(test_device, wires=(0, 1))

    # Input: RY(0), RY(0) -> |00> -> CNOT -> |00> -> <Z0>=1, <Z1>=1
    out_00 = [float(v) for v in circ(0.0, 0.0)]
    assert math.isclose(out_00[0], 1.0, abs_tol=1e-7)
    assert math.isclose(out_00[1], 1.0, abs_tol=1e-7)

    # Input: RY(pi), RY(0) -> |10> -> CNOT -> |11> -> <Z0>=-1, <Z1>=-1
    out_11 = [float(v) for v in circ(math.pi, 0.0)]
    assert math.isclose(out_11[0], -1.0, abs_tol=1e-7)
    assert math.isclose(out_11[1], -1.0, abs_tol=1e-7)


def test_bell_state_perfect_correlation(test_device):
    """Bell state (|00> + |11>)/sqrt(2) produces <Z0>=0, <Z1>=0, and joint parity <Z0*Z1>=1."""
    bell = build_bell_state_circuit(test_device, wires=(0, 1))
    z0, z1, z0z1 = [float(v) for v in bell()]

    assert math.isclose(z0, 0.0, abs_tol=1e-7)
    assert math.isclose(z1, 0.0, abs_tol=1e-7)
    assert math.isclose(z0z1, 1.0, abs_tol=1e-7)


# ---------------------------------------------------------------------------
# Test: Parameterized Circuit & Autodiff Gradients
# ---------------------------------------------------------------------------


def test_autodiff_gradient_computation(test_device):
    """Verify autodiff computes a finite, non-zero gradient vector."""
    n_wires = 2
    circ = build_parameterized_circuit(test_device, n_wires=n_wires)
    params = [0.4, -0.6, 0.3, 0.8]  # 2 * n_wires = 4 parameters

    grad_ad = compute_circuit_gradient(circ, params)
    assert len(grad_ad) == 4
    assert np.all(np.isfinite(grad_ad))
    assert np.any(np.abs(grad_ad) > 1e-4)


@pytest.mark.parametrize(
    "params",
    [
        [0.5, 0.8, -0.3, 1.2],
        [-0.7, 0.2, 0.9, -0.4],
        [math.pi / 3.0, -math.pi / 4.0, 0.1, 0.5],
    ],
)
def test_autodiff_vs_finite_difference(test_device, params):
    """
    Rigorously compare PennyLane autodiff with central finite-difference gradient.
    Maximum absolute difference must be < 1e-5.
    """
    circ = build_parameterized_circuit(test_device, n_wires=2)
    res = verify_gradient_correctness(circ, params, epsilon=1e-5, atol=1e-4, rtol=1e-4)

    assert res["is_correct"] is True
    assert res["all_finite"] is True
    assert res["is_nonzero"] is True
    assert res["max_abs_diff"] < 1e-5, f"Discrepancy too large: {res['max_abs_diff']}"


# ---------------------------------------------------------------------------
# Test: Deterministic Execution
# ---------------------------------------------------------------------------


def test_deterministic_circuit_execution(test_device):
    """Executing identical statevector circuits repeatedly yields 0.0 variation."""
    circ = build_parameterized_circuit(test_device, n_wires=2)
    params = [0.35, -0.85, 0.65, 0.15]

    det_res = verify_circuit_determinism(circ, params, n_runs=5, atol=1e-12)
    assert det_res["is_deterministic"] is True
    assert det_res["max_variation"] < 1e-12
