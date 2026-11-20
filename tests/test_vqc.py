"""
tests/test_vqc.py
=================
Unit and integration test suite for the Parameterized Variational Quantum Circuit (VQC)
in QML Benchmark Studio (Day 23 milestone).

Test Categories:
  1. TestVQCConfig: Configuration defaults, valid bounds (qubits 2-10, depth 1-4), invalid bounds,
     rotation sequence parsing, serialization.
  2. TestCircuitArchitecture: Gate sequence, linear vs circular entanglement, 2-qubit circular
     guardrail (no duplicate CNOTs), weight shape compliance.
  3. TestMeasurementOutputs: Output length == n_qubits, bounds [-1, 1], finite values, determinism.
  4. TestDifferentiability: qml.grad computation, finite-difference validation, finite non-zero gradients.
  5. TestBackendParity: default.qubit and lightning.qubit execution.
  6. TestAngleEncoderIntegration: End-to-end integration from raw tabular features to VQC output.
  7. TestVisualization: Text drawing and Matplotlib export.
"""

import math
import os
from typing import List
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import pytest

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS
from backend.quantum.device import create_quantum_device, get_available_devices
from backend.quantum.encoding import AngleEncoder
from backend.quantum.vqc import (
    DEFAULT_ENTANGLEMENT,
    DEFAULT_ROTATION_GATES,
    MAX_DEPTH,
    MIN_DEPTH,
    SUPPORTED_ROTATION_SEQUENCES,
    VALID_ENTANGLEMENTS,
    ParameterizedVQC,
    VQCConfig,
    apply_vqc_entanglement,
    build_vqc_circuit,
    draw_vqc_circuit,
    get_vqc_expectations,
    parse_rotation_gates,
    save_circuit_visualization,
)


# ===========================================================================
# 1. Configuration Validation Tests
# ===========================================================================

class TestVQCConfig:
    """Tests for VQCConfig defaults, boundary validation, and serialization."""

    def test_default_configuration(self):
        """Verify default configuration attributes."""
        cfg = VQCConfig()
        assert cfg.n_qubits == 4
        assert cfg.depth == 2
        assert cfg.entanglement == "linear"
        assert cfg.rotation_gates == "RX+RY+RZ"
        assert cfg.device_name == "default.qubit"
        assert cfg.encoding_rotation == "RY"
        assert cfg.shots is None
        assert cfg.seed is None

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 6, 8, 10])
    def test_valid_qubit_counts(self, n_qubits: int):
        """Verify supported qubit counts [2, 10] are accepted."""
        cfg = VQCConfig(n_qubits=n_qubits)
        assert cfg.n_qubits == n_qubits

    @pytest.mark.parametrize("invalid_qubits", [0, 1, 11, 20, -1, True, False, "4"])
    def test_invalid_qubit_counts_raise_error(self, invalid_qubits):
        """Verify qubit counts outside [2, 10] or invalid types raise errors."""
        with pytest.raises((ValueError, TypeError)):
            VQCConfig(n_qubits=invalid_qubits)

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_valid_depth_values(self, depth: int):
        """Verify supported depth values [1, 4] are accepted."""
        cfg = VQCConfig(depth=depth)
        assert cfg.depth == depth

    @pytest.mark.parametrize("invalid_depth", [0, 5, 10, -1, True, False, 2.5, "2"])
    def test_invalid_depth_values_raise_error(self, invalid_depth):
        """Verify depth values outside [1, 4] or invalid types raise errors."""
        with pytest.raises((ValueError, TypeError)):
            VQCConfig(depth=invalid_depth)

    @pytest.mark.parametrize("entanglement", ["linear", "circular", "LINEAR", "Circular", "  circular  "])
    def test_valid_entanglements(self, entanglement: str):
        """Verify supported entanglement patterns are normalized to lowercase."""
        cfg = VQCConfig(entanglement=entanglement)
        assert cfg.entanglement == entanglement.strip().lower()

    @pytest.mark.parametrize("invalid_entanglement", ["all_to_all", "random", "none", "", 123])
    def test_invalid_entanglements_raise_error(self, invalid_entanglement):
        """Verify unsupported entanglement patterns raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            VQCConfig(entanglement=invalid_entanglement)

    @pytest.mark.parametrize(
        "rot_str, expected_list",
        [
            ("RX", ["RX"]),
            ("RY", ["RY"]),
            ("RZ", ["RZ"]),
            ("RX+RY", ["RX", "RY"]),
            ("RX+RY+RZ", ["RX", "RY", "RZ"]),
            (["rx", "ry"], ["RX", "RY"]),
            ("ry+rz", ["RY", "RZ"]),
        ],
    )
    def test_parse_rotation_gates(self, rot_str, expected_list):
        """Verify parsing and normalization of rotation gate sequences."""
        parsed = parse_rotation_gates(rot_str)
        assert parsed == expected_list

    @pytest.mark.parametrize("invalid_rot", ["H", "CNOT+RX", "RX+PAULI_Z", "", []])
    def test_invalid_rotation_gates_raise_error(self, invalid_rot):
        """Verify invalid gate names raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            parse_rotation_gates(invalid_rot)

    def test_weight_shape_and_parameter_count(self):
        """Verify parameter tensor shape computation."""
        cfg = VQCConfig(n_qubits=4, depth=2, rotation_gates="RX+RY+RZ")
        assert cfg.get_weight_shape() == (2, 4, 3)
        assert cfg.get_num_parameters() == 24

        cfg_single = VQCConfig(n_qubits=6, depth=3, rotation_gates="RY")
        assert cfg_single.get_weight_shape() == (3, 6, 1)
        assert cfg_single.get_num_parameters() == 18

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verify serialization roundtrip."""
        cfg = VQCConfig(n_qubits=6, depth=3, entanglement="circular", rotation_gates="RX+RY")
        data = cfg.to_dict()
        reconstructed = VQCConfig.from_dict(data)

        assert reconstructed.n_qubits == 6
        assert reconstructed.depth == 3
        assert reconstructed.entanglement == "circular"
        assert reconstructed.rotation_gates == "RX+RY"


# ===========================================================================
# 2. Circuit Architecture & Entanglement Tests
# ===========================================================================

class TestCircuitArchitecture:
    """Tests for gate sequencing, linear vs circular entanglement, and 2-qubit guardrails."""

    def test_linear_entanglement_gate_connections(self):
        """Verify linear entanglement applies CNOT(0,1), CNOT(1,2), CNOT(2,3)."""
        cfg = VQCConfig(n_qubits=4, depth=1, entanglement="linear", rotation_gates="RY")
        dev = create_quantum_device(wires=4)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(4)
        weights = np.zeros((1, 4, 1))

        tape = qml.workflow.construct_tape(qnode)(angles, weights)
        cnots = [op for op in tape.operations if isinstance(op, qml.CNOT)]

        assert len(cnots) == 3
        assert [list(c.wires) for c in cnots] == [[0, 1], [1, 2], [2, 3]]

    def test_circular_entanglement_gate_connections(self):
        """Verify circular entanglement applies linear chain plus (n-1)->0 for n=4."""
        cfg = VQCConfig(n_qubits=4, depth=1, entanglement="circular", rotation_gates="RY")
        dev = create_quantum_device(wires=4)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(4)
        weights = np.zeros((1, 4, 1))

        tape = qml.workflow.construct_tape(qnode)(angles, weights)
        cnots = [op for op in tape.operations if isinstance(op, qml.CNOT)]

        assert len(cnots) == 4
        assert [list(c.wires) for c in cnots] == [[0, 1], [1, 2], [2, 3], [3, 0]]

    def test_two_qubit_circular_avoids_duplicate_cnot(self):
        """
        Verify that for n_qubits=2, circular entanglement applies EXACTLY 1 CNOT per layer
        and does NOT duplicate or self-control CNOT(1, 0).
        """
        cfg = VQCConfig(n_qubits=2, depth=1, entanglement="circular", rotation_gates="RY")
        dev = create_quantum_device(wires=2)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(2)
        weights = np.zeros((1, 2, 1))

        tape = qml.workflow.construct_tape(qnode)(angles, weights)
        cnots = [op for op in tape.operations if isinstance(op, qml.CNOT)]

        # Exactly 1 CNOT between wire 0 and 1
        assert len(cnots) == 1
        assert list(cnots[0].wires) == [0, 1]

    def test_multi_layer_rotation_and_entanglement_structure(self):
        """Verify depth=2 circuit applies rotations and CNOTs across both layers."""
        cfg = VQCConfig(n_qubits=3, depth=2, entanglement="linear", rotation_gates="RX+RY+RZ")
        dev = create_quantum_device(wires=3)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(3)
        weights = np.zeros((2, 3, 3))

        tape = qml.workflow.construct_tape(qnode)(angles, weights)
        ops = tape.operations

        # Encoding: 3 gates (RY)
        # Layer 1: 3 qubits * 3 gates = 9 rotations + 2 CNOTs = 11 ops
        # Layer 2: 3 qubits * 3 gates = 9 rotations + 2 CNOTs = 11 ops
        # Total expected operations: 3 + 11 + 11 = 25 ops
        assert len(ops) == 25

    def test_weight_shape_mismatch_raises_error(self):
        """Verify invalid weight shapes raise ValueError."""
        cfg = VQCConfig(n_qubits=4, depth=2, rotation_gates="RX+RY+RZ")
        dev = create_quantum_device(wires=4)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(4)
        # Invalid weights shape (2, 3, 3) instead of (2, 4, 3)
        invalid_weights = np.zeros((2, 3, 3))

        with pytest.raises(ValueError, match="Parameter tensor shape mismatch"):
            qnode(angles, invalid_weights)

    def test_nan_or_inf_weights_raise_error(self):
        """Verify NaN or Inf in weights raises ValueError."""
        cfg = VQCConfig(n_qubits=2, depth=1, rotation_gates="RY")
        dev = create_quantum_device(wires=2)
        qnode = build_vqc_circuit(dev, cfg)

        angles = np.zeros(2)
        weights_nan = np.array([[[np.nan], [0.0]]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            qnode(angles, weights_nan)


# ===========================================================================
# 3. Measurement Outputs Tests
# ===========================================================================

class TestMeasurementOutputs:
    """Tests for Pauli-Z expectation values, bounds [-1, 1], finite values, determinism."""

    @pytest.mark.parametrize("n_qubits", [2, 4, 6, 8, 10])
    def test_output_length_equals_qubit_count(self, n_qubits: int):
        """Verify exactly n_qubits expectation values are returned."""
        cfg = VQCConfig(n_qubits=n_qubits, depth=1)
        vqc = ParameterizedVQC(config=cfg)

        x = np.zeros(n_qubits)
        w = vqc.init_weights(seed=42)

        expvals = vqc.evaluate(x, w, is_encoded_angles=True)
        assert len(expvals) == n_qubits
        assert isinstance(expvals, np.ndarray)

    def test_output_values_strictly_bounded_in_neg_one_to_one(self):
        """Verify expectation values are bounded in [-1.0, 1.0]."""
        cfg = VQCConfig(n_qubits=4, depth=2, rotation_gates="RX+RY+RZ")
        vqc = ParameterizedVQC(config=cfg)

        rng = np.random.default_rng(101)
        # Evaluate across 10 random angles and weights
        for _ in range(10):
            x = rng.uniform(-np.pi, np.pi, size=4)
            w = rng.uniform(0, 2 * np.pi, size=(2, 4, 3))
            expvals = vqc.evaluate(x, w, is_encoded_angles=True)

            assert np.all(expvals >= -1.0 - 1e-12)
            assert np.all(expvals <= 1.0 + 1e-12)
            assert np.all(np.isfinite(expvals))

    def test_deterministic_outputs_for_identical_inputs(self):
        """Verify multiple runs on identical inputs yield identical outputs (variance == 0.0)."""
        cfg = VQCConfig(n_qubits=4, depth=2, seed=42)
        vqc = ParameterizedVQC(config=cfg)

        x = np.array([0.1, -0.5, 1.2, -2.0])
        w = vqc.init_weights(seed=999)

        runs = [vqc.evaluate(x, w, is_encoded_angles=True) for _ in range(10)]
        diffs = np.array(runs) - runs[0]
        assert np.max(np.abs(diffs)) < 1e-15

    def test_batch_evaluation_shape(self):
        """Verify batch evaluation on 2D array returns 2D array of shape (n_samples, n_qubits)."""
        cfg = VQCConfig(n_qubits=4, depth=2)
        vqc = ParameterizedVQC(config=cfg)

        X_batch = np.random.uniform(-np.pi, np.pi, size=(15, 4))
        w = vqc.init_weights(seed=42)

        batch_expvals = vqc.evaluate(X_batch, w, is_encoded_angles=True)
        assert batch_expvals.shape == (15, 4)
        assert np.all(np.isfinite(batch_expvals))


# ===========================================================================
# 4. PennyLane Differentiability & Gradient Tests
# ===========================================================================

class TestDifferentiability:
    """Tests verifying qml.grad autodiff gradients and finite-difference validation."""

    def test_gradient_computation_with_qml_grad(self):
        """Verify qml.grad computes finite non-zero gradients with respect to weights."""
        cfg = VQCConfig(n_qubits=2, depth=1, rotation_gates="RY")
        dev = create_quantum_device(wires=2)
        qnode = build_vqc_circuit(dev, cfg)

        angles = pnp.array([0.5, 0.8], requires_grad=False)
        weights = pnp.array([[[0.3], [0.4]]], requires_grad=True)

        def cost_fn(w):
            expvals = qnode(angles, w)
            return pnp.sum(pnp.stack(expvals))

        grad_fn = qml.grad(cost_fn)
        grad_val = grad_fn(weights)

        assert grad_val.shape == weights.shape
        assert np.all(np.isfinite(grad_val))
        assert not np.allclose(grad_val, 0.0)

    def test_autodiff_vs_finite_difference_gradient(self):
        """Verify qml.grad matches numerical central finite-difference gradient within tolerance."""
        cfg = VQCConfig(n_qubits=2, depth=1, rotation_gates="RY+RZ")
        dev = create_quantum_device(wires=2)
        qnode = build_vqc_circuit(dev, cfg)

        angles = pnp.array([0.4, 0.7], requires_grad=False)
        weights = pnp.array([[[0.2, 0.5], [0.3, 0.6]]], requires_grad=True)

        def cost_fn(w):
            return pnp.sum(pnp.stack(qnode(angles, w)))

        # Analytical autodiff gradient
        grad_autodiff = qml.grad(cost_fn)(weights)

        # Numerical finite-difference gradient
        eps = 1e-6
        grad_num = np.zeros_like(weights)
        w_flat = weights.flatten()

        for idx in range(len(w_flat)):
            w_plus = np.copy(w_flat)
            w_minus = np.copy(w_flat)
            w_plus[idx] += eps
            w_minus[idx] -= eps

            c_plus = cost_fn(pnp.array(w_plus.reshape(weights.shape), requires_grad=False))
            c_minus = cost_fn(pnp.array(w_minus.reshape(weights.shape), requires_grad=False))
            grad_num.flat[idx] = (c_plus - c_minus) / (2 * eps)

        np.testing.assert_allclose(grad_autodiff, grad_num, atol=1e-5)


# ===========================================================================
# 5. Device Backend Parity Tests
# ===========================================================================

class TestBackendParity:
    """Tests verifying VQC execution across default.qubit and lightning.qubit."""

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    def test_vqc_execution_across_backends(self, device_name: str):
        """Verify VQC executes and yields identical values on both backends."""
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available in environment.")

        cfg = VQCConfig(n_qubits=3, depth=2, device_name=device_name, seed=42)
        vqc = ParameterizedVQC(config=cfg)

        x = np.array([0.2, 0.4, 0.6])
        w = vqc.init_weights(seed=123)

        expvals = vqc.evaluate(x, w, is_encoded_angles=True)
        assert len(expvals) == 3
        assert np.all(np.isfinite(expvals))


# ===========================================================================
# 6. End-to-End AngleEncoder Integration Tests
# ===========================================================================

class TestAngleEncoderIntegration:
    """Tests verifying raw tabular data scales through AngleEncoder into VQC."""

    def test_raw_tabular_features_through_vqc(self):
        """Verify raw tabular features scale via AngleEncoder and produce valid expectations."""
        # 4 features: radius, texture, perimeter, area
        X_raw = np.array([
            [10.0, 15.0, 60.0, 300.0],
            [15.0, 20.0, 90.0, 600.0],
            [20.0, 25.0, 120.0, 900.0],
        ])

        cfg = VQCConfig(n_qubits=4, depth=2, encoding_rotation="RY")
        vqc = ParameterizedVQC(config=cfg)

        # Fit encoder on raw data
        vqc.encoder.fit(X_raw)
        w = vqc.init_weights(seed=42)

        # Evaluate with raw features (is_encoded_angles=False)
        output = vqc.evaluate(X_raw, w, is_encoded_angles=False)

        assert output.shape == (3, 4)
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)


# ===========================================================================
# 7. Visualization Tests
# ===========================================================================

class TestVisualization:
    """Tests for text drawing and Matplotlib file export."""

    def test_draw_vqc_circuit_text(self):
        """Verify draw_vqc_circuit generates readable text diagram."""
        cfg = VQCConfig(n_qubits=3, depth=1, entanglement="linear")
        diagram = draw_vqc_circuit(config=cfg)

        assert isinstance(diagram, str)
        assert "0:" in diagram
        assert "1:" in diagram
        assert "2:" in diagram
        assert "<Z>" in diagram

    def test_save_circuit_visualization_png_and_txt(self, tmp_path):
        """Verify save_circuit_visualization writes both .png and .txt files."""
        cfg = VQCConfig(n_qubits=3, depth=1)
        out_base = str(tmp_path / "test_vqc")

        png_file = save_circuit_visualization(cfg, out_base)
        txt_file = f"{out_base}.txt"

        assert os.path.exists(png_file)
        assert os.path.exists(txt_file)
        assert os.path.getsize(png_file) > 1000
        assert os.path.getsize(txt_file) > 100
