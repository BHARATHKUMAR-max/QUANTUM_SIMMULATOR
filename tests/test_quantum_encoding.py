"""
tests/test_quantum_encoding.py
==============================
Unit and integration test suite for quantum feature angle encoding
in QML Benchmark Studio (Day 22 milestone).

Test Categories:
  1. TestAngleEncodingConfig: Configuration defaults, valid gates, custom ranges, serialization, validation.
  2. TestScalingMathematics: Exact min/max/midpoint mapping, negative/positive spans, constant features, clipping, training-only scaling.
  3. TestDimensionalityAndGuardrails: Qubit bounds [2, 10], feature-qubit mismatches, no silent dropping/padding, NaN/Inf rejection.
  4. TestFeatureWireMapping: Sequential wire allocation, preservation of DataFrame column names and order, introspection dicts.
  5. TestDataCompatibility: NumPy arrays, Pandas DataFrames/Series, 1D single samples, batch inputs, PCA feature names.
  6. TestComparisonModeIntegration: Compatibility with existing data pipeline, Mode A, Mode B outputs.
  7. TestCircuitIntegration: End-to-end PennyLane device circuit execution, analytical <Z> = cos(theta) for RY and RX, both default.qubit and lightning.qubit.
"""

import math
from typing import List
import numpy as np
import pandas as pd
import pytest

from backend.data.schemas import MAX_QUBITS, MIN_QUBITS
from backend.quantum.device import create_quantum_device, get_available_devices
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


# ===========================================================================
# 1. Configuration Tests
# ===========================================================================

class TestAngleEncodingConfig:
    """Tests for AngleEncodingConfig defaults, validation, and serialization."""

    def test_default_configuration(self):
        """Verify default configuration values."""
        cfg = AngleEncodingConfig()
        assert cfg.rotation == "RY"
        assert cfg.angle_range == (-math.pi, math.pi)
        assert cfg.n_qubits is None
        assert cfg.clipping is True
        assert cfg.constant_feature_policy == "midpoint"

    @pytest.mark.parametrize("rotation", ["RX", "RY", "RZ", "rx", "ry", "rz", "  ry  "])
    def test_valid_rotations_case_insensitivity(self, rotation: str):
        """Verify valid rotation gates are accepted and normalized to uppercase."""
        cfg = AngleEncodingConfig(rotation=rotation)
        assert cfg.rotation == rotation.strip().upper()

    @pytest.mark.parametrize("invalid_rotation", ["H", "CNOT", "U3", "PAULI_Z", "", 123])
    def test_invalid_rotations_raise_error(self, invalid_rotation):
        """Verify invalid rotation gates raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            AngleEncodingConfig(rotation=invalid_rotation)

    def test_custom_angle_range(self):
        """Verify valid custom angle range."""
        cfg = AngleEncodingConfig(angle_range=(0.0, math.pi))
        assert cfg.angle_range == (0.0, math.pi)

    @pytest.mark.parametrize(
        "invalid_range",
        [
            (math.pi, -math.pi),      # lower > upper
            (1.0, 1.0),               # lower == upper
            (-math.pi, float("nan")), # NaN
            (float("-inf"), 1.0),     # Inf
            [0.0],                    # wrong length
            "[-pi, pi]",              # wrong type
        ],
    )
    def test_invalid_angle_ranges_raise_error(self, invalid_range):
        """Verify inverted, equal, NaN, or non-2-tuple ranges raise errors."""
        with pytest.raises((ValueError, TypeError)):
            AngleEncodingConfig(angle_range=invalid_range)

    @pytest.mark.parametrize("n_qubits", [2, 4, 6, 8, 10])
    def test_valid_qubit_counts(self, n_qubits: int):
        """Verify supported qubit counts [2, 10] are accepted."""
        cfg = AngleEncodingConfig(n_qubits=n_qubits)
        assert cfg.n_qubits == n_qubits

    @pytest.mark.parametrize("invalid_qubits", [0, 1, 11, 20, -1, True, False, 4.5, "4"])
    def test_invalid_qubit_counts_raise_error(self, invalid_qubits):
        """Verify qubit counts outside [2, 10] or invalid types raise errors."""
        with pytest.raises((ValueError, TypeError)):
            AngleEncodingConfig(n_qubits=invalid_qubits)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verify configuration serialization roundtrip."""
        cfg = AngleEncodingConfig(
            rotation="RZ",
            angle_range=(-math.pi / 2, math.pi / 2),
            n_qubits=4,
            clipping=False,
            constant_feature_policy="zero",
        )
        data = cfg.to_dict()
        reconstructed = AngleEncodingConfig.from_dict(data)

        assert reconstructed.rotation == "RZ"
        assert reconstructed.angle_range == (-math.pi / 2, math.pi / 2)
        assert reconstructed.n_qubits == 4
        assert reconstructed.clipping is False
        assert reconstructed.constant_feature_policy == "zero"


# ===========================================================================
# 2. Scaling Mathematics Tests
# ===========================================================================

class TestScalingMathematics:
    """Tests for linear mapping, min/max/midpoint bounds, constant features, and clipping."""

    def test_min_max_midpoint_mapping(self):
        """Verify minimum maps to angle_min, maximum to angle_max, and midpoint to 0.0."""
        # 3 features: [0, 5, 10], [-100, 0, 100], [1, 2, 3]
        X_train = np.array([
            [0.0, -100.0, 1.0],
            [5.0, 0.0, 2.0],
            [10.0, 100.0, 3.0],
        ])

        encoder = AngleEncoder()
        angles = encoder.fit_transform(X_train)

        # Min sample (row 0) -> should be -pi
        np.testing.assert_allclose(angles[0], [-math.pi, -math.pi, -math.pi], atol=1e-12)

        # Midpoint sample (row 1) -> should be 0.0
        np.testing.assert_allclose(angles[1], [0.0, 0.0, 0.0], atol=1e-12)

        # Max sample (row 2) -> should be +pi
        np.testing.assert_allclose(angles[2], [math.pi, math.pi, math.pi], atol=1e-12)

    def test_custom_range_scaling(self):
        """Verify scaling to custom range [0, pi]."""
        X_train = np.array([
            [10.0, 20.0],
            [20.0, 40.0],
            [30.0, 60.0],
        ])

        encoder = AngleEncoder(angle_range=(0.0, math.pi))
        angles = encoder.fit_transform(X_train)

        # Min -> 0.0, Midpoint -> pi/2, Max -> pi
        np.testing.assert_allclose(angles[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(angles[1], [math.pi / 2, math.pi / 2], atol=1e-12)
        np.testing.assert_allclose(angles[2], [math.pi, math.pi], atol=1e-12)

    def test_constant_feature_handling_midpoint(self):
        """Verify constant-valued features map to range midpoint (0.0 for [-pi, pi])."""
        X_train = np.array([
            [1.0, 42.0, 10.0],
            [2.0, 42.0, 20.0],
            [3.0, 42.0, 30.0],
        ])

        encoder = AngleEncoder(constant_feature_policy="midpoint")
        angles = encoder.fit_transform(X_train)

        assert encoder.is_constant_[1] is np.True_ or encoder.is_constant_[1] == True
        assert encoder.is_constant_[0] == False
        assert encoder.is_constant_[2] == False

        # Middle column should all be midpoint (0.0)
        np.testing.assert_allclose(angles[:, 1], [0.0, 0.0, 0.0], atol=1e-12)

    def test_constant_feature_handling_zero(self):
        """Verify constant-valued features map to 0.0 when policy='zero'."""
        X_train = np.array([
            [1.0, 99.0],
            [2.0, 99.0],
        ])
        encoder = AngleEncoder(angle_range=(0.0, math.pi), constant_feature_policy="zero")
        angles = encoder.fit_transform(X_train)
        np.testing.assert_allclose(angles[:, 1], [0.0, 0.0], atol=1e-12)

    def test_out_of_range_clipping_policy(self):
        """Verify values outside training range are clipped to [angle_min, angle_max]."""
        X_train = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
        ])
        # Values outside [10, 20]: e.g. 5.0 (<10) and 35.0 (>20)
        X_test = np.array([
            [5.0, 35.0],
            [0.0, 100.0],
        ])

        encoder_clipped = AngleEncoder(clipping=True)
        encoder_clipped.fit(X_train)
        angles_clipped = encoder_clipped.transform(X_test)

        # All values must be bounded within [-pi, pi]
        assert np.all(angles_clipped >= -math.pi)
        assert np.all(angles_clipped <= math.pi)
        # 5.0 was < 10, clipped to -pi
        assert angles_clipped[0, 0] == -math.pi
        # 35.0 was > 20, clipped to +pi
        assert angles_clipped[0, 1] == math.pi

    def test_unclipped_linear_extrapolation(self):
        """Verify unclipped configuration performs direct linear extrapolation."""
        X_train = np.array([
            [0.0, 0.0],
            [10.0, 10.0],
        ])
        X_test = np.array([
            [20.0, -10.0], # 20 is 2x max, -10 is below min
        ])
        encoder = AngleEncoder(clipping=False)
        encoder.fit(X_train)
        angles = encoder.transform(X_test)

        # For [0, 10] -> [-pi, pi], slope is 2*pi / 10 = 0.2*pi per unit
        # At x=20: -pi + 20 * 0.2*pi = 3*pi
        assert angles[0, 0] > math.pi
        assert angles[0, 1] < -math.pi

    def test_training_only_scaling_invariance(self):
        """Verify transforming test data does NOT alter fitted scaling parameters."""
        X_train = np.array([
            [1.0, 2.0],
            [5.0, 10.0],
        ])
        encoder = AngleEncoder()
        encoder.fit(X_train)

        saved_min = np.copy(encoder.x_min_)
        saved_max = np.copy(encoder.x_max_)
        saved_mapping = list(encoder.get_feature_wire_mapping())

        # Test data with extreme outliers
        X_test = np.array([
            [-1000.0, 5000.0],
            [9999.0, -9999.0],
        ])
        _ = encoder.transform(X_test)

        # Check fitted attributes remain exactly unchanged
        np.testing.assert_array_equal(encoder.x_min_, saved_min)
        np.testing.assert_array_equal(encoder.x_max_, saved_max)
        assert encoder.get_feature_wire_mapping() == saved_mapping

    def test_nan_values_rejected(self):
        """Verify NaN values in fit or transform raise ValueError."""
        X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
        encoder = AngleEncoder()
        with pytest.raises(ValueError, match="NaN"):
            encoder.fit(X_nan)

        X_valid = np.array([[1.0, 2.0], [3.0, 4.0]])
        encoder.fit(X_valid)
        with pytest.raises(ValueError, match="NaN"):
            encoder.transform(X_nan)

    def test_inf_values_rejected(self):
        """Verify Infinite values in fit or transform raise ValueError."""
        X_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
        encoder = AngleEncoder()
        with pytest.raises(ValueError, match="infinite"):
            encoder.fit(X_inf)


# ===========================================================================
# 3. Dimensionality & Qubit Guardrails Tests
# ===========================================================================

class TestDimensionalityAndGuardrails:
    """Tests for wire counts [2, 10], feature mismatches, and no silent padding/truncation."""

    @pytest.mark.parametrize("n_features", [2, 4, 6, 8, 10])
    def test_valid_feature_counts_fit_correctly(self, n_features: int):
        """Verify feature counts from 2 to 10 are accepted when n_qubits is inferred."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, n_features))

        encoder = AngleEncoder()
        angles = encoder.fit_transform(X)

        assert encoder.n_qubits_ == n_features
        assert encoder.n_features_in_ == n_features
        assert angles.shape == (20, n_features)

    def test_fewer_than_two_features_rejected(self):
        """Verify 1 feature is rejected (MIN_QUBITS=2)."""
        X_1d = np.array([[1.0], [2.0], [3.0]])
        encoder = AngleEncoder()
        with pytest.raises(ValueError, match=r"outside the supported qubit simulator range"):
            encoder.fit(X_1d)

    def test_more_than_ten_features_rejected(self):
        """Verify 11 features are rejected (MAX_QUBITS=10)."""
        X_11 = np.ones((5, 11))
        encoder = AngleEncoder()
        with pytest.raises(ValueError, match=r"outside the supported qubit simulator range"):
            encoder.fit(X_11)

    def test_configured_n_qubits_matches_features_success(self):
        """Verify matching configured n_qubits with input features succeeds."""
        X = np.zeros((10, 4))
        encoder = AngleEncoder(n_qubits=4)
        encoder.fit(X)
        assert encoder.n_qubits_ == 4

    def test_configured_n_qubits_mismatch_raises_informative_error(self):
        """Verify feature count mismatch with configured n_qubits raises clear ValueError."""
        X_3 = np.zeros((10, 3))
        encoder = AngleEncoder(n_qubits=4)

        with pytest.raises(ValueError) as excinfo:
            encoder.fit(X_3)

        assert "Dimensionality mismatch" in str(excinfo.value)
        assert "Configured n_qubits=4" in str(excinfo.value)
        assert "3 features" in str(excinfo.value)

    def test_transform_feature_mismatch_raises_error(self):
        """Verify transform input with different feature count than fit raises ValueError."""
        X_train = np.zeros((10, 4))
        X_test = np.zeros((5, 3))

        encoder = AngleEncoder()
        encoder.fit(X_train)

        with pytest.raises(ValueError, match=r"Feature dimension mismatch"):
            encoder.transform(X_test)

    def test_unfitted_encoder_raises_runtime_error(self):
        """Verify calling transform or mapping before fit raises RuntimeError."""
        encoder = AngleEncoder()
        with pytest.raises(RuntimeError, match="not fitted yet"):
            encoder.transform(np.zeros((2, 2)))

        with pytest.raises(RuntimeError, match="not fitted yet"):
            encoder.get_feature_wire_mapping()


# ===========================================================================
# 4. Feature-to-Wire Mapping Tests
# ===========================================================================

class TestFeatureWireMapping:
    """Tests for 1-to-1 sequential wire allocation and metadata preservation."""

    def test_sequential_feature_to_wire_mapping(self):
        """Verify sequential wire assignment: feature 0 -> wire 0, ..., feature k -> wire k."""
        features = ["radius", "texture", "perimeter", "area"]
        df = pd.DataFrame(np.random.normal(size=(10, 4)), columns=features)

        encoder = AngleEncoder()
        encoder.fit(df)

        mapping = encoder.get_feature_wire_mapping()
        assert len(mapping) == 4

        for i, item in enumerate(mapping):
            assert item["wire"] == i
            assert item["feature_name"] == features[i]
            assert item["feature_index"] == i
            assert item["rotation"] == "RY"

        feature_to_wire = encoder.get_feature_to_wire_dict()
        assert feature_to_wire == {"radius": 0, "texture": 1, "perimeter": 2, "area": 3}

        wire_to_feature = encoder.get_wire_to_feature_dict()
        assert wire_to_feature == {0: "radius", 1: "texture", 2: "perimeter", 3: "area"}

    def test_explicit_feature_names_on_numpy_array(self):
        """Verify explicit feature_names list provided with NumPy array is preserved."""
        X = np.random.normal(size=(5, 3))
        names = ["feat_A", "feat_B", "feat_C"]

        encoder = AngleEncoder()
        encoder.fit(X, feature_names=names)

        assert encoder.feature_names_ == names
        mapping = encoder.get_feature_to_wire_dict()
        assert mapping == {"feat_A": 0, "feat_B": 1, "feat_C": 2}

    def test_feature_names_length_mismatch_raises_error(self):
        """Verify length mismatch between feature_names and array columns raises ValueError."""
        X = np.random.normal(size=(5, 3))
        with pytest.raises(ValueError, match="Length of feature_names"):
            AngleEncoder().fit(X, feature_names=["f1", "f2"])

    def test_dataframe_feature_names_mismatch_raises_error(self):
        """Verify mismatch between DataFrame columns and explicit feature_names raises ValueError."""
        df = pd.DataFrame(np.random.normal(size=(5, 2)), columns=["x", "y"])
        with pytest.raises(ValueError, match="does not match DataFrame column names"):
            AngleEncoder().fit(df, feature_names=["a", "b"])

    def test_dataframe_columns_checked_on_transform(self):
        """Verify transforming DataFrame with altered column names raises ValueError."""
        df_train = pd.DataFrame(np.ones((5, 2)), columns=["col_a", "col_b"])
        df_test = pd.DataFrame(np.ones((5, 2)), columns=["col_x", "col_y"])

        encoder = AngleEncoder()
        encoder.fit(df_train)

        with pytest.raises(ValueError, match="do not match fitted feature names"):
            encoder.transform(df_test)


# ===========================================================================
# 5. Data Compatibility Tests
# ===========================================================================

class TestDataCompatibility:
    """Tests for NumPy, Pandas DataFrame, 1D single samples, batch inputs, and PCA names."""

    def test_numpy_batch_input(self):
        """Verify batch NumPy array transformation."""
        X = np.arange(20, dtype=np.float64).reshape(5, 4)
        encoder = AngleEncoder()
        angles = encoder.fit_transform(X)

        assert angles.shape == (5, 4)
        assert isinstance(angles, np.ndarray)

    def test_pandas_dataframe_input(self):
        """Verify Pandas DataFrame input transforms to NumPy array with preserved names."""
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
        encoder = AngleEncoder()
        angles = encoder.fit_transform(df)

        assert isinstance(angles, np.ndarray)
        assert angles.shape == (3, 2)
        assert encoder.feature_names_ == ["f1", "f2"]

    def test_single_sample_1d_array(self):
        """Verify 1D single-sample array (n_features,) transforms to 1D array of angles."""
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
        encoder = AngleEncoder()
        encoder.fit(X_train)

        # Single 1D sample: [5.0, 10.0]
        sample_1d = np.array([5.0, 10.0])
        angles_1d = encoder.transform(sample_1d)

        assert angles_1d.ndim == 1
        assert angles_1d.shape == (2,)
        # 5.0 is midpoint (0.0), 10.0 is max (pi)
        np.testing.assert_allclose(angles_1d, [0.0, math.pi], atol=1e-12)

    def test_single_sample_pandas_series(self):
        """Verify single sample passed as pd.Series transforms to 1D array of angles."""
        df_train = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]})
        encoder = AngleEncoder()
        encoder.fit(df_train)

        series = pd.Series({"a": 5.0, "b": 0.0})
        angles = encoder.transform(series)

        assert angles.ndim == 1
        assert angles.shape == (2,)
        np.testing.assert_allclose(angles, [0.0, -math.pi], atol=1e-12)

    def test_pca_generated_feature_names(self):
        """Verify compatibility with PCA column naming conventions (pca_0, pca_1, ...)."""
        pca_cols = [f"pca_{i}" for i in range(4)]
        df_pca = pd.DataFrame(np.random.normal(size=(30, 4)), columns=pca_cols)

        encoder = AngleEncoder()
        angles = encoder.fit_transform(df_pca)

        assert encoder.feature_names_ == pca_cols
        assert angles.shape == (30, 4)
        for i, col in enumerate(pca_cols):
            assert encoder.wire_mapping_[i]["feature_name"] == col
            assert encoder.wire_mapping_[i]["wire"] == i


# ===========================================================================
# 6. Comparison Mode Integration Tests
# ===========================================================================

class TestComparisonModeIntegration:
    """Tests verifying angle encoding integrates cleanly with existing comparison modes."""

    def test_mode_a_matched_feature_encoding(self):
        """Verify Mode A matched reduced feature dimensions map directly to qubits."""
        from backend.data.comparison_modes import validate_qubit_compatibility

        # Suppose feature selector selected 4 features
        n_features = 4
        n_qubits = 4
        compat = validate_qubit_compatibility(n_features=n_features, n_qubits=n_qubits, encoding_strategy="angle")
        assert compat.is_compatible is True

        # Fit encoder on 4 features
        X_reduced = np.random.normal(size=(25, n_features))
        encoder = AngleEncoder(n_qubits=n_qubits)
        angles = encoder.fit_transform(X_reduced)

        assert angles.shape == (25, n_qubits)
        assert encoder.n_qubits_ == n_qubits

    def test_mode_b_quantum_limited_encoding(self):
        """Verify Mode B quantum-limited feature selection output encodes correctly."""
        # Configured for 6 qubits
        n_qubits = 6
        X_quantum_limited = np.random.uniform(low=0.0, high=10.0, size=(40, n_qubits))

        encoder = AngleEncoder(n_qubits=n_qubits, rotation="RY")
        angles = encoder.fit_transform(X_quantum_limited)

        assert angles.shape == (40, 6)
        assert np.all(angles >= -math.pi)
        assert np.all(angles <= math.pi)


# ===========================================================================
# 7. Circuit Integration & Numerical Correctness Tests
# ===========================================================================

class TestCircuitIntegration:
    """Tests verifying encoded angles execute in PennyLane circuits with analytical parity."""

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    def test_ry_encoding_analytical_expectation(self, device_name: str):
        """
        Verify that RY angle encoding on |0> produces <Z> = cos(theta).
        Test known mathematical checkpoints:
          theta = 0     -> <Z> = 1.0
          theta = pi/2  -> <Z> = 0.0
          theta = pi    -> <Z> = -1.0
          theta = -pi/2 -> <Z> = 0.0
        """
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available in environment.")

        # Create 2-qubit device
        dev = create_quantum_device(device_name=device_name, wires=2)
        qnode = build_angle_encoding_circuit(dev, n_qubits=2, rotation="RY")

        test_angles = [
            (np.array([0.0, 0.0]), [1.0, 1.0]),
            (np.array([math.pi / 2, math.pi / 2]), [0.0, 0.0]),
            (np.array([math.pi, math.pi]), [-1.0, -1.0]),
            (np.array([0.0, math.pi]), [1.0, -1.0]),
            (np.array([-math.pi / 2, math.pi / 4]), [0.0, math.cos(math.pi / 4)]),
        ]

        for angles_in, expected_z in test_angles:
            expval = qnode(angles_in)
            np.testing.assert_allclose(expval, expected_z, atol=1e-12)

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    def test_rx_encoding_analytical_expectation(self, device_name: str):
        """
        Verify RX angle encoding: RX(theta)|0> also yields <Z> = cos(theta).
        """
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available in environment.")

        dev = create_quantum_device(device_name=device_name, wires=2)
        qnode = build_angle_encoding_circuit(dev, n_qubits=2, rotation="RX")

        angles = np.array([math.pi / 3, 2 * math.pi / 3])
        expected = [math.cos(math.pi / 3), math.cos(2 * math.pi / 3)]

        expval = qnode(angles)
        np.testing.assert_allclose(expval, expected, atol=1e-12)

    def test_rz_encoding_circuit_execution(self):
        """
        Verify RZ encoding executes cleanly on device without error.
        RZ on |0> leaves state in |0> with global phase e^{-i theta/2}, so <Z> = 1.0.
        """
        dev = create_quantum_device(device_name="default.qubit", wires=3)
        qnode = build_angle_encoding_circuit(dev, n_qubits=3, rotation="RZ")

        angles = np.array([0.1, 0.5, 1.2])
        expval = qnode(angles)

        # <Z> on |0> after RZ is identically 1.0
        np.testing.assert_allclose(expval, [1.0, 1.0, 1.0], atol=1e-12)

    def test_full_pipeline_to_circuit_execution(self):
        """
        End-to-end integration:
          1. Tabular data (4 features).
          2. AngleEncoder scales features to [-pi, pi].
          3. Encoded angles passed to 4-qubit circuit.
          4. Expectation values match analytical cos(angle).
        """
        # Create tabular data
        X_train = np.array([
            [1.0, 10.0, 100.0, 0.0],
            [3.0, 30.0, 300.0, 50.0],
            [5.0, 50.0, 500.0, 100.0],
        ])

        encoder = AngleEncoder(n_qubits=4, rotation="RY")
        angles_matrix = encoder.fit_transform(X_train)

        dev = create_quantum_device(device_name="default.qubit", wires=4)
        qnode = build_angle_encoding_circuit(dev, n_qubits=4, rotation="RY")

        for sample_angles in angles_matrix:
            expval = qnode(sample_angles)
            expected_expval = [math.cos(a) for a in sample_angles]
            np.testing.assert_allclose(expval, expected_expval, atol=1e-12)


# ===========================================================================
# 8. RZ Angle Encoding Mathematical Verification Tests
# ===========================================================================

class TestRZAngleEncodingMathematicalVerification:
    """
    Mathematical verification of RZ angle encoding behavior:
      1. RZ(theta)|0> = exp(-i theta/2)|0>, producing <Z> = 1.0 identically for all theta.
      2. Superposition (Hadamard) converts global phase to relative phase:
         H|0> -> RZ(theta) -> <X> = cos(theta), <Y> = -sin(theta).
      3. Noncommuting rotations map relative phase to observable Z populations:
         H|0> -> RZ(theta) -> RY(pi/2) -> <Z> = -cos(theta).
      4. Verification of RZ gate dispatching and feature-to-wire mapping without mutation.
    """

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    @pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2])
    def test_rz_on_ground_state_constant_pauli_z(self, device_name: str, theta: float):
        """
        Confirm RZ(theta)|0> produces <Z> = 1.0 identically for all theta values.
        Since |0> is an eigenstate of RZ, the transformation is purely a global phase.
        """
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available.")

        dev = create_quantum_device(device_name=device_name, wires=2)
        circuit = build_angle_encoding_circuit(dev, n_qubits=2, rotation="RZ")

        # Encode theta on wire 0, 0.0 on wire 1
        angles = np.array([theta, 0.0])
        expvals = circuit(angles)

        # Expected: exactly 1.0 on both wires regardless of theta
        np.testing.assert_allclose(expvals, [1.0, 1.0], atol=1e-14)

    def test_rz_ground_state_independence_across_theta(self):
        """
        Verify that <Z> for RZ(theta)|0> is mathematically independent of theta.
        Variance of <Z> across 20 distinct angles must be 0.0.
        """
        dev = create_quantum_device(device_name="default.qubit", wires=2)
        circuit = build_angle_encoding_circuit(dev, n_qubits=2, rotation="RZ")

        thetas = np.linspace(-math.pi, math.pi, 20)
        results = [circuit(np.array([t, t]))[0] for t in thetas]

        # Max difference across all outputs must be virtually 0.0
        assert np.max(np.abs(np.array(results) - 1.0)) < 1e-15
        assert np.var(results) < 1e-30

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    @pytest.mark.parametrize(
        "theta",
        [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3, 3 * math.pi / 4, math.pi, -math.pi / 2],
    )
    def test_rz_hadamard_pauli_x_expectation(self, device_name: str, theta: float):
        """
        Verify observable phase with Hadamard:
          |0> -> H -> RZ(theta) -> Measure <X>.
        Analytical expectation: <X> = cos(theta).
        """
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available.")

        dev = create_quantum_device(device_name=device_name, wires=2)
        qnode_x = build_rz_phase_diagnostic_circuit(dev, wire=0, observable="X")

        measured_x = qnode_x(theta)
        expected_x = math.cos(theta)
        np.testing.assert_allclose(measured_x, expected_x, atol=1e-14)

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    @pytest.mark.parametrize(
        "theta",
        [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3, 3 * math.pi / 4, math.pi, -math.pi / 2],
    )
    def test_rz_hadamard_pauli_y_expectation(self, device_name: str, theta: float):
        """
        Verify observable phase with Hadamard:
          |0> -> H -> RZ(theta) -> Measure <Y>.
        PennyLane convention: RZ(theta) = exp(-i*theta*Z/2).
        For |+> = (|0> + |1>)/sqrt(2), counter-clockwise rotation about Z rotates into +Y hemisphere:
          <Y> = sin(theta).
        """
        available = get_available_devices()
        if device_name not in available:
            pytest.skip(f"Device '{device_name}' not available.")

        dev = create_quantum_device(device_name=device_name, wires=2)
        qnode_y = build_rz_phase_diagnostic_circuit(dev, wire=0, observable="Y")

        measured_y = qnode_y(theta)
        expected_y = math.sin(theta)
        np.testing.assert_allclose(measured_y, expected_y, atol=1e-14)

    def test_rz_hadamard_pauli_z_expectation(self):
        """
        Verify that |0> -> H -> RZ(theta) has <Z> = 0.0 for all theta (equatorial state on Bloch sphere).
        """
        dev = create_quantum_device(device_name="default.qubit", wires=2)
        qnode_z = build_rz_phase_diagnostic_circuit(dev, wire=0, observable="Z")

        for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
            measured_z = qnode_z(theta)
            np.testing.assert_allclose(measured_z, 0.0, atol=1e-14)

    @pytest.mark.parametrize(
        "post_rot, post_angle, sign",
        [
            ("RY", math.pi / 2, -1.0),   # <Z> = -cos(theta)
            ("RY", -math.pi / 2, 1.0),   # <Z> = +cos(theta)
        ],
    )
    def test_rz_superposition_noncommuting_gate(self, post_rot: str, post_angle: float, sign: float):
        """
        Verify that applying a noncommuting gate (RY) after RZ in superposition maps relative phase
        to observable Z populations:
          |0> -> H -> RZ(theta) -> RY(post_angle) -> <Z> = sign * cos(theta).
        """
        dev = create_quantum_device(device_name="default.qubit", wires=2)
        circuit = build_rz_noncommuting_diagnostic_circuit(
            dev, wire=0, post_rotation=post_rot, post_angle=post_angle
        )

        for theta in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]:
            measured = circuit(theta)
            expected = sign * math.cos(theta)
            np.testing.assert_allclose(measured, expected, atol=1e-14)

    def test_rz_without_superposition_noncommuting_gate_remains_independent_of_theta(self):
        """
        Demonstrate that applying RZ directly to |0> followed by noncommuting RY(pi/2)
        remains COMPLETELY independent of theta, because RZ only imparts an unobservable global phase:
          |0> -> RZ(theta) -> RY(pi/2) -> <Z> = 0.0 (independent of theta).
        """
        import pennylane as qml

        dev = create_quantum_device(device_name="default.qubit", wires=2)

        @qml.qnode(dev)
        def no_superposition_rz(theta: float):
            qml.RZ(theta, wires=0)
            qml.RY(math.pi / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        thetas = [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]
        for t in thetas:
            val = no_superposition_rz(t)
            # RY(pi/2)|0> = (|0> + |1>)/sqrt(2), which has <Z> = 0.0 identically
            np.testing.assert_allclose(val, 0.0, atol=1e-14)

    def test_rz_encoding_applies_correct_gate_to_wires(self):
        """
        Inspect PennyLane circuit operations to verify RZ gate is correctly applied to target wires
        with exact numerical angles and without conversion to RY or RX.
        """
        import pennylane as qml

        dev = create_quantum_device(device_name="default.qubit", wires=3)
        circuit = build_angle_encoding_circuit(dev, n_qubits=3, rotation="RZ")

        angles = np.array([0.25, 0.50, 0.75])

        # Construct tape from QNode to inspect dispatched operations
        tape = qml.workflow.construct_tape(circuit)(angles)
        ops = tape.operations

        assert len(ops) == 3
        for i, op in enumerate(ops):
            assert isinstance(op, qml.RZ)
            assert op.wires[0] == i
            np.testing.assert_allclose(float(op.data[0]), angles[i], atol=1e-14)

    def test_rz_encoder_batch_and_single_sample_consistency(self):
        """
        Verify AngleEncoder with rotation='RZ' transforms single samples and batches consistently.
        """
        X_train = np.array([[10.0, 20.0], [50.0, 100.0]])
        encoder = AngleEncoder(rotation="RZ", n_qubits=2)
        encoder.fit(X_train)

        # Batch transform
        batch_angles = encoder.transform(X_train)
        assert encoder.config.rotation == "RZ"
        assert encoder.get_feature_wire_mapping()[0]["rotation"] == "RZ"
        assert encoder.get_feature_wire_mapping()[1]["rotation"] == "RZ"

        # Single-sample transform
        single_sample_0 = encoder.transform(X_train[0])
        single_sample_1 = encoder.transform(X_train[1])

        np.testing.assert_allclose(batch_angles[0], single_sample_0, atol=1e-14)
        np.testing.assert_allclose(batch_angles[1], single_sample_1, atol=1e-14)

