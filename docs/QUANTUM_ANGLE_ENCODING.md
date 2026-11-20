# Quantum Angle Encoding Specification

This document specifies the architecture, mathematics, configuration, and verification of the quantum angle encoding subsystem implemented in **QML Benchmark Studio** (`backend/quantum/encoding.py`, Day 22 milestone).

---

## 1. Overview & Conceptual Definition

In Quantum Machine Learning (QML), **Angle Encoding** (also known as *tensor product encoding* or *qubit rotation encoding*) is a quantum feature-mapping strategy that maps a continuous classical feature vector $x = (x_0, x_1, \dots, x_{d-1}) \in \mathbb{R}^d$ into the quantum state of a multi-qubit register.

For a system initialized in the computational ground state $|0\rangle^{\otimes n}$, each normalized feature value $\theta_i = f(x_i)$ is applied as the rotation angle of a single-qubit rotation gate $R_G(\theta_i)$ acting on qubit wire $i$:

$$|\psi(x)\rangle = \bigotimes_{i=0}^{n-1} R_G(\theta_i) |0\rangle = R_G(\theta_0)|0\rangle \otimes R_G(\theta_1)|0\rangle \otimes \cdots \otimes R_G(\theta_{n-1})|0\rangle$$

where $R_G \in \{R_X, R_Y, R_Z\}$.

```
              |0⟩ ─────[ R_G(θ₀) ]───── |ψ₀⟩
              |0⟩ ─────[ R_G(θ₁) ]───── |ψ₁⟩
              |0⟩ ─────[ R_G(θ₂) ]───── |ψ₂⟩
              |0⟩ ─────[ R_G(θ₃) ]───── |ψ₃⟩
```

Angle encoding prepares an unentangled product state that provides the input layer for subsequent parameterised variational ansatzes (such as the Variational Quantum Classifier).

---

## 2. Feature-to-Angle Scaling Mathematics

### 2.1 Default Angle Interval
The default target angle range is:
$$[\theta_{\min}, \theta_{\max}] = [-\pi, \pi]$$

This interval covers the full period of single-qubit rotation gates without phase ambiguity. The range is configurable (e.g. $[0, \pi]$ or $[-\pi/2, \pi/2]$), subject to the strict constraint $\theta_{\min} < \theta_{\max}$.

### 2.2 Linear Mapping Formula
For each feature $i$, given training minimum $x_{\min, i}$ and maximum $x_{\max, i}$:

$$\theta_i = \theta_{\min} + (x_i - x_{\min, i}) \cdot \frac{\theta_{\max} - \theta_{\min}}{x_{\max, i} - x_{\min, i}}$$

- When $x_i = x_{\min, i} \implies \theta_i = \theta_{\min} = -\pi$
- When $x_i = \frac{x_{\min, i} + x_{\max, i}}{2} \implies \theta_i = \frac{\theta_{\min} + \theta_{\max}}{2} = 0.0$
- When $x_i = x_{\max, i} \implies \theta_i = \theta_{\max} = \pi$

### 2.3 Strict Training-Only Fitting (Leakage Safety)
- The scaling bounds $(x_{\min, i}, x_{\max, i})$ and constant flags are computed **strictly on training partitions** during `fit()`.
- During validation, cross-validation evaluation, or final test inference, `transform()` applies the frozen training parameters.
- Transforming validation/test data **never modifies** the fitted encoder parameters.

### 2.4 Constant Feature Handling
If a feature exhibits zero variance in the training set ($x_{\max, i} == x_{\min, i}$):
- Division by zero is avoided.
- By default (`constant_feature_policy="midpoint"`), the feature is mapped deterministically to the interval midpoint:
  $$\theta_i = \frac{\theta_{\min} + \theta_{\max}}{2} \quad (\text{i.e. } 0.0 \text{ for } [-\pi, \pi])$$
- An alternative policy `constant_feature_policy="zero"` maps the angle to $0.0$.

### 2.5 Out-of-Range Test Values & Clipping
If a test sample contains an unseen extreme value ($x_i < x_{\min, i}$ or $x_i > x_{\max, i}$):
- With `clipping=True` (default), the computed angle is clamped to $[\theta_{\min}, \theta_{\max}]$. This prevents uncalibrated phase wraps beyond the configured range.
- With `clipping=False`, linear extrapolation occurs directly.

### 2.6 Strict Rejection of Non-Finite Values
In accordance with platform safety guidelines:
- Any input array or DataFrame containing `NaN` or `±Inf` is immediately rejected with an explicit `ValueError`.
- Values are never silently imputed or overwritten inside the quantum encoder.

---

## 3. Rotation Gate Architectures

The module supports three primary Pauli rotation gates:

| Gate | Unitary Matrix Representation | Action on $|0\rangle$ | Pauli-$Z$ Expectation Value $\langle Z \rangle$ |
|---|---|---|---|
| **$R_Y(\theta)$** *(Default)* | $\begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$ | $\cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$ | $\langle Z \rangle = \cos(\theta)$ |
| **$R_X(\theta)$** | $\begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$ | $\cos(\theta/2)|0\rangle - i\sin(\theta/2)|1\rangle$ | $\langle Z \rangle = \cos(\theta)$ |
| **$R_Z(\theta)$** | $\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ | $e^{-i\theta/2}|0\rangle$ | $\langle Z \rangle = 1.0$ (phase-only shift) |

### Why $R_Y$ is Default
$R_Y$ rotations keep state amplitudes strictly real-valued:
$$R_Y(\theta)|0\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + \sin\left(\frac{\theta}{2}\right)|1\rangle \in \mathbb{R}^2$$
This directly modulates computational basis populations $P(|0\rangle) = \cos^2(\theta/2)$ and $P(|1\rangle) = \sin^2(\theta/2)$, producing $\langle Z \rangle = \cos(\theta)$ directly from $|0\rangle$.

### 3.1 Mathematical Analysis of $R_Z$ Encoding: Global vs Relative Phase
An important mathematical distinction exists when configuring `rotation="RZ"`:

#### 1. Action on Ground State $|0\rangle$ (Global Phase Invariance)
Since $|0\rangle$ is an eigenstate of $R_Z(\theta)$:
$$R_Z(\theta)|0\rangle = e^{-i\theta/2}|0\rangle$$
The factor $e^{-i\theta/2}$ is an unobservable **global phase**. Consequently:
$$\langle Z \rangle = \langle 0| R_Z(\theta)^\dagger Z R_Z(\theta) |0\rangle = \langle 0| Z |0\rangle = 1.0$$
identically for **every** $\theta \in [-\pi, \pi]$. Measuring Pauli-$Z$ immediately after $R_Z(\theta)|0\rangle$ reveals zero feature information.

Furthermore, applying any subsequent unitary $U$ directly to $R_Z(\theta)|0\rangle$ without prior superposition preserves this global phase:
$$\langle \psi | M | \psi \rangle = \langle 0| R_Z(\theta)^\dagger U^\dagger M U R_Z(\theta) |0\rangle = \langle 0| U^\dagger M U |0\rangle$$
which remains completely independent of $\theta$.

#### 2. Making $R_Z$ Phase Observable via Superposition
When the qubit is first placed into an equal superposition (e.g. via Hadamard $H|0\rangle = |+\rangle$), $R_Z(\theta)$ acts non-trivially to impart a **relative phase**:
$$R_Z(\theta)|+\rangle = \frac{1}{\sqrt{2}} \left( e^{-i\theta/2}|0\rangle + e^{i\theta/2}|1\rangle \right) = e^{-i\theta/2} \left( \frac{|0\rangle + e^{i\theta}|1\rangle}{\sqrt{2}} \right)$$

This relative phase is directly observable via orthogonal Pauli measurements on the Bloch sphere equator:
$$\langle X \rangle = \cos(\theta), \quad \langle Y \rangle = \sin(\theta), \quad \langle Z \rangle = 0.0$$

#### 3. Mapping Phase to Computational Basis ($Z$ Measurement)
Applying a noncommuting rotation after $R_Z$ converts relative phase into longitudinal population differences:
$$|0\rangle \xrightarrow{H} |+\rangle \xrightarrow{R_Z(\theta)} |\psi_{\text{phase}}\rangle \xrightarrow{R_Y(\pi/2)} |\psi_{\text{out}}\rangle \implies \langle Z \rangle = -\cos(\theta)$$
$$|0\rangle \xrightarrow{H} |+\rangle \xrightarrow{R_Z(\theta)} |\psi_{\text{phase}}\rangle \xrightarrow{R_Y(-\pi/2)} |\psi_{\text{out}}\rangle \implies \langle Z \rangle = +\cos(\theta)$$

### 3.2 Implications for Variational Quantum Classifier (VQC) Design
1. **Encoding Gate Choice**: $R_Y$ remains the recommended default for tabular classification because it directly encodes features into state populations without requiring pre-rotation gates.
2. **$R_Z$ Encoding Prerequisite**: If $R_Z$ encoding is selected for VQC, the circuit must include an initial basis-change layer (e.g. $H^{\otimes n}$ or $R_Y(\pi/2)^{\otimes n}$) prior to data encoding to convert $|0\rangle^{\otimes n}$ into a superposition state.
3. **Ansatz Noncommuting Requirements**: The parameterized ansatz layers following data encoding must contain noncommuting gates ($R_Y$, $R_X$) and entangling CNOT gates to allow the encoded phase $\theta_i$ to rotate into observable computational-basis measurement distributions.
4. **Data Re-uploading**: In architectures combining multiple rotations per feature (e.g. $R_Y(x_i) R_Z(x_i)$ or alternating layers), the noncommuting geometry allows rich non-linear decision boundaries to form.

---

## 4. Feature-to-Wire Mapping & Dimensionality Guardrails

### 4.1 One-to-One Sequential Mapping
The platform enforces a deterministic, sequential 1-to-1 mapping between selected features and quantum circuit wires:

| Feature Index | Feature Name (Example) | Target Wire | Assigned Gate |
|---|---|---|---|
| 0 | `mean_radius` | Wire 0 | $R_Y(\theta_0)$ |
| 1 | `mean_texture` | Wire 1 | $R_Y(\theta_1)$ |
| 2 | `mean_perimeter` | Wire 2 | $R_Y(\theta_2)$ |
| 3 | `mean_area` | Wire 3 | $R_Y(\theta_3)$ |

### 4.2 Guardrail Invariants
1. **Wire Bounding**: Simulated wire counts must satisfy $2 \le n_{\text{qubits}} \le 10$.
2. **Exact Dimensional Matching**:
   $$\text{number of selected features} == n_{\text{qubits}}$$
3. **Forbidden Behaviors**:
   - **No Silent Truncation**: Slicing features (e.g. `X[:, :n_qubits]`) is strictly prohibited.
   - **No Silent Padding**: Zero-padding or duplicating missing features is strictly prohibited.
   - Any dimensional discrepancy immediately raises an informative `ValueError`.

---

## 5. Module API & Example Usage

### 5.1 Python API (`backend/quantum/encoding.py`)

```python
from backend.quantum import (
    AngleEncoder,
    AngleEncodingConfig,
    create_quantum_device,
    build_angle_encoding_circuit,
)
import numpy as np
import pandas as pd

# 1. Prepare tabular dataset (e.g. 4 features selected via PCA or MI)
X_train = pd.DataFrame({
    "radius": [10.0, 15.0, 20.0],
    "texture": [12.0, 18.0, 24.0],
    "perimeter": [60.0, 90.0, 120.0],
    "area": [300.0, 600.0, 900.0],
})

# 2. Configure and fit encoder
config = AngleEncodingConfig(
    rotation="RY",
    angle_range=(-np.pi, np.pi),
    n_qubits=4,
    clipping=True,
)
encoder = AngleEncoder(config=config)
encoded_angles = encoder.fit_transform(X_train)

# 3. Inspect structured feature-to-wire mapping
mapping = encoder.get_feature_wire_mapping()
# [{'wire': 0, 'feature_name': 'radius', 'rotation': 'RY', ...}, ...]

# 4. Connect to PennyLane circuit
dev = create_quantum_device(wires=4)
circuit = build_angle_encoding_circuit(dev, n_qubits=4, rotation="RY")

# 5. Evaluate circuit for first sample
sample_0_angles = encoded_angles[0]
expvals = circuit(sample_0_angles)
# Result: <Z_i> = cos(sample_0_angles[i]) == -1.0 (since min value maps to -pi)
```

---

## 6. Integration with Classical Preprocessing Pipeline

Angle encoding directly receives the output of the existing feature engineering pipeline:

```
Raw Tabular Data
       │
       ▼
[StandardScaler / MinMaxScaler]
       │
       ▼
[Feature Selection / PCA]  ───▶ Selects exactly k features (2 <= k <= 10)
       │
       ▼
[Comparison Mode Coordinator]
       ├── Mode A (Matched Reduced): Same k features fed to Classical Models and QML
       └── Mode B (Quantum Limited): Exactly k features fed to QML
       │
       ▼
[AngleEncoder]  ───▶ Scales features to [-π, π]
       │
       ▼
[PennyLane Quantum Device] ───▶ Evaluates parameterized circuit
```

---

## 7. Fundamental Limitations of Angle Encoding

While angle encoding provides a simple and direct feature representation, its theoretical limitations must be acknowledged:

1. **No Automatic High-Dimensional Compression**:
   Angle encoding requires exactly 1 qubit per feature. It **does not compress** 30-feature datasets into 4 qubits. Upstream dimensionality reduction (PCA, Mutual Information, ANOVA) is mandatory.
2. **Linear State Separability**:
   Without subsequent entangling layers or non-linear feature maps (e.g. $ZZ$-feature maps or repeated data re-uploading), single-layer angle encoding produces unentangled product states with limited representational capacity.
3. **No Claimed Quantum Advantage**:
   Angle encoding runs on classical statevector simulators (`default.qubit` and `lightning.qubit`). All mathematical transformations execute on classical CPU hardware.

---

## 8. Verification & Test Summary

The implementation is verified by **74 automated tests** in [tests/test_quantum_encoding.py](file:///D:/QUANTUM_PROJECT/tests/test_quantum_encoding.py):

| Test Class | Test Count | Scope |
|---|---|---|
| `TestAngleEncodingConfig` | 36 | Default settings, case insensitivity, invalid gates, custom angle ranges, qubit bounds, serialization |
| `TestScalingMathematics` | 9 | Min/max/midpoint parity, constant feature handling, clipping vs unclipped extrapolation, training-only invariance, NaN/Inf rejection |
| `TestDimensionalityAndGuardrails` | 11 | Qubit bounds [2, 10], feature mismatch errors, unfitted state errors |
| `TestFeatureWireMapping` | 5 | Sequential wire mapping, DataFrame column preservation, metadata dictionaries |
| `TestDataCompatibility` | 5 | NumPy arrays, DataFrames, Pandas Series, 1D single-sample, PCA column conventions |
| `TestComparisonModeIntegration` | 2 | Mode A matched features and Mode B quantum-limited feature flows |
| `TestCircuitIntegration` | 6 | PennyLane circuit execution, analytical cosine expectations for RY/RX across `default.qubit` and `lightning.qubit` |
| **Total Tests** | **74** | **100% Pass Rate** |
