# Parameterized Variational Quantum Circuit (VQC) Specification

This document specifies the technical architecture, mathematical formulation, configuration schema, and execution of the parameterized Variational Quantum Circuit (VQC) implemented in **QML Benchmark Studio** (`backend/quantum/vqc.py`, Day 23 milestone).

---

## 1. Architectural Overview

The Variational Quantum Circuit (VQC) serves as the core parameterized quantum component for quantum machine learning classification. Built on PennyLane's differentiable quantum device abstraction (`default.qubit` and `lightning.qubit`), the circuit implements a 4-step execution lifecycle:

```
  Input Features (x ∈ ℝⁿ)
             │
             ▼
   [Step A: Data Encoding]        ───▶ AngleEncoder (RY / RX / RZ) on |0⟩⊗ⁿ
             │
             ▼
   [Step B: Variational Layer l]  ───▶ Trainable Rotations (RX, RY, RZ) on each qubit
             │
             ▼
   [Step C: Entanglement Layer l] ───▶ CNOT Entangling Chain (Linear or Circular)
             │
   (Repeated for l = 1 .. depth)
             │
             ▼
   [Step D: Measurement]         ───▶ Multi-qubit Pauli-Z Expectations: [<Z₀>, ..., <Zₙ₋₁>]
```

---

## 2. Circuit Stages & Logical Invariants

### 2.1 Step A — Input Feature Encoding
- Reuses the Day 22 `AngleEncoder` module without duplicate scaling logic.
- Takes scaled angles $\theta \in [-\pi, \pi]^n$ (or raw tabular features that are transformed on the fly).
- Applies single-qubit rotations to each wire:
  $$|\psi_0(x)\rangle = \bigotimes_{i=0}^{n-1} R_{\text{enc}}(\theta_i) |0\rangle$$
- Default encoding gate: $R_Y$.
- Strict dimensional requirement: $\text{number of features} == n_{\text{qubits}}$. No silent feature truncation or padding is permitted.

### 2.2 Step B — Trainable Variational Layers
For each layer $l \in \{0, \dots, \text{depth}-1\}$, trainable rotation gates act independently on every qubit $q \in \{0, \dots, n-1\}$.

Supported rotation gate sequences:
- `"RX"`: 1 parameter per qubit per layer.
- `"RY"`: 1 parameter per qubit per layer.
- `"RZ"`: 1 parameter per qubit per layer (requires prior superposition or preceding noncommuting rotation).
- `"RX+RY"`: 2 parameters per qubit per layer.
- `"RX+RY+RZ"` *(Default)*: 3 parameters per qubit per layer (universal Euler-like parameterized rotation).

Each gate receives a dedicated scalar parameter from the 3D weight tensor:
$$U_{\text{rot}}^{(l, q)}(\mathbf{w}_{l, q}) = \prod_{k=0}^{n_{\text{gates}}-1} R_{G_k}(w_{l, q, k})$$

### 2.3 Step C — Entanglement Patterns
Following the trainable rotations in each layer, an entangling layer is applied to create quantum correlations across qubits.

#### 1. Linear Entanglement (`"linear"`, Default)
Adjacent qubits are connected sequentially via nearest-neighbor CNOT gates:
$$\text{CNOT}(q_i \to q_{i+1}) \quad \text{for } i \in \{0, 1, \dots, n-2\}$$
Total CNOTs per layer: $n - 1$.

```
Wire 0: ────╭●────────────
Wire 1: ────╰X────╭●──────
Wire 2: ──────────╰X────╭●
Wire 3: ────────────────╰X
```

#### 2. Circular Entanglement (`"circular"`)
Applies the linear chain, plus a feedback CNOT from the last qubit to the first:
$$\text{CNOT}(q_{n-1} \to q_0) \quad \text{for } n \ge 3$$
Total CNOTs per layer: $n$.

**Two-Qubit Guardrail**:
For $n = 2$ qubits, circular entanglement does **not** add a reverse $\text{CNOT}(1 \to 0)$, because wire 0 and wire 1 are already entangled by the linear $\text{CNOT}(0 \to 1)$. Applying a reverse CNOT would introduce duplicate or inverse gates.

### 2.4 Step D — Multi-Qubit Pauli-$Z$ Measurements
The circuit terminates by measuring the expectation value of the Pauli-$Z$ operator on **every** qubit wire:
$$\langle Z_i \rangle = \langle \psi(\theta, \mathbf{w}) | Z_i | \psi(\theta, \mathbf{w}) \rangle \in [-1.0, 1.0]$$

Output vector:
$$\mathbf{y}_{\text{pred\_raw}} = \begin{pmatrix} \langle Z_0 \rangle \\ \langle Z_1 \rangle \\ \vdots \\ \langle Z_{n-1} \rangle \end{pmatrix} \in [-1.0, 1.0]^n$$

---

## 3. Configuration & Parameter Tensor Shapes

### 3.1 `VQCConfig` Specification
The circuit configuration is managed by the validated dataclass `VQCConfig`:

| Parameter | Type | Default | Permitted Values | Description |
|---|---|---|---|---|
| `n_qubits` | `int` | `4` | `2` to `10` | Number of qubits (`MIN_QUBITS` to `MAX_QUBITS`) |
| `depth` | `int` | `2` | `1` to `4` | Number of variational layers (`MIN_DEPTH` to `MAX_DEPTH`) |
| `entanglement` | `str` | `"linear"` | `"linear"`, `"circular"` | CNOT entanglement topology |
| `rotation_gates`| `str` | `"RX+RY+RZ"` | `"RX"`, `"RY"`, `"RZ"`, `"RX+RY"`, `"RX+RY+RZ"` | Trainable rotation sequence |
| `device_name` | `str` | `"default.qubit"`| `"default.qubit"`, `"lightning.qubit"` | PennyLane simulator backend |
| `encoding_rotation` | `str`| `"RY"` | `"RX"`, `"RY"`, `"RZ"` | Feature encoding gate |
| `shots` | `int` | `None` | Positive `int` or `None` | `None` for analytic statevector |

### 3.2 Parameter Tensor Shape
The trainable parameters are structured as a 3D NumPy array or PennyLane tensor:
$$\text{Shape: } (\text{depth}, n_{\text{qubits}}, n_{\text{rotation\_gates}})$$

Examples:
- $n=4$ qubits, $\text{depth}=2$, `"RX+RY+RZ"` ($n_{\text{gates}}=3$):
  $$\text{Shape: } (2, 4, 3) \implies 24 \text{ parameters}$$
- $n=2$ qubits, $\text{depth}=1$, `"RY"` ($n_{\text{gates}}=1$):
  $$\text{Shape: } (1, 2, 1) \implies 2 \text{ parameters}$$
- $n=10$ qubits, $\text{depth}=4$, `"RX+RY+RZ"` ($n_{\text{gates}}=3$):
  $$\text{Shape: } (4, 10, 3) \implies 120 \text{ parameters}$$

---

## 4. Differentiability & Autodiff Verification

The VQC is fully differentiable with respect to its variational parameter tensor $\mathbf{w}$ using PennyLane's automatic differentiation engine (`qml.grad`).

### 4.1 Parameter-Shift & Backpropagation Parity
In statevector mode (`shots=None`), analytic gradients are computed either via backpropagation (`default.qubit`) or the adjoint differentiation method (`lightning.qubit`).

Autodiff gradients were cross-checked against central finite differences:
$$\frac{\partial C}{\partial w_{l, q, k}} \approx \frac{C(\mathbf{w} + \epsilon \mathbf{e}_{l, q, k}) - C(\mathbf{w} - \epsilon \mathbf{e}_{l, q, k})}{2\epsilon}$$
with step size $\epsilon = 10^{-6}$.

**Empirical Result**: Autodiff gradients match numerical finite-difference gradients within $\Delta < 1.0 \times 10^{-5}$ across all tested layers and rotation gates.

---

## 5. Circuit Visualizations

Generated diagrams are stored in `reports/quantum_circuits/`:

### 5.1 4-Qubit, Depth-2, Linear Entanglement
- Image: [reports/quantum_circuits/vqc_4qubit_depth2_linear.png](file:///D:/QUANTUM_PROJECT/reports/quantum_circuits/vqc_4qubit_depth2_linear.png)
- Text diagram: [reports/quantum_circuits/vqc_4qubit_depth2_linear.txt](file:///D:/QUANTUM_PROJECT/reports/quantum_circuits/vqc_4qubit_depth2_linear.txt)

```text
0: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)─╭●──RX(0.00)──RY(0.00)──RZ(0.00)───────────╭●──────────────┤  <Z>
1: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)─╰X─╭●─────────RX(0.00)──RY(0.00)──RZ(0.00)─╰X────────╭●────┤  <Z>
2: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)────╰X────────╭●─────────RX(0.00)──RY(0.00)──RZ(0.00)─╰X─╭●─┤  <Z>
3: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)──────────────╰X─────────RX(0.00)──RY(0.00)──RZ(0.00)────╰X─┤  <Z>
```

### 5.2 4-Qubit, Depth-2, Circular Entanglement
- Image: [reports/quantum_circuits/vqc_4qubit_depth2_circular.png](file:///D:/QUANTUM_PROJECT/reports/quantum_circuits/vqc_4qubit_depth2_circular.png)
- Text diagram: [reports/quantum_circuits/vqc_4qubit_depth2_circular.txt](file:///D:/QUANTUM_PROJECT/reports/quantum_circuits/vqc_4qubit_depth2_circular.txt)

```text
0: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)─╭●───────╭X──RX(0.00)──RY(0.00)──RZ(0.00)─╭●───────╭X─┤  <Z>
1: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)─╰X─╭●────│───RX(0.00)──RY(0.00)──RZ(0.00)─╰X─╭●────│──┤  <Z>
2: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)────╰X─╭●─│───RX(0.00)──RY(0.00)──RZ(0.00)────╰X─╭●─│──┤  <Z>
3: ──RY(0.00)──RX(0.00)──RY(0.00)──RZ(0.00)───────╰X─╰●──RX(0.00)──RY(0.00)──RZ(0.00)───────╰X─╰●─┤  <Z>
```

---

## 6. Example Python API Usage

```python
from backend.quantum import VQCConfig, ParameterizedVQC
import numpy as np

# 1. Configure a 4-qubit, depth-2 VQC with circular entanglement
config = VQCConfig(
    n_qubits=4,
    depth=2,
    entanglement="circular",
    rotation_gates="RX+RY+RZ",
    device_name="default.qubit",
)

# 2. Instantiate VQC manager
vqc = ParameterizedVQC(config=config)

# 3. Initialize trainable weights
weights = vqc.init_weights(seed=42)  # Shape: (2, 4, 3)

# 4. Prepare raw tabular batch (4 features)
X_batch = np.array([
    [10.0, 15.0, 60.0, 300.0],
    [12.0, 18.0, 75.0, 450.0],
])

# 5. Evaluate circuit
expectations = vqc.evaluate(X_batch, weights, is_encoded_angles=False)
# Shape: (2, 4), values in [-1.0, 1.0]

# 6. Generate circuit visualization
text_diag = vqc.draw()
print(text_diag)
```

---

## 7. Supported Configurations & Limitations

1. **Qubit Range**: Strictly $2 \le n_{\text{qubits}} \le 10$. Bounded by simulator memory scaling ($2^n$ statevector amplitudes).
2. **Depth Range**: Strictly $1 \le \text{depth} \le 4$. Keeps classical simulation runtime responsive.
3. **No Classical Neural Network Head**: Measurement produces raw Pauli-$Z$ expectations. Output processing (e.g. linear layer, softmax, or sign thresholding) is deferred to the VQC classifier wrapper (Day 24).
4. **Classical Hardware Only**: Runs via CPU simulation (`default.qubit` or `lightning.qubit`). No quantum speedup is claimed or expected.

---

## 8. Verification & Test Summary

The VQC circuit implementation is validated by **72 automated tests** in [tests/test_vqc.py](file:///D:/QUANTUM_PROJECT/tests/test_vqc.py):

| Test Category | Count | Scope |
|---|---|---|
| `TestVQCConfig` | 51 | Boundary checks (qubits 2-10, depth 1-4), gate sequence parsing, serialization |
| `TestCircuitArchitecture` | 6 | Linear vs circular entanglement, 2-qubit guardrail, multi-layer gate sequencing, shape validation |
| `TestMeasurementOutputs` | 8 | Length == $n$, $[-1, 1]$ bounds, finite values, determinism, batching |
| `TestDifferentiability` | 2 | `qml.grad` execution, central finite-difference gradient agreement |
| `TestBackendParity` | 2 | Parity across `default.qubit` and `lightning.qubit` |
| `TestAngleEncoderIntegration`| 1 | End-to-end raw tabular feature flow through encoder and circuit |
| `TestVisualization` | 2 | Text drawing and Matplotlib file export |
| **Total Tests** | **72** | **100% Pass Rate** |
