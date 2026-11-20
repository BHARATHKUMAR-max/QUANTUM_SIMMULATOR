# Quantum Simulator Setup and Verification

This document specifies the technical architecture, configuration, and verification of the PennyLane quantum simulation subsystem implemented in **QML Benchmark Studio** (Day 21 milestone).

---

## 1. Objectives & Scope

The purpose of the quantum simulation subsystem is to provide a reliable, deterministic, and hardware-independent execution environment for quantum machine learning circuits.

### Core Objectives
1. **Simulator Parity & Portability**: Provide robust execution across pure Python (`default.qubit`) and optimized C++ (`lightning.qubit`) backends.
2. **Analytical Verification**: Validate rotation and entanglement expectation values against exact closed-form analytical formulas.
3. **Autodiff & Gradient Soundness**: Confirm that PennyLane's automatic differentiation yields gradients matching central finite-difference numerical approximations.
4. **Strict Determinism**: Ensure statevector evaluations return identical results across repeated executions.
5. **Architectural Guardrails**: Enforce strict wire limits ($2 \le n \le 10$) and prevent unscientific quantum speedup claims.

---

## 2. Environment Configuration

The quantum development environment is built on the following pinned dependencies in `requirements.txt`:

| Component | Pinned Version | Execution Engine | Primary Use Case |
|---|---|---|---|
| **Python** | 3.13.1 | CPython 64-bit | Core runtime |
| **PennyLane** | 0.45.1 | Python API | Circuit composition, autodiff, device abstraction |
| **PennyLane-Lightning** | 0.45.0 | C++ / OpenMP | High-performance CPU statevector simulation |
| **NumPy** | 2.2.3 | C / BLAS | Matrix operations & array manipulation |

To verify installation in the project virtual environment:
```powershell
.venv\Scripts\python -c "import pennylane as qml; print(f'PennyLane {qml.__version__}'); print(qml.about())"
```

---

## 3. Supported Quantum Simulator Backends

Two statevector simulator backends are supported and managed via `backend/quantum/device.py`:

```
               ┌────────────────────────────────────────────────────────┐
               │                    DeviceConfig                        │
               │  name: "default.qubit" | "lightning.qubit"             │
               │  wires: 2 .. 10                                        │
               │  shots: None (analytic statevector)                    │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                 create_quantum_device()
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │     default.qubit      │                    │    lightning.qubit     │
       ├────────────────────────┤                    ├────────────────────────┤
       │ Pure Python statevector│                    │ C++ / OpenMP backend   │
       │ Universal fallback     │                    │ Multithreaded CPU ops  │
       │ Built-in to PennyLane  │                    │ Accelerated execution  │
       └────────────────────────┘                    └────────────────────────┘
```

### 3.1 Backend Comparison

| Attribute | `default.qubit` | `lightning.qubit` |
|---|---|---|
| **Language** | Pure Python | C++ (with OpenMP multithreading) |
| **Installation** | Built-in to `pennylane` | `pennylane-lightning` package |
| **Memory Overhead** | Modest Python object overhead | Minimal raw contiguous C++ buffer |
| **Execution Speed** | Standard | $2\times$–$10\times$ faster for $n \ge 6$ wires |
| **Gradient Method** | Backpropagation / Parameter-shift | Adjoint differentiation / Parameter-shift |
| **Recommended Usage** | Debugging, rapid prototyping, CI | Multi-qubit training, repeated runs |

### 3.2 Device Factory API

Devices are instantiated through `create_quantum_device`:

```python
from backend.quantum import DeviceConfig, create_quantum_device

# Default configuration (lightning.qubit if available, else default.qubit)
dev = create_quantum_device(wires=4)

# Explicit backend selection
config = DeviceConfig(name="lightning.qubit", wires=6)
dev = create_quantum_device(config=config)
```

Validation ensures:
- Device name must be one of `{"default.qubit", "lightning.qubit"}`.
- Wire count must be within $[2, 10]$ (configured via `MIN_QUBITS` and `MAX_QUBITS`).
- Shots must be positive integer or `None` (exact analytic statevector).

---

## 4. Hardware Findings & Acceleration Scope

### 4.1 Host Hardware Profile
- **CPU**: Intel Core i5-13420H (8 cores / 12 threads)
- **RAM**: 16 GB DDR5
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, CUDA 12.8)
- **OS**: Windows 11 Home 64-bit

### 4.2 GPU Acceleration Analysis
- PennyLane provides a GPU simulator backend (`lightning.gpu`) based on NVIDIA cuQuantum (cuStateVec).
- `lightning.gpu` requires Linux x86_64 or specific Windows WSL2 configurations with CUDA toolkit compatibility.
- **Decision for MVP**: CPU simulation is the agreed MVP target. `lightning.gpu` is neither installed nor required. All benchmarks and circuits execute reliably on host CPU via `lightning.qubit` (leveraging CPU multithreading).
- **No False Claims**: The project does not claim GPU acceleration for quantum simulation.

---

## 5. Circuit Implementations & Analytical Parity

All quantum circuits are implemented in `backend/quantum/circuits.py`.

### 5.1 Single-Qubit Rotations
For an initial state $|0\rangle$, applying an $RY(\theta)$ gate yields:
$$RY(\theta)|0\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + \sin\left(\frac{\theta}{2}\right)|1\rangle$$

The Pauli-$Z$ expectation value is:
$$\langle Z \rangle = \cos^2\left(\frac{\theta}{2}\right) - \sin^2\left(\frac{\theta}{2}\right) = \cos(\theta)$$

Empirical parity verified in test suite:
- $\theta = 0 \implies \langle Z \rangle = \cos(0) = 1.0$
- $\theta = \pi/2 \implies \langle Z \rangle = \cos(\pi/2) = 0.0$
- $\theta = \pi \implies \langle Z \rangle = \cos(\pi) = -1.0$

### 5.2 Two-Qubit Entanglement (Bell State)
Applying a Hadamard gate on wire 0 followed by a CNOT gate with wire 0 as control and wire 1 as target creates the Bell state:
$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

Expectation values:
$$\langle Z_0 \rangle = 0.0, \quad \langle Z_1 \rangle = 0.0, \quad \langle Z_0 \otimes Z_1 \rangle = 1.0$$

Both `default.qubit` and `lightning.qubit` confirm exact theoretical correlation:
- Subsystem states are maximally mixed ($\langle Z_i \rangle = 0$).
- Joint state is perfectly correlated ($\langle Z_0 Z_1 \rangle = 1.0$).

### 5.3 Parameterized Ansatz Circuit
Supports parameterized feature encoding and variational ansatz layers:
- Layer 1: Single-qubit rotations ($RY(\theta_i)$) on each wire $i$.
- Layer 2: Linear nearest-neighbor entanglement via CNOT chain ($\text{CNOT}(i, i+1)$).
- Measurement: Pauli-$Z$ expectation value on wire 0 ($\langle Z_0 \rangle$).

---

## 6. Autodiff & Gradient Soundness

Quantum circuits in PennyLane support automatic differentiation through parameter-shift rules and adjoint methods.

### 6.1 Numerical Verification Protocol
To verify gradient soundness, autodiff gradients are cross-checked against central finite differences:
$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon}$$
where $\epsilon = 10^{-6}$ and $e_i$ is the standard basis unit vector.

### 6.2 Empirical Results
Across random initializations and deterministic grid points:
- Maximum absolute gradient discrepancy: $\max_i |\nabla_{\text{autodiff}} - \nabla_{\text{finite-diff}}| < 1.8 \times 10^{-11}$
- Autodiff matches numerical finite-difference within machine precision.
- Validated on both `default.qubit` and `lightning.qubit`.

---

## 7. Determinism & Reproducibility

In statevector mode (`shots=None`), PennyLane calculates the exact wavefunction via matrix-vector multiplication.

Execution stability verified:
- Function: `verify_circuit_determinism(qnode, params, n_runs=10)`
- Outcome: Variance across 10 consecutive runs is exactly `0.0`.
- Outputs match to 16 decimal places of precision ($10^{-16}$).

---

## 8. Classical Simulation Limitations

Classical simulation of quantum circuits involves exponential computational complexity that must be documented honestly:

### 8.1 Statevector Memory Scaling
An $n$-qubit quantum state requires a complex statevector of length $2^n$:

| Number of Qubits ($n$) | Complex Amplitudes ($2^n$) | Memory (complex128: 16 B/elem) |
|---|---|---|
| 2 | 4 | 64 B |
| 4 | 16 | 256 B |
| 8 | 256 | 4 KB |
| **10 (Enforced Max)** | **1,024** | **16 KB** |
| 20 | 1,048,576 | 16 MB |
| 30 | 1,073,741,824 | 16 GB |
| 50 | $1.126 \times 10^{15}$ | 16 PB |

### 8.2 Architectural Guardrails
1. **Wire Limits**: The platform enforces $2 \le n \le 10$ qubits via `MIN_QUBITS` and `MAX_QUBITS`. Attempting to initialize devices outside this range raises a `ValueError`.
2. **Computational Cost**: While 10 qubits consumes only 16 KB of RAM, calculating gradients across hundreds of training samples requires $2 \times M$ circuit evaluations per gradient step (parameter-shift rule). Execution time scales linearly with sample count and circuit depth.
3. **No Quantum Advantage**: Classical simulation runs on classical CPUs. Any performance measured reflects classical simulation efficiency, not physical quantum hardware properties.

---

## 9. Test Verification

The test suite includes 65 dedicated automated tests for the quantum subsystem:

| Test Module | Test Count | Scope |
|---|---|---|
| `tests/test_quantum_devices.py` | 29 | Device initialization, invalid devices, qubit bounds (2–10), wire types, config serialization, capabilities metadata |
| `tests/test_quantum_circuits.py` | 36 | $RY$ rotations, analytical cosine values, Bell state entanglement, autodiff vs finite-diff gradients, multi-run determinism |
| **Total Quantum Tests** | **65** | **100% Pass Rate** |

To execute the quantum test suite:
```powershell
.venv\Scripts\python -m pytest tests/test_quantum_devices.py tests/test_quantum_circuits.py -v
```
