"""
backend/quantum/circuits.py
===========================
Foundational quantum circuits, analytical expectation value verifiers,
autodiff gradient calculators, and determinism testers for QML Benchmark Studio.

Implements small, reusable circuits:
  1. Single-qubit rotation circuits (RX, RY, RZ with Pauli-Z measurement).
  2. Two-qubit entanglement circuits (Rotations + CNOT, measuring single and joint Pauli-Z).
  3. Parameterized quantum ansatzes for gradient verification.
  4. Numerical finite-difference vs. autodiff validation tools.
  5. Deterministic execution verification.

Analytical Invariants:
----------------------
- For RY(theta)|0>: <Z> = cos(theta).
- For theta=0: <Z> = 1.0.
- For theta=pi/2: <Z> = 0.0.
- For theta=pi: <Z> = -1.0.
- Pauli-Z expectation values are strictly bounded in [-1.0, 1.0].
- Statevector simulation on classical hardware is deterministic.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ---------------------------------------------------------------------------
# 1. Single-Qubit Rotation Circuits
# ---------------------------------------------------------------------------


def build_ry_rotation_circuit(device: Any, wire: int = 0) -> qml.QNode:
    """
    Construct a QNode evaluating pure RY rotation: RY(theta)|0> -> <Z>.

    Analytical expectation: <Z> = cos(theta).

    Args:
        device: PennyLane Device (e.g. default.qubit or lightning.qubit).
        wire: Target qubit index (default 0).

    Returns:
        Callable QNode accepting single float parameter theta.
    """
    @qml.qnode(device)
    def ry_circuit(theta: Union[float, pnp.ndarray]) -> float:
        qml.RY(theta, wires=wire)
        return qml.expval(qml.PauliZ(wire))

    return ry_circuit


def build_single_qubit_rotation_circuit(device: Any, wire: int = 0) -> qml.QNode:
    """
    Construct a QNode evaluating consecutive Euler rotations: RZ(omega) * RY(theta) * RX(phi)|0> -> <Z>.

    Args:
        device: PennyLane Device.
        wire: Target qubit index (default 0).

    Returns:
        Callable QNode accepting (phi, theta, omega).
    """
    @qml.qnode(device)
    def single_qubit_rotations(
        phi: Union[float, pnp.ndarray],
        theta: Union[float, pnp.ndarray],
        omega: Union[float, pnp.ndarray],
    ) -> float:
        qml.RX(phi, wires=wire)
        qml.RY(theta, wires=wire)
        qml.RZ(omega, wires=wire)
        return qml.expval(qml.PauliZ(wire))

    return single_qubit_rotations


# ---------------------------------------------------------------------------
# 2. Two-Qubit Entanglement Circuits
# ---------------------------------------------------------------------------


def build_two_qubit_entanglement_circuit(
    device: Any,
    wires: Tuple[int, int] = (0, 1),
) -> qml.QNode:
    """
    Construct a QNode evaluating a two-qubit rotation and CNOT entanglement circuit:
      RY(theta0) on wire 0, RY(theta1) on wire 1, CNOT(wire 0 -> wire 1).
      Measures Pauli-Z expectation on both qubits.

    Args:
        device: PennyLane Device with at least 2 wires.
        wires: Tuple of (control_wire, target_wire).

    Returns:
        Callable QNode accepting (theta0, theta1) and returning array of shape (2,).
    """
    wire0, wire1 = wires

    @qml.qnode(device)
    def two_qubit_entangled(
        theta0: Union[float, pnp.ndarray],
        theta1: Union[float, pnp.ndarray],
    ) -> List[float]:
        qml.RY(theta0, wires=wire0)
        qml.RY(theta1, wires=wire1)
        qml.CNOT(wires=[wire0, wire1])
        return [qml.expval(qml.PauliZ(wire0)), qml.expval(qml.PauliZ(wire1))]

    return two_qubit_entangled


def build_bell_state_circuit(
    device: Any,
    wires: Tuple[int, int] = (0, 1),
) -> qml.QNode:
    """
    Construct a QNode preparing the canonical Bell state |Phi+> = (|00> + |11>) / sqrt(2):
      Hadamard(wire 0) -> CNOT(wire 0 -> wire 1).
      Measures [<Z0>, <Z1>, <Z0 * Z1>].

    Analytical expectations:
      <Z0> = 0.0
      <Z1> = 0.0
      <Z0 * Z1> = 1.0 (perfect correlation)

    Args:
        device: PennyLane Device with at least 2 wires.
        wires: Tuple of (control_wire, target_wire).

    Returns:
        Callable QNode returning [expval(Z0), expval(Z1), expval(Z0 @ Z1)].
    """
    w0, w1 = wires

    @qml.qnode(device)
    def bell_circuit() -> List[float]:
        qml.Hadamard(wires=w0)
        qml.CNOT(wires=[w0, w1])
        return [
            qml.expval(qml.PauliZ(w0)),
            qml.expval(qml.PauliZ(w1)),
            qml.expval(qml.PauliZ(w0) @ qml.PauliZ(w1)),
        ]

    return bell_circuit


# ---------------------------------------------------------------------------
# 3. Parameterized Differentiable Circuit
# ---------------------------------------------------------------------------


def build_parameterized_circuit(
    device: Any,
    n_wires: int = 2,
    diff_method: str = "best",
) -> qml.QNode:
    """
    Construct a parameterized, differentiable quantum circuit suitable for autodiff:
      Layer 1: RY(params[0..n-1]) across all wires.
      Layer 2: Linear CNOT entanglement chain (wire_i -> wire_{i+1}).
      Layer 3: RY(params[n..2n-1]) across all wires.
      Measurement: Pauli-Z expectation value on wire 0.

    Total parameters required: 2 * n_wires.

    Args:
        device: PennyLane Device with at least n_wires.
        n_wires: Number of active wires used in the circuit.
        diff_method: Differentiation strategy ('best', 'backprop', 'parameter-shift').

    Returns:
        Callable QNode accepting parameter vector of length 2 * n_wires.
    """
    n_params_expected = 2 * n_wires

    @qml.qnode(device, diff_method=diff_method)
    def parameterized_ansatz(params: Union[np.ndarray, pnp.ndarray]) -> float:
        # Layer 1: Rotations
        for i in range(n_wires):
            qml.RY(params[i], wires=i)

        # Layer 2: Entanglement chain
        for i in range(n_wires - 1):
            qml.CNOT(wires=[i, i + 1])

        # Layer 3: Rotations
        for i in range(n_wires):
            qml.RY(params[n_wires + i], wires=i)

        return qml.expval(qml.PauliZ(0))

    return parameterized_ansatz


# ---------------------------------------------------------------------------
# 4. Gradient Computation and Numerical Verification
# ---------------------------------------------------------------------------


def compute_circuit_gradient(
    circuit: Callable,
    params: Union[np.ndarray, pnp.ndarray, List[float]],
) -> np.ndarray:
    """
    Compute analytical/autodiff gradient vector of a QNode using PennyLane's Autograd engine.

    Args:
        circuit: Differentiable PennyLane QNode returning a scalar expectation value.
        params: 1D parameter array.

    Returns:
        1D gradient vector as np.ndarray.
    """
    pnp_params = pnp.array(params, requires_grad=True, dtype=float)
    grad_fn = qml.grad(circuit)
    grad_val = grad_fn(pnp_params)
    return np.asarray(grad_val, dtype=float)


def finite_difference_gradient(
    circuit: Callable,
    params: Union[np.ndarray, List[float]],
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Compute numerical gradient using central finite-difference approximation:
      df/d(theta_i) ~ [f(theta + eps * e_i) - f(theta - eps * e_i)] / (2 * eps)

    Args:
        circuit: Scalar-valued callable circuit.
        params: 1D parameter vector.
        epsilon: Finite difference perturbation step size (default 1e-5).

    Returns:
        1D numerical gradient vector as np.ndarray.
    """
    p_arr = np.asarray(params, dtype=float)
    n_dim = len(p_arr)
    grad_fd = np.zeros(n_dim, dtype=float)

    for i in range(n_dim):
        p_plus = p_arr.copy()
        p_minus = p_arr.copy()
        p_plus[i] += epsilon
        p_minus[i] -= epsilon

        f_plus = float(circuit(p_plus))
        f_minus = float(circuit(p_minus))
        grad_fd[i] = (f_plus - f_minus) / (2.0 * epsilon)

    return grad_fd


def verify_gradient_correctness(
    circuit: Callable,
    params: Union[np.ndarray, List[float]],
    epsilon: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Rigorously compare PennyLane autodiff gradient with numerical finite difference.

    Args:
        circuit: Differentiable PennyLane QNode.
        params: 1D parameter vector.
        epsilon: Finite difference step.
        atol: Absolute tolerance for np.allclose.
        rtol: Relative tolerance for np.allclose.

    Returns:
        Dictionary with autodiff_gradient, finite_diff_gradient, max_abs_diff,
        max_rel_diff, and is_correct flag.
    """
    p_arr = np.asarray(params, dtype=float)
    g_ad = compute_circuit_gradient(circuit, p_arr)
    g_fd = finite_difference_gradient(circuit, p_arr, epsilon=epsilon)

    abs_diff = np.abs(g_ad - g_fd)
    max_abs_diff = float(np.max(abs_diff))

    # Compute relative diff with epsilon guard
    denom = np.maximum(np.abs(g_ad), np.abs(g_fd))
    rel_diff = np.zeros_like(abs_diff)
    mask = denom > 1e-9
    rel_diff[mask] = abs_diff[mask] / denom[mask]
    max_rel_diff = float(np.max(rel_diff))

    is_correct = bool(np.allclose(g_ad, g_fd, atol=atol, rtol=rtol))

    return {
        "is_correct": is_correct,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "autodiff_gradient": [round(float(v), 8) for v in g_ad],
        "finite_diff_gradient": [round(float(v), 8) for v in g_fd],
        "all_finite": bool(np.all(np.isfinite(g_ad))),
        "is_nonzero": bool(np.any(np.abs(g_ad) > 1e-6)),
    }


# ---------------------------------------------------------------------------
# 5. Deterministic Execution Verification
# ---------------------------------------------------------------------------


def verify_circuit_determinism(
    circuit: Callable,
    *args,
    n_runs: int = 5,
    atol: float = 1e-10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Execute a circuit repeatedly with identical inputs to confirm exact determinism.

    Args:
        circuit: Callable quantum circuit QNode.
        *args: Positional arguments forwarded to circuit.
        n_runs: Number of consecutive executions (default 5).
        atol: Absolute numerical tolerance.
        **kwargs: Keyword arguments forwarded to circuit.

    Returns:
        Dictionary with is_deterministic flag, run outputs, and max variation.
    """
    outputs = []
    for _ in range(n_runs):
        out = circuit(*args, **kwargs)
        # Convert to numpy array or float
        if isinstance(out, (list, tuple)):
            out_val = [float(v) for v in out]
        else:
            out_val = float(out)
        outputs.append(out_val)

    # Calculate maximum pairwise difference across runs
    arr = np.array(outputs)
    if arr.ndim == 1:
        variation = float(np.max(np.abs(arr - arr[0])))
    else:
        variation = float(np.max(np.abs(arr - arr[0, :])))

    is_deterministic = bool(variation <= atol)

    return {
        "is_deterministic": is_deterministic,
        "n_runs": n_runs,
        "max_variation": variation,
        "sample_output": outputs[0],
    }
