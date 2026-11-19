"""
tests/test_environment.py
=========================
Day 1 smoke test for QML Benchmark Studio.

Verifies that all core dependencies are importable and that the PennyLane
default.qubit simulator executes a basic quantum circuit correctly.
"""

import pytest


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_import_numpy():
    import numpy as np  # noqa: F401
    assert np.__version__, "numpy imported but version string is empty"


def test_import_pandas():
    import pandas as pd  # noqa: F401
    assert pd.__version__, "pandas imported but version string is empty"


def test_import_sklearn():
    import sklearn  # noqa: F401
    assert sklearn.__version__, "scikit-learn imported but version string is empty"


def test_import_pennylane():
    import pennylane as qml  # noqa: F401
    assert qml.__version__, "pennylane imported but version string is empty"


def test_import_fastapi():
    import fastapi  # noqa: F401
    assert fastapi.__version__, "fastapi imported but version string is empty"


def test_import_pytest():
    import pytest as pt  # noqa: F401
    assert pt.__version__, "pytest imported but version string is empty"


# ---------------------------------------------------------------------------
# PennyLane simulator smoke test
# ---------------------------------------------------------------------------

def test_pennylane_default_qubit_hadamard():
    """
    Create a 1-qubit circuit on default.qubit, apply a Hadamard gate, and
    measure the expectation value of Pauli-Z.

    After H|0>, the state is (|0> + |1>) / sqrt(2).
    The expectation value of Z in this state is:
        <Z> = <+|Z|+> = 0

    We verify the result is approximately zero (within floating-point tolerance).
    """
    import pennylane as qml
    import numpy as np

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    result = circuit()
    assert abs(result) < 1e-6, (
        f"Expected <Z> ≈ 0 after Hadamard on |0>, got {result}"
    )
