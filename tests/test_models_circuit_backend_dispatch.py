"""Circuit-level hardware capabilities must not alter native simulation guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qibo import Circuit, gates
from qibo.backends import NumpyBackend, _Global


def conditional_circuit(kind):
    circuit = Circuit(2)
    measured = circuit.add(gates.M(0, collapse=True))
    if kind == "if_else":
        circuit.add(gates.IfElse(measured.is_one(), then_do=gates.X(1)))
    else:
        circuit.add(gates.Switch(measured).case(1, gates.X(1)).default([]))
    circuit.add(gates.M(1))
    return circuit


def adapter(monkeypatch, *, control_flow=True, own_transpilation=True):
    backend = NumpyBackend()
    backend.name = "external-hardware"
    backend.supports_circuit_control_flow = control_flow
    backend.handles_circuit_transpilation = own_transpilation
    backend.execute_circuit = Mock(return_value=object())
    monkeypatch.setattr(_Global, "_backend", backend)
    return backend


@pytest.mark.parametrize("kind", ["if_else", "switch"])
def test_circuit_level_feedback_bypasses_global_transpiler(monkeypatch, kind):
    backend = adapter(monkeypatch)
    circuit = conditional_circuit(kind)
    forbidden = Mock(side_effect=AssertionError("Global transpiler must not be consulted"))
    monkeypatch.setattr(_Global, "transpiler", forbidden)
    result = circuit(nshots=23)
    assert result is backend.execute_circuit.return_value
    backend.execute_circuit.assert_called_once_with(circuit, None, 23)
    forbidden.assert_not_called()


def test_nonconditional_circuit_also_uses_backend_pipeline(monkeypatch):
    backend = adapter(monkeypatch)
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    monkeypatch.setattr(_Global, "_transpiler", Mock(side_effect=AssertionError("No global passes")))
    assert circuit() is backend.execute_circuit.return_value
    backend.execute_circuit.assert_called_once_with(circuit, None, 1000)


def test_transpilation_ownership_does_not_implicitly_enable_feedback(monkeypatch):
    backend = adapter(monkeypatch, control_flow=False)
    with pytest.raises(NotImplementedError, match="circuit-level"):
        conditional_circuit("if_else")()
    backend.execute_circuit.assert_not_called()


def test_native_gate_evaluation_remains_unsupported_for_hardware(monkeypatch):
    backend = adapter(monkeypatch)
    operation = conditional_circuit("if_else").queue[1]
    with pytest.raises(NotImplementedError, match="only NumPy and qibojit-numba"):
        operation.apply(backend, state=None, nqubits=2)


def test_other_backends_keep_global_transpilation(monkeypatch):
    backend = adapter(monkeypatch, control_flow=False, own_transpilation=False)
    circuit = Circuit(1)
    translated = Circuit(1)
    transpiler = Mock(return_value=(translated, {}))
    monkeypatch.setattr(_Global, "_transpiler", transpiler)
    circuit(nshots=7)
    transpiler.assert_called_once_with(circuit)
    backend.execute_circuit.assert_called_once_with(translated, None, 7)


def test_compiled_simulator_does_not_bypass_hardware_selection(monkeypatch):
    backend = adapter(monkeypatch)
    circuit = Circuit(1)
    executor = Mock(side_effect=AssertionError("Must not simulate"))
    circuit.compiled = SimpleNamespace(executor=executor)
    with pytest.raises(NotImplementedError, match="Compiled simulator"):
        circuit()
    executor.assert_not_called()
    backend.execute_circuit.assert_not_called()
