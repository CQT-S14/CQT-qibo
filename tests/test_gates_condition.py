"""Tests for classically conditioned gates, :class:`qibo.gates.Condition`."""

import json

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.measurements import MeasurementResult


def feed_forward_circuit():
    """H on qubit 0, mid-circuit measurement, conditional X on qubit 1."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(1)))
    circuit.add(gates.M(0, 1))
    return circuit, result


def test_add_collapsing_measurement_returns_result():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    assert isinstance(result, MeasurementResult)
    assert circuit.queue[0].register_name == "register0"


@pytest.mark.parametrize("source", ["symbol", "result"])
def test_condition_source(source):
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, 1, collapse=True))
    if source == "symbol":
        condition = gates.Condition(result.symbols[1], 1, gates.X(0))
        assert condition.bit == 1
        assert condition.nbits == 1
    else:
        condition = gates.Condition(result, 3, gates.X(0))
        assert condition.bit is None
        assert condition.nbits == 2
    assert condition.result is result


def test_condition_invalid_source():
    with pytest.raises(TypeError):
        gates.Condition("register0", 1, gates.X(0))


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (3, 3), (True, 1), (np.int64(2), 2), ("10", 2), ("01", 1)],
)
def test_condition_values(value, expected):
    result = Circuit(2).add(gates.M(0, 1, collapse=True))
    condition = gates.Condition(result, value, gates.X(0))
    assert condition.cases == [((expected,), condition.gates)]


@pytest.mark.parametrize(
    "value,error",
    [
        (4, ValueError),
        (-1, ValueError),
        ("1", ValueError),
        ("12", ValueError),
        (1.0, TypeError),
        (None, TypeError),
    ],
)
def test_condition_invalid_values(value, error):
    result = Circuit(2).add(gates.M(0, 1, collapse=True))
    with pytest.raises(error):
        gates.Condition(result, value, gates.X(0))


def test_condition_bit_value_range():
    result = Circuit(2).add(gates.M(0, 1, collapse=True))
    with pytest.raises(ValueError):
        gates.Condition(result.symbols[0], 2, gates.X(0))


def test_condition_qubits():
    result = Circuit(3).add(gates.M(2, collapse=True))
    condition = gates.Condition(result.symbols[0], 1, gates.CNOT(1, 0))
    assert condition.target_qubits == (0, 1)
    assert condition.control_qubits == ()
    assert condition.qubits == (0, 1)


@pytest.mark.parametrize(
    "gate", [gates.M(1), gates.CallbackGate(None)], ids=["measurement", "callback"]
)
def test_condition_invalid_gates(gate):
    result = Circuit(2).add(gates.M(0, collapse=True))
    with pytest.raises(NotImplementedError):
        gates.Condition(result.symbols[0], 1, gate)
    with pytest.raises(TypeError):
        gates.Condition(result.symbols[0], 1, "x")


def test_condition_unsupported_operations():
    circuit, _ = feed_forward_circuit()
    condition = circuit.queue[2]
    with pytest.raises(NotImplementedError):
        condition.matrix()
    with pytest.raises(NotImplementedError):
        condition.controlled_by(0)
    with pytest.raises(NotImplementedError):
        condition.dagger()
    with pytest.raises(NotImplementedError):
        circuit.invert()
    with pytest.raises(NotImplementedError):
        circuit(nshots=10)
    with pytest.raises(NotImplementedError):
        circuit.to_qasm()


@pytest.mark.parametrize("density_matrix", [False, True])
def test_condition_simulation_error(density_matrix):
    """The error is raised before any check on the measurements."""
    circuit = Circuit(2, density_matrix=density_matrix)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(1)))
    with pytest.raises(NotImplementedError, match="Simulation of ``Condition``"):
        circuit()


def test_condition_turns_final_measurement_into_mid_circuit():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0))
    assert circuit.measurements
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(1)))
    assert circuit.queue[0].collapse
    assert not circuit.measurements
    assert circuit.has_collapse


def test_condition_parameters_can_be_updated():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.RX(0, 0.1))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.RY(1, 0.2)))
    circuit.set_parameters([0.3, 0.4])
    assert circuit.queue[-1].gates[0].parameters == (0.4,)
    assert circuit.get_parameters() == [(0.3,), (0.4,)]


def test_condition_moments_after_measurement():
    circuit, _ = feed_forward_circuit()
    moments = circuit.queue.moments
    measured = next(i for i, m in enumerate(moments) if isinstance(m[0], gates.M))
    conditioned = next(
        i for i, m in enumerate(moments) if isinstance(m[1], gates.Condition)
    )
    assert conditioned > measured
    assert circuit.depth == 4


def test_condition_draw():
    circuit, _ = feed_forward_circuit()
    assert circuit.diagram() == "0: ─H─M─────────M─\n1: ─────if[─X─]─M─"


def test_condition_on_qubits():
    small, result = feed_forward_circuit()
    large = Circuit(4)
    large.add(small.on_qubits(3, 1))
    condition = large.queue[2]
    assert condition.result is result
    assert condition.qubits == (1,)
    assert "if (register0[0]) {\n  x q[1];\n}" in large.to_qasm3()
    assert "register0[0] = measure q[3];" in large.to_qasm3()


def test_condition_raw_roundtrip():
    circuit, _ = feed_forward_circuit()
    raw = json.loads(json.dumps(circuit.raw))
    assert raw["queue"][2]["source"] == {"measurement": 0, "bit": 0}
    new = Circuit.from_dict(raw)
    assert new.to_qasm3() == circuit.to_qasm3()
    assert new.queue[2].result is new.queue[1].result
    with pytest.raises(NotImplementedError):
        _ = circuit.queue[2].raw


def test_condition_raw_unresolved():
    result = Circuit(1).add(gates.M(0, collapse=True))
    circuit = Circuit(1)
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(0)))
    with pytest.raises(ValueError):
        _ = circuit.raw


def test_condition_deep_copy():
    circuit, result = feed_forward_circuit()
    new = circuit.copy(deep=True)
    assert new.queue[1].result is not result
    assert new.queue[2].result is new.queue[1].result
    assert new.queue[2].gates[0] is not circuit.queue[2].gates[0]
    assert new.to_qasm3() == circuit.to_qasm3()


def test_condition_light_cone():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.H(2))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(1)))
    small, qubit_map = circuit.light_cone(1)
    assert qubit_map == {0: 0, 1: 1}
    assert [g.__class__.__name__ for g in small.queue] == ["H", "M", "Condition"]
    assert small.queue[2].result is small.queue[1].result


def test_condition_decompose():
    circuit = Circuit(3)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.SX(1)))
    new = circuit.decompose()
    condition = new.queue[1]
    assert isinstance(condition, gates.Condition)
    assert condition.result is new.queue[0].result
    assert [g.__class__.__name__ for g in condition.gates] == ["RX"]


def test_condition_not_fused():
    circuit = Circuit(2)
    circuit.add(gates.H(1))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result.symbols[0], 1, gates.X(1)))
    circuit.add(gates.H(1))
    fused = circuit.fuse()
    names = [g.__class__.__name__ for g in fused.queue]
    assert names.index("Condition") > names.index("M")
    assert fused.to_qasm3() == circuit.to_qasm3()
