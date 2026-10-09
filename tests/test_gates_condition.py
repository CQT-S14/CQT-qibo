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


# ---------------------------------------------------------------------------
# Several gates per branch, if/else and switch
# ---------------------------------------------------------------------------


def names(gate_list):
    return [(g.__class__.__name__, g.qubits) for g in gate_list]


def test_condition_body_forms():
    result = Circuit(3).add(gates.M(0, collapse=True))
    small = Circuit(2)
    small.add(gates.H(0))
    small.add(gates.CNOT(0, 1))
    expected = [("H", (1,)), ("CNOT", (1, 2))]
    for body in (
        [gates.H(1), gates.CNOT(1, 2)],
        (gates.H(1), gates.CNOT(1, 2)),
        [gates.H(1), [gates.CNOT(1, 2)]],
        small.on_qubits(1, 2),
    ):
        condition = gates.Condition(result.symbols[0], 1, body)
        assert names(condition.gates) == expected
    condition = gates.Condition(result.symbols[0], 1, small)
    assert names(condition.gates) == [("H", (0,)), ("CNOT", (0, 1))]
    assert condition.gates[0] is small.queue[0]


def test_condition_body_circuit_with_measurement():
    result = Circuit(2).add(gates.M(0, collapse=True))
    body = Circuit(2)
    body.add(gates.M(1))
    with pytest.raises(NotImplementedError):
        gates.Condition(result.symbols[0], 1, body)


def test_condition_if_else():
    result = Circuit(2).add(gates.M(0, collapse=True))
    condition = gates.Condition(
        result.symbols[0], 1, gates.X(1), default=[gates.H(1), gates.Z(0)]
    )
    assert condition.cases == [((1,), condition.gates[:1])]
    assert names(condition.default) == [("H", (1,)), ("Z", (0,))]
    assert condition.qubits == (0, 1)
    assert condition.draw_label == "if"


def test_condition_switch_forms():
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    x, h, t = gates.X(2), gates.H(2), gates.T(2)
    from_dict = gates.Condition(result, {0: x, (1, "10"): [h, t]}, default=gates.Y(2))
    from_lists = gates.Condition(
        result, [0, (1, "10")], [x, [h, t]], default=gates.Y(2)
    )
    for condition in (from_dict, from_lists):
        assert condition.cases == [((0,), [x]), ((1, 2), [h, t])]
        assert names(condition.default) == [("Y", (2,))]
        assert condition.draw_label == "switch"


def test_condition_switch_empty_case():
    result = Circuit(2).add(gates.M(0, collapse=True))
    condition = gates.Condition(result.symbols[0], {0: [], 1: gates.X(1)})
    assert condition.cases[0] == ((0,), [])
    assert condition.qubits == (1,)


@pytest.mark.parametrize(
    "value,body,error",
    [
        ({0: gates.X(1)}, gates.X(1), ValueError),
        ([0, 1], [gates.X(1)], ValueError),
        ([0, 1], gates.X(1), ValueError),
        ({0: gates.X(1), (1, 0): gates.Z(1)}, None, ValueError),
        ({}, None, ValueError),
        ({(): gates.X(1)}, None, ValueError),
        (1, None, ValueError),
        (1, [], ValueError),
        ({0: [], 1: []}, None, ValueError),
    ],
)
def test_condition_switch_errors(value, body, error):
    result = Circuit(2).add(gates.M(0, collapse=True))
    with pytest.raises(error):
        gates.Condition(result.symbols[0], value, body)


def nested_circuit():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    first = circuit.add(gates.M(0, collapse=True))
    second = circuit.add(gates.M(1, collapse=True))
    inner = gates.Condition(second.symbols[0], 1, gates.X(2), default=gates.RY(2, 0.3))
    circuit.add(gates.Condition(first.symbols[0], 1, [gates.H(2), inner]))
    circuit.add(gates.M(2))
    return circuit


def test_condition_nested():
    circuit = nested_circuit()
    condition = circuit.queue[4]
    assert [c.result for c in condition._nested_conditions()] == [
        circuit.queue[2].result,
        circuit.queue[3].result,
    ]
    assert names(condition._nested_gates()) == [("H", (2,)), ("X", (2,)), ("RY", (2,))]
    # parameters of nested gates are registered
    circuit.set_parameters([0.7])
    assert condition.gates[1].default[0].parameters == (0.7,)
    assert circuit.diagram() == (
        "0: ─H─M─────────────────────────\n"
        "1: ─H─M─────────────────────────\n"
        "2: ─────if[─H─if[─X─][─RY─]─]─M─"
    )


def test_condition_nested_raw_and_copy():
    circuit = nested_circuit()
    raw = json.loads(json.dumps(circuit.raw))
    assert raw["queue"][4]["cases"][0]["gates"][1]["source"] == {
        "measurement": 1,
        "bit": 0,
    }
    loaded = Circuit.from_dict(raw)
    copied = circuit.copy(deep=True)
    for new in (loaded, copied):
        assert new.to_qasm3() == circuit.to_qasm3()
        inner = new.queue[4].gates[1]
        assert inner.result is new.queue[3].result
    with pytest.raises(ValueError):
        gates.Gate.from_dict(raw["queue"][4])


def test_condition_switch_raw_roundtrip():
    circuit = Circuit(3)
    result = circuit.add(gates.M(0, 1, collapse=True))
    circuit.add(
        gates.Condition(result, {0: gates.X(2), (1, 2): []}, default=gates.Y(2))
    )
    loaded = Circuit.from_dict(json.loads(json.dumps(circuit.raw)))
    assert loaded.queue[1].cases[1][0] == (1, 2)
    assert loaded.to_qasm3(switch_statement=True) == circuit.to_qasm3(
        switch_statement=True
    )
