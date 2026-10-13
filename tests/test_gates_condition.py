"""Tests for classically conditioned gates, :class:`qibo.gates.Condition`."""

import copy
import json
import pickle

import numpy as np
import pytest
import sympy

from qibo import Circuit, gates
from qibo.measurements import MeasurementResult, MeasurementSymbol, MeasurementValue


def feed_forward_circuit():
    """H on qubit 0, mid-circuit measurement, conditional X on qubit 1."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result, gates.X(1)))
    circuit.add(gates.M(0, 1))
    return circuit, result


def names(gate_list):
    return [(g.__class__.__name__, g.qubits) for g in gate_list]


# ---------------------------------------------------------------------------
# Measurement results as conditions
# ---------------------------------------------------------------------------


def test_add_collapsing_measurement_returns_result():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    assert isinstance(result, MeasurementResult)
    assert circuit.queue[0].register_name == "register0"


def test_result_bits():
    result = Circuit(3).add(gates.M(0, 1, 2, collapse=True))
    assert result[0] is result.symbols[0]
    assert result[-1] is result.symbols[2]
    assert isinstance(result[1], MeasurementSymbol)
    assert list(result) == result.symbols


def test_result_boolean_operators():
    circuit = Circuit(4)
    a = circuit.add(gates.M(0, collapse=True))
    b = circuit.add(gates.M(1, collapse=True))
    register = circuit.add(gates.M(2, 3, collapse=True))
    x, y = a[0], b[0]
    assert ~a == sympy.Not(x)
    assert a & b == sympy.And(x, y)
    assert a | register[1] == sympy.Or(x, register[1])
    assert a ^ b == sympy.Xor(x, y)
    assert register[0] & a == sympy.And(register[0], x)
    assert (a & ~b) | (register[1] ^ a) == sympy.Or(
        sympy.And(x, sympy.Not(y)), sympy.Xor(register[1], x)
    )


def test_result_comparisons():
    circuit = Circuit(3)
    a = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    # single bits compare to the bit itself or its negation
    assert (a == 1) is a[0]
    assert (a == 0) == sympy.Not(a[0])
    assert (a != 0) is a[0]
    # registers give a comparison symbol, reused for equal values
    equals = register == 2
    assert isinstance(equals, MeasurementValue)
    assert equals.result is register and equals.value == 2
    assert (register == "10") is equals
    assert (register != 2) == sympy.Not(equals)
    # results compared with results keep identity semantics and stay hashable
    assert register.__eq__(register) is NotImplemented
    assert a != register
    assert len({a, register, a}) == 2


@pytest.mark.parametrize(
    "value,error",
    [(4, ValueError), (-1, ValueError), ("1", ValueError), ("12", ValueError)],
)
def test_result_invalid_comparisons(value, error):
    register = Circuit(2).add(gates.M(0, 1, collapse=True))
    with pytest.raises(error):
        _ = register == value


def test_register_used_as_bit():
    circuit = Circuit(3)
    a = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    for operation in (
        lambda: ~register,
        lambda: register & a,
        lambda: a & register,
        lambda: a[0] | register,
        lambda: gates.Condition(register, gates.X(0)),
    ):
        with pytest.raises(TypeError, match="cannot be used as a single bit"):
            operation()


def test_condition_expressions():
    circuit = Circuit(4)
    a = circuit.add(gates.M(0, collapse=True))
    b = circuit.add(gates.M(1, collapse=True))
    register = circuit.add(gates.M(2, 3, collapse=True))
    condition = gates.Condition(a, gates.X(0))
    assert condition.expression is a[0]
    assert condition.results == [a]
    condition = gates.Condition((register == 3) & ~b | a, gates.X(0))
    assert condition.atoms == [a[0], b[0], register == 3]
    assert condition.results == [a, b, register]
    assert condition.target is None and condition.cases is None
    assert names(condition.body) == [("X", (0,))]
    assert condition.draw_label == "if"


@pytest.mark.parametrize(
    "build,error,match",
    [
        (lambda a, x: a[0] == 1, TypeError, "Python ``bool``"),
        (lambda a, x: sympy.true, ValueError, "always True"),
        (lambda a, x: a[0] & x, TypeError, "can only depend on measurement"),
        (lambda a, x: sympy.Eq(a[0], 1), TypeError, "must be a measured bit"),
        (lambda a, x: a[0] & sympy.Eq(x, 1), TypeError, "sympy relations"),
        (lambda a, x: "register0", TypeError, "must be a measured bit"),
        (lambda a, x: 2 * a[0], TypeError, "must be a measured bit"),
    ],
)
def test_condition_invalid_expressions(build, error, match):
    a = Circuit(1).add(gates.M(0, collapse=True))
    with pytest.raises(error, match=match):
        gates.Condition(build(a, sympy.Symbol("x")), gates.X(0))


def test_condition_former_value_argument():
    result = Circuit(2).add(gates.M(0, collapse=True))
    with pytest.raises(TypeError, match="result == value"):
        gates.Condition(result[0], 1, gates.X(1))


def test_condition_qubits():
    result = Circuit(3).add(gates.M(2, collapse=True))
    condition = gates.Condition(result, gates.CNOT(1, 0))
    assert condition.target_qubits == (0, 1)
    assert condition.control_qubits == ()
    assert condition.qubits == (0, 1)


@pytest.mark.parametrize(
    "gate", [gates.M(1), gates.CallbackGate(None)], ids=["measurement", "callback"]
)
def test_condition_invalid_gates(gate):
    result = Circuit(2).add(gates.M(0, collapse=True))
    with pytest.raises(NotImplementedError):
        gates.Condition(result, gate)
    with pytest.raises(TypeError):
        gates.Condition(result, "x")


# ---------------------------------------------------------------------------
# Integration with Circuit
# ---------------------------------------------------------------------------


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
    circuit.add(gates.Condition(result, gates.X(1)))
    with pytest.raises(NotImplementedError, match="Simulation of ``Condition``"):
        circuit()


def test_condition_turns_final_measurements_into_mid_circuit():
    circuit = Circuit(3)
    a = circuit.add(gates.M(0))
    b = circuit.add(gates.M(1))
    assert len(circuit.measurements) == 2
    circuit.add(gates.Condition(a & ~b, gates.X(2)))
    assert circuit.queue[0].collapse and circuit.queue[1].collapse
    assert not circuit.measurements
    assert circuit.has_collapse


def test_condition_parameters_can_be_updated():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.RX(0, 0.1))
    circuit.add(gates.Condition(result, gates.RY(1, 0.2)))
    circuit.set_parameters([0.3, 0.4])
    assert circuit.queue[-1].gates[0].parameters == (0.4,)
    assert circuit.get_parameters() == [(0.3,), (0.4,)]


def test_condition_moments_after_measurements():
    circuit = Circuit(3)
    a = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.H(1))
    circuit.add(gates.H(1))
    b = circuit.add(gates.M(1, collapse=True))
    circuit.add(gates.Condition(a | b, gates.X(2)))
    moments = circuit.queue.moments
    # the condition on qubit 2 waits for the second measurement (moment 2)
    assert isinstance(moments[3][2], gates.Condition)
    assert all(moment[2] is None for moment in moments[:3])


def test_condition_draw():
    circuit, _ = feed_forward_circuit()
    assert circuit.diagram() == "0: ─H─M─────────M─\n1: ─────if[─X─]─M─"


def test_condition_on_qubits():
    small, result = feed_forward_circuit()
    large = Circuit(4)
    large.add(small.on_qubits(3, 1))
    condition = large.queue[2]
    assert condition.results == [result]
    assert condition.qubits == (1,)
    assert "if (register0[0]) {\n  x q[1];\n}" in large.to_qasm3()
    assert "register0[0] = measure q[3];" in large.to_qasm3()


def expression_circuit():
    circuit = Circuit(4)
    circuit.add([gates.H(0), gates.H(1), gates.H(2)])
    a = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    circuit.add(
        gates.Condition(
            (register != 1) & ~a | (register[1] ^ a), gates.X(3), default=gates.Z(3)
        )
    )
    circuit.add(gates.M(3))
    return circuit


def test_condition_raw_roundtrip():
    circuit = expression_circuit()
    raw = json.loads(json.dumps(circuit.raw))
    condition = raw["queue"][5]["condition"]
    assert set(condition) == {"or"}
    conjunction = next(arg["and"] for arg in condition["or"] if "and" in arg)
    assert {"not": [{"bit": [0, 0]}]} in conjunction
    new = Circuit.from_dict(raw)
    assert new.to_qasm3() == circuit.to_qasm3()
    assert new.queue[5].results == [new.queue[3].result, new.queue[4].result]
    with pytest.raises(NotImplementedError):
        _ = circuit.queue[5].raw


@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_condition_pickle_and_deepcopy(method):
    """Copied symbols keep their names, but must stay linked to the copied
    measurements instead of being confused with the original ones."""
    circuit = expression_circuit()
    if method == "pickle":
        new = pickle.loads(pickle.dumps(circuit))
    else:
        new = copy.deepcopy(circuit)
    assert new.queue[5].results == [new.queue[3].result, new.queue[4].result]
    assert [atom.name for atom in new.queue[5].atoms] == [
        atom.name for atom in circuit.queue[5].atoms
    ]
    assert new.to_qasm3() == circuit.to_qasm3()


def test_measurement_symbols_are_distinct():
    result = Circuit(1).add(gates.M(0, collapse=True))
    copied = pickle.loads(pickle.dumps(result[0]))
    assert copied.name == result[0].name
    assert copied != result[0]
    # each symbol is still found in sympy expressions that contain it
    assert (copied & result[0]).subs(copied, True) == result[0]


def test_condition_raw_unresolved():
    result = Circuit(1).add(gates.M(0, collapse=True))
    circuit = Circuit(1)
    circuit.add(gates.Condition(result, gates.X(0)))
    with pytest.raises(ValueError):
        _ = circuit.raw


def test_condition_deep_copy():
    circuit = expression_circuit()
    new = circuit.copy(deep=True)
    old_results = circuit.queue[5].results
    new_results = new.queue[5].results
    assert new_results == [new.queue[3].result, new.queue[4].result]
    assert all(r is not s for r, s in zip(new_results, old_results))
    assert new.queue[5].gates[0] is not circuit.queue[5].gates[0]
    assert new.to_qasm3() == circuit.to_qasm3()


def test_condition_light_cone():
    circuit = Circuit(4)
    circuit.add(gates.H(0))
    a = circuit.add(gates.M(0, collapse=True))
    b = circuit.add(gates.M(1, collapse=True))
    circuit.add(gates.H(3))
    circuit.add(gates.Condition(a & b, gates.X(2)))
    small, qubit_map = circuit.light_cone(2)
    assert qubit_map == {0: 0, 1: 1, 2: 2}
    assert [g.__class__.__name__ for g in small.queue] == ["H", "M", "M", "Condition"]
    assert small.queue[3].results == [small.queue[1].result, small.queue[2].result]


def test_condition_decompose():
    circuit = Circuit(3)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result, gates.SX(1)))
    new = circuit.decompose()
    condition = new.queue[1]
    assert isinstance(condition, gates.Condition)
    assert condition.results == [new.queue[0].result]
    assert [g.__class__.__name__ for g in condition.gates] == ["RX"]


def test_condition_not_fused():
    circuit = Circuit(2)
    circuit.add(gates.H(1))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.Condition(result, gates.X(1)))
    circuit.add(gates.H(1))
    fused = circuit.fuse()
    names_ = [g.__class__.__name__ for g in fused.queue]
    assert names_.index("Condition") > names_.index("M")
    assert fused.to_qasm3() == circuit.to_qasm3()


# ---------------------------------------------------------------------------
# Bodies, if/else and switch
# ---------------------------------------------------------------------------


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
        condition = gates.Condition(result, body)
        assert names(condition.gates) == expected
    condition = gates.Condition(result, small)
    assert names(condition.gates) == [("H", (0,)), ("CNOT", (0, 1))]
    assert condition.gates[0] is small.queue[0]


def test_condition_body_circuit_with_measurement():
    result = Circuit(2).add(gates.M(0, collapse=True))
    body = Circuit(2)
    body.add(gates.M(1))
    with pytest.raises(NotImplementedError):
        gates.Condition(result, body)


def test_condition_empty_default():
    result = Circuit(2).add(gates.M(0, collapse=True))
    assert gates.Condition(result, gates.X(1), default=[]).default is None


def test_condition_if_else():
    result = Circuit(2).add(gates.M(0, collapse=True))
    condition = gates.Condition(result, gates.X(1), default=[gates.H(1), gates.Z(0)])
    assert names(condition.body) == [("X", (1,))]
    assert names(condition.default) == [("H", (1,)), ("Z", (0,))]
    assert condition.branches == [condition.body, condition.default]
    assert condition.qubits == (0, 1)


@pytest.mark.parametrize(
    "cases,expected",
    [
        ({0: "x", (1, "10"): "ht"}, [((0,), "x"), ((1, 2), "ht")]),
        ({3: "x", np.int64(1): "ht"}, [((3,), "x"), ((1,), "ht")]),
        ({"11": "x", (0, 2): "ht"}, [((3,), "x"), ((0, 2), "ht")]),
    ],
)
def test_condition_switch(cases, expected):
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    bodies = {"x": [gates.X(2)], "ht": [gates.H(2), gates.T(2)]}
    cases = {values: bodies[body] for values, body in cases.items()}
    condition = gates.Condition(result, cases, default=gates.Y(2))
    assert condition.target is result and condition.expression is None
    assert condition.cases == [(values, bodies[body]) for values, body in expected]
    assert names(condition.default) == [("Y", (2,))]
    assert condition.draw_label == "switch"
    assert condition.results == [result]


def test_condition_switch_on_bit():
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    condition = gates.Condition(result[1], {0: [], 1: gates.X(2)})
    assert condition.target is result[1]
    assert condition.cases[0] == ((0,), [])
    assert condition.qubits == (2,)


@pytest.mark.parametrize(
    "target,cases,error",
    [
        ("result", {0: gates.X(1), (1, 0): gates.Z(1)}, ValueError),
        ("result", {}, ValueError),
        ("result", {(): gates.X(1)}, ValueError),
        ("result", {2: gates.X(1)}, ValueError),
        ("result", {0: [], 1: []}, ValueError),
        ("result", {0: "x"}, TypeError),
        ("expression", {0: gates.X(1)}, TypeError),
        ("string", {0: gates.X(1)}, TypeError),
    ],
)
def test_condition_switch_errors(target, cases, error):
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    other = circuit.add(gates.M(1, collapse=True))
    target = {"result": result, "expression": result & other, "string": "a"}[target]
    with pytest.raises(error):
        gates.Condition(target, cases)


def nested_circuit():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    first = circuit.add(gates.M(0, collapse=True))
    second = circuit.add(gates.M(1, collapse=True))
    inner = gates.Condition(second, gates.X(2), default=gates.RY(2, 0.3))
    circuit.add(gates.Condition(first, [gates.H(2), inner]))
    circuit.add(gates.M(2))
    return circuit


def test_condition_nested():
    circuit = nested_circuit()
    condition = circuit.queue[4]
    assert [c.results for c in condition._nested_conditions()] == [
        [circuit.queue[2].result],
        [circuit.queue[3].result],
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
    assert raw["queue"][4]["body"][1]["condition"] == {"bit": [1, 0]}
    loaded = Circuit.from_dict(raw)
    copied = circuit.copy(deep=True)
    for new in (loaded, copied):
        assert new.to_qasm3() == circuit.to_qasm3()
        inner = new.queue[4].gates[1]
        assert inner.results == [new.queue[3].result]
    with pytest.raises(ValueError):
        gates.Gate.from_dict(raw["queue"][4])


@pytest.mark.parametrize("target", ["register", "bit"])
def test_condition_switch_raw_roundtrip(target):
    circuit = Circuit(3)
    result = circuit.add(gates.M(0, 1, collapse=True))
    target = result if target == "register" else result[1]
    circuit.add(gates.Condition(target, {0: gates.X(2), (1,): []}, default=gates.Y(2)))
    raw = json.loads(json.dumps(circuit.raw))
    loaded = Circuit.from_dict(raw)
    assert loaded.queue[1].cases[1][0] == (1,)
    for compatible in (True, False):
        assert loaded.to_qasm3(qiskit_compatible=compatible) == circuit.to_qasm3(
            qiskit_compatible=compatible
        )
