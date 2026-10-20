"""Tests for explicit classical control-flow operations."""

import copy
import json
import pickle

import numpy as np
import pytest
import sympy

from qibo import Circuit, conditions, gates, get_backend, set_backend
from qibo.backends import NumpyBackend
from qibo.measurements import MeasurementResult, MeasurementSymbol, MeasurementValue


def names(gate_list):
    return [(gate.__class__.__name__, gate.qubits) for gate in gate_list]


def feed_forward_circuit():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(gates.IfElse(result.is_one(), then_do=gates.X(1), else_do=gates.Z(1)))
    circuit.add(gates.M(0, 1))
    return circuit, result


def expression_circuit():
    circuit = Circuit(4)
    circuit.add([gates.H(0), gates.H(1), gates.H(2)])
    first = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    expression = conditions.any_of(
        conditions.all_of(register.not_equals(1), first.is_zero()),
        conditions.xor_of(register.bit(1).is_one(), first.is_one()),
    )
    circuit.add(gates.IfElse(expression, then_do=gates.X(3), else_do=[]))
    circuit.add(gates.M(3))
    return circuit


def test_measurement_explicit_api_and_shorthand_equivalence():
    circuit = Circuit(4)
    first = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, 3, collapse=True))

    assert first.is_one() is first.bit(0).is_one()
    assert first.is_zero() == ~first[0]
    assert register.bit(1).is_one() is register[1]
    assert register.bit(1).is_zero() == ~register[1]
    assert register.equals("101") == (register == 5)
    assert register.not_equals(5) == (register != "101")
    assert isinstance(register.equals(5), MeasurementValue)
    assert isinstance(register.bit(0).is_one(), MeasurementSymbol)

    explicit = conditions.any_of(
        conditions.all_of(first.is_one(), register.bit(0).is_zero()),
        register.equals(3),
    )
    shorthand = (first & ~register[0]) | (register == 3)
    assert explicit == shorthand


def test_measurement_explicit_api_errors_are_clear():
    register = Circuit(2).add(gates.M(0, 1, collapse=True))
    with pytest.raises(TypeError, match="cannot be used as a single bit"):
        register.is_one()
    with pytest.raises(TypeError, match="index must be an integer"):
        register.bit("0")
    with pytest.raises(IndexError, match="out of range"):
        register.bit(2)
    with pytest.raises(ValueError, match="2 measured bits"):
        register.equals("1")


def test_condition_helpers_validate_arity():
    with pytest.raises(ValueError, match="at least one"):
        conditions.all_of()
    with pytest.raises(ValueError, match="at least one"):
        conditions.any_of()
    with pytest.raises(ValueError, match="at least two"):
        conditions.xor_of(sympy.true)
    with pytest.raises(TypeError, match="Python bool"):
        conditions.any_of(True, sympy.Symbol("x"))
    with pytest.raises(TypeError, match="Python bool"):
        conditions.not_(False)


def test_ifelse_properties_and_selection():
    circuit = Circuit(4)
    first = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    expression = conditions.all_of(first.is_one(), register.equals("10"))
    operation = gates.IfElse(
        expression,
        then_do=[gates.X(3), gates.T(3)],
        else_do=gates.Z(3),
    )

    assert operation.expression == expression
    assert operation.results == [first, register]
    assert operation.branches == [operation.then_do, operation.else_do]
    assert names(operation.then_do) == [("X", (3,)), ("T", (3,))]
    assert names(operation.else_do) == [("Z", (3,))]
    assert operation.select({first: 1, register: "10"}) is operation.then_do
    assert operation.select({first: 0, register: "10"}) is operation.else_do
    with pytest.raises(KeyError, match="no outcome"):
        operation.select({first: 1})


@pytest.mark.parametrize(
    "condition,error,match",
    [
        (True, TypeError, "Python bool"),
        (sympy.true, ValueError, "always True"),
        (sympy.Symbol("x"), TypeError, "measurement outcomes"),
    ],
)
def test_ifelse_rejects_invalid_conditions(condition, error, match):
    with pytest.raises(error, match=match):
        gates.IfElse(condition, then_do=gates.X(0))


def test_ifelse_rejects_invalid_bodies():
    result = Circuit(2).add(gates.M(0, collapse=True))
    with pytest.raises(TypeError, match="then_do"):
        gates.IfElse(result.is_one(), then_do="x")
    with pytest.raises(NotImplementedError, match="cannot be used"):
        gates.IfElse(result.is_one(), then_do=gates.M(1))
    with pytest.raises(ValueError, match="at least one gate"):
        gates.IfElse(result.is_one(), then_do=[], else_do=[])


def test_ifchain_lowers_to_nested_ifelse():
    circuit = Circuit(4)
    first = circuit.add(gates.M(0, collapse=True))
    register = circuit.add(gates.M(1, 2, collapse=True))
    operation = (
        gates.IfChain()
        .if_(first.is_one(), gates.X(3))
        .elif_(register.equals("01"), [gates.H(3), gates.T(3)])
        .else_(gates.Z(3))
    )

    assert isinstance(operation, gates.IfElse)
    nested = operation.else_do[0]
    assert isinstance(nested, gates.IfElse)
    assert names(operation.then_do) == [("X", (3,))]
    assert names(nested.then_do) == [("H", (3,)), ("T", (3,))]
    assert names(nested.else_do) == [("Z", (3,))]
    assert operation.select({first: 0, register: "01"}) == operation.else_do
    assert nested.select({first: 0, register: "01"}) == nested.then_do


def test_unfinished_ifchain_fails_when_added():
    result = Circuit(1).add(gates.M(0, collapse=True))
    chain = gates.IfChain().if_(result.is_one(), gates.X(0))
    with pytest.raises(ValueError, match="unfinished IfChain"):
        Circuit(1).add(chain)


def test_switch_constructor_builder_and_selection():
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    direct = gates.Switch(
        target=result,
        cases={0: gates.X(2), (1, 2): [gates.H(2), gates.T(2)]},
        default_do=gates.Z(2),
    )
    fluent = (
        gates.Switch(result)
        .case(0, gates.X(2))
        .case((1, 2), [gates.H(2), gates.T(2)])
        .default(gates.Z(2))
    )

    assert [values for values, _ in direct.cases] == [
        values for values, _ in fluent.cases
    ]
    assert [names(body) for _, body in direct.cases] == [
        names(body) for _, body in fluent.cases
    ]
    assert names(direct.select({result: 0})) == [("X", (2,))]
    assert names(direct.select({result: "10"})) == [("H", (2,)), ("T", (2,))]
    assert names(direct.select({result: 3})) == [("Z", (2,))]


def test_switch_bit_target_and_noop_default():
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    operation = gates.Switch(result.bit(1)).case(1, gates.X(2)).default([])
    assert names(operation.select({result: "01"})) == [("X", (2,))]
    assert operation.select({result: "10"}) == []


def test_switch_builder_must_be_completed():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    unfinished = gates.Switch(result).case(1, gates.X(1))
    with pytest.raises(ValueError, match="unfinished Switch"):
        circuit.add(unfinished)
    with pytest.raises(TypeError, match="requires ``default_do``"):
        gates.Switch(result, {1: gates.X(1)})
    with pytest.raises(TypeError, match="not a mixture"):
        gates.Switch(result).case(0, gates.X(1)).case("1", gates.Z(1))


@pytest.mark.parametrize(
    "cases,error,match",
    [
        ({}, ValueError, "at least one case"),
        ({(0, 1): gates.X(2), 1: gates.Z(2)}, ValueError, "more than one case"),
        ({0: gates.X(2), "01": gates.Z(2)}, TypeError, "not a mixture"),
        ({(): gates.X(2)}, ValueError, "at least one value"),
        ({4: gates.X(2)}, ValueError, "not a possible outcome"),
    ],
)
def test_switch_case_validation(cases, error, match):
    result = Circuit(3).add(gates.M(0, 1, collapse=True))
    with pytest.raises(error, match=match):
        gates.Switch(result, cases, default_do=[])


def test_circuit_integration_and_numpy_native_execution():
    circuit, _ = feed_forward_circuit()
    operation = circuit.queue[2]
    assert circuit.has_collapse
    assert isinstance(operation, gates.ControlFlowOp)

    set_backend("numpy")
    get_backend().set_seed(123)
    result = circuit(nshots=100)
    frequencies = result.frequencies(binary=True)
    assert set(frequencies) <= {"00", "11"}
    assert sum(frequencies.values()) == 100
    assert all(frequencies[outcome] > 0 for outcome in ("00", "11"))

    with pytest.raises(NotImplementedError):
        operation.matrix()
    with pytest.raises(NotImplementedError):
        circuit.to_qasm()


@pytest.mark.parametrize(
    "prepare_one, expected",
    [(False, [0, 0]), (True, [1, 1])],
)
def test_numpy_ifelse_deterministic_branches(prepare_one, expected):
    circuit = Circuit(2)
    if prepare_one:
        circuit.add(gates.X(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(
        gates.IfElse(
            result.is_one(),
            then_do=gates.X(1),
            else_do=gates.I(1),
        )
    )
    circuit.add(gates.M(0, 1))

    samples = NumpyBackend().execute_circuit(circuit, nshots=10).samples()
    np.testing.assert_array_equal(samples, np.tile(expected, (10, 1)))


def test_numpy_switch_and_nested_control_flow():
    circuit = Circuit(3)
    circuit.add(gates.X(0))
    first = circuit.add(gates.M(0, collapse=True))
    syndrome = circuit.add(gates.M(1, collapse=True))
    correction = (
        gates.Switch(syndrome).case(0, gates.X(2)).case(1, gates.Z(2)).default([])
    )
    circuit.add(
        gates.IfElse(
            conditions.all_of(first.is_one(), syndrome.is_zero()),
            then_do=correction,
            else_do=gates.I(2),
        )
    )
    circuit.add(gates.M(0, 1, 2))

    samples = NumpyBackend().execute_circuit(circuit, nshots=10).samples()
    np.testing.assert_array_equal(samples, np.tile([1, 0, 1], (10, 1)))


def test_numpy_control_flow_uses_current_execution_outcome():
    circuit = Circuit(2)
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(
        gates.IfElse(
            result.is_one(),
            then_do=gates.X(1),
            else_do=gates.I(1),
        )
    )
    circuit.add(gates.M(0, 1))
    backend = NumpyBackend()

    state_one = np.zeros(4, dtype=complex)
    state_one[2] = 1
    first = backend.execute_circuit(circuit, initial_state=state_one, nshots=1)
    np.testing.assert_array_equal(first.samples()[0], [1, 1])

    state_zero = np.zeros(4, dtype=complex)
    state_zero[0] = 1
    second = backend.execute_circuit(circuit, initial_state=state_zero, nshots=1)
    np.testing.assert_array_equal(second.samples()[0], [0, 0])


def test_numpy_control_flow_density_matrix():
    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.X(0))
    result = circuit.add(gates.M(0, collapse=True))
    circuit.add(
        gates.IfElse(
            result.is_one(),
            then_do=gates.X(1),
            else_do=gates.I(1),
        )
    )

    final_state = NumpyBackend().execute_circuit(circuit, nshots=5)
    np.testing.assert_allclose(final_state.probabilities(), [0, 0, 0, 1])


def test_numpy_control_flow_requires_prior_measurement():
    external = Circuit(1).add(gates.M(0, collapse=True))
    circuit = Circuit(1)
    circuit.add(gates.IfElse(external.is_one(), then_do=gates.X(0)))

    with pytest.raises(ValueError, match="not performed before"):
        NumpyBackend().execute_circuit(circuit, nshots=1)


def test_control_flow_compilation_fails_loudly():
    circuit, _ = feed_forward_circuit()
    with pytest.raises(NotImplementedError, match="compilation"):
        circuit.compile(backend=NumpyBackend())


def test_parameters_moments_drawing_and_fusion():
    circuit = Circuit(3)
    first = circuit.add(gates.M(0))
    circuit.add(gates.H(1))
    second = circuit.add(gates.M(1))
    circuit.add(
        gates.IfElse(
            conditions.any_of(first.is_one(), second.is_one()),
            then_do=gates.RY(2, 0.2),
            else_do=gates.RX(2, 0.3),
        )
    )
    circuit.set_parameters([0.4, 0.5])
    assert [gate.parameters for gate in circuit.queue[-1].gates] == [(0.4,), (0.5,)]
    assert isinstance(circuit.queue.moments[-1][2], gates.IfElse)
    assert "if[" in circuit.diagram()
    fused = circuit.fuse()
    assert isinstance(fused.queue[-1], gates.IfElse)


def test_raw_roundtrip_pickle_and_deepcopy():
    circuit = expression_circuit()
    raw = json.loads(json.dumps(circuit.raw))
    assert raw["queue"][5]["_class"] == "IfElse"
    assert raw["queue"][5]["else_do"] == []
    loaded = Circuit.from_dict(raw)
    assert loaded.to_qasm3() == circuit.to_qasm3()
    with pytest.raises(NotImplementedError):
        _ = circuit.queue[5].raw
    with pytest.raises(ValueError, match="only be loaded with"):
        gates.Gate.from_dict(raw["queue"][5])

    for copied in (copy.deepcopy(circuit), pickle.loads(pickle.dumps(circuit))):
        assert copied.to_qasm3() == circuit.to_qasm3()
        assert copied.queue[5].results == [
            copied.queue[3].result,
            copied.queue[4].result,
        ]


def test_deep_copy_decompose_and_light_cone():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    first = circuit.add(gates.M(0, collapse=True))
    second = circuit.add(gates.M(1, collapse=True))
    circuit.add(
        gates.IfElse(
            conditions.all_of(first.is_one(), second.is_one()),
            then_do=gates.SX(2),
            else_do=[],
        )
    )

    copied = circuit.copy(deep=True)
    assert copied.queue[3].results == [copied.queue[1].result, copied.queue[2].result]
    assert copied.queue[3].gates[0] is not circuit.queue[3].gates[0]

    decomposed = circuit.decompose()
    assert isinstance(decomposed.queue[3], gates.IfElse)
    assert decomposed.queue[3].gates[0].__class__.__name__ == "RX"

    small, qubit_map = circuit.light_cone(2)
    assert qubit_map == {0: 0, 1: 1, 2: 2}
    assert isinstance(small.queue[-1], gates.IfElse)


def test_nested_control_flow():
    circuit = Circuit(3)
    first = circuit.add(gates.M(0, collapse=True))
    second = circuit.add(gates.M(1, collapse=True))
    inner = gates.Switch(
        second,
        cases={1: gates.RY(2, 0.3)},
        default_do=gates.X(2),
    )
    outer = gates.IfElse(first.is_one(), then_do=[gates.H(2), inner], else_do=[])
    circuit.add(outer)

    assert [node.results for node in outer._nested_control_flow()] == [
        [first],
        [second],
    ]
    assert names(outer._nested_gates()) == [("H", (2,)), ("RY", (2,)), ("X", (2,))]
    circuit.set_parameters([0.7])
    assert inner.cases[0][1][0].parameters == (0.7,)


def test_switch_raw_roundtrip_and_qasm_defaults():
    circuit = Circuit(3)
    result = circuit.add(gates.M(0, 1, collapse=True))
    circuit.add(
        gates.Switch(
            result,
            cases={0: gates.X(2), (1, 2): gates.H(2)},
            default_do=gates.Z(2),
        )
    )
    loaded = Circuit.from_dict(json.loads(json.dumps(circuit.raw)))
    assert loaded.to_qasm3() == circuit.to_qasm3()
    native = circuit.to_qasm3(qiskit_compatible=False)
    compatible = circuit.to_qasm3(qiskit_compatible=True)
    assert "default {\n    z q[2];\n" in native
    assert "} else {\n  z q[2];\n}" in compatible


def test_unresolved_measurement_fails_loudly():
    result = Circuit(1).add(gates.M(0, collapse=True))
    circuit = Circuit(1)
    circuit.add(gates.IfElse(result.is_one(), then_do=gates.X(0)))
    with pytest.raises(ValueError, match="not performed before"):
        _ = circuit.raw
    with pytest.raises(ValueError, match="not performed before"):
        circuit.to_qasm3()


def test_public_legacy_condition_is_removed():
    assert not hasattr(gates, "Condition")
    assert isinstance(MeasurementResult((0,)), MeasurementResult)
