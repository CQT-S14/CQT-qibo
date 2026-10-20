"""Classical control-flow operations driven by mid-circuit measurements."""

import copy
from collections.abc import Mapping

import sympy
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Boolean, Not, Or, Xor, to_nnf

from qibo.config import raise_error
from qibo.gates.abstract import Gate, SpecialGate
from qibo.gates.measurements import M
from qibo.measurements import (
    MeasurementBit,
    MeasurementResult,
    MeasurementSymbol,
    MeasurementValue,
    parse_outcome,
)

__all__ = ["ControlFlowOp", "IfChain", "IfElse", "Switch"]

_MISSING = object()


class ControlFlowOp(Gate):
    """Base class for classical control-flow operations.

    This is an implementation detail shared by :class:`IfElse` and
    :class:`Switch`; users normally construct one of those concrete operations.
    """

    def _setup(self, name: str, draw_label: str, branches: list[list[Gate]]):
        self.name = name
        self.draw_label = draw_label
        for gate in (gate for branch in branches for gate in branch):
            if not isinstance(gate, Gate):
                raise_error(
                    TypeError,
                    "Conditional branch bodies must contain gates, but found "
                    + f"{type(gate).__name__}.",
                )
            if isinstance(gate, (M, SpecialGate)):
                raise_error(
                    NotImplementedError,
                    f"{gate.__class__.__name__} gates cannot be used inside "
                    + "conditional branches.",
                )
        qubits = {
            qubit for branch in branches for gate in branch for qubit in gate.qubits
        }
        if not qubits:
            raise_error(
                ValueError,
                f"A {self.__class__.__name__} needs at least one gate across its branches.",
            )
        self.target_qubits = tuple(sorted(qubits))

    @property
    def gates(self) -> list[Gate]:
        """All direct child gates, without descending into nested control flow."""
        return [gate for branch in self.branches for gate in branch]

    def _nested_control_flow(self):
        """Yields this operation and all control-flow operations nested in it."""
        yield self
        for gate in self.gates:
            if isinstance(gate, ControlFlowOp):
                yield from gate._nested_control_flow()

    def _nested_gates(self):
        """Yields ordinary gates, descending into nested control flow."""
        for gate in self.gates:
            if isinstance(gate, ControlFlowOp):
                yield from gate._nested_gates()
            else:
                yield gate

    def on_qubits(self, qubit_map: dict) -> "ControlFlowOp":
        """Copies this operation, remapping its branch gates to other qubits."""
        return self._map_bodies(
            lambda body: [gate.on_qubits(qubit_map) for gate in body]
        )

    def decompose(
        self, *free, use_toffolis: bool = True, method: str = "standard", **kwargs
    ):
        """Decomposes the gates in every branch while preserving control flow."""

        def decompose_body(body):
            return [
                new
                for gate in body
                for new in gate.decompose(
                    *free, use_toffolis=use_toffolis, method=method, **kwargs
                )
            ]

        return [self._map_bodies(decompose_body)]

    @property
    def raw(self) -> dict:
        raise_error(
            NotImplementedError,
            f"A {self.__class__.__name__} refers to measurements of its circuit, "
            + "so it can only be serialized together with it using ``Circuit.raw``.",
        )

    @classmethod
    def _deserialize(cls, raw: dict, measurements: list) -> "ControlFlowOp":
        def deserialize_body(body):
            return [
                (
                    ControlFlowOp._deserialize(gate, measurements)
                    if gate["_class"] in {"IfElse", "Switch"}
                    else Gate.from_dict(gate)
                )
                for gate in body
            ]

        if raw["_class"] == "IfElse":
            return IfElse(
                _deserialize_expression(raw["condition"], measurements),
                then_do=deserialize_body(raw["then_do"]),
                else_do=deserialize_body(raw["else_do"]),
            )
        if raw["_class"] == "Switch":
            target = _deserialize_expression(raw["target"], measurements)
            cases = [
                (tuple(case["values"]), deserialize_body(case["do"]))
                for case in raw["cases"]
            ]
            return Switch._from_normalized(
                target, cases, deserialize_body(raw["default_do"])
            )
        raise_error(ValueError, f"Unknown control-flow operation {raw['_class']!r}.")

    @property
    def qasm_label(self):
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} is only supported by OpenQASM 3; use "
            + "``Circuit.to_qasm3``.",
        )

    def controlled_by(self, *q):
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} operations cannot be quantum-controlled.",
        )

    def _dagger(self):
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} operations cannot be inverted.",
        )

    def matrix(self, backend=None):
        raise_error(
            NotImplementedError,
            f"{self.__class__.__name__} operations do not have a matrix representation.",
        )

    def apply(self, backend, state, nqubits):
        """Selects and applies one branch using live NumPy measurement outcomes."""
        if backend.name != "numpy":
            raise_error(
                NotImplementedError,
                "Native Qibo execution of classical control flow currently supports "
                + "only the NumPy backend, but received "
                + f"{backend.name!r}.",
            )

        classical_state = [(result, _latest_outcome(result)) for result in self.results]
        for gate in self.select(classical_state):
            if gate.symbolic_parameters:
                gate.substitute_symbols()
            state = gate.apply(backend, state, nqubits)
        return state

    def apply_clifford(self, backend, state, nqubits):
        return self.apply(backend, state, nqubits)

    def apply_hamming_weight(self, backend, state, nqubits, weight):
        return self.apply(backend, state, nqubits)


class IfElse(ControlFlowOp):
    """Applies one of two gate bodies according to a measured condition.

    Args:
        condition: A measured bit or boolean expression of measured outcomes.
        then_do: Gate, circuit, or iterable of gates applied when true.
        else_do: Gate, circuit, or iterable of gates applied when false. Defaults
            to an empty (no-op) branch.
    """

    def __init__(self, condition, then_do, else_do=()):
        super().__init__()
        self.expression = _expression(condition)
        self.then_do = _body(then_do, "then_do")
        self.else_do = _body(else_do, "else_do")
        self._setup("ifelse", "if", self.branches)

    @property
    def branches(self) -> list[list[Gate]]:
        return [self.then_do, self.else_do]

    @property
    def atoms(self) -> list:
        return sorted(self.expression.free_symbols, key=_atom_order)

    @property
    def results(self) -> list[MeasurementResult]:
        return _results(self.atoms)

    def select(self, classical_state) -> list[Gate]:
        """Returns the branch selected by ``classical_state`` without executing it."""
        return (
            self.then_do
            if _evaluate(self.expression, classical_state)
            else self.else_do
        )

    def _map_bodies(self, func) -> "IfElse":
        return IfElse(self.expression, func(self.then_do), func(self.else_do))

    def _copy(self, results: dict) -> "IfElse":
        expression = _remap_expression(self.expression, results)
        return IfElse(
            expression,
            _copy_body(self.then_do, results),
            _copy_body(self.else_do, results),
        )

    def _serialize(self, measurements: list) -> dict:
        reference = _measurement_reference(measurements, self.__class__.__name__)
        return {
            "_class": "IfElse",
            "name": self.name,
            "condition": _serialize_expression(self.expression, reference),
            "then_do": _serialize_body(self.then_do, measurements),
            "else_do": _serialize_body(self.else_do, measurements),
        }


class Switch(ControlFlowOp):
    """Selects a gate body using the value of a measured bit or register.

    Use the complete constructor ``Switch(target, cases, default_do=...)`` or
    the fluent form ``Switch(target).case(...).default(...)``. The fluent form
    deliberately requires ``default``; pass ``[]`` to acknowledge a no-op default.
    """

    def __init__(self, target, cases=None, default_do=_MISSING):
        super().__init__()
        self.target = _switch_target(target)
        entries = _case_entries(cases)
        self._value_kind = _case_kind(entries)
        self._cases = _normalize_cases(self.target, entries)
        self._default_do = (
            _MISSING if default_do is _MISSING else _body(default_do, "default_do")
        )
        self.name = "switch"
        self.draw_label = "switch"
        self.target_qubits = ()
        if cases is not None and default_do is _MISSING:
            raise_error(
                TypeError,
                "Switch(target, cases=...) requires ``default_do``. Use "
                + "``default_do=[]`` for an explicit no-op default.",
            )
        if self.complete:
            self._finish_setup()

    @classmethod
    def _from_normalized(cls, target, cases, default_do) -> "Switch":
        switch = cls.__new__(cls)
        Gate.__init__(switch)
        switch.target = _switch_target(target)
        switch._cases = [(tuple(values), list(body)) for values, body in cases]
        switch._value_kind = "integer"
        switch._default_do = list(default_do)
        switch._finish_setup()
        return switch

    def _finish_setup(self):
        if not self._cases:
            raise_error(
                ValueError, "A Switch needs at least one case before its default."
            )
        self._setup("switch", "switch", self.branches)

    @property
    def complete(self) -> bool:
        return self._default_do is not _MISSING

    @property
    def cases(self) -> list[tuple[tuple[int, ...], list[Gate]]]:
        return self._cases

    @property
    def default_do(self) -> list[Gate]:
        if not self.complete:
            raise_error(
                RuntimeError,
                "This Switch builder is unfinished. Call ``.default(gates)``; "
                + "use ``.default([])`` for a no-op default.",
            )
        return self._default_do

    @property
    def branches(self) -> list[list[Gate]]:
        branches = [body for _, body in self._cases]
        if self.complete:
            branches.append(self._default_do)
        return branches

    @property
    def atoms(self) -> list:
        return [self.target]

    @property
    def results(self) -> list[MeasurementResult]:
        return _results(self.atoms)

    def case(self, values, do) -> "Switch":
        """Returns a new builder with one additional case."""
        if self.complete:
            raise_error(
                RuntimeError,
                "Cannot add a case after ``Switch.default`` has completed the switch.",
            )
        raw_cases = list(self._cases)
        entry = [(values, do)]
        value_kind = _case_kind(entry)
        if self._value_kind is not None and value_kind != self._value_kind:
            raise_error(
                TypeError,
                "Switch case values must be all integers or all bitstrings, not a mixture.",
            )
        raw_cases.extend(_normalize_cases(self.target, entry))
        _check_duplicate_cases(raw_cases)
        switch = self.__class__.__new__(self.__class__)
        Gate.__init__(switch)
        switch.target = self.target
        switch._cases = raw_cases
        switch._value_kind = self._value_kind or value_kind
        switch._default_do = _MISSING
        return switch

    def default(self, do) -> "Switch":
        """Completes the builder with an explicit default branch."""
        if self.complete:
            raise_error(RuntimeError, "This Switch already has a default branch.")
        return self._from_normalized(self.target, self._cases, _body(do, "default"))

    def validate_complete(self):
        if not self.complete:
            raise_error(
                ValueError,
                "Cannot add an unfinished Switch to a circuit. Call "
                + "``.default(gates)`` or ``.default([])`` first.",
            )

    def select(self, classical_state) -> list[Gate]:
        """Returns the selected case/default body without executing it."""
        self.validate_complete()
        value = _target_outcome(self.target, classical_state)
        for values, body in self._cases:
            if value in values:
                return body
        return self._default_do

    def _map_bodies(self, func) -> "Switch":
        self.validate_complete()
        return self._from_normalized(
            self.target,
            [(values, func(body)) for values, body in self._cases],
            func(self._default_do),
        )

    def _copy(self, results: dict) -> "Switch":
        self.validate_complete()
        target = _remap_atom(self.target, results)
        return self._from_normalized(
            target,
            [(values, _copy_body(body, results)) for values, body in self._cases],
            _copy_body(self._default_do, results),
        )

    def _serialize(self, measurements: list) -> dict:
        self.validate_complete()
        reference = _measurement_reference(measurements, self.__class__.__name__)
        return {
            "_class": "Switch",
            "name": self.name,
            "target": _serialize_expression(self.target, reference),
            "cases": [
                {"values": list(values), "do": _serialize_body(body, measurements)}
                for values, body in self._cases
            ],
            "default_do": _serialize_body(self._default_do, measurements),
        }


class IfChain:
    """Fluent builder for readable ``if``/``elif``/``else`` control flow.

    ``else_`` completes the builder and returns nested :class:`IfElse` nodes.
    """

    def __init__(self, branches=()):
        self._branches = tuple(branches)

    def if_(self, condition, then_do) -> "IfChain":
        if self._branches:
            raise_error(
                RuntimeError,
                "``IfChain.if_`` can only start an empty chain; use ``elif_`` next.",
            )
        return IfChain(((_expression(condition), _body(then_do, "then_do")),))

    def elif_(self, condition, then_do) -> "IfChain":
        if not self._branches:
            raise_error(RuntimeError, "Call ``IfChain().if_(...)`` before ``elif_``.")
        return IfChain(
            self._branches + ((_expression(condition), _body(then_do, "then_do")),)
        )

    def else_(self, else_do) -> IfElse:
        if not self._branches:
            raise_error(RuntimeError, "Call ``IfChain().if_(...)`` before ``else_``.")
        tail = _body(else_do, "else_do")
        operation = None
        for condition, then_do in reversed(self._branches):
            operation = IfElse(
                condition, then_do, tail if operation is None else operation
            )
        return operation

    def _unfinished_error(self):
        raise_error(
            ValueError,
            "Cannot add an unfinished IfChain to a circuit. Complete it with "
            + "``.else_(gates)``; use ``.else_([])`` for a no-op else branch.",
        )


def _expression(condition):
    """Validates a boolean expression over measurement outcomes."""
    if isinstance(condition, MeasurementResult):
        condition = condition._bit()
    if isinstance(condition, bool):
        raise_error(
            TypeError,
            "The condition is a Python bool, not a measurement expression. Use "
            + "``bit.is_one()`` / ``bit.is_zero()`` or a register's "
            + "``equals(value)`` method.",
        )
    if not isinstance(condition, Boolean) or isinstance(condition, Relational):
        raise_error(
            TypeError,
            "A condition must be a measured bit or a boolean expression of "
            + f"measurement outcomes, but got {condition!r}.",
        )
    if condition.atoms(Relational):
        raise_error(
            TypeError,
            "SymPy relations are not valid measurement conditions. Use the explicit "
            + "measurement API such as ``result.equals(value)``.",
        )
    atoms = condition.free_symbols
    if not atoms:
        raise_error(
            ValueError,
            f"The condition is always {condition}; it must depend on a measurement.",
        )
    for atom in atoms:
        if not isinstance(atom, (MeasurementSymbol, MeasurementValue)):
            raise_error(
                TypeError,
                f"Conditions can only depend on measurement outcomes, but {atom} is not one.",
            )
    return condition


def _switch_target(target):
    if isinstance(target, MeasurementBit):
        target = target.is_one()
    if not isinstance(target, (MeasurementResult, MeasurementSymbol)):
        raise_error(
            TypeError,
            "A Switch target must be a MeasurementResult or a bit selected with "
            + f"``result.bit(index)``, but got {type(target).__name__}.",
        )
    return target


def _nbits(target) -> int:
    return len(target.target_qubits) if isinstance(target, MeasurementResult) else 1


def _case_values(values) -> tuple:
    if isinstance(values, (tuple, list, set, frozenset)):
        if not values:
            raise_error(ValueError, "A Switch case needs at least one value.")
        return tuple(values)
    return (values,)


def _case_entries(cases) -> list:
    if cases is None:
        return []
    return list(cases.items()) if isinstance(cases, Mapping) else list(cases)


def _case_kind(entries) -> str | None:
    if not entries:
        return None
    kinds = {
        "bitstring" if isinstance(value, str) else "integer"
        for values, _ in entries
        for value in _case_values(values)
    }
    if len(kinds) > 1:
        raise_error(
            TypeError,
            "Switch case values must be all integers or all bitstrings, not a mixture.",
        )
    return kinds.pop()


def _normalize_cases(target, entries) -> list:
    if not entries:
        return []
    nbits = _nbits(target)
    raw = [(_case_values(values), do) for values, do in entries]
    normalized = [
        (tuple(parse_outcome(value, nbits) for value in values), _body(do, "case"))
        for values, do in raw
    ]
    _check_duplicate_cases(normalized)
    return normalized


def _check_duplicate_cases(cases):
    seen = set()
    for values, _ in cases:
        for value in values:
            if value in seen:
                raise_error(
                    ValueError,
                    f"Switch value {value} appears in more than one case.",
                )
            seen.add(value)


def _body(body, argument: str) -> list:
    """Flattens a gate, circuit, or nested iterable into a branch body."""
    from qibo.models.circuit import Circuit

    if isinstance(body, Gate):
        return [body]
    if isinstance(body, Circuit):
        return list(body.queue)
    if isinstance(body, (str, bytes, dict)) or not hasattr(body, "__iter__"):
        raise_error(
            TypeError,
            f"``{argument}`` must be a gate, circuit, or iterable of gates; got "
            + f"{type(body).__name__}.",
        )
    return [gate for item in body for gate in _body(item, argument)]


def _atom_order(atom) -> tuple:
    return (isinstance(atom, MeasurementValue), int(atom.name[1:]))


def _results(atoms) -> list[MeasurementResult]:
    results = []
    for atom in atoms:
        result = atom if isinstance(atom, MeasurementResult) else atom.result
        if not any(result is other for other in results):
            results.append(result)
    return results


def _latest_outcome(result: MeasurementResult) -> int:
    """Returns the latest sampled register value produced during execution."""
    if not result.has_samples():
        raise_error(
            RuntimeError,
            "Cannot evaluate classical control flow because the measurement on "
            + f"qubits {result.target_qubits} has no outcome. Ensure that the "
            + "referenced measurement is executed before the control-flow operation.",
        )
    bits = result._samples[-1]
    return sum(int(bit) << (len(bits) - index - 1) for index, bit in enumerate(bits))


def _lookup_result(classical_state, result: MeasurementResult) -> int:
    from qibo.conditions import ClassicalState

    state = (
        classical_state
        if isinstance(classical_state, ClassicalState)
        else ClassicalState(classical_state)
    )
    return state.value(result)


def _target_outcome(target, classical_state) -> int:
    value = _lookup_result(
        classical_state,
        target if isinstance(target, MeasurementResult) else target.result,
    )
    if isinstance(target, MeasurementSymbol):
        nbits = len(target.result.target_qubits)
        return (value >> (nbits - target.index - 1)) & 1
    return value


def _evaluate(expression, classical_state) -> bool:
    substitutions = {}
    for atom in expression.free_symbols:
        if isinstance(atom, MeasurementSymbol):
            substitutions[atom] = bool(_target_outcome(atom, classical_state))
        else:
            substitutions[atom] = (
                _lookup_result(classical_state, atom.result) == atom.value
            )
    value = expression.subs(substitutions)
    if value not in (sympy.true, sympy.false):
        raise_error(
            ValueError,
            f"Could not fully evaluate condition {expression!r} from the classical state.",
        )
    return value == sympy.true


def _remap_atom(atom, results: dict):
    old = atom if isinstance(atom, MeasurementResult) else atom.result
    result = results.get(id(old), old)
    if isinstance(atom, MeasurementResult):
        return result
    if isinstance(atom, MeasurementSymbol):
        return result[atom.index]
    return result.equals(atom.value)


def _remap_expression(expression, results: dict):
    return expression.xreplace(
        {
            atom: _remap_atom(atom, results)
            for atom in sorted(expression.free_symbols, key=_atom_order)
        }
    )


def _copy_body(body, results: dict) -> list:
    return [
        gate._copy(results) if isinstance(gate, ControlFlowOp) else copy.copy(gate)
        for gate in body
    ]


def _measurement_reference(measurements: list, operation: str):
    def reference(result):
        index = _find_measurement(measurements, result)
        if index is None:
            raise_error(
                ValueError,
                f"Cannot serialize {operation}: one of its measurements is not "
                + "performed before it in the circuit.",
            )
        return index

    return reference


def _serialize_body(body: list, measurements: list) -> list[dict]:
    return [
        gate._serialize(measurements) if isinstance(gate, ControlFlowOp) else gate.raw
        for gate in body
    ]


def _serialize_expression(expression, reference) -> dict:
    if isinstance(expression, MeasurementResult):
        return {"register": reference(expression)}
    if isinstance(expression, MeasurementSymbol):
        return {"bit": [reference(expression.result), expression.index]}
    if isinstance(expression, MeasurementValue):
        return {"equals": [reference(expression.result), expression.value]}
    for operation, name in ((Not, "not"), (And, "and"), (Or, "or"), (Xor, "xor")):
        if isinstance(expression, operation):
            return {
                name: [_serialize_expression(arg, reference) for arg in expression.args]
            }
    return _serialize_expression(to_nnf(expression, simplify=False), reference)


def _deserialize_expression(raw: dict, measurements: list):
    (name, value), *_ = raw.items()
    if name == "register":
        return measurements[value].result
    if name == "bit":
        return measurements[value[0]].result[value[1]]
    if name == "equals":
        return measurements[value[0]].result.equals(value[1])
    args = [_deserialize_expression(arg, measurements) for arg in value]
    return {"not": Not, "and": And, "or": Or, "xor": Xor}[name](*args)


def _find_measurement(measurements: list, result: MeasurementResult) -> int | None:
    for index in range(len(measurements) - 1, -1, -1):
        if measurements[index].result is result:
            return index
    return None
