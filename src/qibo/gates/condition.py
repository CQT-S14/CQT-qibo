"""Gates applied conditionally on the outcome of mid-circuit measurements."""

import copy
from numbers import Integral

from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Boolean, Not, Or, Xor, to_nnf

from qibo.config import raise_error
from qibo.gates.abstract import Gate, SpecialGate
from qibo.gates.measurements import M
from qibo.measurements import (
    MeasurementResult,
    MeasurementSymbol,
    MeasurementValue,
    parse_outcome,
)

__all__ = ["Condition"]


class Condition(Gate):
    """Applies gates only if mid-circuit measurements have given outcomes.

    The outcomes are referenced through the result returned when a measurement
    is added to a circuit, ``result = circuit.add(gates.M(..., collapse=True))``:

    - ``result[i]`` is the :math:`i`-th measured bit, which is true when it was
      measured in state 1. A result of a single qubit is itself a bit;
    - bits are combined with ``&`` (and), ``|`` (or), ``^`` (xor) and ``~`` (not);
    - ``result == value`` and ``result != value`` compare the whole register with
      an integer or a bitstring.

    Register values follow qibo's bitstring convention: ``result[0]`` is the
    most significant bit. For example, after
    ``result = circuit.add(gates.M(0, 1, collapse=True))``, ``result == 2`` (or
    ``result == "10"``) means that the first measured bit is 1 and the second is 0.

    If ``body`` is a dictionary, the condition is a ``switch``: ``condition`` is a
    measured register or bit, and the body of the matching value is applied. The
    values of a ``switch`` are either all integers or all bitstrings, and a tuple
    of values, as in ``{(1, 2): body}``, shares one body.

    A body is a gate, a list of gates, a generator of gates such as
    :meth:`qibo.models.circuit.Circuit.on_qubits`, or a
    :class:`qibo.models.circuit.Circuit`, whose gates are used. Bodies may
    contain other conditions, but not measurements.

    Simulation of conditional gates is not supported yet. Circuits containing
    them can be exported with :meth:`qibo.models.circuit.Circuit.to_qasm3`.

    Example:
        .. code-block:: python

            from qibo import Circuit, gates

            circuit = Circuit(4)
            circuit.add([gates.H(0), gates.H(1), gates.H(2)])
            first = circuit.add(gates.M(0, collapse=True))
            register = circuit.add(gates.M(1, 2, collapse=True))
            # flip qubit 3 if qubit 0 was measured in state 1
            circuit.add(gates.Condition(first, gates.X(3)))
            # boolean expression, with several gates
            circuit.add(gates.Condition(~first & register[1], [gates.H(3), gates.S(3)]))
            # comparison of a register
            circuit.add(gates.Condition(register == "10", gates.X(3)))
            # switch on the value of a register
            circuit.add(gates.Condition(register, {0: gates.X(3), (1, 2): gates.T(3)}))
            circuit.add(gates.M(3))
            print(circuit.to_qasm3())

    Args:
        condition (:class:`qibo.measurements.MeasurementResult` or sympy.Basic):
            boolean expression of measured bits and register comparisons. For a
            ``switch``, the measured register or bit whose value selects the body.
        body (:class:`qibo.gates.Gate` or list or :class:`qibo.models.circuit.Circuit` or dict):
            gates applied when the condition is true. For a ``switch``, a
            dictionary that maps values to bodies.
    """

    def __init__(self, condition, body):
        super().__init__()
        if isinstance(body, (Integral, str)):
            raise_error(
                TypeError,
                "Conditions are expressions of measurement outcomes: write "
                + "``Condition(result == value, gates)`` instead of "
                + "``Condition(result, value, gates)``.",
            )
        if isinstance(body, dict):
            target = _switch_target(condition)
            nbits = _nbits(target)
            raw_cases = [
                (_case_values(values), case_body) for values, case_body in body.items()
            ]
            if not raw_cases:
                raise_error(ValueError, "A ``switch`` needs at least one case.")
            if len({isinstance(v, str) for values, _ in raw_cases for v in values}) > 1:
                raise_error(
                    TypeError,
                    "The values of a ``switch`` must be all integers or all "
                    + "bitstrings, not both.",
                )
            cases = []
            seen = set()
            for values, case_body in raw_cases:
                values = tuple(parse_outcome(value, nbits) for value in values)
                for value in values:
                    if value in seen:
                        raise_error(
                            ValueError,
                            f"Value {value} is used by more than one case of the condition.",
                        )
                    seen.add(value)
                cases.append((values, _body(case_body)))
            self._setup(None, target, cases)
        else:
            self._setup(_expression(condition), None, [((), _body(body))])

    @classmethod
    def _from_parts(cls, expression, target, cases):
        """Creates a condition from already validated parts."""
        gate = cls.__new__(cls)
        Gate.__init__(gate)
        gate._setup(expression, target, cases)
        return gate

    def _setup(self, expression, target, cases):
        self.name = "condition"
        self.draw_label = "if" if target is None else "switch"
        self.expression = expression
        self.target = target
        self._cases = cases
        for gate in self.gates:
            if not isinstance(gate, Gate):
                raise_error(
                    TypeError,
                    f"Conditional operations must be gates, not {type(gate)}.",
                )
            if isinstance(gate, (M, SpecialGate)):
                raise_error(
                    NotImplementedError,
                    f"{gate.__class__.__name__} gates cannot be applied conditionally.",
                )
        qubits = {q for gate in self.gates for q in gate.qubits}
        if not qubits:
            raise_error(ValueError, "A ``Condition`` needs at least one gate to apply.")
        self.target_qubits = tuple(sorted(qubits))

    @property
    def body(self) -> list | None:
        """Gates applied when the condition is true, ``None`` for a ``switch``."""
        return self._cases[0][1] if self.target is None else None

    @property
    def cases(self) -> list | None:
        """Pairs of values and bodies of a ``switch``, ``None`` otherwise.

        Values are integers in qibo's bit ordering.
        """
        return None if self.target is None else self._cases

    @property
    def atoms(self) -> list:
        """Measured bits and register comparisons in the condition.

        For a ``switch``, the measured register or bit.
        """
        if self.target is not None:
            return [self.target]
        return sorted(self.expression.free_symbols, key=_atom_order)

    @property
    def results(self) -> list:
        """Measurement results that the condition reads, without repetitions."""
        results = []
        for atom in self.atoms:
            result = atom if isinstance(atom, MeasurementResult) else atom.result
            if not any(result is other for other in results):
                results.append(result)
        return results

    @property
    def branches(self) -> list:
        """Lists of gates of all branches."""
        return [body for _, body in self._cases]

    @property
    def gates(self) -> list:
        """All gates of all branches, without descending into nested conditions."""
        return [gate for branch in self.branches for gate in branch]

    def _nested_conditions(self):
        """Yields this condition and all conditions nested in its branches."""
        yield self
        for gate in self.gates:
            if isinstance(gate, Condition):
                yield from gate._nested_conditions()

    def _nested_gates(self):
        """Yields all non-conditional gates, descending into nested conditions."""
        for gate in self.gates:
            if isinstance(gate, Condition):
                yield from gate._nested_gates()
            else:
                yield gate

    def _map_bodies(self, func) -> "Condition":
        """New condition with the same test and ``func`` applied to each branch."""
        cases = [(values, func(body)) for values, body in self._cases]
        return self._from_parts(self.expression, self.target, cases)

    def on_qubits(self, qubit_map: dict) -> "Condition":
        """Creates the same condition with its gates acting on different qubits.

        The measurements the condition depends on are not changed, so mapping a
        whole circuit (for example with
        :meth:`qibo.models.circuit.Circuit.on_qubits`) keeps conditions linked
        to the mapped measurements.
        """
        return self._map_bodies(lambda body: [g.on_qubits(qubit_map) for g in body])

    def decompose(
        self, *free, use_toffolis: bool = True, method: str = "standard", **kwargs
    ):
        """Decomposes the conditional gates, keeping the condition."""

        def decompose_body(body):
            return [
                new
                for gate in body
                for new in gate.decompose(
                    *free, use_toffolis=use_toffolis, method=method, **kwargs
                )
            ]

        return [self._map_bodies(decompose_body)]

    def _copy(self, results: dict) -> "Condition":
        """Copy whose gates are new objects and whose measurements are remapped.

        Args:
            results (dict): maps ``id`` of old measurement results to the new
                :class:`qibo.measurements.MeasurementResult` objects.
        """

        def remap(atom):
            old = atom if isinstance(atom, MeasurementResult) else atom.result
            new = results.get(id(old), old)
            return _rebuild_atom(atom, new)

        def copy_body(body):
            return [
                g._copy(results) if isinstance(g, Condition) else copy.copy(g)
                for g in body
            ]

        expression, target = self.expression, self.target
        if target is not None:
            target = remap(target)
        else:
            expression = expression.xreplace({atom: remap(atom) for atom in self.atoms})
        cases = [(values, copy_body(body)) for values, body in self._cases]
        return self._from_parts(expression, target, cases)

    @property
    def raw(self) -> dict:
        raise_error(
            NotImplementedError,
            "A ``Condition`` refers to measurements of its circuit, so it can only "
            + "be serialized together with it using ``Circuit.raw``.",
        )

    def _serialize(self, measurements: list) -> dict:
        """Serializes the condition referring to measurements by position.

        Args:
            measurements (list): measurement gates that precede the condition
                in the circuit, in order.
        """

        def reference(result):
            index = _find_measurement(measurements, result)
            if index is None:
                raise_error(
                    ValueError,
                    "Cannot serialize a ``Condition`` whose measurement is not "
                    + "performed before it in the circuit.",
                )
            return index

        def serialize_body(body):
            return [
                g._serialize(measurements) if isinstance(g, Condition) else g.raw
                for g in body
            ]

        raw = {"_class": "Condition", "name": self.name}
        if self.target is None:
            raw["condition"] = _serialize_expression(self.expression, reference)
            raw["body"] = serialize_body(self.body)
        else:
            raw["switch"] = _serialize_expression(self.target, reference)
            raw["cases"] = [
                {"values": list(values), "gates": serialize_body(body)}
                for values, body in self._cases
            ]
        return raw

    @classmethod
    def _deserialize(cls, raw: dict, measurements: list) -> "Condition":
        """Counter-part of :meth:`_serialize`."""

        def deserialize_body(body):
            return [
                (
                    cls._deserialize(g, measurements)
                    if g["_class"] == "Condition"
                    else Gate.from_dict(g)
                )
                for g in body
            ]

        if "switch" in raw:
            target = _deserialize_expression(raw["switch"], measurements)
            cases = [
                (tuple(case["values"]), deserialize_body(case["gates"]))
                for case in raw["cases"]
            ]
            return cls._from_parts(None, target, cases)
        expression = _deserialize_expression(raw["condition"], measurements)
        return cls._from_parts(expression, None, [((), deserialize_body(raw["body"]))])

    @property
    def qasm_label(self):
        raise_error(
            NotImplementedError,
            "``Condition`` is only supported by OpenQASM 3, use ``Circuit.to_qasm3``.",
        )

    def controlled_by(self, *q):
        raise_error(NotImplementedError, "``Condition`` gates cannot be controlled.")

    def _dagger(self):
        raise_error(NotImplementedError, "``Condition`` gates cannot be inverted.")

    def matrix(self, backend=None):
        raise_error(
            NotImplementedError,
            "``Condition`` gates do not have a matrix representation.",
        )

    def apply(self, backend, state, nqubits):
        raise_error(
            NotImplementedError,
            "Simulation of ``Condition`` gates is not supported yet. "
            + "Circuits with conditions can be exported with ``Circuit.to_qasm3``.",
        )

    def apply_clifford(self, backend, state, nqubits):
        return self.apply(backend, state, nqubits)

    def apply_hamming_weight(self, backend, state, nqubits, weight):
        return self.apply(backend, state, nqubits)


def _expression(condition):
    """Validates the boolean expression of a condition."""
    if isinstance(condition, MeasurementResult):
        condition = condition._bit()
    if isinstance(condition, bool):
        raise_error(
            TypeError,
            "The condition is a Python ``bool``. Comparing a single bit with ``==``, "
            + "as in ``result[0] == 1``, does not build a condition: use "
            + "``result[0]`` or ``~result[0]``. Results build conditions only when "
            + "compared with integers or bitstrings, as in ``result == 1``.",
        )
    if not isinstance(condition, Boolean) or isinstance(condition, Relational):
        raise_error(
            TypeError,
            "A condition must be a measured bit or a boolean expression of measured "
            + f"bits and register comparisons, but got {condition!r}.",
        )
    if condition.atoms(Relational):
        raise_error(
            TypeError,
            "Conditions cannot contain sympy relations such as ``Eq``. Use measured "
            + "bits, ``~`` for their negation, and ``result == value``.",
        )
    atoms = condition.free_symbols
    if not atoms:
        raise_error(
            ValueError,
            f"The condition is always {condition}, so it does not depend on any "
            + "measurement.",
        )
    for atom in atoms:
        if not isinstance(atom, (MeasurementSymbol, MeasurementValue)):
            raise_error(
                TypeError,
                f"Conditions can only depend on measurement outcomes, but {atom} is not one.",
            )
    return condition


def _switch_target(target):
    if not isinstance(target, (MeasurementResult, MeasurementSymbol)):
        raise_error(
            TypeError,
            "The target of a ``switch`` must be a measurement result or one of its "
            + f"bits, but is {type(target)}.",
        )
    return target


def _nbits(target) -> int:
    """Number of bits of a ``switch`` target."""
    if isinstance(target, MeasurementResult):
        return len(target.target_qubits)
    return 1


def _case_values(values) -> tuple:
    """Values of a ``switch`` case, as given: a single value or a tuple of values."""
    if isinstance(values, (tuple, frozenset)):
        if not values:
            raise_error(ValueError, "A case of a ``switch`` needs at least one value.")
        return tuple(values)
    return (values,)


def _body(body) -> list:
    """List of gates of a conditional branch.

    Accepts a gate, a circuit, or (nested) iterables of gates and circuits.
    """
    from qibo.models.circuit import Circuit

    if isinstance(body, Gate):
        return [body]
    if isinstance(body, Circuit):
        return list(body.queue)
    if isinstance(body, (str, bytes, dict)) or not hasattr(body, "__iter__"):
        raise_error(
            TypeError, f"Conditional operations must be gates, not {type(body)}."
        )
    return [gate for item in body for gate in _body(item)]


def _atom_order(atom) -> tuple:
    """Order of creation of measurement symbols and register comparisons."""
    return (isinstance(atom, MeasurementValue), int(atom.name[1:]))


def _rebuild_atom(atom, result: MeasurementResult):
    """The same bit, comparison or register as ``atom``, for another result."""
    if isinstance(atom, MeasurementResult):
        return result
    if isinstance(atom, MeasurementSymbol):
        return result[atom.index]
    return result == atom.value


def _serialize_expression(expression, reference) -> dict:
    """JSON-compatible form of a condition, with measurements given by position."""
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
    """Counter-part of :func:`_serialize_expression`."""
    (name, value), *_ = raw.items()
    if name == "register":
        return measurements[value].result
    if name == "bit":
        return measurements[value[0]].result[value[1]]
    if name == "equals":
        return measurements[value[0]].result == value[1]
    args = [_deserialize_expression(arg, measurements) for arg in value]
    operations = {"not": Not, "and": And, "or": Or, "xor": Xor}
    return operations[name](*args)


def _find_measurement(measurements: list, result: MeasurementResult) -> int | None:
    """Index of the last measurement gate in ``measurements`` that writes ``result``.

    Measurement gates copied with :meth:`qibo.gates.M.on_qubits` share their
    result object, so a condition always refers to the closest preceding one.
    """
    for index in range(len(measurements) - 1, -1, -1):
        if measurements[index].result is result:
            return index
    return None
