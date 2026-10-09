"""Gates applied conditionally on the outcome of mid-circuit measurements."""

import copy
from numbers import Integral

from qibo.config import raise_error
from qibo.gates.abstract import Gate, SpecialGate
from qibo.gates.measurements import M
from qibo.measurements import MeasurementResult, MeasurementSymbol

__all__ = ["Condition"]


class Condition(Gate):
    """Applies gates only if a mid-circuit measurement has a given outcome.

    The outcome is referenced through the objects returned when a measurement
    is added to a circuit, in the same way as measurement symbols are used to
    parametrize gates:

    - a single measured bit, :class:`qibo.measurements.MeasurementSymbol`
      (``result.symbols[i]``), compared against ``0`` or ``1``;
    - the full measurement register, :class:`qibo.measurements.MeasurementResult`,
      compared against an integer or a bitstring.

    Register values follow qibo's bitstring convention: ``result.symbols[0]``
    is the most significant bit. For example, after
    ``result = circuit.add(gates.M(0, 1, collapse=True))`` the value ``2``
    (or ``"10"``) means that the first measured bit is ``1`` and the second is ``0``.

    A condition can test a single value (``if``), or several values each with
    its own gates, like a ``switch`` statement:

    - ``Condition(source, value, body)`` applies ``body`` if the outcome is ``value``;
    - ``Condition(source, {value: body, ...})`` applies the body of the matching value;
    - ``Condition(source, [value, ...], [body, ...])`` is equivalent, with the values
      and bodies given as two sequences of the same length.

    A value can also be a tuple of values that share the same body, as in
    ``{(1, 2): body}``. If ``default`` is given, it is applied when no value
    matches (``else``). A body is a gate, a list of gates, a generator of
    gates such as :meth:`qibo.models.circuit.Circuit.on_qubits`, or a
    :class:`qibo.models.circuit.Circuit`, whose gates are used. Bodies may contain
    other conditions, but not measurements.

    Simulation of conditional gates is not supported yet. Circuits containing
    them can be exported with :meth:`qibo.models.circuit.Circuit.to_qasm3`.

    Example:
        .. code-block:: python

            from qibo import Circuit, gates

            circuit = Circuit(3)
            circuit.add(gates.H(0))
            circuit.add(gates.H(1))
            bit = circuit.add(gates.M(0, collapse=True))
            # flip qubit 2 if qubit 0 was measured in state 1
            circuit.add(gates.Condition(bit.symbols[0], 1, gates.X(2)))
            # if/else with several gates in each branch
            circuit.add(
                gates.Condition(
                    bit.symbols[0], 0, [gates.H(2), gates.S(2)], default=gates.Z(2)
                )
            )
            # switch on the outcome of a two-qubit measurement
            register = circuit.add(gates.M(0, 1, collapse=True))
            circuit.add(
                gates.Condition(
                    register,
                    {0: gates.X(2), (1, 2): [gates.H(2), gates.T(2)]},
                    default=gates.Y(2),
                )
            )
            circuit.add(gates.M(2))
            print(circuit.to_qasm3())

    Args:
        source (:class:`qibo.measurements.MeasurementSymbol` or :class:`qibo.measurements.MeasurementResult`):
            measured bit or register that the condition depends on.
        value (int or str or tuple or dict or list): outcome for which ``body``
            is applied. For a single bit, ``0`` or ``1`` (``False`` or ``True``).
            For a register, an integer in :math:`[0, 2^{n})` or a bitstring of
            length :math:`n`. A dictionary maps values to bodies, and a list or
            tuple gives one value per entry of ``body``.
        body (:class:`qibo.gates.Gate` or list or :class:`qibo.models.circuit.Circuit`, optional):
            gates applied when the outcome is ``value``. When ``value`` is a list
            or tuple, a list with one body per value. Not used when ``value`` is a
            dictionary.
        default (:class:`qibo.gates.Gate` or list or :class:`qibo.models.circuit.Circuit`, optional):
            gates applied when the outcome matches no value. Defaults to ``None``.
    """

    def __init__(self, source, value, body=None, default=None):
        super().__init__()
        self._set_source(source)
        if isinstance(value, dict):
            if body is not None:
                raise_error(
                    ValueError,
                    "``body`` cannot be given when the cases are a dictionary.",
                )
            pairs = list(value.items())
        elif isinstance(value, (list, tuple, range)):
            if not isinstance(body, (list, tuple)) or len(body) != len(value):
                raise_error(
                    ValueError,
                    "When several values are given, ``body`` must be a list "
                    + "or tuple with one body per value.",
                )
            pairs = list(zip(value, body))
        else:
            if body is None:
                raise_error(ValueError, "Missing the ``body`` of the condition.")
            pairs = [(value, body)]
        if not pairs:
            raise_error(ValueError, "A ``Condition`` needs at least one value.")

        cases = [(self._parse_values(values), _body(body)) for values, body in pairs]
        seen = set()
        for values, _ in cases:
            for outcome in values:
                if outcome in seen:
                    raise_error(
                        ValueError,
                        f"Value {outcome} is used by more than one case of the condition.",
                    )
                seen.add(outcome)
        self._setup(cases, None if default is None else _body(default))

    def _parse_values(self, values) -> tuple[int, ...]:
        """Values of a case, as a tuple of integers in qibo's bit ordering."""
        if isinstance(values, (list, tuple, set, frozenset)):
            if not values:
                raise_error(
                    ValueError, "A case of a condition needs at least one value."
                )
            return tuple(self._parse_value(value) for value in values)
        return (self._parse_value(values),)

    @classmethod
    def _from_cases(cls, source, cases, default):
        """Creates a condition from already validated cases."""
        gate = cls.__new__(cls)
        Gate.__init__(gate)
        gate._set_source(source)
        gate._setup(cases, default)
        return gate

    def _set_source(self, source):
        if not isinstance(source, (MeasurementSymbol, MeasurementResult)):
            raise_error(
                TypeError,
                "The source of a ``Condition`` must be a measurement symbol "
                + "(``result.symbols[i]``) or the measurement result returned by "
                + f"``circuit.add(gates.M(...))``, but is {type(source)}.",
            )
        self.source = source

    def _setup(self, cases, default):
        self.name = "condition"
        self.draw_label = "if" if len(cases) == 1 else "switch"
        self.cases = cases
        self.default = default
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

    def _parse_value(self, value) -> int:
        """Converts a case value to an integer in qibo's bit ordering."""
        nbits = self.nbits
        if isinstance(value, str):
            if len(value) != nbits or set(value) - {"0", "1"}:
                raise_error(
                    ValueError,
                    f"Bitstring {value!r} is not a valid outcome of {nbits} measured bits.",
                )
            return int(value, 2)
        if not isinstance(value, Integral):
            raise_error(
                TypeError,
                f"Condition values must be integers or bitstrings, but got {value!r}.",
            )
        if not 0 <= value < 2**nbits:
            raise_error(
                ValueError,
                f"Value {value} is not a possible outcome of {nbits} measured bits.",
            )
        return int(value)

    @property
    def result(self) -> MeasurementResult:
        """Measurement result that the condition depends on."""
        if isinstance(self.source, MeasurementSymbol):
            return self.source.result
        return self.source

    @property
    def bit(self) -> int | None:
        """Index of the measured bit, or ``None`` if the full register is used."""
        if isinstance(self.source, MeasurementSymbol):
            return self.source.index
        return None

    @property
    def nbits(self) -> int:
        """Number of measured bits that the condition depends on."""
        return 1 if self.bit is not None else len(self.result.target_qubits)

    @property
    def gates(self) -> list:
        """All gates of all branches, without descending into nested conditions."""
        gates = [gate for _, body in self.cases for gate in body]
        if self.default is not None:
            gates += self.default
        return gates

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
        """New condition with the same source and ``func`` applied to each branch."""
        cases = [(values, func(body)) for values, body in self.cases]
        default = None if self.default is None else func(self.default)
        return self._from_cases(self.source, cases, default)

    def on_qubits(self, qubit_map: dict) -> "Condition":
        """Creates the same condition with its gates acting on different qubits.

        The measurement the condition depends on is not changed, so mapping a
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
        """Copy whose gates are new objects and whose source is remapped.

        Args:
            results (dict): maps ``id`` of old measurement results to the new
                :class:`qibo.measurements.MeasurementResult` objects.
        """
        result = results.get(id(self.result), self.result)
        source = result if self.bit is None else result.symbols[self.bit]

        def copy_body(body):
            return [
                g._copy(results) if isinstance(g, Condition) else copy.copy(g)
                for g in body
            ]

        cases = [(values, copy_body(body)) for values, body in self.cases]
        default = None if self.default is None else copy_body(self.default)
        return self._from_cases(source, cases, default)

    @property
    def raw(self) -> dict:
        raise_error(
            NotImplementedError,
            "A ``Condition`` refers to a measurement of its circuit, so it can only "
            + "be serialized together with it using ``Circuit.raw``.",
        )

    def _serialize(self, measurements: list) -> dict:
        """Serializes the condition referring to its measurement by position.

        Args:
            measurements (list): measurement gates that precede the condition
                in the circuit, in order.
        """
        index = _find_measurement(measurements, self.result)
        if index is None:
            raise_error(
                ValueError,
                "Cannot serialize a ``Condition`` whose measurement is not "
                + "performed before it in the circuit.",
            )

        def serialize_body(body):
            return [
                g._serialize(measurements) if isinstance(g, Condition) else g.raw
                for g in body
            ]

        return {
            "_class": "Condition",
            "name": self.name,
            "source": {"measurement": index, "bit": self.bit},
            "cases": [
                {"values": list(values), "gates": serialize_body(body)}
                for values, body in self.cases
            ],
            "default": None if self.default is None else serialize_body(self.default),
        }

    @classmethod
    def _deserialize(cls, raw: dict, measurements: list) -> "Condition":
        """Counter-part of :meth:`_serialize`."""
        result = measurements[raw["source"]["measurement"]].result
        bit = raw["source"]["bit"]
        source = result if bit is None else result.symbols[bit]

        def deserialize_body(body):
            return [
                (
                    cls._deserialize(g, measurements)
                    if g["_class"] == "Condition"
                    else Gate.from_dict(g)
                )
                for g in body
            ]

        cases = [
            (tuple(case["values"]), deserialize_body(case["gates"]))
            for case in raw["cases"]
        ]
        default = raw["default"]
        default = None if default is None else deserialize_body(default)
        return cls._from_cases(source, cases, default)

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


def _body(body) -> list:
    """List of gates of a conditional branch.

    Accepts a gate, a circuit, or (nested) iterables of gates and circuits.
    """
    from qibo.models.circuit import Circuit

    if isinstance(body, Gate):
        return [body]
    if isinstance(body, Circuit):
        return list(body.queue)
    if isinstance(body, (str, bytes)) or not hasattr(body, "__iter__"):
        raise_error(
            TypeError, f"Conditional operations must be gates, not {type(body)}."
        )
    return [gate for item in body for gate in _body(item)]


def _find_measurement(measurements: list, result: MeasurementResult) -> int | None:
    """Index of the last measurement gate in ``measurements`` that writes ``result``.

    Measurement gates copied with :meth:`qibo.gates.M.on_qubits` share their
    result object, so a condition always refers to the closest preceding one.
    """
    for index in range(len(measurements) - 1, -1, -1):
        if measurements[index].result is result:
            return index
    return None
