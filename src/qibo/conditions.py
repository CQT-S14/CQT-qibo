"""Readable helpers for conditions on mid-circuit measurement outcomes."""

from collections.abc import Mapping

from sympy.logic.boolalg import And, Not, Or, Xor

from qibo.config import raise_error
from qibo.measurements import MeasurementResult, parse_outcome

__all__ = ["ClassicalState", "all_of", "any_of", "not_", "xor_of"]


class ClassicalState:
    """Measurement outcomes used to inspect which control-flow branch is selected.

    Args:
        outcomes: A mapping or iterable of ``(MeasurementResult, outcome)`` pairs.
            Outcomes may be integers or bitstrings of the measurement's width.

    Measurement results are tracked by identity, matching their role inside a
    circuit. This class selects branches only; it does not simulate a circuit.
    """

    def __init__(self, outcomes=()):
        items = outcomes.items() if isinstance(outcomes, Mapping) else outcomes
        self._values = {}
        for result, outcome in items:
            if not isinstance(result, MeasurementResult):
                raise_error(
                    TypeError,
                    "ClassicalState keys must be MeasurementResult objects, but got "
                    + f"{type(result).__name__}.",
                )
            self._values[id(result)] = parse_outcome(outcome, len(result.target_qubits))

    def value(self, result: MeasurementResult) -> int:
        """Returns the normalized integer outcome for ``result``."""
        try:
            return self._values[id(result)]
        except KeyError:
            raise_error(
                KeyError,
                "The classical state has no outcome for the measurement on qubits "
                + f"{result.target_qubits}.",
            )


def all_of(*expressions):
    """Condition that is true only when every expression is true."""
    _require_expressions("all_of", expressions)
    return And(*expressions)


def any_of(*expressions):
    """Condition that is true when at least one expression is true."""
    _require_expressions("any_of", expressions)
    return Or(*expressions)


def xor_of(*expressions):
    """Exclusive-or condition over two or more expressions."""
    if len(expressions) < 2:
        raise_error(ValueError, "``conditions.xor_of`` needs at least two expressions.")
    _reject_python_booleans("xor_of", expressions)
    return Xor(*expressions)


def not_(expression):
    """Logical negation of a measurement expression."""
    _reject_python_booleans("not_", (expression,))
    return Not(expression)


def _require_expressions(name: str, expressions):
    if not expressions:
        raise_error(ValueError, f"``conditions.{name}`` needs at least one expression.")
    _reject_python_booleans(name, expressions)


def _reject_python_booleans(name: str, expressions):
    if any(isinstance(expression, bool) for expression in expressions):
        raise_error(
            TypeError,
            f"``conditions.{name}`` received a Python bool. Use explicit measurement "
            + "expressions such as ``bit.is_one()`` or ``result.equals(value)``.",
        )
