"""Runnable examples of native conditional execution with Qibo's NumPy backend.

Run from the repository root with:

    .venv/bin/python examples/conditional_measurement_example/native_conditional_execution.py
"""

from qibo import Circuit, gates, get_backend, set_backend


def stochastic_if_else(nshots=1000):
    """Correlate qubit 1 with a random mid-circuit measurement of qubit 0."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    first = circuit.add(gates.M(0, collapse=True))
    circuit.add(
        gates.IfElse(
            condition=first.is_one(),
            then_do=gates.X(1),
            else_do=gates.I(1),
        )
    )
    circuit.add(gates.M(0, 1, register_name="result"))

    frequencies = circuit(nshots=nshots).frequencies(binary=True)
    assert set(frequencies).issubset({"00", "11"})
    assert sum(frequencies.values()) == nshots
    return frequencies


def deterministic_if_else():
    """Force the true branch and verify that it applies X to qubit 1."""
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    first = circuit.add(gates.M(0, collapse=True))
    circuit.add(
        gates.IfElse(
            condition=first.is_one(),
            then_do=gates.X(1),
            else_do=gates.I(1),
        )
    )
    circuit.add(gates.M(0, 1, register_name="result"))

    frequencies = circuit(nshots=10).frequencies(binary=True)
    assert frequencies == {"11": 10}
    return frequencies


def switch_on_syndrome():
    """Measure syndrome 10, match grouped case (1, 2), and apply X to qubit 2."""
    circuit = Circuit(3)
    circuit.add(gates.X(0))
    syndrome = circuit.add(gates.M(0, 1, collapse=True))
    circuit.add(
        gates.Switch(syndrome)
        .case(0, gates.I(2))
        .case((1, 2), gates.X(2))
        .case(3, gates.Z(2))
        .default([])
    )
    circuit.add(gates.M(0, 1, 2, register_name="result"))

    frequencies = circuit(nshots=10).frequencies(binary=True)
    assert frequencies == {"101": 10}
    return frequencies


def if_elif_else_on_syndrome():
    """Measure syndrome 01 and select the first branch of an IfChain."""
    circuit = Circuit(3)
    circuit.add(gates.X(1))
    syndrome = circuit.add(gates.M(0, 1, collapse=True))
    correction = (
        gates.IfChain()
        .if_(syndrome.equals("01"), gates.X(2))
        .elif_(syndrome.equals("10"), gates.Z(2))
        .else_([])
    )
    circuit.add(correction)
    circuit.add(gates.M(0, 1, 2, register_name="result"))

    frequencies = circuit(nshots=10).frequencies(binary=True)
    assert frequencies == {"011": 10}
    return frequencies


if __name__ == "__main__":
    set_backend("numpy")
    get_backend().set_seed(123)

    print("stochastic IfElse:", stochastic_if_else())
    print("deterministic IfElse:", deterministic_if_else())
    print("Switch:", switch_on_syndrome())
    print("IfChain:", if_elif_else_on_syndrome())
