"""Conditional operations on mid-circuit measurements, exported to OpenQASM 3.

Simulation of ``gates.Condition`` is not supported yet, so the circuit is
converted to OpenQASM 3 and, if Qiskit is installed, executed with Qiskit Aer.
"""

from qibo import Circuit, gates

circuit = Circuit(3)
circuit.add(gates.H(0))
circuit.add(gates.H(1))

# mid-circuit measurement of qubit 0, its outcome is ``bit.symbols[0]``
bit = circuit.add(gates.M(0, collapse=True))
# active reset: flip qubit 0 back to |0> if it was measured in |1>
circuit.add(gates.Condition(bit.symbols[0], 1, gates.X(0)))
# if/else with several gates per branch
circuit.add(
    gates.Condition(bit.symbols[0], 0, [gates.H(2), gates.S(2)], default=gates.X(2))
)

# switch on the outcome of a two-qubit measurement: register values use
# qibo's bit order, where ``register.symbols[0]`` is the most significant bit
register = circuit.add(gates.M(0, 1, collapse=True))
circuit.add(
    gates.Condition(
        register,
        {0: gates.Z(2), (1, "10"): [gates.H(2), gates.T(2)]},
        default=gates.Y(2),
    )
)
circuit.add(gates.M(0, 1, 2, register_name="result"))

circuit.draw()
print()
print(circuit.to_qasm3())
print()
# the same program with a ``switch`` statement and reused registers
print(circuit.to_qasm3(switch_statement=True, reuse_registers=True))

try:
    import qiskit.qasm3
    from qiskit_aer import AerSimulator
except ImportError:
    print("\nInstall qiskit, qiskit-qasm3-import and qiskit-aer to run the circuit.")
else:
    qiskit_circuit = qiskit.qasm3.loads(circuit.to_qasm3())
    counts = AerSimulator().run(qiskit_circuit, shots=1000).result().get_counts()
    print("\nQiskit Aer counts (registers printed last-declared first):")
    print(counts)
