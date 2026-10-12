"""Conditional operations on mid-circuit measurements, exported to OpenQASM 3.

Simulation of ``gates.Condition`` is not supported yet, so the circuit is
converted to OpenQASM 3 and, if Qiskit is installed, executed with Qiskit Aer.
"""

from qibo import Circuit, gates

circuit = Circuit(4)
circuit.add([gates.H(0), gates.H(1), gates.H(2)])

# mid-circuit measurements: a single qubit is a bit, several qubits a register
first = circuit.add(gates.M(0, collapse=True))
register = circuit.add(gates.M(1, 2, collapse=True))

# active reset: flip qubit 0 back to |0> if it was measured in |1>
circuit.add(gates.Condition(first, gates.X(0)))
# boolean expression of bits, with several gates and an else branch
circuit.add(
    gates.Condition(first & ~register[1], [gates.H(3), gates.S(3)], default=gates.X(3))
)
# comparison of a register: qibo's bit order, register[0] is the most significant
circuit.add(gates.Condition(register != "01", gates.Z(3)))
# switch on the value of a register; a tuple of values shares the gates
circuit.add(
    gates.Condition(
        register,
        {0: gates.Z(3), (1, "10"): [gates.H(3), gates.T(3)]},
        default=gates.Y(3),
    )
)
circuit.add(gates.M(0, 1, 2, 3, register_name="result"))

circuit.draw()
print()
print(circuit.to_qasm3())
print()
# the same program with OpenQASM 3 operators and switch statements
print(circuit.to_qasm3(qiskit_compatible=False, reuse_registers=True))

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
