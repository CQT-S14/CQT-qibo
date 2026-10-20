"""Export conditional operations to OpenQASM 3 and execute them with Qiskit Aer.

Qibo can now execute classical control flow natively with its NumPy backend and
with qibojit's Numba platform. This example focuses instead on interoperability:
it constructs the conditional circuit using Qibo's public API, shows both QASM 3
export modes, and, when Qiskit is installed, executes the compatible export with
Qiskit Aer.

For focused native Qibo examples using ``IfElse``, ``Switch``, and ``IfChain``,
see ``native_conditional_execution.py`` in this directory.
"""

from qibo import Circuit, conditions, gates

circuit = Circuit(4)
circuit.add([gates.H(0), gates.H(1), gates.H(2)])

# mid-circuit measurements: a single qubit is a bit, several qubits a register
first = circuit.add(gates.M(0, collapse=True))
register = circuit.add(gates.M(1, 2, collapse=True))

# active reset: flip qubit 0 back to |0> if it was measured in |1>
circuit.add(gates.IfElse(first.is_one(), then_do=gates.X(0)))
# boolean expression of bits, with several gates
circuit.add(
    gates.IfElse(
        conditions.all_of(first.is_one(), register.bit(1).is_zero()),
        then_do=[gates.H(3), gates.S(3)],
    )
)
# comparison of a register: qibo's bit order, register[0] is the most significant
circuit.add(gates.IfElse(register.not_equals("01"), then_do=gates.Z(3), else_do=[]))
# switch on the value of a register; a tuple of values shares the gates
circuit.add(
    gates.Switch(register)
    .case(0, gates.Z(3))
    .case((1, 2), [gates.H(3), gates.T(3)])
    .default([])
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
