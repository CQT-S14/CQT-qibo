from qibo import Circuit, gates
from qibo.measurements import MeasurementResult
import qiskit.qasm2


# Build the circuit
circuit = Circuit(1, density_matrix=True)
# Add some gates
circuit.add(gates.H(0))
output, register_name = circuit.add(gates.M(0, collapse=True))
circuit.add(gates.Condition(register_name, 1, gates.X(0)))
circuit.add(gates.H(0))

circuit.add(gates.M(0))

print(circuit.to_qasm3())

# Execute the circuit and obtain the final state
result = circuit(nshots=100) # circuit.execute(initial_state) also works

print(circuit.to_qasm())

