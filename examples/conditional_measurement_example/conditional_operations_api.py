"""Template showing construction, native NumPy execution, and QASM 3 export."""

from qibo import Circuit, conditions, gates, set_backend

circuit = Circuit(5)

# Measurements return readable references to later classical outcomes.
first = circuit.add(gates.M(0, collapse=True))
syndrome = circuit.add(gates.M(1, 2, collapse=True))

# One bit: read this as "if first is one, then do X; otherwise do Z".
circuit.add(
    gates.IfElse(
        condition=first.is_one(),
        then_do=gates.X(3),
        else_do=gates.Z(3),
    )
)

# A register comparison. Bit 0 is the most-significant measured bit.
circuit.add(
    gates.IfElse(
        condition=syndrome.equals("01"),
        then_do=[gates.H(3), gates.T(3)],
        else_do=[],  # explicit no-op
    )
)

# Compose conditions with named helpers. This reads as normal prose.
needs_correction = conditions.all_of(
    first.is_zero(),
    conditions.any_of(
        syndrome.bit(0).is_one(),
        syndrome.bit(1).is_one(),
    ),
)
circuit.add(gates.IfElse(needs_correction, then_do=gates.X(4)))

# A readable if/elif/else chain. else_ completes it and returns an IfElse.
circuit.add(
    gates.IfChain()
    .if_(syndrome.equals("01"), gates.X(4))
    .elif_(syndrome.equals("10"), gates.Z(4))
    .else_([gates.H(4), gates.S(4)])
)

# A switch with grouped values and a required, explicit default.
circuit.add(
    gates.Switch(syndrome)
    .case(0, gates.I(4))
    .case((1, 2), gates.X(4))
    .case(3, gates.Z(4))
    .default([])
)

# The complete constructor is equivalent to the fluent switch above.
standalone_switch = gates.Switch(
    target=syndrome,
    cases={0: gates.I(4), (1, 2): gates.X(4), 3: gates.Z(4)},
    default_do=[],
)

# Branch selection is backend-independent and does not simulate quantum gates.
selected_gates = standalone_switch.select({syndrome: "10"})
print("Selected branch:", selected_gates)

# A compact circuit executing conditional control flow natively with NumPy.
set_backend("numpy")
native_circuit = Circuit(2)
native_circuit.add(gates.H(0))
native_first = native_circuit.add(gates.M(0, collapse=True))
native_circuit.add(
    gates.IfElse(
        native_first.is_one(),
        then_do=gates.X(1),
        else_do=gates.I(1),
    )
)
native_circuit.add(gates.M(0, 1, register_name="result"))
native_result = native_circuit(nshots=1000)
print("Native NumPy frequencies:", native_result.frequencies(binary=True))

# Export remains available. Set qiskit_compatible=False for native switch syntax.
print(circuit.to_qasm3())
print(circuit.to_qasm3(qiskit_compatible=False))
