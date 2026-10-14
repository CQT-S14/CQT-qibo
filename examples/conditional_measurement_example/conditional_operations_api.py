"""Template showing the main conditional-operations API forms.

This example builds and exports control flow. Native Qibo simulation of these
operations is intentionally not implemented yet.
"""

from qibo import Circuit, conditions, gates

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

# Export is supported today. Set qiskit_compatible=False for native switch syntax.
print(circuit.to_qasm3())
print(circuit.to_qasm3(qiskit_compatible=False))
