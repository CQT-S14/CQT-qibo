# Conditional operations

Qibo circuits can represent classical control flow driven by mid-circuit
measurement outcomes. The public API has two operations:

- `gates.IfElse` chooses between two bodies using a Boolean condition.
- `gates.Switch` chooses a body using the value of a measured bit or register.

`gates.IfChain` is a fluent builder for `if`/`elif`/`else`; it lowers to nested
`IfElse` operations, so it does not introduce a third control-flow node.

> Native execution is available with Qibo's NumPy backend. Other backends,
> compiled circuits, and distributed execution fail explicitly until they gain
> dedicated support. Circuits can also be exported with `Circuit.to_qasm3()`.

## Measurement expressions

Adding a measurement returns a `MeasurementResult`. Use its explicit methods to
say what the condition means:

```python
from qibo import Circuit, conditions, gates

circuit = Circuit(4)
first = circuit.add(gates.M(0, collapse=True))
syndrome = circuit.add(gates.M(1, 2, collapse=True))

first.is_one()               # the one-bit result is 1
first.is_zero()              # the one-bit result is 0
syndrome.bit(0).is_one()     # the first bit of the register is 1
syndrome.bit(1).is_zero()    # the second bit of the register is 0
syndrome.equals("01")        # the whole register is 01
syndrome.not_equals(2)       # the whole register is not 10
```

Register bit zero is the most-significant bit. Therefore, for a measurement of
qubits 1 and 2, `syndrome.equals("10")` and `syndrome.equals(2)` mean that
qubit 1 was measured as one and qubit 2 as zero. Bitstrings must have exactly
the register width.

Combine expressions using readable, namespaced helpers:

```python
needs_correction = conditions.all_of(
    first.is_one(),
    conditions.any_of(
        syndrome.bit(0).is_one(),
        syndrome.bit(1).is_one(),
    ),
)

odd_parity = conditions.xor_of(
    syndrome.bit(0).is_one(),
    syndrome.bit(1).is_one(),
)

not_first = conditions.not_(first.is_one())
```

The shorthand forms remain available for compact code: `result[i]`, `==`,
`!=`, `&`, `|`, `^`, and `~`. For example,
`first.is_one() & syndrome.bit(0).is_zero()` is equivalent to
`first & ~syndrome[0]`. The explicit API is recommended for application code
because it is harder to confuse with Python Boolean evaluation.

Calling `is_one()` or `is_zero()` on a multi-bit result is an error. Select a
bit or compare the whole register instead.

## If/else

Use `then_do` and `else_do` for branch bodies:

```python
circuit.add(
    gates.IfElse(
        condition=syndrome.equals("01"),
        then_do=[gates.X(3), gates.T(3)],
        else_do=gates.Z(3),
    )
)
```

`else_do` defaults to `[]`, an empty no-op branch:

```python
circuit.add(
    gates.IfElse(
        condition=first.is_one(),
        then_do=gates.X(3),
    )
)
```

A body may be one gate, nested iterables of gates, or a `Circuit`. Measurements,
callbacks, and other special gates cannot be placed inside branches. Control
flow may be nested.

## If/elif/else chains

For several mutually exclusive conditions, the fluent builder keeps the source
close to ordinary control-flow syntax:

```python
correction = (
    gates.IfChain()
    .if_(syndrome.equals("01"), gates.X(3))
    .elif_(syndrome.equals("10"), gates.Z(3))
    .elif_(syndrome.equals("11"), [gates.X(3), gates.Z(3)])
    .else_([])
)
circuit.add(correction)
```

`else_()` completes the builder and returns nested `IfElse` operations. An
unfinished `IfChain` cannot be added to a circuit. Use `.else_([])` when the
intended final branch is a no-op.

## Switch

Use the fluent API when cases are assembled progressively:

```python
circuit.add(
    gates.Switch(syndrome)
    .case(0, gates.I(3))
    .case((1, 2), gates.X(3))  # values 1 and 2 share this body
    .case(3, gates.Z(3))
    .default([])
)
```

The default is mandatory in the fluent API. This makes an omitted case visible
in code; `.default([])` explicitly acknowledges a no-op. Cases can use either
all integers or all bitstrings, but cannot mix the two forms. Values cannot be
duplicated across cases.

The complete constructor is useful when the mapping already exists:

```python
operation = gates.Switch(
    target=syndrome,
    cases={
        "00": gates.I(3),
        ("01", "10"): gates.X(3),
        "11": gates.Z(3),
    },
    default_do=[],
)
circuit.add(operation)
```

`Switch` also accepts a single bit as its target:

```python
operation = (
    gates.Switch(syndrome.bit(0))
    .case(1, gates.X(3))
    .default([])
)
```

## Native NumPy execution

Select the NumPy backend and execute the circuit normally. During every shot,
Qibo collapses the referenced measurement, evaluates the condition using that
shot's outcome, applies exactly one branch, and continues with the circuit:

```python
from qibo import Circuit, gates, set_backend

set_backend("numpy")

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

result = circuit(nshots=1000)
print(result.frequencies(binary=True))  # only 00 and 11
```

Every measurement referenced by an `IfElse` or `Switch`, including nested
operations, must occur earlier in the same circuit. Reusing a circuit is safe:
measurement outcomes are cleared at the beginning of each execution.

For state-vector execution with several shots, add a final non-collapsing
measurement as above so Qibo can return shot outcomes. To retrieve an averaged
final quantum state after stochastic control flow, use a density-matrix circuit.

## Inspecting branch selection

Both operations provide `select(classical_state)`. It returns the list of gates
in the selected branch without applying them:

```python
operation = gates.IfElse(
    syndrome.equals("01"),
    then_do=gates.X(3),
    else_do=gates.Z(3),
)

selected = operation.select({syndrome: "01"})
assert isinstance(selected[0], gates.X)
```

For a reusable state, construct `conditions.ClassicalState`:

```python
state = conditions.ClassicalState({first: 1, syndrome: "10"})
selected = operation.select(state)
```

Every measurement used by the expression or target must have an outcome in the
state. Missing outcomes fail with an error identifying the unresolved
measurement.

## Exporting OpenQASM 3

The default export uses nested `if`/`else` statements compatible with Qiskit's
OpenQASM 3 importer:

```python
qasm = circuit.to_qasm3()
```

Native OpenQASM 3 Boolean operators and `switch` statements are available with:

```python
qasm = circuit.to_qasm3(qiskit_compatible=False)
```

`IfElse` exports an explicit `else`, and native `Switch` exports an explicit
`default`, including when the branch is empty. A measurement referenced by a
control-flow operation must occur earlier in the same circuit.

See the complete runnable template in
`examples/conditional_measurement_example/conditional_operations_api.py`.
