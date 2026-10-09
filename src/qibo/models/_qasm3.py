"""Conversion of :class:`qibo.models.circuit.Circuit` to OpenQASM 3."""

import re

from qibo import __version__, gates
from qibo.config import raise_error

# Gates of the OpenQASM 3 standard library ``stdgates.inc``.
STDGATES = frozenset(
    {
        "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz",
        "cx", "cy", "cz", "cp", "crx", "cry", "crz", "ch", "swap", "ccx", "cswap",
        "cu", "CX", "phase", "cphase", "id", "u1", "u2", "u3",
    }
)  # fmt: skip

# qibo labels with an exact equivalent in ``stdgates.inc``, as functions of the
# gate parameters returning the ``stdgates.inc`` name and parameters.
STDGATE_EQUIVALENTS = {
    "cu1": lambda lam: ("cp", (lam,)),
    # qibo's U3 differs from OpenQASM's U by a phase exp(-i (phi + lam) / 2),
    # which is a relative phase once the gate is controlled
    "cu3": lambda theta, phi, lam: ("cu", (theta, phi, lam, -(phi + lam) / 2)),
}

# qibo labels renamed to follow Qiskit's naming of the same gate.
QISKIT_NAMES = {"prx": "r"}

# Definitions of the gates that are not in ``stdgates.inc``, written in terms
# of ``stdgates.inc`` gates and named as in Qiskit when Qiskit has the gate.
# Each entry holds the definition and the custom gates it uses. Parameters are
# named so that their alphabetical and positional orders agree, because
# Qiskit's importer binds them in alphabetical order.
DEFINITIONS = {
    "sxdg": ("gate sxdg _q0 { rx(-pi/2) _q0; gphase(-pi/4); }", ()),
    "csx": ("gate csx _q0, _q1 { crx(pi/2) _q0, _q1; p(pi/4) _q0; }", ()),
    "csxdg": ("gate csxdg _q0, _q1 { crx(-pi/2) _q0, _q1; p(-pi/4) _q0; }", ()),
    "rxx": (
        (
            "gate rxx(_p0) _q0, _q1 { h _q0; h _q1; cx _q0, _q1; rz(_p0) _q1; "
            "cx _q0, _q1; h _q0; h _q1; }"
        ),
        (),
    ),
    "ryy": (
        (
            "gate ryy(_p0) _q0, _q1 { rx(pi/2) _q0; rx(pi/2) _q1; cx _q0, _q1; "
            "rz(_p0) _q1; cx _q0, _q1; rx(-pi/2) _q0; rx(-pi/2) _q1; }"
        ),
        (),
    ),
    "rzz": ("gate rzz(_p0) _q0, _q1 { cx _q0, _q1; rz(_p0) _q1; cx _q0, _q1; }", ()),
    "iswap": (
        "gate iswap _q0, _q1 { s _q0; s _q1; h _q0; cx _q0, _q1; cx _q1, _q0; h _q1; }",
        (),
    ),
    "fswap": ("gate fswap _q0, _q1 { swap _q0, _q1; cz _q0, _q1; }", ()),
    "ccz": ("gate ccz _q0, _q1, _q2 { h _q2; ccx _q0, _q1, _q2; h _q2; }", ()),
    "r": ("gate r(_p0, _p1) _q0 { U(_p0, _p1 - pi/2, pi/2 - _p1) _q0; }", ()),
    "gpi": ("gate gpi(_p0) _q0 { U(pi, _p0, pi - _p0) _q0; }", ()),
    "gpi2": ("gate gpi2(_p0) _q0 { U(pi/2, _p0 - pi/2, pi/2 - _p0) _q0; }", ()),
    "ms": (
        (
            "gate ms(_p0, _p1, _p2) _q0, _q1 { rz(-_p0) _q0; rz(-_p1) _q1; "
            "rxx(_p2) _q0, _q1; rz(_p0) _q0; rz(_p1) _q1; }"
        ),
        ("rxx",),
    ),
}

# Identifiers that cannot be used as register names.
RESERVED = frozenset(
    {
        "OPENQASM", "include", "defcalgrammar", "def", "cal", "defcal", "gate",
        "extern", "box", "let", "break", "continue", "if", "else", "end",
        "return", "for", "while", "in", "switch", "case", "default", "input",
        "output", "const", "readonly", "mutable", "qreg", "qubit", "creg",
        "bool", "bit", "int", "uint", "float", "angle", "complex", "array",
        "void", "duration", "stretch", "gphase", "inv", "pow", "ctrl",
        "negctrl", "dim", "durationof", "delay", "reset", "measure", "barrier",
        "true", "false", "pi", "tau", "euler", "U", "q", "sizeof", "nop",
    }
) | STDGATES | frozenset(DEFINITIONS)  # fmt: skip

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class QASM3Exporter:
    """Writes a circuit as an OpenQASM 3 program.

    Measurements without ``collapse`` are written at the end of the program,
    as in :meth:`qibo.models.circuit.Circuit.to_qasm`. Mid-circuit
    measurements (``collapse=True``, or measurements whose outcome is used by
    a :class:`qibo.gates.Condition`) are written in place, each into its own
    ``bit`` register. Their bit ``i`` holds ``result.symbols[i]``.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to convert.
        reuse_registers (bool): if ``True``, a mid-circuit measurement writes
            into the register of an earlier one of the same size whose value is
            no longer needed, instead of declaring a new register.
    """

    def __init__(self, circuit, reuse_registers: bool = False):
        self.circuit = circuit
        self.reuse_registers = reuse_registers
        self.queue = list(circuit.queue)
        # custom gate definitions used by the program
        self.definitions = []
        self._analyse()
        self._allocate_registers()

    def _analyse(self):
        """Finds the mid-circuit measurements and what each condition reads."""
        latest = {}
        # (position of a condition, id of a condition) -> position of its measurement
        self.sources = {}
        # position of a measurement -> position of its last condition
        self.last_use = {}
        for position, gate in enumerate(self.queue):
            if isinstance(gate, gates.M):
                latest[id(gate.result)] = position
            elif isinstance(gate, gates.Condition):
                for condition in gate._nested_conditions():
                    source = latest.get(id(condition.result))
                    if source is None:
                        raise_error(
                            ValueError,
                            "A ``Condition`` depends on a measurement that is not "
                            + "performed before it in the circuit.",
                        )
                    self.sources[(position, id(condition))] = source
                    self.last_use[source] = position

        self.mid_circuit = [
            position
            for position, gate in enumerate(self.queue)
            if isinstance(gate, gates.M)
            and (gate.collapse or position in self.last_use)
        ]
        mid_circuit = set(self.mid_circuit)
        self.final = [
            gate
            for position, gate in enumerate(self.queue)
            if isinstance(gate, gates.M) and position not in mid_circuit
        ]

    def _allocate_registers(self):
        """Assigns a ``bit`` register to every measurement."""
        # registers of measurements at the end keep their names, so they are
        # named first and declared after the mid-circuit ones
        final = []
        taken = set()
        for gate in self.final:
            name = _check_identifier(gate.register_name)
            if name in taken:
                raise_error(KeyError, f"Register {name} is used by two measurements.")
            taken.add(name)
            final.append((name, len(gate.target_qubits)))
        self.declarations = []

        # position of a mid-circuit measurement -> register name
        self.registers = {}
        # registers available for reuse: [name, size, position of last use]
        pool = []
        for position in self.mid_circuit:
            gate = self.queue[position]
            size = len(gate.target_qubits)
            register = None
            if self.reuse_registers:
                for entry in pool:
                    if entry[1] == size and entry[2] < position:
                        register = entry
                        break
            if register is None:
                name = _check_identifier(gate.register_name)
                suffix = 0
                while name in taken:
                    suffix += 1
                    name = f"{gate.register_name}_{suffix}"
                taken.add(name)
                self.declarations.append((name, size))
                register = [name, size, position]
                pool.append(register)
            register[2] = self.last_use.get(position, position)
            self.registers[position] = register[0]
        self.declarations.extend(final)

    def dumps(self) -> str:
        """Returns the OpenQASM 3 program."""
        statements = []
        for position, gate in enumerate(self.queue):
            if isinstance(gate, gates.M):
                if position in self.registers:
                    register = self.registers[position]
                    # the bits of collapsing measurements follow the qubit order
                    for i, qubit in enumerate(sorted(gate.target_qubits)):
                        statements.append(f"{register}[{i}] = measure q[{qubit}];")
            elif isinstance(gate, gates.Condition):
                statements.extend(self._condition(gate, position, 0))
            else:
                statements.extend(self._gate(gate))

        for gate in self.final:
            for i, qubit in enumerate(gate.target_qubits):
                statements.append(f"{gate.register_name}[{i}] = measure q[{qubit}];")

        code = [
            f"// Generated by QIBO {__version__}",
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
        ]
        code.extend(DEFINITIONS[name][0] for name in self.definitions)
        code.append(f"qubit[{self.circuit.nqubits}] q;")
        code.extend(f"bit[{size}] {name};" for name, size in self.declarations)
        code.extend(statements)
        return "\n".join(code)

    def _gate(self, gate) -> list[str]:
        """Statements applying a (non-conditional) gate."""
        if gate.is_controlled_by:
            raise_error(
                ValueError,
                f"{gate.__class__.__name__} gate controlled with ``controlled_by`` "
                + "cannot be converted to OpenQASM 3.",
            )
        try:
            label = gate.qasm_label
        except NotImplementedError:
            raise_error(
                NotImplementedError,
                f"{gate.__class__.__name__} is not supported by OpenQASM 3, "
                + "so the circuit cannot be converted.",
            )
        if isinstance(label, tuple):
            label = label[0]

        params = ()
        if isinstance(gate, gates.ParametrizedGate):
            try:
                params = tuple(float(param) for param in gate.parameters)
            except (TypeError, ValueError):
                raise_error(
                    ValueError,
                    f"{gate.__class__.__name__} has symbolic parameters, "
                    + "which cannot be converted to OpenQASM 3.",
                )
        if label in STDGATE_EQUIVALENTS:
            label, params = STDGATE_EQUIVALENTS[label](*params)
        label = QISKIT_NAMES.get(label, label)
        if label not in STDGATES:
            if label not in DEFINITIONS:
                raise_error(
                    NotImplementedError,
                    f"{gate.__class__.__name__} is not supported by OpenQASM 3, "
                    + "so the circuit cannot be converted.",
                )
            self._add_definition(label)

        name = label
        if params:
            name += "(" + ", ".join(str(param) for param in params) + ")"
        if label == "id":
            # ``id`` acts on a single qubit, while ``gates.I`` may act on several
            return [f"{name} q[{qubit}];" for qubit in gate.qubits]
        qubits = ", ".join(f"q[{qubit}]" for qubit in gate.qubits)
        return [f"{name} {qubits};"]

    def _add_definition(self, label: str):
        if label in self.definitions:
            return
        for dependency in DEFINITIONS[label][1]:
            self._add_definition(dependency)
        self.definitions.append(label)

    def _condition(self, condition, position: int, depth: int) -> list[str]:
        """Statements of a conditional block."""
        source = self.sources[(position, id(condition))]
        register = self.registers[source]
        if condition.bit is not None:
            register = f"{register}[{condition.bit}]"

        def test(value):
            if condition.bit is not None:
                return register if value else f"!{register}"
            return f"{register} == {_qasm_integer(value, condition.nbits)}"

        # a case with several values repeats its body for each of them
        branches = [
            (test(value), body) for values, body in condition.cases for value in values
        ]
        indent = "  " * depth
        lines = []
        for i, (expression, body) in enumerate(branches):
            keyword = "if" if i == 0 else "} else if"
            lines.append(f"{indent}{keyword} ({expression}) {{")
            lines.extend(self._block(body, position, depth))
        if condition.default is not None:
            lines.append(f"{indent}}} else {{")
            lines.extend(self._block(condition.default, position, depth))
        lines.append(f"{indent}}}")
        return lines

    def _block(self, body: list, position: int, depth: int) -> list[str]:
        """Indented statements of the gates in a conditional branch."""
        indent = "  " * (depth + 1)
        lines = []
        for gate in body:
            if isinstance(gate, gates.Condition):
                lines.extend(self._condition(gate, position, depth + 1))
            else:
                lines.extend(indent + line for line in self._gate(gate))
        return lines


def _check_identifier(name: str) -> str:
    if not IDENTIFIER.fullmatch(name) or name in RESERVED:
        raise_error(
            NameError,
            f"Register name {name!r} is not a valid OpenQASM 3 identifier.",
        )
    return name


def _qasm_integer(value: int, nbits: int) -> int:
    """Converts a register value from qibo's bit order, where the first bit is
    the most significant, to OpenQASM's, where bit ``0`` is the least significant."""
    return int(format(value, f"0{nbits}b")[::-1], 2)
