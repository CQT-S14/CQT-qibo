from collections import Counter
from numbers import Integral

import sympy

from qibo.config import raise_error


def _check_backend(backend):
    """This is only needed due to the circular import with qibo.backends."""
    from qibo.backends import _check_backend

    return _check_backend(backend)


def frequencies_to_binary(frequencies, nqubits):
    return Counter({f"{k:b}".zfill(nqubits): v for k, v in frequencies.items()})


def apply_bitflips(result, p0, p1=None):
    gate = result.measurement_gate
    if p1 is None:
        probs = 2 * (gate._get_bitflip_tuple(gate.qubits, p0),)
    else:
        probs = (
            gate._get_bitflip_tuple(gate.qubits, p0),
            gate._get_bitflip_tuple(gate.qubits, p1),
        )
    noiseless_samples = result.samples()
    probs = result.backend.cast(probs, dtype=result.backend.float64)
    return result.backend.apply_bitflips(noiseless_samples, probs)


class MeasurementSymbol(sympy.Symbol):
    """``sympy.Symbol`` connected to measurement results.

    Used by :class:`qibo.gates.measurements.M` with ``collapse=True`` to allow
    controlling subsequent gates from the measurement results.
    """

    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = f"m{cls._counter}"
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, index, result):
        self.index = index
        self.result = result

    def __getstate__(self):
        return {"index": self.index, "result": self.result, "name": self.name}

    def __setstate__(self, data):
        self.index = data.get("index")
        self.result = data.get("result")
        self.name = data.get("name")

    def _hashable_content(self):
        # copies made by ``pickle`` or ``copy.deepcopy`` keep the name of the
        # symbol, but refer to other measurement results, so sympy (and its
        # caches) must not consider them equal to the original
        return super()._hashable_content() + (id(self),)

    def outcome(self):
        return self.result.samples(binary=True)[-1][self.index]

    def evaluate(self, expr):
        """Substitutes the symbol's value in the given expression.

        Args:
            expr (sympy.Expr): Sympy expression that involves the current
                measurement symbol.
        """
        return expr.subs(self, self.outcome())


class MeasurementValue(sympy.Symbol):
    """``sympy.Symbol`` that is true when a measured register has a given value.

    Created by comparing a :class:`qibo.measurements.MeasurementResult` with an
    outcome, as in ``result == 2``, to be used in the conditions of
    :class:`qibo.gates.Condition`.
    """

    _counter = 0

    def __new__(cls, *args, **kwargs):
        name = f"v{cls._counter}"
        cls._counter += 1
        return super().__new__(cls=cls, name=name)

    def __init__(self, result, value):
        self.result = result
        self.value = value

    def __getstate__(self):
        return {"result": self.result, "value": self.value, "name": self.name}

    def __setstate__(self, data):
        self.result = data.get("result")
        self.value = data.get("value")
        self.name = data.get("name")

    def _hashable_content(self):
        # see :meth:`MeasurementSymbol._hashable_content`
        return super()._hashable_content() + (id(self),)


def parse_outcome(value, nbits: int) -> int:
    """Converts an outcome of ``nbits`` measured bits to an integer.

    Args:
        value (int or str): integer in :math:`[0, 2^{n})`, or bitstring of length
            :math:`n` where the first bit is the most significant.
        nbits (int): number of measured bits.

    Returns:
        int: the outcome as an integer.
    """
    if isinstance(value, str):
        if len(value) != nbits or set(value) - {"0", "1"}:
            raise_error(
                ValueError,
                f"Bitstring {value!r} is not a valid outcome of {nbits} measured bits.",
            )
        return int(value, 2)
    if not isinstance(value, Integral):
        raise_error(
            TypeError,
            f"Measurement outcomes must be integers or bitstrings, but got {value!r}.",
        )
    if not 0 <= value < 2**nbits:
        raise_error(
            ValueError,
            f"Value {value} is not a possible outcome of {nbits} measured bits.",
        )
    return int(value)


class MeasurementResult:
    """Data structure for holding measurement outcomes.

    :class:`qibo.measurements.MeasurementResult` objects can be obtained
    when adding measurement gates to a circuit.

    The outcomes can also be used to build the conditions of
    :class:`qibo.gates.Condition`: ``result[i]`` is the symbol of the
    :math:`i`-th measured bit (the same as ``result.symbols[i]``), a result of a
    single qubit can be used directly as a bit, bits are combined with ``&``,
    ``|``, ``^`` and ``~``, and ``result == value`` (or ``!=``) compares the
    whole register with an integer or a bitstring.

    Args:
        gate (:class:`qibo.gates.M`): Measurement gate associated with
            this result object.
        nshots (int): Number of measurement shots.
        backend (:class:`qibo.backends.abstract.AbstractBackend`): Backend
            to use for calculations.
    """

    def __init__(self, qubits):
        self.target_qubits = qubits
        self.circuit = None

        self._samples = None
        self._frequencies = None
        self._bitflip_p0 = None
        self._bitflip_p1 = None
        self._symbols = None
        self._values = {}

    # ``==`` builds conditions on the outcome, results stay hashable by identity
    __hash__ = object.__hash__

    def __getitem__(self, index: int) -> MeasurementSymbol:
        """Symbol of a measured bit, the same as ``result.symbols[index]``."""
        return self.symbols[index]

    def _bit(self) -> MeasurementSymbol:
        """Symbol of the measured bit, for results of a single qubit."""
        nbits = len(self.target_qubits)
        if nbits != 1:
            raise_error(
                TypeError,
                f"A measurement of {nbits} qubits cannot be used as a single bit. "
                + "Use one of its bits, as in ``result[0]``, or compare it with a "
                + "value, as in ``result == 2``.",
            )
        return self.symbols[0]

    def _sympy_(self):
        return self._bit()

    def __invert__(self):
        return ~self._bit()

    def __and__(self, other):
        return self._bit() & other

    def __rand__(self, other):
        return other & self._bit()

    def __or__(self, other):
        return self._bit() | other

    def __ror__(self, other):
        return other | self._bit()

    def __xor__(self, other):
        return self._bit() ^ other

    def __rxor__(self, other):
        return other ^ self._bit()

    def __eq__(self, value):
        if not isinstance(value, (Integral, str)):
            return NotImplemented
        nbits = len(self.target_qubits)
        value = parse_outcome(value, nbits)
        if nbits == 1:
            return self.symbols[0] if value else ~self.symbols[0]
        if value not in self._values:
            self._values[value] = MeasurementValue(self, value)
        return self._values[value]

    def __ne__(self, value):
        condition = self.__eq__(value)
        return condition if condition is NotImplemented else ~condition

    def __repr__(self):
        qubits = self.target_qubits
        nshots = self.nshots
        return f"MeasurementResult(qubits={qubits}, nshots={nshots})"

    @property
    def raw(self) -> dict:
        samples = self._samples.tolist() if self.has_samples() else self._samples
        return {"samples": samples}

    @property
    def nshots(self) -> int:
        if self.has_samples():
            return len(self._samples)

        if self._frequencies is not None:
            return sum(self._frequencies.values())

    def add_shot(self, probs, backend=None):
        backend = _check_backend(backend)
        qubits = sorted(self.target_qubits)
        shot = backend.sample_shots(probs, 1)
        bshot = backend.samples_to_binary(shot, len(qubits))
        if self._samples:
            self._samples.append(bshot[0])
        else:
            self._samples = [bshot[0]]
        return shot

    def add_shot_from_sample(self, sample):
        if self._samples:
            self._samples.append(sample)
        else:
            self._samples = [sample]

    def has_samples(self):
        return self._samples is not None

    def register_samples(self, samples):
        """Register samples array to the ``MeasurementResult`` object."""
        self._samples = samples

    def register_frequencies(self, frequencies):
        """Register frequencies to the ``MeasurementResult`` object."""
        self._frequencies = frequencies

    def reset(self):
        """Remove all registered samples and frequencies."""
        self._samples = None
        self._frequencies = None

    @property
    def symbols(self):
        """List of ``sympy.Symbols`` associated with the results of the measurement.

        These symbols are useful for conditioning parametrized gates on measurement outcomes.
        """
        if self._symbols is None:
            qubits = self.target_qubits
            self._symbols = [
                MeasurementSymbol(qubit, self) for qubit in range(len(qubits))
            ]

        return self._symbols

    def samples(self, binary=True, registers=False, backend=None):
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
        """
        backend = _check_backend(backend)

        if self._samples is None:
            if self.circuit is None:
                raise_error(
                    RuntimeError, "Cannot calculate samples if circuit is not provided."
                )
            # calculate samples for the whole circuit so that
            # individual register samples are registered here
            self.circuit.final_state.samples()

        if binary:
            return self._samples

        qubits = self.target_qubits
        return backend.samples_to_decimal(self._samples, len(qubits))

    def frequencies(self, binary=True, registers=False, backend=None):
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
        """
        backend = _check_backend(backend)

        if self._frequencies is None:
            self._frequencies = backend.calculate_frequencies(
                self.samples(binary=False, backend=backend)
            )

        if binary:
            qubits = self.target_qubits
            return frequencies_to_binary(self._frequencies, len(qubits))

        return self._frequencies

    def apply_bitflips(self, p0, p1=None):  # pragma: no cover
        return apply_bitflips(self, p0, p1)
