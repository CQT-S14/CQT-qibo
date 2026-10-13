import json

from qibo import gates
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Z
from qibo.measurements import MeasurementResult
from typing import Tuple, Union


class Condition(Gate):
    def __init__(self, register_name:str, condition:int,gate: Gate):
        super().__init__()
        self.name = "condition"
        self.draw_label = "Cd"
        self.target_qubits = gate.target_qubits
        self.init_args = [gate.target_qubits]
        self.unitary = False
        self.conditional_gate = gate
        self.register_name = register_name
        self.condition = condition

    @property
    def qasm_label(self) -> str:
        return "if"
