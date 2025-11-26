from abc import ABC
from typing import List, Optional, Any, Union
from enum import Enum
from besser.BUML.metamodel.structural import NamedElement, Model

class ControlState(Enum):
    """
    Represents the state of a control qubit.
    """
    CONTROL = 'CONTROL'       # Active on |1> (filled circle)
    ANTI_CONTROL = 'ANTI_CONTROL' # Active on |0> (empty circle)

class QuantumCircuit(Model):
    """
    Represents a quantum circuit containing registers and a sequence of operations.
    
    Args:
        name (str): The name of the circuit.
        qubits (int): The number of qubits in the circuit (default register).
    """
    def __init__(self, name: str, qubits: int = 0):
        super().__init__(name)
        self.qregs: List[QuantumRegister] = []
        self.cregs: List[ClassicalRegister] = []
        self.operations: List[QuantumOperation] = []
        
        # Create a default quantum register if qubits > 0
        if qubits > 0:
            self.add_qreg(QuantumRegister("q", qubits))

    def add_qreg(self, qreg: 'QuantumRegister'):
        """Add a quantum register to the circuit."""
        self.qregs.append(qreg)

    def add_creg(self, creg: 'ClassicalRegister'):
        """Add a classical register to the circuit."""
        self.cregs.append(creg)

    def add_operation(self, operation: 'QuantumOperation'):
        """Add an operation (gate, measurement, etc.) to the circuit."""
        self.operations.append(operation)

    def __repr__(self):
        return f"QuantumCircuit(name='{self.name}', qregs={self.qregs}, cregs={self.cregs}, ops={len(self.operations)})"

class QuantumRegister(NamedElement):
    """
    Represents a register of qubits.
    """
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size: int = size

    def __repr__(self):
        return f"QuantumRegister(name='{self.name}', size={self.size})"

class ClassicalRegister(NamedElement):
    """
    Represents a register of classical bits.
    """
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size: int = size

    def __repr__(self):
        return f"ClassicalRegister(name='{self.name}', size={self.size})"

class QuantumOperation(NamedElement, ABC):
    """
    Abstract base class for all quantum operations.
    """
    def __init__(self, name: str, target_qubits: List[int], control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name)
        self.target_qubits: List[int] = target_qubits
        self.control_qubits: List[int] = control_qubits if control_qubits else []
        self.control_states: List[ControlState] = control_states if control_states else [ControlState.CONTROL] * len(self.control_qubits)

class Gate(QuantumOperation):
    """
    Base class for unitary quantum gates.
    """
    pass

# --- Primitive Gates ---
class PrimitiveGate(Gate):
    """
    Represents standard primitive gates (H, X, Y, Z, S, T, SWAP, etc.).
    """
    def __init__(self, type_name: str, target_qubits: List[int], control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(type_name, target_qubits, control_qubits, control_states)
        self.type_name = type_name

    def __repr__(self):
        return f"PrimitiveGate(type='{self.type_name}', targets={self.target_qubits}, controls={self.control_qubits})"

# --- Parametric Gates ---
class ParametricGate(Gate):
    """
    Represents gates that take a parameter (angle), e.g., RX, RY, RZ, Phase.
    """
    def __init__(self, type_name: str, target_qubits: List[int], parameter: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(type_name, target_qubits, control_qubits, control_states)
        self.type_name = type_name
        self.parameter = parameter

    def __repr__(self):
        return f"ParametricGate(type='{self.type_name}', param={self.parameter}, targets={self.target_qubits}, controls={self.control_qubits})"

# --- Arithmetic Gates ---
class ArithmeticGate(Gate):
    """
    Represents arithmetic operations (Add, Sub, Mul, etc.).
    """
    def __init__(self, operation: str, target_qubits: List[int], input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation, target_qubits, control_qubits, control_states)
        self.operation = operation
        self.input_qubits = input_qubits if input_qubits else []

    def __repr__(self):
        return f"ArithmeticGate(op='{self.operation}', targets={self.target_qubits}, inputs={self.input_qubits})"

# --- Modular Arithmetic Gates ---
class ModularArithmeticGate(Gate):
    """
    Represents modular arithmetic operations (ModAdd, ModMul, etc.).
    """
    def __init__(self, operation: str, target_qubits: List[int], modulo: int, input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation, target_qubits, control_qubits, control_states)
        self.operation = operation
        self.modulo = modulo
        self.input_qubits = input_qubits if input_qubits else []

    def __repr__(self):
        return f"ModularArithmeticGate(op='{self.operation}', mod={self.modulo}, targets={self.target_qubits})"

# --- Comparison Gates ---
class ComparisonGate(Gate):
    """
    Represents comparison operations (>, <, ==, etc.).
    """
    def __init__(self, operation: str, target_qubits: List[int], input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation, target_qubits, control_qubits, control_states)
        self.operation = operation
        self.input_qubits = input_qubits if input_qubits else []

    def __repr__(self):
        return f"ComparisonGate(op='{self.operation}', targets={self.target_qubits})"

# --- Frequency Gates (QFT) ---
class QFTGate(Gate):
    """
    Quantum Fourier Transform gate.
    """
    def __init__(self, target_qubits: List[int], inverse: bool = False, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        name = "QFT_DAG" if inverse else "QFT"
        super().__init__(name, target_qubits, control_qubits, control_states)
        self.inverse = inverse

    def __repr__(self):
        return f"QFTGate(targets={self.target_qubits}, inverse={self.inverse})"

# --- Phase Gradient Gates ---
class PhaseGradientGate(Gate):
    """
    Phase Gradient gate.
    """
    def __init__(self, target_qubits: List[int], inverse: bool = False, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        name = "PHASE_GRADIENT_DAG" if inverse else "PHASE_GRADIENT"
        super().__init__(name, target_qubits, control_qubits, control_states)
        self.inverse = inverse

    def __repr__(self):
        return f"PhaseGradientGate(targets={self.target_qubits}, inverse={self.inverse})"

# --- Display Operations ---
class DisplayOperation(QuantumOperation):
    """
    Represents display/probe operations (Bloch, Density, Chance, etc.).
    """
    def __init__(self, display_type: str, target_qubits: List[int]):
        super().__init__(display_type, target_qubits)
        self.display_type = display_type

    def __repr__(self):
        return f"DisplayOperation(type='{self.display_type}', targets={self.target_qubits})"

# --- Measurement ---
class Measurement(QuantumOperation):
    """
    Represents a measurement operation.
    """
    def __init__(self, target_qubit: int, output_bit: int = None, basis: str = 'Z'):
        super().__init__(f"Measure_{basis}", [target_qubit])
        self.output_bit = output_bit
        self.basis = basis # 'X', 'Y', 'Z'

    def __repr__(self):
        return f"Measurement(qubit={self.target_qubits[0]}, out={self.output_bit}, basis='{self.basis}')"

# --- Post-Selection ---
class PostSelection(QuantumOperation):
    """
    Represents a post-selection operation.
    """
    def __init__(self, target_qubit: int, value: int, basis: str = 'Z'):
        super().__init__(f"PostSelect_{basis}_{value}", [target_qubit])
        self.value = value # 0 or 1
        self.basis = basis # 'X', 'Y', 'Z'

    def __repr__(self):
        return f"PostSelection(qubit={self.target_qubits[0]}, val={self.value}, basis='{self.basis}')"

# --- Input Gates ---
class InputGate(Gate):
    """
    Represents input setting gates (InputA, InputB, etc.).
    """
    def __init__(self, input_type: str, target_qubits: List[int], value: int = None):
        super().__init__(input_type, target_qubits)
        self.input_type = input_type
        self.value = value

    def __repr__(self):
        return f"InputGate(type='{self.input_type}', targets={self.target_qubits}, val={self.value})"

# --- Order Gates ---
class OrderGate(Gate):
    """
    Represents order manipulation gates (Interleave, Reverse, etc.).
    """
    def __init__(self, order_type: str, target_qubits: List[int]):
        super().__init__(order_type, target_qubits)
        self.order_type = order_type

    def __repr__(self):
        return f"OrderGate(type='{self.order_type}', targets={self.target_qubits})"

# --- Scalar Gates ---
class ScalarGate(Gate):
    """
    Represents global phase/scalar gates.
    """
    def __init__(self, scalar_type: str):
        super().__init__(scalar_type, [])
        self.scalar_type = scalar_type

    def __repr__(self):
        return f"ScalarGate(type='{self.scalar_type}')"

# --- Custom/User Defined Gate ---
class CustomGate(Gate):
    """
    Represents a user-defined or custom gate.
    """
    def __init__(self, name: str, target_qubits: List[int], definition: 'QuantumCircuit' = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name, target_qubits, control_qubits, control_states)
        self.definition = definition

    def __repr__(self):
        return f"CustomGate(name='{self.name}', targets={self.target_qubits})"

# --- Specific Gate Classes for Convenience ---

class HadamardGate(PrimitiveGate):
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("H", [target_qubit], control_qubits, control_states)

class PauliXGate(PrimitiveGate):
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("X", [target_qubit], control_qubits, control_states)

class PauliYGate(PrimitiveGate):
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Y", [target_qubit], control_qubits, control_states)

class PauliZGate(PrimitiveGate):
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Z", [target_qubit], control_qubits, control_states)

class SwapGate(PrimitiveGate):
    def __init__(self, qubit1: int, qubit2: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("SWAP", [qubit1, qubit2], control_qubits, control_states)

class RXGate(ParametricGate):
    def __init__(self, target_qubit: int, angle: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RX", [target_qubit], angle, control_qubits, control_states)

class RYGate(ParametricGate):
    def __init__(self, target_qubit: int, angle: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RY", [target_qubit], angle, control_qubits, control_states)

class RZGate(ParametricGate):
    def __init__(self, target_qubit: int, angle: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RZ", [target_qubit], angle, control_qubits, control_states)

class PhaseGate(ParametricGate):
    def __init__(self, target_qubit: int, angle: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("PHASE", [target_qubit], angle, control_qubits, control_states)

class TimeDependentGate(Gate):
    """
    Represents time-dependent gates (e.g., Z^t, X^t).
    """
    def __init__(self, type_name: str, target_qubits: List[int], parameter_expr: str, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(type_name, target_qubits, control_qubits, control_states)
        self.type_name = type_name
        self.parameter_expr = parameter_expr

    def __repr__(self):
        return f"TimeDependentGate(type='{self.type_name}', expr='{self.parameter_expr}', targets={self.target_qubits})"

class SpacerGate(Gate):
    """
    Represents a visual spacer or identity operation.
    """
    def __init__(self, target_qubits: List[int]):
        super().__init__("SPACER", target_qubits)

    def __repr__(self):
        return f"SpacerGate(targets={self.target_qubits})"

