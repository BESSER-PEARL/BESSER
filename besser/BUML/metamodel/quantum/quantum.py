from abc import ABC
from typing import List, Optional, Any, Union
from enum import Enum
from besser.BUML.metamodel.structural import NamedElement, Model


class ControlState(Enum):
    """
    Represents the state of a control qubit.
    """
    CONTROL = 'CONTROL'       # Active on |1> (filled circle)
    ANTI_CONTROL = 'ANTI_CONTROL'  # Active on |0> (empty circle)


# ============================================
# Target Classes (Hardware/Simulator)
# ============================================

class Target(NamedElement, ABC):
    """
    Abstract base class for execution targets.
    """
    def __init__(self, name: str):
        super().__init__(name)


class Hardware(Target):
    """
    Represents a quantum hardware target.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f"Hardware(name='{self.name}')"


class Simulator(Target):
    """
    Represents a quantum simulator target.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self):
        return f"Simulator(name='{self.name}')"


# ============================================
# Register and Bit Classes
# ============================================

class QuantumRegister(NamedElement):
    """
    Represents a register of qubits.
    """
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size: int = size
        self.qubits: List['Qubit'] = [Qubit(i, i) for i in range(size)]

    def __repr__(self):
        return f"QuantumRegister(name='{self.name}', size={self.size})"


class ClassicalRegister(NamedElement):
    """
    Represents a register of classical bits.
    """
    def __init__(self, name: str, size: int):
        super().__init__(name)
        self.size: int = size
        self.bits: List['ClassicalBit'] = [ClassicalBit(i, i) for i in range(size)]

    def __repr__(self):
        return f"ClassicalRegister(name='{self.name}', size={self.size})"


class Qubit:
    """
    Represents a single qubit.
    """
    def __init__(self, id: int, index: int):
        self.id: int = id
        self.index: int = index

    def __repr__(self):
        return f"Qubit(id={self.id}, index={self.index})"


class ClassicalBit:
    """
    Represents a single classical bit.
    """
    def __init__(self, id: int, index: int):
        self.id: int = id
        self.index: int = index

    def __repr__(self):
        return f"ClassicalBit(id={self.id}, index={self.index})"


# ============================================
# Quantum Circuit
# ============================================

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
        self.operations: List['QuantumOperation'] = []
        
        # Create a default quantum register if qubits > 0
        if qubits > 0:
            self.add_qreg(QuantumRegister("q", qubits))

    def add_qreg(self, qreg: QuantumRegister):
        """Add a quantum register to the circuit."""
        self.qregs.append(qreg)

    def add_creg(self, creg: ClassicalRegister):
        """Add a classical register to the circuit."""
        self.cregs.append(creg)

    def add_operation(self, operation: 'QuantumOperation'):
        """Add an operation (gate, measurement, etc.) to the circuit."""
        self.operations.append(operation)

    @property
    def num_qubits(self) -> int:
        """Returns the total number of qubits in the circuit."""
        return sum(qreg.size for qreg in self.qregs)

    @property
    def num_clbits(self) -> int:
        """Returns the total number of classical bits in the circuit."""
        return sum(creg.size for creg in self.cregs)

    def __repr__(self):
        return f"QuantumCircuit(name='{self.name}', qregs={self.qregs}, cregs={self.cregs}, ops={len(self.operations)})"


# ============================================
# Quantum Oracle
# ============================================

class QuantumOracle(NamedElement):
    """
    Represents a quantum oracle with a specification.
    """
    def __init__(self, name: str, specification: str):
        super().__init__(name)
        self.specification: str = specification

    def __repr__(self):
        return f"QuantumOracle(name='{self.name}', spec='{self.specification}')"


# ============================================
# Quantum Operation Hierarchy
# ============================================

class QuantumOperation(NamedElement, ABC):
    """
    Abstract base class for all quantum operations.
    """
    def __init__(self, name: str, target_qubits: List[int], control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name)
        self.target_qubits: List[int] = target_qubits
        self.control_qubits: List[int] = control_qubits if control_qubits else []
        self.control_states: List[ControlState] = control_states if control_states else [ControlState.CONTROL] * len(self.control_qubits)


class Measurement(QuantumOperation):
    """
    Represents a measurement operation.
    """
    def __init__(self, target_qubit: int, output_bit: int = None, basis: str = 'Z'):
        super().__init__(f"Measure_{basis}", [target_qubit])
        self.output_bit: int = output_bit
        self.basis: str = basis  # 'X', 'Y', 'Z'

    def __repr__(self):
        return f"Measurement(qubit={self.target_qubits[0]}, out={self.output_bit}, basis='{self.basis}')"


# ============================================
# Gate Hierarchy
# ============================================

class Gate(QuantumOperation, ABC):
    """
    Abstract base class for unitary quantum gates.
    """
    pass


class GateDefinition(NamedElement):
    """
    Represents the definition of a custom gate.
    """
    def __init__(self, name: str, circuit: 'QuantumCircuit' = None):
        super().__init__(name)
        self.circuit: QuantumCircuit = circuit

    def __repr__(self):
        return f"GateDefinition(name='{self.name}')"


class CustomGate(Gate):
    """
    Represents a user-defined or custom gate.
    
    Args:
        name: The name/label of the custom gate
        target_qubits: List of target qubit indices
        definition: Optional GateDefinition with a full circuit
        gates: Optional list of Gate objects (alternative to definition)
        control_qubits: Optional list of control qubit indices
        control_states: Optional list of control states
    """
    def __init__(self, name: str, target_qubits: List[int], definition: GateDefinition = None, gates: List['Gate'] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name, target_qubits, control_qubits, control_states)
        self.definition: GateDefinition = definition
        self.gates: List['Gate'] = gates if gates else []

    def __repr__(self):
        return f"CustomGate(name='{self.name}', targets={self.target_qubits}, gates={len(self.gates)})"


class FunctionGate(Gate):
    """
    Represents a function gate defined by a nested circuit.
    """
    def __init__(self, name: str, target_qubits: List[int], definition: GateDefinition = None, gates: List['Gate'] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name, target_qubits, control_qubits, control_states)
        self.definition: GateDefinition = definition
        self.gates: List['Gate'] = gates if gates else []

    def __repr__(self):
        return f"FunctionGate(name='{self.name}', targets={self.target_qubits}, gates={len(self.gates)})"


class InputGate(Gate):
    """
    Represents input setting gates (InputA, InputB, etc.).
    """
    def __init__(self, input_type: str, target_qubits: List[int], value: str = None):
        super().__init__(input_type, target_qubits)
        self.input_type: str = input_type
        self.value: str = value

    def __repr__(self):
        return f"InputGate(type='{self.input_type}', targets={self.target_qubits}, val={self.value})"


# ============================================
# Abstract Gate Categories
# ============================================

class SingleBitGate(Gate, ABC):
    """
    Abstract base class for single-qubit gates.
    """
    def __init__(self, name: str, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(name, [target_qubit], control_qubits, control_states)


class MultiBitGate(Gate, ABC):
    """
    Abstract base class for multi-qubit gates.
    """
    pass


class ArithmeticGate(Gate, ABC):
    """
    Abstract base class for arithmetic operations.
    """
    def __init__(self, operation_type: str, target_qubits: List[int], input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation_type, target_qubits, control_qubits, control_states)
        self.operation_type: str = operation_type
        self.input_qubits: List[int] = input_qubits if input_qubits else []


class ComparisonGate(Gate):
    """
    Represents comparison operations (>, <, ==, etc.).
    """
    def __init__(self, operation: str, target_qubits: List[int], input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation, target_qubits, control_qubits, control_states)
        self.operation: str = operation
        self.input_qubits: List[int] = input_qubits if input_qubits else []

    def __repr__(self):
        return f"ComparisonGate(op='{self.operation}', targets={self.target_qubits})"


# ============================================
# Single Bit Gates
# ============================================

class PauliXGate(SingleBitGate):
    """Pauli-X gate (NOT gate)."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("X", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"PauliXGate(target={self.target_qubits[0]})"


class PauliYGate(SingleBitGate):
    """Pauli-Y gate."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Y", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"PauliYGate(target={self.target_qubits[0]})"


class PauliZGate(SingleBitGate):
    """Pauli-Z gate."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Z", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"PauliZGate(target={self.target_qubits[0]})"


class HadamardGate(SingleBitGate):
    """Hadamard gate."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("H", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"HadamardGate(target={self.target_qubits[0]})"


class TGate(SingleBitGate):
    """T gate (π/4 phase gate)."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("T", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"TGate(target={self.target_qubits[0]})"


class SGate(SingleBitGate):
    """S gate (π/2 phase gate)."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("S", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"SGate(target={self.target_qubits[0]})"


class PhaseGateZ(SingleBitGate):
    """Z phase gate."""
    def __init__(self, target_qubit: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Z", target_qubit, control_qubits, control_states)

    def __repr__(self):
        return f"PhaseGateZ(target={self.target_qubits[0]})"


class RXGate(SingleBitGate):
    """Rotation around X-axis gate."""
    def __init__(self, target_qubit: int, theta: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RX", target_qubit, control_qubits, control_states)
        self.theta: float = theta

    def __repr__(self):
        return f"RXGate(target={self.target_qubits[0]}, theta={self.theta})"


class RYGate(SingleBitGate):
    """Rotation around Y-axis gate."""
    def __init__(self, target_qubit: int, theta: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RY", target_qubit, control_qubits, control_states)
        self.theta: float = theta

    def __repr__(self):
        return f"RYGate(target={self.target_qubits[0]}, theta={self.theta})"


class RZGate(SingleBitGate):
    """Rotation around Z-axis gate."""
    def __init__(self, target_qubit: int, theta: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("RZ", target_qubit, control_qubits, control_states)
        self.theta: float = theta

    def __repr__(self):
        return f"RZGate(target={self.target_qubits[0]}, theta={self.theta})"


class IdentityGate(SingleBitGate):
    """Identity gate (no operation)."""
    def __init__(self, target_qubit: int):
        super().__init__("I", target_qubit)

    def __repr__(self):
        return f"IdentityGate(target={self.target_qubits[0]})"


# ============================================
# Multi Bit Gates
# ============================================

class CNOT(MultiBitGate):
    """Controlled-NOT (CNOT) gate."""
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__("CNOT", [target_qubit], [control_qubit])

    def __repr__(self):
        return f"CNOT(control={self.control_qubits[0]}, target={self.target_qubits[0]})"


class CPhaseGate(MultiBitGate):
    """Controlled phase gate."""
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__("CP", [target_qubit], [control_qubit])
        self.theta: float = theta

    def __repr__(self):
        return f"CPhaseGate(control={self.control_qubits[0]}, target={self.target_qubits[0]}, theta={self.theta})"


class CYGate(MultiBitGate):
    """Controlled-Y gate."""
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__("CY", [target_qubit], [control_qubit])

    def __repr__(self):
        return f"CYGate(control={self.control_qubits[0]}, target={self.target_qubits[0]})"


class CZGate(MultiBitGate):
    """Controlled-Z gate."""
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__("CZ", [target_qubit], [control_qubit])

    def __repr__(self):
        return f"CZGate(control={self.control_qubits[0]}, target={self.target_qubits[0]})"


class CXGate(MultiBitGate):
    """Controlled-X gate (same as CNOT)."""
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__("CX", [target_qubit], [control_qubit])

    def __repr__(self):
        return f"CXGate(control={self.control_qubits[0]}, target={self.target_qubits[0]})"


class CHGate(MultiBitGate):
    """Controlled-Hadamard gate."""
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__("CH", [target_qubit], [control_qubit])

    def __repr__(self):
        return f"CHGate(control={self.control_qubits[0]}, target={self.target_qubits[0]})"


class QFTGate(MultiBitGate):
    """Quantum Fourier Transform gate."""
    def __init__(self, target_qubits: List[int], inverse: bool = False):
        name = "QFT_DAG" if inverse else "QFT"
        super().__init__(name, target_qubits)
        self.inverse: bool = inverse

    def __repr__(self):
        return f"QFTGate(targets={self.target_qubits}, inverse={self.inverse})"


class RXXGate(MultiBitGate):
    """Two-qubit XX rotation gate."""
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__("RXX", [qubit1, qubit2])
        self.theta: float = theta

    def __repr__(self):
        return f"RXXGate(qubits={self.target_qubits}, theta={self.theta})"


class RZZGate(MultiBitGate):
    """Two-qubit ZZ rotation gate."""
    def __init__(self, qubit1: int, qubit2: int, theta: float = None):
        super().__init__("RZZ", [qubit1, qubit2])
        self.theta: float = theta

    def __repr__(self):
        return f"RZZGate(qubits={self.target_qubits})"


class CRXGate(MultiBitGate):
    """Controlled RX gate."""
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__("CRX", [target_qubit], [control_qubit])
        self.theta: float = theta

    def __repr__(self):
        return f"CRXGate(control={self.control_qubits[0]}, target={self.target_qubits[0]}, theta={self.theta})"


class CRYGate(MultiBitGate):
    """Controlled RY gate."""
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__("CRY", [target_qubit], [control_qubit])
        self.theta: float = theta

    def __repr__(self):
        return f"CRYGate(control={self.control_qubits[0]}, target={self.target_qubits[0]}, theta={self.theta})"


class CRZGate(MultiBitGate):
    """Controlled RZ gate."""
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__("CRZ", [target_qubit], [control_qubit])
        self.theta: float = theta

    def __repr__(self):
        return f"CRZGate(control={self.control_qubits[0]}, target={self.target_qubits[0]}, theta={self.theta})"


class BellGate(MultiBitGate):
    """Bell state preparation gate."""
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__("Bell", [qubit1, qubit2])

    def __repr__(self):
        return f"BellGate(qubits={self.target_qubits})"


class iSwapGate(MultiBitGate):
    """iSWAP gate."""
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__("iSWAP", [qubit1, qubit2])

    def __repr__(self):
        return f"iSwapGate(qubits={self.target_qubits})"


class SwapGate(MultiBitGate):
    """SWAP gate."""
    def __init__(self, qubit1: int, qubit2: int, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("SWAP", [qubit1, qubit2], control_qubits, control_states)

    def __repr__(self):
        return f"SwapGate(qubits={self.target_qubits})"


class SqrtSwapGate(MultiBitGate):
    """Square root of SWAP gate."""
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__("SqrtSWAP", [qubit1, qubit2])

    def __repr__(self):
        return f"SqrtSwapGate(qubits={self.target_qubits})"


class RYYGate(MultiBitGate):
    """Two-qubit YY rotation gate."""
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__("RYY", [qubit1, qubit2])
        self.theta: float = theta

    def __repr__(self):
        return f"RYYGate(qubits={self.target_qubits}, theta={self.theta})"


class RZXGate(MultiBitGate):
    """Two-qubit ZX rotation gate."""
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__("RZX", [qubit1, qubit2])
        self.theta: float = theta

    def __repr__(self):
        return f"RZXGate(qubits={self.target_qubits}, theta={self.theta})"


class PhaseGate(SingleBitGate):
    """General Phase gate."""
    def __init__(self, target_qubit: int, angle: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__("Phase", target_qubit, control_qubits, control_states)
        self.parameter: float = angle

    def __repr__(self):
        return f"PhaseGate(target={self.target_qubits[0]}, angle={self.parameter})"


class PhaseGradientGate(MultiBitGate):
    """Phase Gradient gate."""
    def __init__(self, target_qubits: List[int], inverse: bool = False):
        name = "PhaseGradient_DAG" if inverse else "PhaseGradient"
        super().__init__(name, target_qubits)
        self.inverse: bool = inverse

    def __repr__(self):
        return f"PhaseGradientGate(targets={self.target_qubits}, inverse={self.inverse})"


class ModularArithmeticGate(ArithmeticGate):
    """Modular Arithmetic gate."""
    def __init__(self, operation: str, target_qubits: List[int], modulo: int, input_qubits: List[int] = None, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(operation, target_qubits, input_qubits, control_qubits, control_states)
        self.modulo: int = modulo

    def __repr__(self):
        return f"ModularArithmeticGate(op='{self.operation_type}', targets={self.target_qubits}, mod={self.modulo})"


class DisplayOperation(QuantumOperation):
    """Display operation (Bloch sphere, Density matrix, etc.)."""
    def __init__(self, display_type: str, target_qubits: List[int]):
        super().__init__(display_type, target_qubits)
        self.display_type: str = display_type

    def __repr__(self):
        return f"DisplayOperation(type='{self.display_type}', targets={self.target_qubits})"


class TimeDependentGate(Gate):
    """Time-dependent gate."""
    def __init__(self, type_name: str, target_qubits: List[int], parameter_expr: str, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(type_name, target_qubits, control_qubits, control_states)
        self.type_name: str = type_name
        self.parameter_expr: str = parameter_expr

    def __repr__(self):
        return f"TimeDependentGate(type='{self.type_name}', targets={self.target_qubits}, expr='{self.parameter_expr}')"


class SpacerGate(Gate):
    """Spacer gate (visual only)."""
    def __init__(self, target_qubits: List[int]):
        super().__init__("Spacer", target_qubits)

    def __repr__(self):
        return f"SpacerGate(targets={self.target_qubits})"


class PostSelection(QuantumOperation):
    """Post-selection operation."""
    def __init__(self, target_qubit: int, value: int, basis: str = 'Z'):
        super().__init__(f"PostSelect_{value}", [target_qubit])
        self.value: int = value
        self.basis: str = basis

    def __repr__(self):
        return f"PostSelection(qubit={self.target_qubits[0]}, val={self.value}, basis='{self.basis}')"


class PrimitiveGate(Gate):
    """Generic primitive gate."""
    def __init__(self, type_name: str, target_qubits: List[int]):
        super().__init__(type_name, target_qubits)
        self.type_name: str = type_name

    def __repr__(self):
        return f"PrimitiveGate(type='{self.type_name}', targets={self.target_qubits})"


class ParametricGate(Gate):
    """Generic parametric gate."""
    def __init__(self, type_name: str, target_qubits: List[int], parameter: float, control_qubits: List[int] = None, control_states: List[ControlState] = None):
        super().__init__(type_name, target_qubits, control_qubits, control_states)
        self.type_name: str = type_name
        self.parameter: float = parameter

    def __repr__(self):
        return f"ParametricGate(type='{self.type_name}', targets={self.target_qubits}, param={self.parameter})"


class OrderGate(Gate):
    """Order gate (for QFT ordering etc)."""
    def __init__(self, order_type: str, target_qubits: List[int]):
        super().__init__(order_type, target_qubits)
        self.order_type: str = order_type

    def __repr__(self):
        return f"OrderGate(type='{self.order_type}', targets={self.target_qubits})"


class ScalarGate(QuantumOperation):
    """Scalar gate (global phase etc)."""
    def __init__(self, scalar_type: str):
        super().__init__(scalar_type, [])
        self.scalar_type: str = scalar_type

    def __repr__(self):
        return f"ScalarGate(type='{self.scalar_type}')"
