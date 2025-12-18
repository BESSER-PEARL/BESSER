import os
import sys


try:
    import besser
    print(f"besser package found at: {besser.__file__}")
except ImportError as e:
    print(f"Failed to import besser: {e}")

from besser.BUML.metamodel.quantum import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    HadamardGate, PauliXGate, RXGate, Measurement, ControlState
)
from besser.generators.qiskit import QiskitGenerator

def test_generator():
    print("Creating Quantum Circuit Model...")
    # Create Circuit
    circuit = QuantumCircuit("TestCircuit", qubits=3)
    
    # Add Registers (already added default 'q' with 3 qubits)
    c_reg = ClassicalRegister("c", 3)
    circuit.add_creg(c_reg)
    
    # Add Gates
    # H on q[0]
    circuit.add_operation(HadamardGate(target_qubit=0))
    
    # X on q[1] controlled by q[0] (CNOT)
    circuit.add_operation(PauliXGate(target_qubit=1, control_qubits=[0]))
    
    # RX(pi/2) on q[2] controlled by q[1] (Anti-Control)
    circuit.add_operation(RXGate(target_qubit=2, angle=1.57, control_qubits=[1], control_states=[ControlState.ANTI_CONTROL]))
    
    # Measure all
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    circuit.add_operation(Measurement(target_qubit=1, output_bit=1))
    circuit.add_operation(Measurement(target_qubit=2, output_bit=2))
    
    print("Initializing Generator...")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    generator = QiskitGenerator(model=circuit, output_dir=output_dir)
    
    print("Generating Code...")
    generator.generate()
    
    print("Done!")

if __name__ == "__main__":
    test_generator()
