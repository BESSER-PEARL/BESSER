"""
BUML Quantum Circuit Example - Bell State and CNOT
This example demonstrates:
1. Creating a quantum circuit with the BUML metamodel
2. Adding registers (quantum and classical)
3. Creating a CNOT gate (Controlled-X) to generate a Bell state
4. Adding measurements
5. Generating Qiskit code using the generator
"""

import os
from besser.BUML.metamodel.quantum import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SGate, TGate, RXGate, RYGate, RZGate,
    Measurement, ControlState, SwapGate,
    QFTGate, DisplayOperation
)
from besser.generators.qiskit import QiskitGenerator

def create_bell_state_circuit():
    """
    Creates a Bell state using Hadamard + CNOT
    
    Circuit:
         ┌───┐     
    q_0: ┤ H ├──■──
         └───┘┌─┴─┐
    q_1: ─────┤ X ├
              └───┘
    """
    print("=" * 60)
    print("EXAMPLE 1: Bell State (Hadamard + CNOT)")
    print("=" * 60)
    
    # Create a quantum circuit with 2 qubits
    circuit = QuantumCircuit("BellState", qubits=2)
    
    # Add classical register for measurements
    circuit.add_creg(ClassicalRegister("c", 2))
    
    # Step 1: Apply Hadamard to qubit 0 (creates superposition)
    circuit.add_operation(HadamardGate(target_qubit=0))
    print("Added: H gate on qubit 0")
    
    # Step 2: Apply CNOT gate (qubit 0 controls qubit 1)
    # This is a controlled-X where qubit 0 is the control
    circuit.add_operation(PauliXGate(
        target_qubit=1, 
        control_qubits=[0],
        control_states=[ControlState.CONTROL]
    ))
    print("Added: CNOT gate (control=0, target=1)")
    
    # Step 3: Measure both qubits
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    circuit.add_operation(Measurement(target_qubit=1, output_bit=1))
    print("Added: Measurements on both qubits")
    
    return circuit

def create_grover_like_circuit():
    """
    Creates a more complex circuit with multiple gates
    
    Demonstrates:
    - Multiple controlled gates
    - Different gate types
    - Multi-qubit operations
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Complex Circuit (Multi-gate, Multi-control)")
    print("=" * 60)
    
    circuit = QuantumCircuit("ComplexCircuit", qubits=4)
    circuit.add_creg(ClassicalRegister("c", 4))
    
    # Initialize with Hadamard on all qubits
    for i in range(4):
        circuit.add_operation(HadamardGate(target_qubit=i))
    print("Added: Hadamard on all 4 qubits")
    
    # CNOT chain: 0->1, 1->2, 2->3
    circuit.add_operation(PauliXGate(target_qubit=1, control_qubits=[0]))
    circuit.add_operation(PauliXGate(target_qubit=2, control_qubits=[1]))
    circuit.add_operation(PauliXGate(target_qubit=3, control_qubits=[2]))
    print("Added: CNOT chain (0→1→2→3)")
    
    # Toffoli gate (CCX): Double controlled-X on qubit 3
    circuit.add_operation(PauliXGate(
        target_qubit=3, 
        control_qubits=[0, 1],
        control_states=[ControlState.CONTROL, ControlState.CONTROL]
    ))
    print("Added: Toffoli gate (controls=0,1, target=3)")
    
    # Controlled Hadamard
    circuit.add_operation(HadamardGate(target_qubit=2, control_qubits=[0]))
    print("Added: Controlled Hadamard (control=0, target=2)")
    
    # Parametric rotation
    circuit.add_operation(RYGate(target_qubit=1, angle=1.57))  # π/2
    print("Added: RY(π/2) on qubit 1")
    
    # Anti-controlled gate (control on |0⟩ instead of |1⟩)
    circuit.add_operation(PauliZGate(
        target_qubit=2, 
        control_qubits=[3],
        control_states=[ControlState.ANTI_CONTROL]
    ))
    print("Added: Anti-controlled Z (anti-control=3, target=2)")
    
    # SWAP gate
    circuit.add_operation(SwapGate(qubit1=0, qubit2=3))
    print("Added: SWAP gate (qubits 0 and 3)")
    
    # Measurements
    for i in range(4):
        circuit.add_operation(Measurement(target_qubit=i, output_bit=i))
    print("Added: Measurements on all qubits")
    
    return circuit

def create_qft_circuit():
    """
    Creates a Quantum Fourier Transform circuit
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Quantum Fourier Transform (QFT)")
    print("=" * 60)
    
    circuit = QuantumCircuit("QFT_Circuit", qubits=3)
    circuit.add_creg(ClassicalRegister("c", 3))
    
    # Initialize with some state
    circuit.add_operation(PauliXGate(target_qubit=0))
    circuit.add_operation(PauliXGate(target_qubit=2))
    print("Added: Initialize state |101⟩")
    
    # Apply QFT on all 3 qubits
    circuit.add_operation(QFTGate(target_qubits=[0, 1, 2], inverse=False))
    print("Added: QFT on qubits [0, 1, 2]")
    
    # Some operations in frequency space
    circuit.add_operation(PauliZGate(target_qubit=1))
    print("Added: Z gate in frequency space")
    
    # Inverse QFT
    circuit.add_operation(QFTGate(target_qubits=[0, 1, 2], inverse=True))
    print("Added: Inverse QFT")
    
    # Measure
    for i in range(3):
        circuit.add_operation(Measurement(target_qubit=i, output_bit=i))
    print("Added: Measurements")
    
    return circuit

def main():
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " BUML Quantum Circuit Metamodel Examples ".center(58) + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "quantum_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Bell State with CNOT
    bell_circuit = create_bell_state_circuit()
    print("\nGenerating Qiskit code for Bell State...")
    generator1 = QiskitGenerator(model=bell_circuit, output_dir=output_dir)
    generator1.generate()
    
    # Example 2: Complex Multi-gate Circuit
    complex_circuit = create_grover_like_circuit()
    print("\nGenerating Qiskit code for Complex Circuit...")
    output_dir2 = os.path.join(output_dir, "complex")
    os.makedirs(output_dir2, exist_ok=True)
    generator2 = QiskitGenerator(model=complex_circuit, output_dir=output_dir2)
    generator2.generate()
    
    # Example 3: QFT Circuit
    qft_circuit = create_qft_circuit()
    print("\nGenerating Qiskit code for QFT Circuit...")
    output_dir3 = os.path.join(output_dir, "qft")
    os.makedirs(output_dir3, exist_ok=True)
    generator3 = QiskitGenerator(model=qft_circuit, output_dir=output_dir3)
    generator3.generate()
    
    print("\n" + "=" * 60)
    print("✓ All circuits generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nTo run the generated circuits:")
    print("  1. Install Qiskit: pip install qiskit qiskit-aer")
    print("  2. Navigate to the output directory")
    print("  3. Run: python qiskit_circuit.py")
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
