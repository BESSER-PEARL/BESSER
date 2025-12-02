"""
BUML Quantum Circuit - Multi-Qubit Gates Examples
This example demonstrates various multi-qubit quantum gates.
"""

from besser.BUML.metamodel.quantum import (
    QuantumCircuit, ClassicalRegister, ControlState,
    # Primitive gates
    HadamardGate, PauliXGate, PauliYGate, PauliZGate, SwapGate,
    # Parametric gates
    RYGate, RXGate, RZGate,
    # Other gates
    QFTGate, Measurement
)

def example_1_cnot():
    """
    Example 1: CNOT Gate (2-qubit gate)
    Control: qubit 0, Target: qubit 1
    """
    print("=" * 70)
    print("EXAMPLE 1: CNOT (Controlled-NOT)")
    print("=" * 70)
    
    circuit = QuantumCircuit("CNOT_Example", qubits=2)
    circuit.add_creg(ClassicalRegister("c", 2))
    
    # CNOT: qubit 0 controls qubit 1
    cnot = PauliXGate(
        target_qubit=1,
        control_qubits=[0],
        control_states=[ControlState.CONTROL]
    )
    circuit.add_operation(cnot)
    
    print(f"✓ Created CNOT gate:")
    print(f"  Control qubit: 0")
    print(f"  Target qubit: 1")
    print(f"  Operation: {cnot}")
    
    return circuit

def example_2_toffoli():
    """
    Example 2: Toffoli Gate (CCX - 3-qubit gate)
    Controls: qubits 0 and 1, Target: qubit 2
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Toffoli (CCNOT - Controlled-Controlled-NOT)")
    print("=" * 70)
    
    circuit = QuantumCircuit("Toffoli_Example", qubits=3)
    
    # Toffoli: qubits 0 and 1 control qubit 2
    toffoli = PauliXGate(
        target_qubit=2,
        control_qubits=[0, 1],
        control_states=[ControlState.CONTROL, ControlState.CONTROL]
    )
    circuit.add_operation(toffoli)
    
    print(f"✓ Created Toffoli gate:")
    print(f"  Control qubits: [0, 1]")
    print(f"  Target qubit: 2")
    print(f"  Operation: {toffoli}")
    
    return circuit

def example_3_swap():
    """
    Example 3: SWAP Gate (2-qubit gate)
    Swaps the states of two qubits
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: SWAP Gate")
    print("=" * 70)
    
    circuit = QuantumCircuit("SWAP_Example", qubits=3)
    
    # SWAP qubits 0 and 2
    swap = SwapGate(qubit1=0, qubit2=2)
    circuit.add_operation(swap)
    
    print(f"✓ Created SWAP gate:")
    print(f"  Qubit 1: 0")
    print(f"  Qubit 2: 2")
    print(f"  Operation: {swap}")
    
    return circuit

def example_4_controlled_swap():
    """
    Example 4: Controlled-SWAP (Fredkin Gate - 3-qubit gate)
    Control: qubit 0, Swap targets: qubits 1 and 2
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Fredkin Gate (Controlled-SWAP)")
    print("=" * 70)
    
    circuit = QuantumCircuit("Fredkin_Example", qubits=3)
    
    # Controlled-SWAP: qubit 0 controls swap of qubits 1 and 2
    fredkin = SwapGate(
        qubit1=1,
        qubit2=2,
        control_qubits=[0],
        control_states=[ControlState.CONTROL]
    )
    circuit.add_operation(fredkin)
    
    print(f"✓ Created Fredkin gate:")
    print(f"  Control qubit: 0")
    print(f"  Swap qubits: [1, 2]")
    print(f"  Operation: {fredkin}")
    
    return circuit

def example_5_multi_controlled():
    """
    Example 5: Multi-Controlled Gate (4+ qubits)
    Controls: qubits 0, 1, 2, Target: qubit 3
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Controlled Gate (C³X)")
    print("=" * 70)
    
    circuit = QuantumCircuit("MultiControl_Example", qubits=4)
    
    # Triple-controlled X: qubits 0, 1, 2 control qubit 3
    c3x = PauliXGate(
        target_qubit=3,
        control_qubits=[0, 1, 2],
        control_states=[ControlState.CONTROL, ControlState.CONTROL, ControlState.CONTROL]
    )
    circuit.add_operation(c3x)
    
    print(f"✓ Created C³X gate:")
    print(f"  Control qubits: [0, 1, 2]")
    print(f"  Target qubit: 3")
    print(f"  Operation: {c3x}")
    
    return circuit

def example_6_mixed_controls():
    """
    Example 6: Mixed Control States
    Anti-control on qubit 0, Control on qubit 1, Target: qubit 2
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Mixed Control States")
    print("=" * 70)
    
    circuit = QuantumCircuit("MixedControl_Example", qubits=3)
    
    # Mixed controls: qubit 0 (anti-control), qubit 1 (control), target qubit 2
    mixed = PauliZGate(
        target_qubit=2,
        control_qubits=[0, 1],
        control_states=[ControlState.ANTI_CONTROL, ControlState.CONTROL]
    )
    circuit.add_operation(mixed)
    
    print(f"✓ Created mixed control gate:")
    print(f"  Control qubits: [0, 1]")
    print(f"  Control states: [ANTI_CONTROL (○), CONTROL (●)]")
    print(f"  Target qubit: 2")
    print(f"  Gate activates when qubit 0 is |0⟩ AND qubit 1 is |1⟩")
    print(f"  Operation: {mixed}")
    
    return circuit

def example_7_qft():
    """
    Example 7: Quantum Fourier Transform (multi-qubit operation)
    Operates on multiple qubits simultaneously
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Quantum Fourier Transform (QFT)")
    print("=" * 70)
    
    circuit = QuantumCircuit("QFT_Example", qubits=4)
    
    # QFT on qubits 0, 1, 2, 3
    qft = QFTGate(target_qubits=[0, 1, 2, 3], inverse=False)
    circuit.add_operation(qft)
    
    print(f"✓ Created QFT gate:")
    print(f"  Target qubits: [0, 1, 2, 3]")
    print(f"  Inverse: False")
    print(f"  Operation: {qft}")
    
    return circuit

def example_8_controlled_rotation():
    """
    Example 8: Controlled Rotation Gates
    Multi-qubit parametric gates
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Controlled Rotation Gates")
    print("=" * 70)
    
    circuit = QuantumCircuit("ControlledRotation_Example", qubits=3)
    
    # Controlled RY gate
    cry = RYGate(
        target_qubit=2,
        angle=1.57,  # π/2
        control_qubits=[0, 1],
        control_states=[ControlState.CONTROL, ControlState.CONTROL]
    )
    circuit.add_operation(cry)
    
    print(f"✓ Created Controlled-RY gate:")
    print(f"  Control qubits: [0, 1]")
    print(f"  Target qubit: 2")
    print(f"  Angle: π/2 (1.57 rad)")
    print(f"  Operation: {cry}")
    
    return circuit

def example_9_grover_oracle():
    """
    Example 9: Grover Oracle Pattern
    Multi-qubit oracle for |11⟩ state
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Grover Oracle for |11⟩")
    print("=" * 70)
    
    circuit = QuantumCircuit("GroverOracle_Example", qubits=3)
    
    # Initialize superposition
    circuit.add_operation(HadamardGate(0))
    circuit.add_operation(HadamardGate(1))
    circuit.add_operation(HadamardGate(2))
    
    # Oracle: mark |11⟩ state (on first two qubits)
    # Multi-controlled Z on ancilla qubit
    oracle = PauliZGate(
        target_qubit=2,
        control_qubits=[0, 1],
        control_states=[ControlState.CONTROL, ControlState.CONTROL]
    )
    circuit.add_operation(oracle)
    
    print(f"✓ Created Grover oracle:")
    print(f"  1. Hadamard on qubits [0, 1, 2]")
    print(f"  2. CCZ gate marking |11⟩")
    print(f"     Control qubits: [0, 1]")
    print(f"     Target qubit: 2")
    print(f"  Operation: {oracle}")
    
    return circuit

def example_10_entanglement_chain():
    """
    Example 10: Entanglement Chain
    Creating multi-qubit entanglement
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Multi-Qubit Entanglement Chain")
    print("=" * 70)
    
    circuit = QuantumCircuit("EntanglementChain_Example", qubits=5)
    circuit.add_creg(ClassicalRegister("c", 5))
    
    # Create GHZ state: (|00000⟩ + |11111⟩)/√2
    circuit.add_operation(HadamardGate(0))
    
    # Chain of CNOTs
    circuit.add_operation(PauliXGate(1, control_qubits=[0]))
    circuit.add_operation(PauliXGate(2, control_qubits=[1]))
    circuit.add_operation(PauliXGate(3, control_qubits=[2]))
    circuit.add_operation(PauliXGate(4, control_qubits=[3]))
    
    # Measure all
    for i in range(5):
        circuit.add_operation(Measurement(i, i))
    
    print(f"✓ Created 5-qubit GHZ state:")
    print(f"  1. H on qubit 0")
    print(f"  2. CNOT chain: 0→1→2→3→4")
    print(f"  3. Measure all qubits")
    print(f"  Expected result: all 0s or all 1s")
    
    return circuit

def main():
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " Multi-Qubit Quantum Gates Examples ".center(68) + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    circuits = [
        example_1_cnot(),
        example_2_toffoli(),
        example_3_swap(),
        example_4_controlled_swap(),
        example_5_multi_controlled(),
        example_6_mixed_controls(),
        example_7_qft(),
        example_8_controlled_rotation(),
        example_9_grover_oracle(),
        example_10_entanglement_chain()
    ]
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(circuits)}")
    print("\nMulti-qubit gate types demonstrated:")
    print("  1. 2-qubit gates: CNOT, SWAP")
    print("  2. 3-qubit gates: Toffoli (CCX), Fredkin (CSWAP)")
    print("  3. 4+ qubit gates: Multi-controlled gates, QFT")
    print("  4. Mixed controls: CONTROL and ANTI_CONTROL combinations")
    print("  5. Parametric: Controlled rotation gates")
    print("\nKey concepts:")
    print("  • control_qubits: List of control qubit indices")
    print("  • control_states: List of ControlState (CONTROL or ANTI_CONTROL)")
    print("  • target_qubit: The qubit that the gate acts on")
    print("  • For SWAP: both qubits are targets (qubit1, qubit2)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
