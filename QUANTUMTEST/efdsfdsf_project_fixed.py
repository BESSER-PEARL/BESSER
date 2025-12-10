from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SGate, TGate, SwapGate, RXGate, RYGate, RZGate, PhaseGate,
    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,
    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,
    PrimitiveGate, ParametricGate, OrderGate, ScalarGate, FunctionGate,
    GateDefinition
)

# Quantum Circuit: Quantum_Circuit
qc = QuantumCircuit(name='Quantum_Circuit', qubits=15)

gate_HadamardGate_2333504131536 = HadamardGate(target_qubit=0)
qc.add_operation(gate_HadamardGate_2333504131536)

gate_HadamardGate_2333477654480 = HadamardGate(target_qubit=1)
qc.add_operation(gate_HadamardGate_2333477654480)

gate_HadamardGate_2333504131728 = HadamardGate(target_qubit=2)
qc.add_operation(gate_HadamardGate_2333504131728)

gates_Uf_2333504132880 = []
gate_PauliXGate_2333504131664 = PauliXGate(target_qubit=2)
gates_Uf_2333504132880.append(gate_PauliXGate_2333504131664)
gate_HadamardGate_2333417889616 = HadamardGate(target_qubit=2)
gates_Uf_2333504132880.append(gate_HadamardGate_2333417889616)
gate_PauliXGate_2333504131984 = PauliXGate(target_qubit=2)
gates_Uf_2333504132880.append(gate_PauliXGate_2333504131984)
gate_HadamardGate_2333504132240 = HadamardGate(target_qubit=2)
gates_Uf_2333504132880.append(gate_HadamardGate_2333504132240)
gate_PauliXGate_2333504132496 = PauliXGate(target_qubit=2)
gates_Uf_2333504132880.append(gate_PauliXGate_2333504132496)
gate_FunctionGate_2333504132880 = FunctionGate(name='Uf', target_qubits=[0, 1, 2], gates=gates_Uf_2333504132880)
qc.add_operation(gate_FunctionGate_2333504132880)

gates_V_2333504135952 = []
gate_HadamardGate_2333504132944 = HadamardGate(target_qubit=0)
gates_V_2333504135952.append(gate_HadamardGate_2333504132944)
gate_HadamardGate_2333504133328 = HadamardGate(target_qubit=1)
gates_V_2333504135952.append(gate_HadamardGate_2333504133328)
gate_HadamardGate_2333504133584 = HadamardGate(target_qubit=2)
gates_V_2333504135952.append(gate_HadamardGate_2333504133584)
gate_PauliXGate_2333504133840 = PauliXGate(target_qubit=2)
gates_V_2333504135952.append(gate_PauliXGate_2333504133840)
gate_HadamardGate_2333504134096 = HadamardGate(target_qubit=2)
gates_V_2333504135952.append(gate_HadamardGate_2333504134096)
gate_PauliXGate_2333504134416 = PauliXGate(target_qubit=2)
gates_V_2333504135952.append(gate_PauliXGate_2333504134416)
gate_HadamardGate_2333504134672 = HadamardGate(target_qubit=0)
gates_V_2333504135952.append(gate_HadamardGate_2333504134672)
gate_HadamardGate_2333504134928 = HadamardGate(target_qubit=1)
gates_V_2333504135952.append(gate_HadamardGate_2333504134928)
gate_HadamardGate_2333504135184 = HadamardGate(target_qubit=2)
gates_V_2333504135952.append(gate_HadamardGate_2333504135184)
gate_PauliXGate_2333504134352 = PauliXGate(target_qubit=2)
gates_V_2333504135952.append(gate_PauliXGate_2333504134352)
gate_HadamardGate_2333504135632 = HadamardGate(target_qubit=2)
gates_V_2333504135952.append(gate_HadamardGate_2333504135632)
gate_FunctionGate_2333504135952 = FunctionGate(name='V', target_qubits=[0, 1, 2], gates=gates_V_2333504135952)
qc.add_operation(gate_FunctionGate_2333504135952)

gates_Uf_2333504137680 = []
gate_PauliXGate_2333504136272 = PauliXGate(target_qubit=2)
gates_Uf_2333504137680.append(gate_PauliXGate_2333504136272)
gate_HadamardGate_2333504136528 = HadamardGate(target_qubit=2)
gates_Uf_2333504137680.append(gate_HadamardGate_2333504136528)
gate_PauliXGate_2333504136784 = PauliXGate(target_qubit=2)
gates_Uf_2333504137680.append(gate_PauliXGate_2333504136784)
gate_HadamardGate_2333504137040 = HadamardGate(target_qubit=2)
gates_Uf_2333504137680.append(gate_HadamardGate_2333504137040)
gate_PauliXGate_2333504137296 = PauliXGate(target_qubit=2)
gates_Uf_2333504137680.append(gate_PauliXGate_2333504137296)
gate_FunctionGate_2333504137680 = FunctionGate(name='Uf', target_qubits=[0, 1, 2], gates=gates_Uf_2333504137680)
qc.add_operation(gate_FunctionGate_2333504137680)

gate_FunctionGate_2333504137744 = FunctionGate(name='V', target_qubits=[0, 1, 2])
qc.add_operation(gate_FunctionGate_2333504137744)

gate_Measurement_2333504138064 = Measurement(target_qubit=0, output_bit=0, basis='X')
qc.add_operation(gate_Measurement_2333504138064)

gate_Measurement_2333504138384 = Measurement(target_qubit=1, output_bit=1, basis='X')
qc.add_operation(gate_Measurement_2333504138384)

gate_Measurement_2333504138768 = Measurement(target_qubit=2, output_bit=2, basis='Z')
qc.add_operation(gate_Measurement_2333504138768)


