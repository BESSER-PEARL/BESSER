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

gate_HadamardGate_2258755758384 = HadamardGate(target_qubit=0)
qc.add_operation(gate_HadamardGate_2258755758384)

gate_HadamardGate_2258755365216 = HadamardGate(target_qubit=1)
qc.add_operation(gate_HadamardGate_2258755365216)

gate_HadamardGate_2258753324000 = HadamardGate(target_qubit=2)
qc.add_operation(gate_HadamardGate_2258753324000)

gates_Uf_2258755758480 = []
gate_PauliXGate_2258755758672 = PauliXGate(target_qubit=2)
gates_Uf_2258755758480.append(gate_PauliXGate_2258755758672)
gate_HadamardGate_2258755758816 = HadamardGate(target_qubit=2)
gates_Uf_2258755758480.append(gate_HadamardGate_2258755758816)
gate_PauliXGate_2258755758912 = PauliXGate(target_qubit=2)
gates_Uf_2258755758480.append(gate_PauliXGate_2258755758912)
gate_HadamardGate_2258755759008 = HadamardGate(target_qubit=2)
gates_Uf_2258755758480.append(gate_HadamardGate_2258755759008)
gate_PauliXGate_2258755759104 = PauliXGate(target_qubit=2)
gates_Uf_2258755758480.append(gate_PauliXGate_2258755759104)
gate_FunctionGate_2258755758480 = FunctionGate(name='Uf', target_qubits=[0], gates=gates_Uf_2258755758480)
qc.add_operation(gate_FunctionGate_2258755758480)

gates_V_2258753559952 = []
gate_HadamardGate_2258755759200 = HadamardGate(target_qubit=0)
gates_V_2258753559952.append(gate_HadamardGate_2258755759200)
gate_HadamardGate_2258755759344 = HadamardGate(target_qubit=1)
gates_V_2258753559952.append(gate_HadamardGate_2258755759344)
gate_HadamardGate_2258755759440 = HadamardGate(target_qubit=2)
gates_V_2258753559952.append(gate_HadamardGate_2258755759440)
gate_PauliXGate_2258755759536 = PauliXGate(target_qubit=2)
gates_V_2258753559952.append(gate_PauliXGate_2258755759536)
gate_HadamardGate_2258755759632 = HadamardGate(target_qubit=2)
gates_V_2258753559952.append(gate_HadamardGate_2258755759632)
gate_PauliXGate_2258755759728 = PauliXGate(target_qubit=2)
gates_V_2258753559952.append(gate_PauliXGate_2258755759728)
gate_HadamardGate_2258755759824 = HadamardGate(target_qubit=0)
gates_V_2258753559952.append(gate_HadamardGate_2258755759824)
gate_HadamardGate_2258755759920 = HadamardGate(target_qubit=1)
gates_V_2258753559952.append(gate_HadamardGate_2258755759920)
gate_HadamardGate_2258755760016 = HadamardGate(target_qubit=2)
gates_V_2258753559952.append(gate_HadamardGate_2258755760016)
gate_PauliXGate_2258755760112 = PauliXGate(target_qubit=2)
gates_V_2258753559952.append(gate_PauliXGate_2258755760112)
gate_HadamardGate_2258755760208 = HadamardGate(target_qubit=2)
gates_V_2258753559952.append(gate_HadamardGate_2258755760208)
gate_FunctionGate_2258753559952 = FunctionGate(name='V', target_qubits=[0], gates=gates_V_2258753559952)
qc.add_operation(gate_FunctionGate_2258753559952)

gates_Uf_2258755758432 = []
gate_PauliXGate_2258755758576 = PauliXGate(target_qubit=2)
gates_Uf_2258755758432.append(gate_PauliXGate_2258755758576)
gate_HadamardGate_2258755760496 = HadamardGate(target_qubit=2)
gates_Uf_2258755758432.append(gate_HadamardGate_2258755760496)
gate_PauliXGate_2258755760592 = PauliXGate(target_qubit=2)
gates_Uf_2258755758432.append(gate_PauliXGate_2258755760592)
gate_HadamardGate_2258755760688 = HadamardGate(target_qubit=2)
gates_Uf_2258755758432.append(gate_HadamardGate_2258755760688)
gate_PauliXGate_2258755760784 = PauliXGate(target_qubit=2)
gates_Uf_2258755758432.append(gate_PauliXGate_2258755760784)
gate_FunctionGate_2258755758432 = FunctionGate(name='Uf', target_qubits=[0], gates=gates_Uf_2258755758432)
qc.add_operation(gate_FunctionGate_2258755758432)

gate_FunctionGate_2258755760400 = FunctionGate(name='V', target_qubits=[0])
qc.add_operation(gate_FunctionGate_2258755760400)

gate_Measurement_2258755760976 = Measurement(target_qubit=0, output_bit=0, basis='X')
qc.add_operation(gate_Measurement_2258755760976)

gate_Measurement_2258740071024 = Measurement(target_qubit=1, output_bit=1, basis='X')
qc.add_operation(gate_Measurement_2258740071024)

gate_Measurement_2258755761120 = Measurement(target_qubit=2, output_bit=2, basis='Z')
qc.add_operation(gate_Measurement_2258755761120)




