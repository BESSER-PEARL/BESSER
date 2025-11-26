"""
Quantum Model Builder: Generates Python code for BESSER QuantumCircuit models.
"""

from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, QuantumRegister, ClassicalRegister, 
    HadamardGate, PauliXGate, PauliYGate, PauliZGate, 
    SGate, TGate, SwapGate, RXGate, RYGate, RZGate, PhaseGate,
    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,
    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,
    PrimitiveGate, ParametricGate, OrderGate, ScalarGate
)

def quantum_model_to_code(model: QuantumCircuit, file_path: str):
    """
    Generates Python code for a QuantumCircuit model.
    
    Args:
        model (QuantumCircuit): The quantum circuit model.
        file_path (str): The path to save the generated code.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # Imports
        f.write("from besser.BUML.metamodel.quantum.quantum import (\n")
        f.write("    QuantumCircuit, QuantumRegister, ClassicalRegister,\n")
        f.write("    HadamardGate, PauliXGate, PauliYGate, PauliZGate,\n")
        f.write("    SGate, TGate, SwapGate, RXGate, RYGate, RZGate, PhaseGate,\n")
        f.write("    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,\n")
        f.write("    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,\n")
        f.write("    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,\n")
        f.write("    PrimitiveGate, ParametricGate, OrderGate, ScalarGate\n")
        f.write(")\n\n")
        
        # Circuit definition
        f.write(f"# Quantum Circuit: {model.name}\n")
        total_qubits = sum(reg.size for reg in model.qregs)
        f.write(f"qc = QuantumCircuit(name='{model.name}', qubits={total_qubits})\n\n")
        
        # Operations
        for op in model.operations:
            _write_operation(f, op)
            
        f.write("\n")

def _write_operation(f, op):
    """Helper to write a single operation."""
    
    gate_var = "gate"
    
    # === Specific Gate Classes (check these first as they're subclasses) ===
    if isinstance(op, HadamardGate):
        f.write(f"{gate_var} = HadamardGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, PauliXGate):
        f.write(f"{gate_var} = PauliXGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, PauliYGate):
        f.write(f"{gate_var} = PauliYGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, PauliZGate):
        f.write(f"{gate_var} = PauliZGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, SGate):
        f.write(f"{gate_var} = SGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, TGate):
        f.write(f"{gate_var} = TGate(target_qubit={op.target_qubits[0]})\n")
    elif isinstance(op, SwapGate):
        f.write(f"{gate_var} = SwapGate(qubit1={op.target_qubits[0]}, qubit2={op.target_qubits[1]})\n")
        
    # === Parametric Gates (specific subclasses first) ===
    elif isinstance(op, RXGate):
        f.write(f"{gate_var} = RXGate(target_qubit={op.target_qubits[0]}, angle={op.parameter})\n")
    elif isinstance(op, RYGate):
        f.write(f"{gate_var} = RYGate(target_qubit={op.target_qubits[0]}, angle={op.parameter})\n")
    elif isinstance(op, RZGate):
        f.write(f"{gate_var} = RZGate(target_qubit={op.target_qubits[0]}, angle={op.parameter})\n")
    elif isinstance(op, PhaseGate):
        f.write(f"{gate_var} = PhaseGate(target_qubit={op.target_qubits[0]}, angle={op.parameter})\n")
    elif isinstance(op, ParametricGate):
        f.write(f"{gate_var} = ParametricGate(type_name='{op.type_name}', target_qubits={op.target_qubits}, parameter={op.parameter})\n")
        
    # === Frequency Gates ===
    elif isinstance(op, QFTGate):
        f.write(f"{gate_var} = QFTGate(target_qubits={op.target_qubits}, inverse={op.inverse})\n")
    elif isinstance(op, PhaseGradientGate):
        f.write(f"{gate_var} = PhaseGradientGate(target_qubits={op.target_qubits}, inverse={op.inverse})\n")
        
    # === Arithmetic Gates ===
    elif isinstance(op, ArithmeticGate):
        input_qubits_str = f", input_qubits={op.input_qubits}" if op.input_qubits else ""
        f.write(f"{gate_var} = ArithmeticGate(operation='{op.operation}', target_qubits={op.target_qubits}{input_qubits_str})\n")
    elif isinstance(op, ModularArithmeticGate):
        input_qubits_str = f", input_qubits={op.input_qubits}" if op.input_qubits else ""
        f.write(f"{gate_var} = ModularArithmeticGate(operation='{op.operation}', target_qubits={op.target_qubits}, modulo={op.modulo}{input_qubits_str})\n")
    elif isinstance(op, ComparisonGate):
        input_qubits_str = f", input_qubits={op.input_qubits}" if op.input_qubits else ""
        f.write(f"{gate_var} = ComparisonGate(operation='{op.operation}', target_qubits={op.target_qubits}{input_qubits_str})\n")
        
    # === Order Gates ===
    elif isinstance(op, OrderGate):
        f.write(f"{gate_var} = OrderGate(order_type='{op.order_type}', target_qubits={op.target_qubits})\n")
        
    # === Scalar Gates ===
    elif isinstance(op, ScalarGate):
        f.write(f"{gate_var} = ScalarGate(scalar_type='{op.scalar_type}')\n")
        f.write(f"qc.add_operation({gate_var})\n\n")
        return  # ScalarGate has no controls
        
    # === Time-Dependent Gates ===
    elif isinstance(op, TimeDependentGate):
        f.write(f"{gate_var} = TimeDependentGate(type_name='{op.type_name}', target_qubits={op.target_qubits}, parameter_expr='{op.parameter_expr}')\n")
        
    # === Spacer Gate ===
    elif isinstance(op, SpacerGate):
        f.write(f"{gate_var} = SpacerGate(target_qubits={op.target_qubits})\n")
        
    # === Measurement ===
    elif isinstance(op, Measurement):
        f.write(f"{gate_var} = Measurement(target_qubit={op.target_qubits[0]}, output_bit={op.output_bit}, basis='{op.basis}')\n")
        
    # === Post-Selection ===
    elif isinstance(op, PostSelection):
        f.write(f"{gate_var} = PostSelection(target_qubit={op.target_qubits[0]}, value={op.value}, basis='{op.basis}')\n")
        
    # === Display Operations ===
    elif isinstance(op, DisplayOperation):
        f.write(f"{gate_var} = DisplayOperation(display_type='{op.display_type}', target_qubits={op.target_qubits})\n")
        
    # === Input Gates ===
    elif isinstance(op, InputGate):
        value_str = f", value={op.value}" if op.value is not None else ""
        f.write(f"{gate_var} = InputGate(input_type='{op.input_type}', target_qubits={op.target_qubits}{value_str})\n")
        
    # === Custom/User-Defined Gates ===
    elif isinstance(op, CustomGate):
        f.write(f"{gate_var} = CustomGate(name='{op.name}', target_qubits={op.target_qubits})\n")
        
    # === Generic PrimitiveGate (fallback) ===
    elif isinstance(op, PrimitiveGate):
        f.write(f"{gate_var} = PrimitiveGate(type_name='{op.type_name}', target_qubits={op.target_qubits})\n")
    
    else:
        # Fallback for generic or unhandled gates
        f.write(f"# Unknown gate type: {type(op).__name__}\n")
        return

    # Add controls (for gates that support them)
    if hasattr(op, 'control_qubits') and op.control_qubits:
        for i, control_qubit in enumerate(op.control_qubits):
            state = op.control_states[i] if op.control_states and i < len(op.control_states) else ControlState.CONTROL
            state_str = "ControlState.CONTROL" if state == ControlState.CONTROL else "ControlState.ANTI_CONTROL"
            f.write(f"{gate_var}.control_qubits.append({control_qubit})\n")
            f.write(f"{gate_var}.control_states.append({state_str})\n")

    f.write(f"qc.add_operation({gate_var})\n\n")
