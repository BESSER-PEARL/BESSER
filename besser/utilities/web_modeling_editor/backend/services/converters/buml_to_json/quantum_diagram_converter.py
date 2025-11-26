"""
Quantum diagram conversion from BUML to JSON format.
"""

from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, QuantumOperation, PrimitiveGate, ParametricGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, QFTGate,
    PhaseGradientGate, DisplayOperation, Measurement, PostSelection,
    InputGate, OrderGate, ScalarGate, CustomGate, TimeDependentGate,
    SpacerGate, HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SwapGate, SGate, TGate, RXGate, RYGate, RZGate, PhaseGate,
    ControlState
)


def quantum_circuit_to_json(circuit: QuantumCircuit) -> dict:
    """
    Convert a BUML QuantumCircuit to JSON format matching the frontend structure.
    
    Args:
        circuit: The QuantumCircuit model to convert.
        
    Returns:
        A dictionary in the Quirk-compatible JSON format.
    """
    cols = []
    
    # Get total number of qubits
    num_qubits = sum(qreg.size for qreg in circuit.qregs) if circuit.qregs else 0
    
    # Group operations by column (we'll need to infer columns from operation order)
    # For simplicity, each operation gets its own column unless it shares qubits with controls
    
    for op in circuit.operations:
        col = _operation_to_column(op, num_qubits)
        if col:
            cols.append(col)
    
    return {
        "title": circuit.name,
        "model": {
            "cols": cols
        }
    }


def _operation_to_column(op: QuantumOperation, num_qubits: int) -> list:
    """
    Convert a single operation to a column array.
    
    Args:
        op: The quantum operation to convert.
        num_qubits: Total number of qubits in the circuit.
        
    Returns:
        A list representing a column in the circuit.
    """
    # Initialize column with identity (1) for all qubits
    col = [1] * num_qubits
    
    # Get the gate symbol
    symbol = _get_gate_symbol(op)
    
    # Place controls
    for i, ctrl_qubit in enumerate(op.control_qubits):
        if ctrl_qubit < num_qubits:
            ctrl_state = op.control_states[i] if i < len(op.control_states) else ControlState.CONTROL
            if ctrl_state == ControlState.CONTROL:
                col[ctrl_qubit] = "•"  # Filled circle for control
            else:
                col[ctrl_qubit] = "◦"  # Empty circle for anti-control
    
    # Place the gate on target qubits
    for target in op.target_qubits:
        if target < num_qubits:
            col[target] = symbol
    
    # Handle special cases
    if isinstance(op, SwapGate) and len(op.target_qubits) >= 2:
        col[op.target_qubits[0]] = "Swap"
        col[op.target_qubits[1]] = "Swap"
    
    return col


def _get_gate_symbol(op: QuantumOperation) -> str:
    """
    Get the frontend symbol for a quantum operation.
    
    Args:
        op: The quantum operation.
        
    Returns:
        The symbol string used by the frontend.
    """
    # === Primitive Gates ===
    if isinstance(op, HadamardGate):
        return "H"
    if isinstance(op, PauliXGate):
        return "X"
    if isinstance(op, PauliYGate):
        return "Y"
    if isinstance(op, PauliZGate):
        return "Z"
    if isinstance(op, SGate):
        return "S"
    if isinstance(op, TGate):
        return "T"
    if isinstance(op, SwapGate):
        return "Swap"
    
    # === Parametric Gates ===
    if isinstance(op, RXGate):
        return "Rx"
    if isinstance(op, RYGate):
        return "Ry"
    if isinstance(op, RZGate):
        return "Rz"
    if isinstance(op, PhaseGate):
        return "Phase"
    
    if isinstance(op, ParametricGate):
        type_name = op.type_name
        # Quarter turns
        if type_name == 'S_DAG':
            return "S_DAG"
        if type_name == 'V':
            return "V"
        if type_name == 'V_DAG':
            return "V_DAG"
        if type_name in ('SQRT_Y', 'Y^1/2'):
            return "SQRT_Y"
        if type_name in ('SQRT_Y_DAG', 'Y^-1/2'):
            return "SQRT_Y_DAG"
        # Eighth turns
        if type_name == 'T_DAG':
            return "T_DAG"
        if type_name in ('X^1/4', 'SQRT_SQRT_X'):
            return "SQRT_SQRT_X"
        if type_name in ('X^-1/4', 'SQRT_SQRT_X_DAG'):
            return "SQRT_SQRT_X_DAG"
        if type_name in ('Y^1/4', 'SQRT_SQRT_Y'):
            return "SQRT_SQRT_Y"
        if type_name in ('Y^-1/4', 'SQRT_SQRT_Y_DAG'):
            return "SQRT_SQRT_Y_DAG"
        # Generic parametric
        return type_name
    
    if isinstance(op, PrimitiveGate):
        return op.type_name
    
    # === Frequency Gates ===
    if isinstance(op, QFTGate):
        return "QFT_DAG" if op.inverse else "QFT"
    
    if isinstance(op, PhaseGradientGate):
        return "PHASE_GRADIENT_DAG" if op.inverse else "PhaseGradient"
    
    # === Arithmetic Gates ===
    if isinstance(op, ArithmeticGate):
        op_map = {
            'Increment': '+1',
            'Decrement': '-1',
            'Add': '+A',
            'Subtract': '-A',
            'Multiply': '×A',
            'AddAB': '+AB',
            'SubtractAB': '-AB',
            'MultiplyInverse': '×A⁻¹',
            'CountOnes': 'COUNT_1S',
            'XOR': 'XOR',
        }
        return op_map.get(op.operation, op.operation)
    
    # === Modular Arithmetic Gates ===
    if isinstance(op, ModularArithmeticGate):
        op_map = {
            'Increment': 'MOD_INC',
            'Decrement': 'MOD_DEC',
            'Add': 'MOD_ADD',
            'Subtract': 'MOD_SUB',
            'Multiply': 'MOD_MUL',
            'MultiplyInverse': 'MOD_INV_MUL',
            'MultiplyB': 'MOD_MUL_B',
            'MultiplyBInverse': 'MOD_MUL_B_INV',
        }
        return op_map.get(op.operation, f"MOD_{op.operation}")
    
    # === Comparison Gates ===
    if isinstance(op, ComparisonGate):
        op_map = {
            'LessThan': '<',
            'GreaterThan': '>',
            'LessEqual': '≤',
            'GreaterEqual': '≥',
            'Equal': '=',
            'NotEqual': '≠',
            'ALessThan': 'A<',
            'AGreaterThan': 'A>',
            'AEqual': 'A=',
        }
        return op_map.get(op.operation, op.operation)
    
    # === Order Gates ===
    if isinstance(op, OrderGate):
        op_map = {
            'Interleave': 'INTERLEAVE',
            'Deinterleave': 'DEINTERLEAVE',
            'Reverse': 'REVERSE_BITS',
            'RotateLeft': '<<',
            'RotateRight': '>>',
            'CycleBits': 'CYCLE_BITS',
            'TimeShift': 'TIME_SHIFT',
            'TimeShiftInv': 'TIME_SHIFT_INV',
        }
        return op_map.get(op.order_type, op.order_type)
    
    # === Scalar Gates ===
    if isinstance(op, ScalarGate):
        op_map = {
            '1': 'ONE',
            '-1': 'MINUS_ONE',
            'i': 'PHASE_I',
            '-i': 'PHASE_MINUS_I',
            '√i': 'PHASE_SQRT_I',
            '√-i': 'PHASE_SQRT_MINUS_I',
        }
        return op_map.get(op.scalar_type, op.scalar_type)
    
    # === Time-Dependent Gates ===
    if isinstance(op, TimeDependentGate):
        type_map = {
            'Z^t': 'Z_POW_T',
            'Z^-t': 'Z_POW_NEG_T',
            'Y^t': 'Y_POW_T',
            'Y^-t': 'Y_POW_NEG_T',
            'X^t': 'X_POW_T',
            'X^-t': 'X_POW_NEG_T',
            'EXP_X': 'EXP_X',
            'EXP_Y': 'EXP_Y',
            'EXP_Z': 'EXP_Z',
            'Z_FUNC': 'Z_FUNC_T',
            'Y_FUNC': 'Y_FUNC_T',
            'X_FUNC': 'X_FUNC_T',
        }
        return type_map.get(op.type_name, op.type_name)
    
    # === Display Operations ===
    if isinstance(op, DisplayOperation):
        type_map = {
            'Bloch': 'Bloch',
            'Density': 'Density',
            'Amplitude': 'Amps',
            'Chance': 'Chance',
            'Sample': 'SAMPLE',
            'Detect': 'DETECT',
            'AxisSample': 'AXIS_SAMPLE',
        }
        return type_map.get(op.display_type, op.display_type)
    
    # === Measurement ===
    if isinstance(op, Measurement):
        if op.basis == 'X':
            return 'MEASURE_X'
        elif op.basis == 'Y':
            return 'MEASURE_Y'
        else:
            return 'Measure'
    
    # === Post-Selection ===
    if isinstance(op, PostSelection):
        if op.basis == 'Z':
            return 'POST_SELECT_ON' if op.value == 1 else 'POST_SELECT_OFF'
        elif op.basis == 'X':
            return 'POST_SELECT_X_ON' if op.value == 1 else 'POST_SELECT_X_OFF'
        elif op.basis == 'Y':
            return 'POST_SELECT_Y_ON' if op.value == 1 else 'POST_SELECT_Y_OFF'
        return f'POST_SELECT_{op.basis}_{op.value}'
    
    # === Input Gates ===
    if isinstance(op, InputGate):
        return op.input_type
    
    # === Spacer ===
    if isinstance(op, SpacerGate):
        return '…'
    
    # === Custom Gate ===
    if isinstance(op, CustomGate):
        return op.name
    
    # === Fallback ===
    return op.name if hasattr(op, 'name') else 'Unknown'


def quantum_circuit_to_editor_json(circuit: QuantumCircuit) -> dict:
    """
    Convert a BUML QuantumCircuit to the editor JSON format (Quirk-compatible).
    
    This is the format expected by the web modeling editor's quantum canvas.
    It uses the Quirk cols format where each column is an array of gate symbols.
    
    Args:
        circuit: The QuantumCircuit model to convert.
        
    Returns:
        A dictionary in the Quirk-compatible format with version info.
    """
    # Get the basic Quirk format
    quirk_data = quantum_circuit_to_json(circuit)
    
    # Add editor metadata
    return {
        "version": "1.0.0",
        "title": circuit.name,
        "cols": quirk_data["model"]["cols"],
        "gates": []  # Custom gates would go here
    }
