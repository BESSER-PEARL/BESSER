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
    ControlState, RXXGate, RZZGate, RZXGate, RYYGate, BellGate, iSwapGate, SqrtSwapGate,
    CXGate, CYGate, CZGate, CHGate, CRXGate, CRYGate, CRZGate, CPhaseGate,
    FunctionGate
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
    
    # Get total number of classical bits
    num_clbits = sum(creg.size for creg in circuit.cregs) if circuit.cregs else 0
    
    # Group operations by column (we'll need to infer columns from operation order)
    # For simplicity, each operation gets its own column unless it shares qubits with controls
    
    gate_metadata = {}
    
    for op in circuit.operations:
        col = _operation_to_column(op, num_qubits, gate_metadata, len(cols))
        if col:
            cols.append(col)
    
    return {
        "title": circuit.name,
        "model": {
            "cols": cols,
            "gateMetadata": gate_metadata,
            "classicalBitCount": num_clbits
        }
    }


def _operation_to_column(op: QuantumOperation, num_qubits: int, gate_metadata: dict, col_index: int) -> list:
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
    metadata_added = False  # Track if we've already added metadata for this CustomGate
    for target in op.target_qubits:
        if target < num_qubits:
            col[target] = symbol
            
            # Handle CustomGate or FunctionGate with definition (Nested Circuit)
            # Only add metadata once (for the first target) to avoid duplicates
            if isinstance(op, (CustomGate, FunctionGate)) and not metadata_added:
                metadata_key = f"{col_index}_{target}"
                metadata = {
                    "label": op.name,
                    "type": "FUNCTION", # Or derive from op
                    "isFunctionGate": True
                }
                
                if hasattr(op, 'definition') and op.definition and op.definition.circuit:
                    # Recursively convert the nested circuit
                    nested_json = quantum_circuit_to_json(op.definition.circuit)
                    metadata["nestedCircuit"] = nested_json['model']
                elif hasattr(op, 'gates') and op.gates:
                    # Serialize list of gates into new columns format
                    metadata["nestedCircuit"] = _serialize_gates_to_columns(op.gates)
                
                gate_metadata[metadata_key] = metadata
                metadata_added = True
    
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
    if isinstance(op, SwapGate):
        return "Swap"
    if isinstance(op, SqrtSwapGate):
        return "SqrtSwap"
    if isinstance(op, iSwapGate):
        return "iSwap"
    if isinstance(op, BellGate):
        return "Bell"
    
    # === Controlled Gates (Map to base symbol) ===
    if isinstance(op, CXGate):
        return "X"
    if isinstance(op, CYGate):
        return "Y"
    if isinstance(op, CZGate):
        return "Z"
    if isinstance(op, CHGate):
        return "H"
    if isinstance(op, CRXGate):
        return "Rx"
    if isinstance(op, CRYGate):
        return "Ry"
    if isinstance(op, CRZGate):
        return "Rz"
    if isinstance(op, CPhaseGate):
        return "Phase"

    # === Parametric Gates ===
    if isinstance(op, RXGate):
        return "Rx"
    if isinstance(op, RYGate):
        return "Ry"
    if isinstance(op, RZGate):
        return "Rz"
    if isinstance(op, PhaseGate):
        return "Phase"
    if isinstance(op, RXXGate):
        return "RXX"
    if isinstance(op, RZZGate):
        return "RZZ"
    if isinstance(op, RZXGate):
        return "RZX"
    if isinstance(op, RYYGate):
        return "RYY"
    
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
        if type_name in ('X^1_4', 'X^1/4', 'SQRT_SQRT_X'):
            return "SQRT_SQRT_X"
        if type_name in ('X^_1_4', 'X^-1/4', 'SQRT_SQRT_X_DAG'):
            return "SQRT_SQRT_X_DAG"
        if type_name in ('Y^1_4', 'Y^1/4', 'SQRT_SQRT_Y'):
            return "SQRT_SQRT_Y"
        if type_name in ('Y^_1_4', 'Y^-1/4', 'SQRT_SQRT_Y_DAG'):
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
        return op_map.get(op.operation_type, op.operation_type)
    
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
        return op_map.get(op.operation_type, f"MOD_{op.operation_type}")
    
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
        "gateMetadata": quirk_data["model"].get("gateMetadata", {}),
        "gates": []  # Custom gates would go here
    }


def _serialize_gates_to_columns(gates):
    """
    Serialize a list of gates into the columns/gates JSON format.
    
    Args:
        gates: List of Gate objects
        
    Returns:
        Dict with 'columns', 'qubitCount', and 'initialStates'
    """
    if not gates:
        return {"columns": [], "qubitCount": 0, "initialStates": []}
        
    # Determine number of qubits
    max_qubit = 0
    for gate in gates:
        for q in gate.target_qubits:
            max_qubit = max(max_qubit, q)
        if hasattr(gate, 'control_qubits') and gate.control_qubits:
            for q in gate.control_qubits:
                max_qubit = max(max_qubit, q)
            
    num_qubits = max_qubit + 1
    
    # Organize gates into columns
    # This is a simplified placement: one column per gate
    columns = []
    
    for gate in gates:
        col_gates = [None] * num_qubits
        
        # Get gate symbol and type
        symbol = _get_gate_symbol(gate)
        
        # Create gate object for JSON
        gate_obj = {
            "type": symbol,
            "label": symbol,
            "id": f"{symbol}-{id(gate)}",
            "height": 1,
            "canResize": False
        }
        
        # Place control qubits first
        if hasattr(gate, 'control_qubits') and gate.control_qubits:
            for i, ctrl_qubit in enumerate(gate.control_qubits):
                if ctrl_qubit < num_qubits:
                    ctrl_state = gate.control_states[i] if i < len(gate.control_states) else ControlState.CONTROL
                    ctrl_type = "CONTROL" if ctrl_state == ControlState.CONTROL else "ANTI_CONTROL"
                    ctrl_obj = {
                        "type": ctrl_type,
                        "label": "•" if ctrl_state == ControlState.CONTROL else "◦",
                        "id": f"{ctrl_type}-{id(gate)}-{i}",
                        "height": 1,
                        "isControl": True
                    }
                    col_gates[ctrl_qubit] = ctrl_obj
        
        # Place the gate on target qubit
        if gate.target_qubits:
            target = gate.target_qubits[0]
            if target < num_qubits:
                col_gates[target] = gate_obj
                
        columns.append({"gates": col_gates})
        
    return {
        "columns": columns,
        "qubitCount": num_qubits,
        "initialStates": ["|0⟩"] * num_qubits
    }
