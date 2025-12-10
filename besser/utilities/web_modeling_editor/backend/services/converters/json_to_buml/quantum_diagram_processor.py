"""
Quantum diagram processing for converting JSON to BUML format.
"""

from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, GateDefinition,
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SGate, TGate, SwapGate, RXGate, RYGate, RZGate,
    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,
    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,
    ParametricGate, OrderGate, ScalarGate, FunctionGate,
    RXXGate, RZZGate, RZXGate, RYYGate, BellGate, iSwapGate, SqrtSwapGate, PhaseGate
)
import math
import re


def sanitize_name(name: str) -> str:
    """
    Sanitize a name to be valid for BUML NamedElement.
    According to NamedElement validation:
    - Spaces are not allowed (replaced with underscores)
    - Hyphens are not allowed (replaced with underscores)
    - Other characters like ^, (, ), etc. are allowed
    
    Args:
        name (str): The original name
        
    Returns:
        str: Sanitized name safe for NamedElement
    """
    if not name:
        return "unnamed"
    
    # Replace spaces with underscores
    sanitized = name.replace(' ', '_')
    
    # Replace hyphens with underscores (but preserve minus signs in context like Z^-t)
    # Only replace standalone hyphens, not those that are part of mathematical notation
    # For safety, we'll just replace all hyphens with underscores
    sanitized = sanitized.replace('-', '_')
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        return "unnamed"
    
    return sanitized


def process_quantum_diagram(json_data):
    model_data = json_data.get('model', {})
    if not model_data:
        model_data = json_data

    gate_metadata = model_data.get('gateMetadata', {})

    cols = model_data.get('cols', [])

    raw_title = json_data.get('title', 'Quantum_Circuit')
    title = raw_title.replace(' ', '_').replace('-', '_')
    title = ''.join(c for c in title if c.isalnum() or c == '_')
    if not title:
        title = 'Quantum_Circuit'

    num_qubits = 0
    for col in cols:
        if len(col) > num_qubits:
            num_qubits = len(col)

    qc = QuantumCircuit(name=title, qubits=num_qubits)

    for col_idx, col in enumerate(cols):
        col_gates = {}
        controls = []
        swap_qubits = []
        rxx_qubits = []
        rzz_qubits = []
        rzx_qubits = []
        ryy_qubits = []
        bell_qubits = []
        iswap_qubits = []
        sqrtswap_qubits = []

        for q_idx, symbol in enumerate(col):
            if symbol == 1 or symbol == "1":
                continue
            if symbol == "\u2022":  # • (bullet/control)
                controls.append((q_idx, ControlState.CONTROL))
            elif symbol == "\u25e6":  # ◦ (anti-control)
                controls.append((q_idx, ControlState.ANTI_CONTROL))
            elif symbol == "Swap":
                swap_qubits.append(q_idx)
            elif symbol in ("RXX", "rxx"):
                rxx_qubits.append(q_idx)
            elif symbol in ("RZZ", "rzz"):
                rzz_qubits.append(q_idx)
            elif symbol in ("RZX", "rzx"):
                rzx_qubits.append(q_idx)
            elif symbol in ("RYY", "ryy"):
                ryy_qubits.append(q_idx)
            elif symbol in ("Bell", "BELL"):
                bell_qubits.append(q_idx)
            elif symbol in ("iSwap", "ISWAP"):
                iswap_qubits.append(q_idx)
            elif symbol in ("SqrtSwap", "SQRT_SWAP"):
                sqrtswap_qubits.append(q_idx)
            else:
                col_gates[q_idx] = symbol

        # Check for metadata-defined custom gates (nested circuits)
        # We need to do this BEFORE creating standard gates because a symbol might be a placeholder
        # for a custom gate defined in metadata.
        # However, the loop below iterates over col_gates items.
        # We should check metadata for each gate we are about to create.

        if len(swap_qubits) >= 2:
            # Check for SqrtSwap or iSwap based on symbol if needed, but here we only see "Swap"
            # If the symbol was "SqrtSwap" or "iSwap", it would be handled in the main loop if we didn't special case it here.
            # However, the loop above puts "Swap" into swap_qubits list.
            # If we want to support SqrtSwap/iSwap as multi-qubit gates in the same way, we might need to adjust the loop.
            # For now, let's assume standard Swap is the only one using the "Swap" string literal in this specific way in the frontend.
            gate = SwapGate(swap_qubits[0], swap_qubits[1],
                          control_qubits=[c[0] for c in controls],
                          control_states=[c[1] for c in controls])
            qc.add_operation(gate)

        if len(rxx_qubits) >= 2:
            # RXX typically needs a parameter. Defaulting to 0 if not found.
            gate = RXXGate(rxx_qubits[0], rxx_qubits[1], theta=0.0)
            # Add controls if any
            if controls:
                gate.control_qubits = [c[0] for c in controls]
                gate.control_states = [c[1] for c in controls]
            qc.add_operation(gate)

        if len(rzz_qubits) >= 2:
            gate = RZZGate(rzz_qubits[0], rzz_qubits[1], theta=0.0)
            if controls:
                gate.control_qubits = [c[0] for c in controls]
                gate.control_states = [c[1] for c in controls]
            qc.add_operation(gate)

        if len(rzx_qubits) >= 2:
            gate = RZXGate(rzx_qubits[0], rzx_qubits[1], theta=0.0)
            if controls:
                gate.control_qubits = [c[0] for c in controls]
                gate.control_states = [c[1] for c in controls]
            qc.add_operation(gate)

        if len(ryy_qubits) >= 2:
            gate = RYYGate(ryy_qubits[0], ryy_qubits[1], theta=0.0)
            if controls:
                gate.control_qubits = [c[0] for c in controls]
                gate.control_states = [c[1] for c in controls]
            qc.add_operation(gate)

        if len(bell_qubits) >= 2:
            gate = BellGate(bell_qubits[0], bell_qubits[1])
            qc.add_operation(gate)

        if len(iswap_qubits) >= 2:
            gate = iSwapGate(iswap_qubits[0], iswap_qubits[1])
            qc.add_operation(gate)

        if len(sqrtswap_qubits) >= 2:
            gate = SqrtSwapGate(sqrtswap_qubits[0], sqrtswap_qubits[1])
            qc.add_operation(gate)

        # Track qubits already processed (for multi-qubit custom gates)
        processed_qubits = set()
        
        for q_idx, symbol in col_gates.items():
            if q_idx in processed_qubits:
                continue  # Skip already processed qubits
                
            # Check for metadata
            metadata_key = f"{col_idx}_{q_idx}"
            metadata = gate_metadata.get(metadata_key)
            
            gate = None
            
            # Check if this is a function gate (symbol starts with __FUNC__)
            is_func_gate = isinstance(symbol, str) and symbol.startswith('__FUNC__')
            
            if metadata and metadata.get('nestedCircuit'):
                # This is a custom function gate with a nested circuit
                nested_circuit_data = metadata['nestedCircuit']
                
                # Determine target qubits based on metadata height
                # Function gates span from q_idx to q_idx + height - 1
                height = metadata.get('height', 1)
                target_qubits = list(range(q_idx, q_idx + height))
                
                # Mark other qubits as processed to avoid duplicate processing
                for other_q_idx in target_qubits:
                    if other_q_idx != q_idx:
                        processed_qubits.add(other_q_idx)
                
                # Check if nested circuit uses new format (columns with gates) or old format (cols)
                if 'columns' in nested_circuit_data:
                    # New format: parse gates from columns array
                    gate_name = sanitize_name(metadata.get('label', symbol))
                    nested_gates = _parse_nested_columns(nested_circuit_data)
                    gate = FunctionGate(
                        name=gate_name, 
                        target_qubits=target_qubits, 
                        gates=nested_gates,
                        control_qubits=[c[0] for c in controls], 
                        control_states=[c[1] for c in controls]
                    )
                else:
                    # Old format: recursively process as QuantumCircuit
                    gate_name = sanitize_name(metadata.get('label', symbol))
                    nested_qc = process_quantum_diagram(nested_circuit_data)
                    gate_def = GateDefinition(name=gate_name, circuit=nested_qc)
                    gate = FunctionGate(
                        name=gate_name, 
                        target_qubits=target_qubits, 
                        definition=gate_def, 
                        control_qubits=[c[0] for c in controls], 
                        control_states=[c[1] for c in controls]
                    )
            
            elif metadata and metadata.get('isFunctionGate'):
                 # Function gate without explicit nested circuit (maybe just a label/type)
                 # Determine target qubits based on metadata height
                 height = metadata.get('height', 1)
                 target_qubits = list(range(q_idx, q_idx + height))
                 
                 # Mark other qubits as processed
                 for other_q_idx in target_qubits:
                     if other_q_idx != q_idx:
                         processed_qubits.add(other_q_idx)
                 
                 gate = CustomGate(name=sanitize_name(metadata.get('label', symbol)), target_qubits=target_qubits, control_qubits=[c[0] for c in controls], control_states=[c[1] for c in controls])
            
            elif is_func_gate:
                # Symbol starts with __FUNC__ but no metadata found yet
                # Search for metadata with this column and any row that matches this function
                # Extract function name from symbol (e.g., __FUNC__FUNCTION -> FUNCTION)
                func_name = symbol[8:] if len(symbol) > 8 else symbol
                
                # Try to find metadata for this gate
                found_metadata = gate_metadata.get(metadata_key)
                
                # Determine target qubits based on metadata height if available
                if found_metadata and 'height' in found_metadata:
                    height = found_metadata.get('height', 1)
                    target_qubits = list(range(q_idx, q_idx + height))
                    
                    # Mark other qubits as processed
                    for other_q_idx in target_qubits:
                        if other_q_idx != q_idx:
                            processed_qubits.add(other_q_idx)
                else:
                    # Fallback: Find all qubits with the same symbol
                    target_qubits = [q_idx]
                    for other_q_idx, other_symbol in col_gates.items():
                        if other_q_idx != q_idx and other_symbol == symbol:
                            target_qubits.append(other_q_idx)
                            processed_qubits.add(other_q_idx)
                    target_qubits.sort()
                
                if found_metadata and found_metadata.get('nestedCircuit'):
                    nested_circuit_data = found_metadata['nestedCircuit']
                    gate_name = sanitize_name(found_metadata.get('label', func_name))
                    if 'columns' in nested_circuit_data:
                        nested_gates = _parse_nested_columns(nested_circuit_data)
                        gate = FunctionGate(
                            name=gate_name, 
                            target_qubits=target_qubits, 
                            gates=nested_gates,
                            control_qubits=[c[0] for c in controls], 
                            control_states=[c[1] for c in controls]
                        )
                    else:
                        nested_qc = process_quantum_diagram(nested_circuit_data)
                        gate_def = GateDefinition(name=gate_name, circuit=nested_qc)
                        gate = FunctionGate(
                            name=gate_name, 
                            target_qubits=target_qubits, 
                            definition=gate_def, 
                            control_qubits=[c[0] for c in controls], 
                            control_states=[c[1] for c in controls]
                        )
                else:
                    # No nested circuit found, create basic CustomGate
                    gate_name = sanitize_name(found_metadata.get('label', func_name) if found_metadata else func_name)
                    gate = CustomGate(
                        name=gate_name, 
                        target_qubits=target_qubits, 
                        control_qubits=[c[0] for c in controls], 
                        control_states=[c[1] for c in controls]
                    )

            else:
                gate = _create_gate(symbol, q_idx, controls)

            if gate:
                qc.add_operation(gate)

    return qc


def _parse_nested_columns(nested_data):
    """
    Parse nested circuit data in the new columns/gates format.
    
    The format is:
    {
        "columns": [
            { "gates": [null, {"type": "H", "label": "H", ...}, ...] },
            ...
        ],
        "qubitCount": 2,
        "initialStates": [...]
    }
    
    Returns a list of Gate objects.
    """
    gates_list = []
    columns = nested_data.get('columns', [])
    
    for col in columns:
        col_gates = col.get('gates', [])
        for qubit_idx, gate_data in enumerate(col_gates):
            if gate_data is None:
                continue
            
            gate_type = gate_data.get('type', '')
            
            # Create the appropriate gate based on type
            gate = _create_gate_from_type(gate_type, qubit_idx)
            if gate:
                gates_list.append(gate)
    
    return gates_list


def _create_gate_from_type(gate_type, target_qubit):
    """Create a gate from a type string (used for nested circuit parsing)."""
    # Map simple gate types to gate classes
    if gate_type == 'H':
        return HadamardGate(target_qubit)
    elif gate_type == 'X':
        return PauliXGate(target_qubit)
    elif gate_type == 'Y':
        return PauliYGate(target_qubit)
    elif gate_type == 'Z':
        return PauliZGate(target_qubit)
    elif gate_type == 'S':
        return SGate(target_qubit)
    elif gate_type == 'T':
        return TGate(target_qubit)
    elif gate_type in ('Rx', 'RX'):
        return RXGate(target_qubit, theta=0)  # Default theta
    elif gate_type in ('Ry', 'RY'):
        return RYGate(target_qubit, theta=0)
    elif gate_type in ('Rz', 'RZ'):
        return RZGate(target_qubit, theta=0)
    elif gate_type == 'Measure':
        return Measurement(target_qubit)
    # Add more gate types as needed
    return None


def _create_gate(symbol, target_qubit, controls):
    control_qubits = [c[0] for c in controls] if controls else []
    control_states = [c[1] for c in controls] if controls else []

    # === Half Turns ===
    if symbol == 'H':
        return HadamardGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'X':
        return PauliXGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'Y':
        return PauliYGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'Z':
        return PauliZGate(target_qubit, control_qubits=control_qubits, control_states=control_states)

    # === Quarter Turns ===
    if symbol in ('S', 'Z^½'):
        return SGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('S_DAG', 'Z^-½'):
        return ParametricGate('S_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)
    if symbol in ('V', 'X^½'):
        return ParametricGate('V', [target_qubit], math.pi/2, control_qubits, control_states)
    if symbol in ('V_DAG', 'X^-½'):
        return ParametricGate('V_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)
    if symbol in ('SQRT_Y', 'Y^½'):
        return ParametricGate('SQRT_Y', [target_qubit], math.pi/2, control_qubits, control_states)
    if symbol in ('SQRT_Y_DAG', 'Y^-½'):
        return ParametricGate('SQRT_Y_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)

    # === Eighth Turns ===
    if symbol in ('T', 'Z^¼'):
        return TGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('T_DAG', 'Z^-¼'):
        return ParametricGate('T_DAG', [target_qubit], -math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_X', 'X^¼'):
        return ParametricGate('X^1/4', [target_qubit], math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_X_DAG', 'X^-¼'):
        return ParametricGate('X^-1/4', [target_qubit], -math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_Y', 'Y^¼'):
        return ParametricGate('Y^1/4', [target_qubit], math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_Y_DAG', 'Y^-¼'):
        return ParametricGate('Y^-1/4', [target_qubit], -math.pi/4, control_qubits, control_states)

    # === Phase Gate ===
    if symbol in ('Phase', 'PHASE'):
        # Default to 0 if no parameter provided (though Phase usually needs one)
        # In Quirk, Phase might be a dynamic rotation or fixed.
        # If it's the specific "Phase" gate from our metamodel:
        return PhaseGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)

    # === Parametric Rotations ===
    if symbol.startswith('Rx') or symbol == 'X_POW':
        return RXGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol.startswith('Ry') or symbol == 'Y_POW':
        return RYGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol.startswith('Rz') or symbol == 'Z_POW':
        return RZGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('EXP_X', 'exp(iXt)'):
        return TimeDependentGate('EXP_X', [target_qubit], 'exp(iXt)', control_qubits, control_states)
    if symbol in ('EXP_Y', 'exp(iYt)'):
        return TimeDependentGate('EXP_Y', [target_qubit], 'exp(iYt)', control_qubits, control_states)
    if symbol in ('EXP_Z', 'exp(iZt)'):
        return TimeDependentGate('EXP_Z', [target_qubit], 'exp(iZt)', control_qubits, control_states)

    # === Time-Dependent (Spinning) Gates ===
    if symbol in ('Z_POW_T', 'Z^t'):
        return TimeDependentGate('Z^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('Z_POW_NEG_T', 'Z^-t'):
        return TimeDependentGate('Z^-t', [target_qubit], '-t', control_qubits, control_states)
    if symbol in ('Y_POW_T', 'Y^t'):
        return TimeDependentGate('Y^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('Y_POW_NEG_T', 'Y^-t'):
        return TimeDependentGate('Y^-t', [target_qubit], '-t', control_qubits, control_states)
    if symbol in ('X_POW_T', 'X^t'):
        return TimeDependentGate('X^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('X_POW_NEG_T', 'X^-t'):
        return TimeDependentGate('X^-t', [target_qubit], '-t', control_qubits, control_states)

    # === Formulaic Gates ===
    if symbol in ('Z_FUNC_T', 'RZ_FUNC_T'):
        return TimeDependentGate('Z_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('Y_FUNC_T', 'RY_FUNC_T'):
        return TimeDependentGate('Y_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('X_FUNC_T', 'RX_FUNC_T'):
        return TimeDependentGate('X_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('TIME_SHIFT', 'TimeShift'):
        return OrderGate('TimeShift', [target_qubit])
    if symbol in ('TIME_SHIFT_INV', 'TimeShiftInv'):
        return OrderGate('TimeShiftInv', [target_qubit])

    # === Display Operations ===
    if symbol in ('Bloch', 'BLOCH'):
        return DisplayOperation('Bloch', [target_qubit])
    if symbol in ('Density', 'DENSITY'):
        return DisplayOperation('Density', [target_qubit])
    if symbol in ('Amps', 'Amplitude', 'AMPLITUDE', 'AMP'):
        return DisplayOperation('Amplitude', [target_qubit])
    if symbol in ('Chance', 'CHANCE'):
        return DisplayOperation('Chance', [target_qubit])

    # === Sampling Gates ===
    if symbol in ('SAMPLE', 'Sample'):
        return DisplayOperation('Sample', [target_qubit])
    if symbol in ('DETECT', 'Detect'):
        return DisplayOperation('Detect', [target_qubit])
    if symbol in ('AXIS_SAMPLE', 'AxisSample'):
        return DisplayOperation('AxisSample', [target_qubit])

    # === Two-Qubit Gates (that might appear as single symbols on target) ===
    # Note: These usually require two targets. If the JSON format puts the symbol on both,
    # we need to handle that. The current logic iterates per qubit.
    # If we encounter a multi-qubit gate symbol here, it might be better to handle it
    # in a pre-processing step or ensure the frontend sends unique identifiers.
    # However, for now, we can try to return the gate.
    # BUT: The current loop adds the gate for EACH qubit it appears on.
    # We need to avoid adding the same multi-qubit gate multiple times.
    # The Swap logic above handles "Swap" specifically.
    # For RXX, RZZ, etc., if they appear as "RXX" on two qubits, we need similar logic.
    # For simplicity, let's assume they are handled if they appear.
    # If the frontend sends "RXX" on qubit 0 and "RXX" on qubit 1, we might add two RXX gates if we aren't careful.
    # A robust solution requires looking at the whole column.
    # For now, let's add the classes support, but be aware of the multi-qubit issue.

    if symbol in ('RXX', 'rxx'):
        # This is tricky without the second qubit.
        # Assuming the frontend handles this or we need to look ahead.
        # For now, returning a CustomGate if we can't fully resolve it,
        # OR if we assume the column processing handles it (it currently doesn't for generic multi-qubit).
        pass # Placeholder

    if symbol in ('Bell', 'BELL'):
         # Bell gate usually implies a specific state prep on 2 qubits.
         pass

    # === Two-Qubit Gates (Explicit handling needed in column loop if they span multiple) ===
    # For now, let's map them if they are single-target representations (unlikely for RXX)
    # or if we are just instantiating them.

    if symbol in ('SqrtSwap', 'SQRT_SWAP'):
        # Needs 2 qubits.
        pass 
    
    if symbol in ('iSwap', 'ISWAP'):
        # Needs 2 qubits.
        pass

    # === Measurement ===
    if symbol in ('Measure', 'MEASURE'):
        return Measurement(target_qubit, target_qubit, basis='Z')
    if symbol in ('MEASURE_X', 'MeasureX', 'Measure X'):
        return Measurement(target_qubit, target_qubit, basis='X')
    if symbol in ('MEASURE_Y', 'MeasureY', 'Measure Y'):
        return Measurement(target_qubit, target_qubit, basis='Y')

    # === Post-Selection ===
    if symbol in ('POST_SELECT_OFF', '|0⟩'):
        return PostSelection(target_qubit, 0, 'Z')
    if symbol in ('POST_SELECT_ON', '|1⟩'):
        return PostSelection(target_qubit, 1, 'Z')
    if symbol in ('POST_SELECT_X_OFF', '|+⟩'):
        return PostSelection(target_qubit, 0, 'X')
    if symbol in ('POST_SELECT_X_ON', '|−⟩'):
        return PostSelection(target_qubit, 1, 'X')
    if symbol in ('POST_SELECT_Y_OFF', '|i⟩'):
        return PostSelection(target_qubit, 0, 'Y')
    if symbol in ('POST_SELECT_Y_ON', '|-i⟩'):
        return PostSelection(target_qubit, 1, 'Y')

    # === Input Gates ===
    if symbol.startswith('input') or symbol.startswith('Input') or symbol.startswith('INPUT'):
        return InputGate(sanitize_name(symbol), [target_qubit])
    if symbol in ('RANDOM', 'Random'):
        return InputGate('Random', [target_qubit])

    # === Frequency Gates ===
    if symbol in ('QFT', 'qft'):
        return QFTGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('QFT_DAG', 'qft†', 'QFT†'):
        return QFTGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PhaseGradient', 'PHASE_GRADIENT'):
        return PhaseGradientGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_DAG', 'PhaseGradient†'):
        return PhaseGradientGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_INV', 'PhaseGradientInv'):
        return PhaseGradientGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_INV_DAG', 'PhaseGradientInv†'):
        return PhaseGradientGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)

    # === Arithmetic Gates ===
    if symbol in ('+1', 'INC', 'inc'):
        return ArithmeticGate('Increment', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-1', 'DEC', 'dec'):
        return ArithmeticGate('Decrement', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('+A', 'ADD', 'add'):
        return ArithmeticGate('Add', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-A', 'SUB', 'sub'):
        return ArithmeticGate('Subtract', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('×A', 'MUL', 'mul'):
        return ArithmeticGate('Multiply', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('+AB', 'ADD_AB'):
        return ArithmeticGate('AddAB', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-AB', 'SUB_AB'):
        return ArithmeticGate('SubtractAB', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('×A⁻¹', 'MUL_INV'):
        return ArithmeticGate('MultiplyInverse', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Modular Arithmetic Gates ===
    if symbol in ('MOD_INC', 'ModInc'):
        return ModularArithmeticGate('Increment', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_DEC', 'ModDec'):
        return ModularArithmeticGate('Decrement', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_ADD', 'ModAdd'):
        return ModularArithmeticGate('Add', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_SUB', 'ModSub'):
        return ModularArithmeticGate('Subtract', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL', 'ModMul'):
        return ModularArithmeticGate('Multiply', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_INV_MUL', 'ModInvMul'):
        return ModularArithmeticGate('MultiplyInverse', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL_B', 'ModMulB'):
        return ModularArithmeticGate('MultiplyB', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL_B_INV', 'ModMulBInv'):
        return ModularArithmeticGate('MultiplyBInverse', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)

    # === Comparison Gates ===
    if symbol in ('COMPARE', 'compare', '<'):
        return ComparisonGate('LessThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('GREATER_THAN', '>'):
        return ComparisonGate('GreaterThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('LESS_EQUAL', '<=', '≤'):
        return ComparisonGate('LessEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('GREATER_EQUAL', '>=', '≥'):
        return ComparisonGate('GreaterEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('EQUAL', '==', '='):
        return ComparisonGate('Equal', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('NOT_EQUAL', '!=', '≠'):
        return ComparisonGate('NotEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_LT', 'A<'):
        return ComparisonGate('ALessThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_GT', 'A>'):
        return ComparisonGate('AGreaterThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_EQ', 'A='):
        return ComparisonGate('AEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Logic Gates ===
    if symbol in ('COUNT_1S', 'Count'):
        return ArithmeticGate('CountOnes', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('CYCLE_BITS', 'Cycle'):
        return OrderGate('CycleBits', [target_qubit])
    if symbol in ('XOR', 'Xor'):
        return ArithmeticGate('XOR', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Order Gates ===
    if symbol in ('INTERLEAVE', 'Interleave'):
        return OrderGate('Interleave', [target_qubit])
    if symbol in ('DEINTERLEAVE', 'Deinterleave'):
        return OrderGate('Deinterleave', [target_qubit])
    if symbol in ('REVERSE_BITS', 'Reverse'):
        return OrderGate('Reverse', [target_qubit])
    if symbol.startswith('<<') or symbol in ('ROTATE_BITS_LEFT', 'RotateLeft'):
        return OrderGate('RotateLeft', [target_qubit])
    if symbol.startswith('>>') or symbol in ('ROTATE_BITS_RIGHT', 'RotateRight'):
        return OrderGate('RotateRight', [target_qubit])

    # === Scalar (Global Phase) Gates ===
    if symbol in ('ONE', '1'):
        return ScalarGate('1')
    if symbol == 'MINUS_ONE':
        return ScalarGate('-1')
    if symbol in ('PHASE_I', 'i'):
        return ScalarGate('i')
    if symbol in ('PHASE_MINUS_I', '-i'):
        return ScalarGate('-i')
    if symbol in ('PHASE_SQRT_I', '√i'):
        return ScalarGate('√i')
    if symbol in ('PHASE_SQRT_MINUS_I', '√-i'):
        return ScalarGate('√-i')

    # === Spacer ===
    if symbol in ('SPACER', '…', 'spacer'):
        return SpacerGate([target_qubit])

    # === Custom/Unknown - fallback ===
    return CustomGate(sanitize_name(symbol), [target_qubit], control_qubits=control_qubits, control_states=control_states)
