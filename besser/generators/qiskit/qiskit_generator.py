import os
from jinja2 import Environment, FileSystemLoader
from besser.generators import GeneratorInterface
from besser.BUML.metamodel.quantum import (
    QuantumCircuit, PrimitiveGate, ParametricGate, Measurement, 
    ControlState, SwapGate, PauliXGate, PauliYGate, PauliZGate, 
    HadamardGate, RXGate, RYGate, RZGate, PhaseGate, SGate, TGate,
    QFTGate, SpacerGate, ArithmeticGate, ModularArithmeticGate,
    ComparisonGate, DisplayOperation, InputGate, CustomGate, FunctionGate,
    TimeDependentGate, PhaseGradientGate, OrderGate, ScalarGate, PostSelection
)

VALID_BACKENDS = {"aer_simulator", "fake_backend", "ibm_quantum"}

class QiskitGenerator(GeneratorInterface):
    """
    Generates Qiskit code from a BESSER QuantumCircuit model.
    """
    def __init__(self, model: QuantumCircuit, output_dir: str = None, 
                 backend_type: str = "aer_simulator", shots: int = 1024):
        super().__init__(model, output_dir)
        if backend_type not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend: '{backend_type}'. Valid backends are: {VALID_BACKENDS}"
            )
        self.backend_type = backend_type
        self.shots = shots

    def generate(self):
        """
        Generates the Qiskit Python script.
        """
        file_path = self.build_generation_path(file_name="qiskit_circuit.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('qiskit_circuit.py.j2')
        
        # Check if circuit needs classical registers for measurements
        has_measurements = any(isinstance(op, Measurement) for op in self.model.operations)
        if has_measurements and not self.model.cregs:
            # Add a classical register with same size as qubits
            from besser.BUML.metamodel.quantum.quantum import ClassicalRegister
            self.model.add_creg(ClassicalRegister("c", self.model.num_qubits))
        
        # Collect function gates to generate their definitions
        function_gates_code = self._generate_function_gates_code()
        operations_code = self._generate_operations_code()
        
        # Determine which helper classes are needed
        helper_classes = self._get_required_helper_classes()
        
        with open(file_path, "w", encoding="utf-8") as f:
            generated_code = template.render(
                circuit=self.model,
                function_gates_code=function_gates_code,
                operations_code=operations_code,
                backend_type=self.backend_type,
                shots=self.shots,
                helper_classes=helper_classes
            )
            f.write(generated_code)
        
        print(f"Qiskit code generated at: {file_path}")

    def _get_required_helper_classes(self) -> set:
        """
        Analyzes the circuit and returns a set of helper class names that are needed.
        """
        needed = set()
        
        for op in self.model.operations:
            if isinstance(op, PhaseGradientGate):
                needed.add('PhaseGradient')
            elif isinstance(op, OrderGate):
                order_type = op.order_type
                if order_type == 'Reverse':
                    needed.add('ReverseBits')
                elif order_type == 'Interleave':
                    needed.add('Interleave')
                elif order_type == 'Deinterleave':
                    needed.add('Deinterleave')
                elif order_type == 'RotateLeft':
                    needed.add('RotateBitsLeft')
                elif order_type == 'RotateRight':
                    needed.add('RotateBitsRight')
            elif isinstance(op, ArithmeticGate) and not isinstance(op, ModularArithmeticGate):
                op_type = op.operation_type
                if op_type == 'Increment':
                    needed.add('Increment')
                elif op_type == 'Decrement':
                    needed.add('Decrement')
        
        # Check if any gate uses a placeholder
        for op in self.model.operations:
            if self._uses_placeholder(op):
                needed.add('create_placeholder')
        
        return needed
    
    def _uses_placeholder(self, op) -> bool:
        """Check if an operation will use the placeholder function."""
        if isinstance(op, ModularArithmeticGate):
            return True
        if isinstance(op, ComparisonGate):
            return True
        if isinstance(op, CustomGate):
            return True
        if isinstance(op, ArithmeticGate):
            op_type = op.operation_type
            if op_type not in ('Add', 'Subtract', 'Multiply', 'Increment', 'Decrement'):
                return True
        if isinstance(op, OrderGate):
            order_type = op.order_type
            if order_type not in ('Reverse', 'Interleave', 'Deinterleave', 'RotateLeft', 'RotateRight'):
                return True
        if isinstance(op, TimeDependentGate):
            # Check if it maps to a known gate
            type_name = op.type_name
            if not ('Exp' in type_name and any(x in type_name for x in ['X', 'Y', 'Z'])):
                if not ('^' in type_name and any(x in type_name for x in ['X', 'Y', 'Z'])):
                    return True
        return False

    def _generate_function_gates_code(self) -> list[str]:
        """
        Generates function definitions for all FunctionGates in the circuit.
        Returns a list of Python function definitions.
        """
        function_gates = {}
        
        # Collect all unique function gates by their unique ID (not just name)
        # This handles cases where gates with same name have different implementations
        for op in self.model.operations:
            if isinstance(op, FunctionGate):
                # Use id() to ensure unique gates are captured
                gate_key = f"{op.name}_{id(op)}"
                if gate_key not in function_gates:
                    function_gates[gate_key] = op
        
        # Generate function definitions
        functions = []
        for gate_key, gate in function_gates.items():
            gate_name = gate.name.replace(' ', '_').replace('-', '_')
            num_qubits = len(gate.target_qubits)
            func_lines = [
                f"def _function_gate_{gate_name}_{id(gate)}(num_qubits):",
                f"    \"\"\"Custom gate: {gate.name}\"\"\"",
                f"    qc_func = QuantumCircuit(num_qubits, name='{gate.name}')"
            ]
            
            # Get nested operations from definition.circuit or gates list
            nested_operations = []
            if gate.definition and gate.definition.circuit:
                nested_operations = gate.definition.circuit.operations
            elif gate.gates:
                nested_operations = gate.gates
            
            # Add nested gates
            for nested_gate in nested_operations:
                gate_code = self._get_nested_gate_code(nested_gate)
                if gate_code:
                    func_lines.append(f"    {gate_code}")
            
            func_lines.append("    return qc_func.to_instruction()")
            func_lines.append("")  # Empty line after function
            
            functions.append("\n".join(func_lines))
        
        return functions
    
    def _get_nested_gate_code(self, gate) -> str:
        """Generate Qiskit code for gates inside a FunctionGate."""
        if isinstance(gate, HadamardGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.h({target})"
        elif isinstance(gate, PauliXGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            # Check if this is a controlled gate
            if gate.control_qubits:
                ctrl_str = ", ".join([str(c) for c in gate.control_qubits])
                # Generate ctrl_state string if needed
                ctrl_state_str = self._get_control_state_string(gate.control_states)
                if len(gate.control_qubits) == 1:
                    if ctrl_state_str == "1":
                        return f"qc_func.cx({gate.control_qubits[0]}, {target})"
                    else:
                        # Use XGate().control() with ctrl_state for anti-control
                        return f"qc_func.append(XGate().control({len(gate.control_qubits)}, ctrl_state='{ctrl_state_str}'), [{ctrl_str}, {target}])"
                else:
                    # Multi-controlled X with ctrl_state
                    return f"qc_func.append(XGate().control({len(gate.control_qubits)}, ctrl_state='{ctrl_state_str}'), [{ctrl_str}, {target}])"
            else:
                return f"qc_func.x({target})"
        elif isinstance(gate, PauliYGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            if gate.control_qubits and len(gate.control_qubits) == 1:
                return f"qc_func.cy({gate.control_qubits[0]}, {target})"
            return f"qc_func.y({target})"
        elif isinstance(gate, PauliZGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            if gate.control_qubits and len(gate.control_qubits) == 1:
                return f"qc_func.cz({gate.control_qubits[0]}, {target})"
            return f"qc_func.z({target})"
        elif isinstance(gate, SGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.s({target})"
        elif isinstance(gate, TGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.t({target})"
        elif isinstance(gate, SwapGate):
            targets = gate.target_qubits
            return f"qc_func.swap({targets[0]}, {targets[1]})"
        elif isinstance(gate, RXGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.rx({gate.theta}, {target})"
        elif isinstance(gate, RYGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.ry({gate.theta}, {target})"
        elif isinstance(gate, RZGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.rz({gate.theta}, {target})"
        elif isinstance(gate, PhaseGate):
            target = gate.target_qubits[0] if gate.target_qubits else gate.target_qubit
            return f"qc_func.p({gate.parameter}, {target})"
        else:
            return f"# Unsupported nested gate: {type(gate).__name__}"

    def _generate_operations_code(self) -> list[str]:
        """
        Generates the list of Qiskit code lines for the circuit operations.
        """
        lines = []
        for op in self.model.operations:
            lines.append(self._map_operation(op))
        return lines

    def _map_operation(self, op) -> str:
        """
        Maps a single QuantumOperation to a Qiskit code line.
        """
        # Handle Measurement
        if isinstance(op, Measurement):
            target = op.target_qubits[0]
            output = op.output_bit if op.output_bit is not None else target
            basis = getattr(op, 'basis', 'Z')
            
            if basis == 'X':
                # X-basis measurement: apply H before measuring
                return f"qc.h(q[{target}]); qc.measure(q[{target}], c[{output}])"
            elif basis == 'Y':
                # Y-basis measurement: apply S†H before measuring
                return f"qc.sdg(q[{target}]); qc.h(q[{target}]); qc.measure(q[{target}], c[{output}])"
            else:
                # Z-basis (standard) measurement
                return f"qc.measure(q[{target}], c[{output}])"

        # Handle InputGate (State Initialization)
        if isinstance(op, InputGate):
            cmds = []
            if op.value is not None:
                val = op.value
                for i, target in enumerate(op.target_qubits):
                    if (val >> i) & 1:
                        cmds.append(f"qc.x(q[{target}])")
            if not cmds:
                return f"# Input {op.input_type} (No value set)"
            return "\n".join(cmds)
        
        # Handle DisplayOperation (Direct call on qc)
        if isinstance(op, DisplayOperation):
            gate_code = self._get_base_gate_code(op)
            if "save_" in gate_code:
                # It's a save instruction, call directly on qc
                # e.g. qc.save_statevector()
                return gate_code
            else:
                return gate_code

        # Handle Gates
        gate_code = self._get_base_gate_code(op)
        
        # Apply Controls
        if op.control_qubits:
            ctrl_state = self._get_control_state_string(op.control_states)
            # .control(num_ctrl_qubits, label=None, ctrl_state=None)
            gate_code = f"{gate_code}.control({len(op.control_qubits)}, ctrl_state='{ctrl_state}')"
        
        # Append to circuit
        # qc.append(gate, [q[c1], q[c2], ..., q[t1], ...])
        
        # Construct qubit list: controls + inputs + targets
        qubit_indices = op.control_qubits[:]
        
        # Some gates have input_qubits (Arithmetic, Comparison, etc.)
        if hasattr(op, 'input_qubits') and op.input_qubits:
            qubit_indices.extend(op.input_qubits)
            
        qubit_indices.extend(op.target_qubits)
        qubit_args = ", ".join([f"q[{i}]" for i in qubit_indices])
        
        return f"qc.append({gate_code}, [{qubit_args}])"

    def _get_base_gate_code(self, gate) -> str:
        """
        Returns the Qiskit object creation code for the base gate (without controls).
        """
        # Handle specific gate classes first
        if isinstance(gate, HadamardGate):
            return "HGate()"
        elif isinstance(gate, PauliXGate):
            return "XGate()"
        elif isinstance(gate, PauliYGate):
            return "YGate()"
        elif isinstance(gate, PauliZGate):
            return "ZGate()"
        elif isinstance(gate, SGate):
            return "SGate()"
        elif isinstance(gate, TGate):
            return "TGate()"
        elif isinstance(gate, SwapGate):
            return "SwapGate()"
        elif isinstance(gate, RXGate):
            return f"RXGate({gate.theta})"
        elif isinstance(gate, RYGate):
            return f"RYGate({gate.theta})"
        elif isinstance(gate, RZGate):
            return f"RZGate({gate.theta})"
        elif isinstance(gate, PhaseGate):
            return f"PhaseGate({gate.parameter})"
        
        # Handle FunctionGate by creating a custom gate from nested circuit
        elif isinstance(gate, FunctionGate):
            # For function gates, we need to generate a QuantumCircuit definition
            # and convert it to an instruction
            num_qubits = len(gate.target_qubits)
            gate_name = gate.name.replace(' ', '_').replace('-', '_')
            
            # Use unique ID to match the generated function name
            return f"_function_gate_{gate_name}_{id(gate)}({num_qubits})"
        
        elif isinstance(gate, PrimitiveGate):
            if gate.type_name == 'H':
                return "HGate()"
            if gate.type_name == 'X':
                return "XGate()"
            if gate.type_name == 'Y':
                return "YGate()"
            if gate.type_name == 'Z':
                return "ZGate()"
            if gate.type_name == 'S':
                return "SGate()"
            if gate.type_name == 'T':
                return "TGate()"
            if gate.type_name == 'SWAP':
                return "SwapGate()"
            # Add more primitives as needed
            return f"# Unsupported Primitive: {gate.type_name}"

        elif isinstance(gate, ParametricGate):
            param = gate.parameter
            if gate.type_name == 'RX':
                return f"RXGate({param})"
            if gate.type_name == 'RY':
                return f"RYGate({param})"
            if gate.type_name == 'RZ':
                return f"RZGate({param})"
            if gate.type_name == 'PHASE':
                return f"PhaseGate({param})"
            # Quarter turns
            if gate.type_name == 'S_DAG':
                return "SdgGate()"
            if gate.type_name == 'V':
                return "SXGate()"
            if gate.type_name == 'V_DAG':
                return "SXdgGate()"
            if gate.type_name in ('SQRT_Y', 'Y^1/2'):
                return f"RYGate({param})"
            if gate.type_name in ('SQRT_Y_DAG', 'Y^-1/2'):
                return f"RYGate({param})"
            # Eighth turns
            if gate.type_name == 'T_DAG':
                return "TdgGate()"
            if gate.type_name in ('X^1_4', 'X^1/4', 'SQRT_SQRT_X'):
                return f"PhaseGate({param})"
            if gate.type_name in ('X^_1_4', 'X^-1/4', 'SQRT_SQRT_X_DAG'):
                return f"PhaseGate({param})"
            if gate.type_name in ('Y^1_4', 'Y^1/4', 'SQRT_SQRT_Y'):
                return f"RYGate({param})"
            if gate.type_name in ('Y^_1_4', 'Y^-1/4', 'SQRT_SQRT_Y_DAG'):
                return f"RYGate({param})"
            return f"# Unsupported Parametric: {gate.type_name}"
            
        elif isinstance(gate, QFTGate):
            # QFT(num_qubits, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')
            inverse_str = ", inverse=True" if gate.inverse else ""
            return f"QFT({len(gate.target_qubits)}{inverse_str}).to_instruction()"

        elif isinstance(gate, PhaseGradientGate):
            # Use the class defined in the template
            inverse_str = ", inverse=True" if gate.inverse else ""
            return f"PhaseGradient({len(gate.target_qubits)}{inverse_str}).to_instruction()"

        elif isinstance(gate, SpacerGate):
            # Identity or just pass
            return "IGate()"

        elif isinstance(gate, ArithmeticGate):
            # Map to known arithmetic gate implementations
            n_qubits = len(gate.target_qubits)
            op_type = gate.operation_type
            if op_type == 'Add':
                return f"DraperQFTAdder({n_qubits}, kind='fixed').to_instruction()"
            elif op_type == 'Increment':
                return f"Increment({n_qubits}).to_instruction()"
            elif op_type == 'Decrement':
                return f"Decrement({n_qubits}).to_instruction()"
            elif op_type == 'Subtract':
                # Subtraction can be done with inverse adder
                return f"DraperQFTAdder({n_qubits}, kind='fixed').inverse().to_instruction()"
            elif op_type == 'Multiply':
                # Use Qiskit's HRSCumulativeMultiplier if available
                return f"HRSCumulativeMultiplier({n_qubits}).to_instruction()"
            return f"create_placeholder('{gate.operation_type}', {n_qubits})"

        elif isinstance(gate, ModularArithmeticGate):
            # Modular arithmetic - some have Qiskit implementations
            n_qubits = len(gate.target_qubits)
            op_type = gate.operation_type
            mod = gate.modulo
            if op_type == 'Add':
                return f"CDKMRippleCarryAdder({n_qubits}, kind='fixed').to_instruction()"
            return f"create_placeholder('{op_type}_mod_{mod}', {n_qubits})"
            
        elif isinstance(gate, ComparisonGate):
            # Comparison gates - use IntegerComparator from Qiskit
            n_qubits = len(gate.target_qubits)
            op_type = gate.operation
            # IntegerComparator(num_state_qubits, value, geq=False)
            # We don't have a specific value, so use placeholder with proper structure
            return f"create_placeholder('{op_type}', {n_qubits})"
            
        elif isinstance(gate, CustomGate):
            if gate.definition:
                # In a full implementation, we would generate the definition as a Gate/Instruction
                # For now, we create an opaque instruction with the name
                return f"create_placeholder('{gate.name}', {len(gate.target_qubits)})"
            return f"create_placeholder('{gate.name}', {len(gate.target_qubits)})"

        elif isinstance(gate, TimeDependentGate):
            # Map time-dependent gates to parametric rotations where possible
            type_name = gate.type_name
            expr = gate.parameter_expr
            # Common patterns: Exp(iXt), Exp(iYt), Exp(iZt), X^t, Y^t, Z^t
            if 'Exp' in type_name:
                # Exponential gates: Exp(iAt) -> R_A(2t)
                if 'X' in type_name:
                    return f"RXGate(2*{expr})"
                elif 'Y' in type_name:
                    return f"RYGate(2*{expr})"
                elif 'Z' in type_name:
                    return f"RZGate(2*{expr})"
            elif '^' in type_name:
                # Power gates: A^t -> R_A(π*t)
                if 'X' in type_name:
                    return f"RXGate(np.pi*{expr})"
                elif 'Y' in type_name:
                    return f"RYGate(np.pi*{expr})"
                elif 'Z' in type_name:
                    return f"RZGate(np.pi*{expr})"
            return f"create_placeholder('{type_name}({expr})', {len(gate.target_qubits)})"
            
        elif isinstance(gate, DisplayOperation):
            # For AerSimulator, we can use save functions
            if "Bloch" in gate.display_type or "State" in gate.display_type:
                return "qc.save_statevector()"
            elif "Density" in gate.display_type:
                return "qc.save_density_matrix()"
            elif "Prob" in gate.display_type:
                return "qc.save_probabilities()"
            return f"# Display: {gate.display_type}"
            
        elif isinstance(gate, InputGate):
            # InputGate logic is handled in _map_operation because it might involve multiple gates (X)
            # Here we return a marker or pass
            return "IGate()"

        elif isinstance(gate, OrderGate):
            # OrderGate represents order manipulation (Interleave, Reverse, bit shifts, etc.)
            n_qubits = len(gate.target_qubits) if gate.target_qubits else 1
            order_type = gate.order_type
            if order_type == 'Reverse':
                return f"ReverseBits({n_qubits}).to_instruction()"
            elif order_type == 'Interleave':
                return f"Interleave({n_qubits}).to_instruction()"
            elif order_type == 'Deinterleave':
                return f"Deinterleave({n_qubits}).to_instruction()"
            elif order_type == 'RotateLeft':
                return f"RotateBitsLeft({n_qubits}).to_instruction()"
            elif order_type == 'RotateRight':
                return f"RotateBitsRight({n_qubits}).to_instruction()"
            return f"create_placeholder('{gate.order_type}', {n_qubits})"

        elif isinstance(gate, ScalarGate):
            # ScalarGate applies a global phase or scalar multiplication
            # In Qiskit, global phase can be represented with GlobalPhaseGate
            scalar = gate.scalar_type
            if scalar == 'i':
                return "GlobalPhaseGate(np.pi/2)"
            elif scalar == '-i':
                return "GlobalPhaseGate(-np.pi/2)"
            elif scalar == '√i':
                return "GlobalPhaseGate(np.pi/4)"
            elif scalar == '-√i':
                return "GlobalPhaseGate(-np.pi/4)"
            elif scalar == '-1':
                return "GlobalPhaseGate(np.pi)"
            elif scalar == '1':
                return "GlobalPhaseGate(0)"
            return f"GlobalPhaseGate(0)  # {gate.scalar_type}"

        elif isinstance(gate, PostSelection):
            # PostSelection is a measurement-based selection, not a standard gate
            # Create a comment with the expected state information
            return f"# PostSelection: measure qubit in basis {gate.basis}, expect value {gate.value}"
            
        # Fallback for other types
        return f"# Unsupported Gate Type: {type(gate).__name__}"

    def _get_control_state_string(self, control_states: list[ControlState]) -> str:
        """
        Constructs the control state string (e.g., '101') for Qiskit.
        Note: Qiskit expects the control state string to be in the order of the control qubits.
        """
        # Map CONTROL -> '1', ANTI_CONTROL -> '0'
        # Qiskit's ctrl_state string corresponds to the control qubits.
        # If we pass [c0, c1], string '10' means c0 is 1, c1 is 0? 
        # Actually Qiskit docs say: "The string is ordered such that the right-most character corresponds to the first control qubit."
        # So we might need to reverse it depending on how we order the list.
        # Let's assume standard order for now and refine if needed.
        
        chars = []
        for state in control_states:
            if state == ControlState.CONTROL:
                chars.append('1')
            else:
                chars.append('0')
        
        # If Qiskit expects right-most = first control, and our list is [c_first, ..., c_last]
        # Then we should reverse the string.
        # Example: controls=[q0, q1], states=[1, 0]. q0 is 1, q1 is 0.
        # Qiskit expects '01' where rightmost '1' is for q0.
        return "".join(reversed(chars))
