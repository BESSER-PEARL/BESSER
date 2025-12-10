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
        
        with open(file_path, "w") as f:
            generated_code = template.render(
                circuit=self.model,
                function_gates_code=function_gates_code,
                operations_code=operations_code,
                backend_type=self.backend_type,
                shots=self.shots
            )
            f.write(generated_code)
        
        print(f"Qiskit code generated at: {file_path}")

    def _generate_function_gates_code(self) -> list[str]:
        """
        Generates function definitions for all FunctionGates in the circuit.
        Returns a list of Python function definitions.
        """
        function_gates = {}
        
        # Collect all unique function gates
        for op in self.model.operations:
            if isinstance(op, FunctionGate):
                gate_name = op.name.replace(' ', '_').replace('-', '_')
                if gate_name not in function_gates:
                    function_gates[gate_name] = op
        
        # Generate function definitions
        functions = []
        for gate_name, gate in function_gates.items():
            num_qubits = len(gate.target_qubits)
            func_lines = [
                f"def _function_gate_{gate_name}(num_qubits):",
                f"    \"\"\"Custom gate: {gate.name}\"\"\"",
                f"    qc_func = QuantumCircuit(num_qubits, name='{gate.name}')"
            ]
            
            # Add nested gates
            if gate.gates:
                for nested_gate in gate.gates:
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
            return f"qc_func.h({gate.target_qubits[0]})"
        elif isinstance(gate, PauliXGate):
            # Check if this is a controlled gate
            if gate.control_qubits:
                ctrl_str = ", ".join([str(c) for c in gate.control_qubits])
                # Generate ctrl_state string if needed
                ctrl_state_str = self._get_control_state_string(gate.control_states)
                if len(gate.control_qubits) == 1:
                    if ctrl_state_str == "1":
                        return f"qc_func.cx({gate.control_qubits[0]}, {gate.target_qubits[0]})"
                    else:
                        return f"qc_func.cx({gate.control_qubits[0]}, {gate.target_qubits[0]})  # Note: ctrl_state='{ctrl_state_str}' requires gate.control() wrapper"
                else:
                    if ctrl_state_str == "1" * len(gate.control_qubits):
                        return f"qc_func.mcx([{ctrl_str}], {gate.target_qubits[0]})"
                    else:
                        return f"qc_func.mcx([{ctrl_str}], {gate.target_qubits[0]}, ctrl_state='{ctrl_state_str}')"
            else:
                return f"qc_func.x({gate.target_qubits[0]})"
        elif isinstance(gate, PauliYGate):
            if gate.control_qubits and len(gate.control_qubits) == 1:
                return f"qc_func.cy({gate.control_qubits[0]}, {gate.target_qubits[0]})"
            return f"qc_func.y({gate.target_qubits[0]})"
        elif isinstance(gate, PauliZGate):
            if gate.control_qubits and len(gate.control_qubits) == 1:
                return f"qc_func.cz({gate.control_qubits[0]}, {gate.target_qubits[0]})"
            return f"qc_func.z({gate.target_qubits[0]})"
        elif isinstance(gate, SGate):
            return f"qc_func.s({gate.target_qubits[0]})"
        elif isinstance(gate, TGate):
            return f"qc_func.t({gate.target_qubits[0]})"
        elif isinstance(gate, RXGate):
            return f"qc_func.rx({gate.theta}, {gate.target_qubits[0]})"
        elif isinstance(gate, RYGate):
            return f"qc_func.ry({gate.theta}, {gate.target_qubits[0]})"
        elif isinstance(gate, RZGate):
            return f"qc_func.rz({gate.theta}, {gate.target_qubits[0]})"
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
            # qc.measure(q[target], c[output])
            # Assuming single register for simplicity in this generator version
            # In a full version, we'd look up the register name based on the qubit index
            # Here we assume 'q' and 'c' are the main registers
            target = op.target_qubits[0]
            output = op.output_bit if op.output_bit is not None else target
            target = op.target_qubits[0]
            output = op.output_bit if op.output_bit is not None else target
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
            
            # Create inline circuit definition
            # We'll generate this as a helper function call
            return f"_function_gate_{gate_name}({num_qubits})"
        
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
            if gate.type_name in ('X^1/4', 'SQRT_SQRT_X'):
                return f"PhaseGate({param})"
            if gate.type_name in ('X^-1/4', 'SQRT_SQRT_X_DAG'):
                return f"PhaseGate({param})"
            if gate.type_name in ('Y^1/4', 'SQRT_SQRT_Y'):
                return f"RYGate({param})"
            if gate.type_name in ('Y^-1/4', 'SQRT_SQRT_Y_DAG'):
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
            # Try to map to known arithmetic gates or use generic instruction
            if gate.operation == 'Add':
                # DraperQFTAdder or similar if available
                # DraperQFTAdder(num_state_qubits, kind='fixed', name='draper')
                # It usually takes 2 registers.
                return f"DraperQFTAdder({len(gate.target_qubits)}, kind='fixed').to_instruction()"
            if gate.operation == 'Add':
                # DraperQFTAdder or similar if available
                # DraperQFTAdder(num_state_qubits, kind='fixed', name='draper')
                # It usually takes 2 registers.
                return f"DraperQFTAdder({len(gate.target_qubits)}, kind='fixed').to_instruction()"
            return f"create_placeholder('{gate.operation}', {len(gate.target_qubits)})"

        elif isinstance(gate, ModularArithmeticGate):
            return f"create_placeholder('{gate.operation}_mod_{gate.modulo}', {len(gate.target_qubits)})"
            
        elif isinstance(gate, ComparisonGate):
            return f"create_placeholder('{gate.operation}', {len(gate.target_qubits)})"
            
        elif isinstance(gate, CustomGate):
            if gate.definition:
                # In a full implementation, we would generate the definition as a Gate/Instruction
                # For now, we create an opaque instruction with the name
                return f"create_placeholder('{gate.name}', {len(gate.target_qubits)})"
            return f"create_placeholder('{gate.name}', {len(gate.target_qubits)})"

        elif isinstance(gate, TimeDependentGate):
            # Map to a generic Unitary or Instruction with the expression as label
            return f"create_placeholder('{gate.type_name}({gate.parameter_expr})', {len(gate.target_qubits)})"
            
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
            # Qiskit doesn't have native order gates, so create a placeholder
            return f"create_placeholder('{gate.order_type}', {len(gate.target_qubits) if gate.target_qubits else 1})"

        elif isinstance(gate, ScalarGate):
            # ScalarGate applies a global phase or scalar multiplication
            # In Qiskit, global phase can be represented with GlobalPhaseGate
            # The scalar_type might be a symbolic value like 'e^(iπ/4)'
            return f"# GlobalPhase: {gate.scalar_type}"

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
