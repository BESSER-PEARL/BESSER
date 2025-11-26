import os
from jinja2 import Environment, FileSystemLoader
from besser.generators import GeneratorInterface
from besser.BUML.metamodel.quantum import (
    QuantumCircuit, PrimitiveGate, ParametricGate, Measurement, 
    ControlState, SwapGate, PauliXGate, PauliYGate, PauliZGate, 
    HadamardGate, RXGate, RYGate, RZGate, PhaseGate,
    QFTGate, SpacerGate, ArithmeticGate, ModularArithmeticGate,
    ComparisonGate, DisplayOperation, InputGate
)

class QiskitGenerator(GeneratorInterface):
    """
    Generates Qiskit code from a BESSER QuantumCircuit model.
    """
    def __init__(self, model: QuantumCircuit, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        """
        Generates the Qiskit Python script.
        """
        file_path = self.build_generation_path(file_name="qiskit_circuit.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('qiskit_circuit.py.j2')
        
        operations_code = self._generate_operations_code()
        
        with open(file_path, "w") as f:
            generated_code = template.render(
                circuit=self.model,
                operations_code=operations_code
            )
            f.write(generated_code)
        
        print(f"Qiskit code generated at: {file_path}")

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
            return f"qc.measure(q[{target}], c[{output}])"

        # Handle Gates
        gate_code = self._get_base_gate_code(op)
        
        # Apply Controls
        if op.control_qubits:
            ctrl_state = self._get_control_state_string(op.control_states)
            # .control(num_ctrl_qubits, label=None, ctrl_state=None)
            gate_code = f"{gate_code}.control({len(op.control_qubits)}, ctrl_state='{ctrl_state}')"
        
        # Append to circuit
        # qc.append(gate, [q[c1], q[c2], ..., q[t1], ...])
        
        # Construct qubit list: controls + targets
        qubit_indices = op.control_qubits + op.target_qubits
        qubit_args = ", ".join([f"q[{i}]" for i in qubit_indices])
        
        return f"qc.append({gate_code}, [{qubit_args}])"

    def _get_base_gate_code(self, gate) -> str:
        """
        Returns the Qiskit object creation code for the base gate (without controls).
        """
        if isinstance(gate, PrimitiveGate):
            if gate.type_name == 'H': return "HGate()"
            if gate.type_name == 'X': return "XGate()"
            if gate.type_name == 'Y': return "YGate()"
            if gate.type_name == 'Z': return "ZGate()"
            if gate.type_name == 'S': return "SGate()"
            if gate.type_name == 'T': return "TGate()"
            if gate.type_name == 'SWAP': return "SwapGate()"
            # Add more primitives as needed
            return f"# Unsupported Primitive: {gate.type_name}"
            
        elif isinstance(gate, ParametricGate):
            param = gate.parameter
            if gate.type_name == 'RX': return f"RXGate({param})"
            if gate.type_name == 'RY': return f"RYGate({param})"
            if gate.type_name == 'RZ': return f"RZGate({param})"
            if gate.type_name == 'PHASE': return f"PhaseGate({param})"
            return f"# Unsupported Parametric: {gate.type_name}"
            
        elif isinstance(gate, QFTGate):
            # QFT(num_qubits, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name='qft')
            # We need to import QFT from qiskit.circuit.library
            # For now, let's assume it's available or use a library call
            inverse_str = ", inverse=True" if gate.inverse else ""
            return f"QFT({len(gate.target_qubits)}{inverse_str}).to_instruction()"

        elif isinstance(gate, SpacerGate):
            # Identity or just pass
            return "IGate()"

        elif isinstance(gate, ArithmeticGate):
            return f"# ArithmeticGate({gate.operation}) - Placeholder"

        elif isinstance(gate, ModularArithmeticGate):
            return f"# ModularArithmeticGate({gate.operation}, mod={gate.modulo}) - Placeholder"
            
        elif isinstance(gate, ComparisonGate):
            return f"# ComparisonGate({gate.operation}) - Placeholder"
            
        elif isinstance(gate, DisplayOperation):
            return f"# Display: {gate.display_type}"
            
        elif isinstance(gate, InputGate):
            return f"# Input: {gate.input_type} = {gate.value}"
            
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
