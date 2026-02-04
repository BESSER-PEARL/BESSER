"""
Test for Qiskit generator handling of empty FunctionGates.

This test validates that the generator correctly handles FunctionGates
that have no nested operations (empty gates list or no definition).
Such gates should not generate function definitions and should use
placeholders instead.
"""
import os
import tempfile
from besser.BUML.metamodel.quantum import (
    QuantumCircuit, HadamardGate, Measurement, FunctionGate, PauliXGate
)
from besser.generators.qiskit import QiskitGenerator


def test_empty_function_gate_no_definitions():
    """
    Test that empty FunctionGates do not generate function definitions.
    
    Issue: Previously, FunctionGates with empty gates lists would still
    generate function definitions like:
        def _function_gate_f_12345(num_qubits):
            qc_func = QuantumCircuit(num_qubits, name='f')
            return qc_func.to_instruction()
    
    This was happening when JSON-to-BUML conversion incorrectly created
    FunctionGates without nested operations.
    """
    # Create a circuit with an empty FunctionGate
    qc = QuantumCircuit(name='Test_Circuit', qubits=2)
    
    qc.add_operation(HadamardGate(target_qubit=0))
    
    # Add an empty FunctionGate (no nested operations)
    empty_func_gate = FunctionGate(
        name="EmptyFunction",
        target_qubits=[0],
        gates=[]  # Empty gates list
    )
    qc.add_operation(empty_func_gate)
    
    qc.add_operation(Measurement(target_qubit=0, output_bit=0, basis='Z'))
    
    # Generate Qiskit code
    with tempfile.TemporaryDirectory() as output_dir:
        generator = QiskitGenerator(model=qc, output_dir=output_dir)
        generator.generate()
        
        # Read the generated file
        with open(os.path.join(output_dir, "qiskit_circuit.py"), "r") as f:
            content = f.read()
        
        # Verify NO function definitions were generated for empty FunctionGate
        assert "_function_gate_EmptyFunction" not in content, \
            "Empty FunctionGate should not generate function definition"
        
        # Verify placeholder is used instead
        assert "create_placeholder('EmptyFunction'" in content, \
            "Empty FunctionGate should use placeholder"
        
        # Verify the helper function is included
        assert "def create_placeholder" in content, \
            "create_placeholder helper should be included when needed"


def test_function_gate_with_operations_generates_definition():
    """
    Test that FunctionGates WITH nested operations DO generate function definitions.
    
    This ensures the fix doesn't break valid FunctionGates.
    """
    # Create a circuit with a FunctionGate that has operations
    qc = QuantumCircuit(name='Test_Circuit', qubits=2)
    
    # Create a function gate with nested operations
    func_gate = FunctionGate(
        name="MyCustomGate",
        target_qubits=[0],
        gates=[
            HadamardGate(target_qubit=0),
            PauliXGate(target_qubit=0)
        ]
    )
    qc.add_operation(func_gate)
    qc.add_operation(Measurement(target_qubit=0, output_bit=0, basis='Z'))
    
    # Generate Qiskit code
    with tempfile.TemporaryDirectory() as output_dir:
        generator = QiskitGenerator(model=qc, output_dir=output_dir)
        generator.generate()
        
        # Read the generated file
        with open(os.path.join(output_dir, "qiskit_circuit.py"), "r") as f:
            content = f.read()
        
        # Verify function definition WAS generated
        assert "_function_gate_MyCustomGate" in content, \
            "FunctionGate with operations should generate function definition"
        
        # Verify the function contains nested gates
        assert "qc_func.h(0)" in content, \
            "Function definition should contain nested Hadamard gate"
        assert "qc_func.x(0)" in content, \
            "Function definition should contain nested X gate"
        
        # Verify the function is called in operations
        assert "_function_gate_MyCustomGate_" in content and "(1)" in content, \
            "Function should be called with correct number of qubits"


def test_multiple_empty_function_gates():
    """
    Test handling of multiple empty FunctionGates with different names.
    
    This simulates the scenario described in the issue where multiple
    function gates like "f", "f1", etc. might be created during conversion.
    """
    qc = QuantumCircuit(name='Test_Circuit', qubits=2)
    
    qc.add_operation(HadamardGate(target_qubit=0))
    
    # Add multiple empty FunctionGates (simulating bad conversion)
    qc.add_operation(FunctionGate(name="f", target_qubits=[0], gates=[]))
    qc.add_operation(FunctionGate(name="f1", target_qubits=[1], gates=[]))
    
    qc.add_operation(Measurement(target_qubit=0, output_bit=0, basis='Z'))
    
    # Generate Qiskit code
    with tempfile.TemporaryDirectory() as output_dir:
        generator = QiskitGenerator(model=qc, output_dir=output_dir)
        generator.generate()
        
        # Read the generated file
        with open(os.path.join(output_dir, "qiskit_circuit.py"), "r") as f:
            content = f.read()
        
        # Verify NO function definitions were generated
        assert "_function_gate_f_" not in content, \
            "Empty FunctionGate 'f' should not generate function definition"
        assert "_function_gate_f1_" not in content, \
            "Empty FunctionGate 'f1' should not generate function definition"
        
        # Verify placeholders are used
        assert "create_placeholder('f'" in content, \
            "Empty FunctionGate 'f' should use placeholder"
        assert "create_placeholder('f1'" in content, \
            "Empty FunctionGate 'f1' should use placeholder"


def test_only_primitive_gates_no_function_definitions():
    """
    Test that circuits with only primitive gates have no function definitions section.
    
    This tests the original scenario from the issue where a model with only
    HadamardGate and Measurement should not have any function gate code.
    """
    qc = QuantumCircuit(name='Quantum_Circuit', qubits=2)
    
    # Add only primitive gates (like the user's test model)
    qc.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(Measurement(target_qubit=0, output_bit=0, basis='Z'))
    qc.add_operation(Measurement(target_qubit=1, output_bit=1, basis='Z'))
    
    # Generate Qiskit code
    with tempfile.TemporaryDirectory() as output_dir:
        generator = QiskitGenerator(model=qc, output_dir=output_dir)
        generator.generate()
        
        # Read the generated file
        with open(os.path.join(output_dir, "qiskit_circuit.py"), "r") as f:
            content = f.read()
        
        # Verify the function definitions section is empty
        lines = content.split('\n')
        func_section = []
        in_func_section = False
        for line in lines:
            if '# Function Gate Definitions' in line:
                in_func_section = True
            elif in_func_section and line.startswith('# Initialize'):
                break
            elif in_func_section:
                func_section.append(line)
        
        # Remove empty lines
        func_section = [line for line in func_section if line.strip()]
        
        assert len(func_section) == 0, \
            "Circuit with only primitive gates should have no function definitions"
        
        # Verify no _function_gate_ calls
        assert "_function_gate_" not in content, \
            "No function gate calls should be present"


if __name__ == "__main__":
    print("Running Qiskit empty FunctionGate tests...")
    
    print("Test 1: Empty FunctionGate no definitions...")
    test_empty_function_gate_no_definitions()
    print("✓ Passed")
    
    print("Test 2: FunctionGate with operations generates definition...")
    test_function_gate_with_operations_generates_definition()
    print("✓ Passed")
    
    print("Test 3: Multiple empty FunctionGates...")
    test_multiple_empty_function_gates()
    print("✓ Passed")
    
    print("Test 4: Only primitive gates - no function definitions...")
    test_only_primitive_gates_no_function_definitions()
    print("✓ Passed")
    
    print("\nAll tests passed! ✓")
