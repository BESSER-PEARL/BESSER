import os
import pytest
from besser.BUML.metamodel.quantum import (
    QuantumCircuit, ClassicalRegister,
    HadamardGate, PauliXGate, RXGate, Measurement, ControlState
)
from besser.generators.qiskit import QiskitGenerator


@pytest.fixture
def quantum_circuit():
    """Create a simple quantum circuit with a few gates and measurements."""
    circuit = QuantumCircuit("TestCircuit", qubits=3)
    c_reg = ClassicalRegister("c", 3)
    circuit.add_creg(c_reg)

    # H gate on qubit 0
    circuit.add_operation(HadamardGate(target_qubit=0))
    # CNOT: X on qubit 1 controlled by qubit 0
    circuit.add_operation(PauliXGate(target_qubit=1, control_qubits=[0]))
    # RX(pi/4) on qubit 2
    circuit.add_operation(RXGate(target_qubit=2, theta=0.785))
    # Measurements
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    circuit.add_operation(Measurement(target_qubit=1, output_bit=1))
    circuit.add_operation(Measurement(target_qubit=2, output_bit=2))

    return circuit


def test_qiskit_generator_instantiation(quantum_circuit):
    """Test that the QiskitGenerator can be instantiated."""
    generator = QiskitGenerator(model=quantum_circuit)
    assert generator is not None
    assert generator.backend_type == "aer_simulator"
    assert generator.shots == 1024


def test_qiskit_generator_invalid_backend(quantum_circuit):
    """Test that an invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        QiskitGenerator(model=quantum_circuit, backend_type="invalid_backend")


def test_qiskit_generator_generate(quantum_circuit, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = QiskitGenerator(
        model=quantum_circuit,
        output_dir=str(output_dir),
    )
    generator.generate()

    output_file = os.path.join(str(output_dir), "qiskit_circuit.py")
    assert os.path.isfile(output_file), "qiskit_circuit.py should be generated"
    assert os.path.getsize(output_file) > 0, "Generated file should not be empty"


def test_qiskit_generator_output_content(quantum_circuit, tmpdir):
    """Test that generated code contains expected Qiskit constructs."""
    output_dir = tmpdir.mkdir("output")
    generator = QiskitGenerator(
        model=quantum_circuit,
        output_dir=str(output_dir),
    )
    generator.generate()

    output_file = os.path.join(str(output_dir), "qiskit_circuit.py")
    with open(output_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify the generated code contains essential Qiskit elements
    assert "QuantumCircuit" in code, "Generated code should reference QuantumCircuit"
    assert "measure" in code.lower() or "Measurement" in code, "Generated code should include measurement operations"


def test_qiskit_generator_with_controlled_gate(tmpdir):
    """Test generation with a controlled gate using anti-control."""
    circuit = QuantumCircuit("ControlledCircuit", qubits=2)
    c_reg = ClassicalRegister("c", 2)
    circuit.add_creg(c_reg)

    circuit.add_operation(PauliXGate(
        target_qubit=1,
        control_qubits=[0],
        control_states=[ControlState.ANTI_CONTROL],
    ))
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    circuit.add_operation(Measurement(target_qubit=1, output_bit=1))

    output_dir = tmpdir.mkdir("output")
    generator = QiskitGenerator(model=circuit, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "qiskit_circuit.py")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "ctrl_state" in code, "Controlled gate with anti-control should set ctrl_state"
