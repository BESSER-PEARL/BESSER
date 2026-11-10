"""A starter B-UML model: a quantum circuit that prepares and measures a Bell state.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate qiskit MODEL_FILE -o out      generate a Qiskit script ('besser generators' lists them all)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/quantum.html
"""

from besser.BUML.metamodel.quantum import (
    ClassicalRegister,
    HadamardGate,
    Measurement,
    PauliXGate,
    QuantumCircuit,
    QuantumRegister,
)

# The circuit. The besser command reads every model bound to a top-level name of this file.
bell_state = QuantumCircuit(name="BellState")

# Registers: the qubits the gates act on and the classical bits that receive the measurements
bell_state.add_qreg(QuantumRegister(name="q", size=2))
bell_state.add_creg(ClassicalRegister(name="c", size=2))

# Gates, in the order they are applied; qubits are numbered from 0 across the quantum registers
bell_state.add_operation(HadamardGate(target_qubit=0))
# A controlled gate is a gate with control qubits: this one is a CNOT (X on qubit 1 when qubit 0 is 1)
bell_state.add_operation(PauliXGate(target_qubit=1, control_qubits=[0]))

# Measurements: each one writes a qubit to a classical bit
bell_state.add_operation(Measurement(target_qubit=0, output_bit=0))
bell_state.add_operation(Measurement(target_qubit=1, output_bit=1))
