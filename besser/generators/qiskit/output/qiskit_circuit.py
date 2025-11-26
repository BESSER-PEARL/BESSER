import sys
import os

# Sanitize sys.path to remove entries with newlines which cause OSError in stevedore
sys.path = [p.strip() for p in sys.path if p]

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
import numpy as np

def create_placeholder(name, num_qubits):
    """Creates a placeholder instruction with a barrier."""
    qc_temp = QuantumCircuit(num_qubits, name=name)
    qc_temp.barrier()
    return qc_temp.to_instruction()

class PhaseGradient(QuantumCircuit):
    """Phase Gradient Gate"""
    def __init__(self, num_qubits, inverse=False):
        super().__init__(num_qubits, name="PhaseGradient")
        for i in range(num_qubits):
            # Phase = 2*pi / 2^(num_qubits - i)
            # i=0 (LSB) -> 2*pi / 2^n ... wait, Qiskit order?
            # Qiskit Little Endian: q0 is LSB.
            # PhaseGradient: |x> -> exp(2pi i x / 2^n) |x>
            # x = sum x_k 2^k.
            # Phase = sum x_k * 2*pi * 2^k / 2^n = sum x_k * 2*pi / 2^(n-k)
            # So qubit k gets phase 2*pi / 2^(n-k).
            lam = 2 * np.pi / (2 ** (num_qubits - i))
            if inverse:
                lam = -lam
            self.p(lam, i)


# Initialize Registers

q = QuantumRegister(4, 'q')



# Initialize Circuit
qc = QuantumCircuit(q, )

# Operations

qc.append(HGate(), [q[0]])

qc.append(XGate(), [q[1]])

qc.x(q[1])

qc.append(QFT(3).to_instruction(), [q[0], q[1], q[2]])

qc.append(PhaseGradient(2).to_instruction(), [q[0], q[1]])

qc.append(DraperQFTAdder(2, kind='fixed').to_instruction(), [q[2], q[3], q[0], q[1]])

qc.append(create_placeholder('Add_mod_5', 2), [q[0], q[1]])

qc.append(create_placeholder('Greater', 2), [q[0], q[1]])

qc.append(create_placeholder('MyGate', 1), [q[0]])

qc.append(create_placeholder('Evolution(exp(-i*H*t))', 1), [q[0]])

qc.save_statevector()


# Draw
print(qc.draw())

# Execution (Simulation)
print("Simulating circuit...")
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()
print("Counts:", counts)