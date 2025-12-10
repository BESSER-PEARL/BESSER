import sys
import os

# Sanitize sys.path to remove entries with newlines which cause OSError in stevedore
sys.path = [p.strip() for p in sys.path if p]

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
from qiskit.circuit.library.standard_gates import GlobalPhaseGate
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

# Function Gate Definitions

def _function_gate_Uf(num_qubits):
    """Custom gate: Uf"""
    qc_func = QuantumCircuit(num_qubits, name='Uf')
    qc_func.x(2)
    qc_func.h(2)
    qc_func.x(2)
    qc_func.h(2)
    qc_func.x(2)
    return qc_func.to_instruction()


def _function_gate_V(num_qubits):
    """Custom gate: V"""
    qc_func = QuantumCircuit(num_qubits, name='V')
    qc_func.h(0)
    qc_func.h(1)
    qc_func.h(2)
    qc_func.x(2)
    qc_func.h(2)
    qc_func.x(2)
    qc_func.h(0)
    qc_func.h(1)
    qc_func.h(2)
    qc_func.x(2)
    qc_func.h(2)
    return qc_func.to_instruction()




# Initialize Registers

q = QuantumRegister(15, 'q')


c = ClassicalRegister(15, 'c')


# Initialize Circuit
qc = QuantumCircuit(q, c)

# Operations

qc.append(HGate(), [q[0]])

qc.append(HGate(), [q[1]])

qc.append(HGate(), [q[2]])

qc.append(_function_gate_Uf(3), [q[0], q[1], q[2]])

qc.append(_function_gate_V(3), [q[0], q[1], q[2]])

qc.append(_function_gate_Uf(3), [q[0], q[1], q[2]])

qc.append(_function_gate_V(3), [q[0], q[1], q[2]])

qc.measure(q[0], c[0])

qc.measure(q[1], c[1])

qc.measure(q[2], c[2])


# Draw
print(qc.draw())

# Execution (Simulation)
print("Simulating circuit...")
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()
print("Counts:", counts)