import sys
import os

# Sanitize sys.path to remove entries with newlines which cause OSError in stevedore
sys.path = [p.strip() for p in sys.path if p]

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
import numpy as np

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

qc.append(DraperQFTAdder(2, kind='fixed').to_instruction(), [q[0], q[1]])

qc.append(Instruction('Add_mod_5', 2, 0, []), [q[0], q[1]])

qc.append(Instruction('Greater', 2, 0, []), [q[0], q[1]])

qc.append(Instruction('MyGate', 1, 0, []), [q[0]])

qc.append(Instruction('Evolution(exp(-i*H*t))', 1, 0, []), [q[0]])

qc.save_statevector()


# Execution (Simulation)
print("Simulating circuit...")
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()
print("Counts:", counts)

# Draw
print(qc.draw())