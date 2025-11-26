import sys
import os

# Sanitize sys.path to remove entries with newlines which cause OSError in stevedore
sys.path = [p.strip() for p in sys.path if p]

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
import numpy as np

# Initialize Registers

q = QuantumRegister(3, 'q')


c = ClassicalRegister(3, 'c')


# Initialize Circuit
qc = QuantumCircuit(q, c)

# Operations

qc.append(HGate(), [q[0]])

qc.append(XGate().control(1, ctrl_state='1'), [q[0], q[1]])

qc.append(RXGate(1.57).control(1, ctrl_state='0'), [q[1], q[2]])

qc.measure(q[0], c[0])

qc.measure(q[1], c[1])

qc.measure(q[2], c[2])


# Execution (Simulation)
print("Simulating circuit...")
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()
print("Counts:", counts)

# Draw
print(qc.draw())