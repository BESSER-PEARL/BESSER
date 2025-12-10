from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit,transpile
import numpy as np
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def diffuser(n=3,name='V'):
    qc = QuantumCircuit(n, name=name)
    qc.h(range(n))
    qc.x(n-1)
    qc.h(n-1)
    qc.mcx([0,1],2,ctrl_state='00')
    qc.h(n-1)
    qc.x(n-1)
    qc.h(range(n))
    return qc



def phase_oracle(n=3,name = 'Uf'):
    qc = QuantumCircuit(3, name=name)
    qc.x(n-1)
    qc.h(n-1)
    qc.mcx([0,1],2,ctrl_state='10')
    qc.h(n-1)
    qc.x(n-1)
    return qc

if __name__ =="__main__":
    n = 3
    gr = QuantumCircuit(n, n)
    nsol = 1  # number of solutions
    alpha = np.arcsin(np.sqrt(nsol / 2 ** (n)))  # Determine alpha
    r = round(np.pi / (4 * alpha) - 0.5)  # Determine r

    gr.h(range(n))  # step 1: apply Hadamard gates on all working qubits

    # step 2: apply r rounds of the phase oracle and the diffuser
    for j in range(r):
        gr.append(phase_oracle(), range(n))
        gr.append(diffuser(), range(n))

    gr.measure(range(n), range(n))  # step 3: measure all qubits
    print("Measured")
###-------------simulating----------------------
    simulator_aer = AerSimulator()
    gr_trsp = transpile(gr, simulator_aer)
    sampler_aer = Sampler(simulator_aer)
    job = sampler_aer.run([(gr_trsp, None, 1000)])
    result = job.result()
    counts = result[0].data.c.get_counts()
    print(counts)
    from qiskit.visualization import  plot_distribution

    plot_distribution(counts)
    plt.show()


