Qiskit Generator
================

The Qiskit Generator transforms a :doc:`Quantum Metamodel <../buml_language/model_types/quantum>` into executable Qiskit Python code. This allows you to design quantum circuits using BESSER and then run them on simulators or real quantum hardware via Qiskit.

Supported Backends
------------------

The generator supports the following backends:

* ``aer_simulator`` (Default): Uses the local Qiskit Aer simulator.
* ``fake_backend``: Uses a fake backend for testing.
* ``ibm_quantum``: Targets IBM Quantum hardware (requires credentials).

Gate Mapping
------------

The generator automatically maps BESSER quantum gates to their Qiskit equivalents:

* **Standard Gates**: ``H``, ``X``, ``Y``, ``Z``, ``S``, ``T``, ``RX``, ``RY``, ``RZ``, ``Phase``, ``Swap`` map directly to Qiskit gates.
* **Controlled Gates**: Controlled versions (e.g., ``CX``, ``CY``, ``CZ``, ``CRX``) are generated using Qiskit's control mechanism.
* **Function Gates**: Nested circuits defined in ``FunctionGate`` are generated as separate Python functions that return a Qiskit instruction, which is then appended to the main circuit.
* **Arithmetic & Custom**: Complex gates like ``ArithmeticGate`` or ``CustomGate`` are currently generated as placeholders or specific library calls (e.g., ``DraperQFTAdder``) if supported.

Usage
-----

To use the Qiskit Generator, you need a populated ``QuantumCircuit`` model.

.. code-block:: python

    from besser.generators.qiskit_generator import QiskitGenerator

    # Assuming 'quantum_circuit_model' is your BESSER QuantumCircuit object
    generator = QiskitGenerator(
        model=quantum_circuit_model,
        backend_type="aer_simulator",
        shots=1024
    )
    generator.generate()

Output
------

The generator produces a Python file (default: ``qiskit_circuit.py``) containing:

1.  **Imports**: Necessary Qiskit modules.
2.  **Function Definitions**: Helper functions for any ``FunctionGate`` used in the circuit.
3.  **Circuit Construction**: Code to initialize registers and apply gates.
4.  **Measurement**: Measurement operations if defined.
5.  **Execution**: (Optional) Code to execute the circuit on the specified backend.
