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

Web Modeling Editor Support
---------------------------

Quantum circuits can also be designed visually in the :doc:`BESSER Web Modeling Editor <../web_editor>`
using the ``QuantumCircuitDiagram`` type. The backend converts the editor JSON to a
``QuantumCircuit`` metamodel instance and passes it to the Qiskit generator. You can generate
Qiskit code directly from the editor by selecting the **Qiskit** generator.


Usage
-----

To use the Qiskit Generator, you need a populated ``QuantumCircuit`` model.

.. code-block:: python

    from besser.generators.qiskit import QiskitGenerator

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

Trace Links
-----------

After ``generate()``, ``generator.traced_files["qiskit_circuit.py"]`` names the circuit as the source of the file and
lists the line ranges rendered from each quantum register (``quantum_register``), classical register
(``classical_register``), function gate definition (``function_gate``) and operation (``operation``: a gate, a
measurement, an input or a display operation; an operation that expands to several lines, such as an input value,
covers them all) that has an ``id``. A classical register the generator adds for measurements has no region. Elements
without an ``id`` are rendered untraced, and recording the trace does not change the generated file. A circuit drawn
in the web editor has no element ids (its cells have none), so its processor gives each gate the id the editor's
trace knows its cell by, ``<diagram id>:gate:<column>:<wire>`` (a gate over several wires by its first); a circuit
is traced by operation. Its registers get none (the editor draws wires, not registers), so they render untraced. See
:doc:`traceable_generator`.
