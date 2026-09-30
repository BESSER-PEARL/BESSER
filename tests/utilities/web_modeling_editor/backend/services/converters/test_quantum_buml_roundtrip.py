"""Quantum B-UML export -> import through the safe loader gives back the circuit.

``quantum_buml_to_json`` moved from ``exec`` to ``safe_load_buml`` with fewer
builtins, and nothing round-tripped a builder-emitted file through it.
"""
import os
import tempfile

import pytest

from besser.BUML.metamodel.quantum.quantum import (
    ControlState,
    FunctionGate,
    GateDefinition,
    HadamardGate,
    InputGate,
    Measurement,
    PauliXGate,
    QFTGate,
    QuantumCircuit,
    RXGate,
    SwapGate,
)
from besser.utilities.buml_code_builder.quantum_model_builder import quantum_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    safe_load_buml,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.quantum_diagram_converter import (
    quantum_buml_to_json,
    quantum_circuit_to_editor_json,
)


def _export(circuit):
    path = os.path.join(tempfile.mkdtemp(), "qc.py")
    quantum_model_to_code(circuit, path)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _strip_imports(code):
    """The builder's only import is one parenthesised block at the top."""
    return code.split(")\n", 1)[1]


def _quantum_names():
    import besser.BUML.metamodel.quantum.quantum as quantum_module

    return {n: getattr(quantum_module, n) for n in dir(quantum_module) if not n.startswith("_")}


def _without_ids(value):
    """Nested-gate ids embed ``id(gate)``, which differs per load."""
    if isinstance(value, dict):
        return {k: _without_ids(v) for k, v in value.items() if k != "id"}
    if isinstance(value, list):
        return [_without_ids(v) for v in value]
    return value


def _circuit():
    qc = QuantumCircuit(name="Bell's", qubits=3)
    qc.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(PauliXGate(target_qubit=1, control_qubits=[0], control_states=[ControlState.CONTROL]))
    qc.add_operation(PauliXGate(target_qubit=2, control_qubits=[0], control_states=[ControlState.ANTI_CONTROL]))
    qc.add_operation(SwapGate(qubit1=1, qubit2=2))
    qc.add_operation(QFTGate(target_qubits=[0, 1, 2], inverse=True))
    qc.add_operation(Measurement(target_qubit=0, output_bit=0))

    inner = QuantumCircuit(name="inner", qubits=1)
    inner.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(FunctionGate(name="Oracle", target_qubits=[0, 1],
                                  definition=GateDefinition(name="Oracle", circuit=inner)))
    qc.add_operation(FunctionGate(name="Mix", target_qubits=[1], gates=[HadamardGate(target_qubit=0)]))
    return qc


def test_a_builder_emitted_circuit_reimports_to_the_same_editor_json():
    circuit = _circuit()

    out = quantum_buml_to_json(_export(circuit))

    expected = quantum_circuit_to_editor_json(circuit)
    assert _without_ids(out) == _without_ids(expected)
    assert len(out["cols"]) == 8
    assert out["title"] == "Bell's"


def test_a_rotation_gate_reimports():
    """The builder wrote ``angle=``; the gate's parameter is ``theta``."""
    qc = QuantumCircuit(name="rot", qubits=1)
    qc.add_operation(RXGate(target_qubit=0, theta=0.5))

    out = quantum_buml_to_json(_export(qc))

    assert out["cols"] == [["Rx"]]


@pytest.mark.parametrize("value", ["A", "a'b", "3"])
def test_an_input_gate_value_reimports_as_the_same_string(value):
    """``InputGate.value`` is a string; the builder wrote it unquoted."""
    qc = QuantumCircuit(name="inp", qubits=1)
    qc.add_operation(InputGate(input_type="INPUT_A", target_qubits=[0], value=value))
    code = _export(qc)

    out = quantum_buml_to_json(code)

    assert out["cols"] == [["INPUT_A"]]
    namespace = safe_load_buml(_strip_imports(code), _quantum_names())
    gate = next(v for v in namespace.values() if isinstance(v, InputGate))
    assert gate.value == value


@pytest.mark.parametrize("gate_name",["U^2", "f(x)", "a.b"])
def test_a_function_gate_whose_name_is_not_an_identifier_reimports(gate_name):
    """The editor keeps ``^`` / ``(`` in gate names; the builder used the raw name
    inside the emitted variable names, which is a syntax error on re-import."""
    qc = QuantumCircuit(name="fn", qubits=1)
    inner = QuantumCircuit(name="inner", qubits=1)
    inner.add_operation(HadamardGate(target_qubit=0))
    qc.add_operation(FunctionGate(name=gate_name, target_qubits=[0],
                                  definition=GateDefinition(name=gate_name, circuit=inner)))
    qc.add_operation(FunctionGate(name=gate_name, target_qubits=[0], gates=[HadamardGate(target_qubit=0)]))

    out = quantum_buml_to_json(_export(qc))

    assert [meta["label"] for meta in out["gateMetadata"].values()] == [gate_name, gate_name]
