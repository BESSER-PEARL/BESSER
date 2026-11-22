"""QiskitGenerator records which lines of qiskit_circuit.py come from which register, function gate or operation."""

import pytest

from besser.BUML.metamodel.quantum import (
    ClassicalRegister, ControlState, FunctionGate, HadamardGate, InputGate, Measurement, PauliXGate, QuantumCircuit,
    QuantumRegister, RXGate, SwapGate,
)
from besser.generators.qiskit import QiskitGenerator


def _circuit():
    """A circuit whose every element has an id; the elements are returned by name."""
    circuit = QuantumCircuit("Circ", qubits=3)
    anc, creg = QuantumRegister("anc", 2), ClassicalRegister("c", 5)
    circuit.add_qreg(anc)
    circuit.add_creg(creg)
    elements = {
        "input": InputGate("InputA", [0, 1], value=3),
        "h": HadamardGate(target_qubit=0),
        "cx": PauliXGate(target_qubit=1, control_qubits=[0], control_states=[ControlState.ANTI_CONTROL]),
        "rx": RXGate(target_qubit=2, theta=0.785),
        "swap": SwapGate(0, 2),
        "fn": FunctionGate("my_fn", [0, 1], gates=[HadamardGate(target_qubit=0), PauliXGate(target_qubit=1)]),
        "measure": Measurement(target_qubit=0, output_bit=0),
    }
    for operation in elements.values():
        circuit.add_operation(operation)
    elements.update({"circuit": circuit, "anc": anc, "c": creg, "q": circuit.qregs[0]})
    for name, element in elements.items():
        element.id = f"id-{name}"
    return circuit, elements


def _generate(circuit, out, **options):
    generator = QiskitGenerator(circuit, output_dir=str(out), **options)
    generator.generate()
    return generator, (out / "qiskit_circuit.py").read_text(encoding="utf-8")


def _regions(generator):
    return {(r.element_id, r.rule.split("#")[1]): r for r in generator.traced_files["qiskit_circuit.py"].regions}


def _lines(text, region):
    return text.split("\n")[region.start_line - 1:region.end_line]


def test_registers_operations_and_function_gates_are_regions_with_their_code(tmp_path):
    circuit, _ = _circuit()
    generator, text = _generate(circuit, tmp_path)
    assert set(generator.traced_files) == {"qiskit_circuit.py"}
    regions = _regions(generator)

    assert _lines(text, regions[("id-q", "quantum_register")]) == ["q = QuantumRegister(3, 'q')"]
    assert _lines(text, regions[("id-anc", "quantum_register")]) == ["anc = QuantumRegister(2, 'anc')"]
    assert _lines(text, regions[("id-c", "classical_register")]) == ["c = ClassicalRegister(5, 'c')"]
    assert _lines(text, regions[("id-input", "operation")]) == ["qc.x(q[0])", "qc.x(q[1])"]  # one operation, two lines
    assert _lines(text, regions[("id-h", "operation")]) == ["qc.append(HGate(), [q[0]])"]
    assert _lines(text, regions[("id-cx", "operation")]) == ["qc.append(XGate().control(1, ctrl_state='0'), [q[0], q[1]])"]
    assert _lines(text, regions[("id-rx", "operation")]) == ["qc.append(RXGate(0.785), [q[2]])"]
    assert _lines(text, regions[("id-swap", "operation")]) == ["qc.append(SwapGate(), [q[0], q[2]])"]
    assert _lines(text, regions[("id-measure", "operation")]) == ["qc.measure(q[0], c[0])"]
    assert _lines(text, regions[("id-fn", "operation")])[0].startswith("qc.append(_function_gate_my_fn_")
    definition = _lines(text, regions[("id-fn", "function_gate")])
    assert definition[0].startswith("def _function_gate_my_fn_") and definition[-1] == "    return qc_func.to_instruction()"
    assert "    qc_func.h(0)" in definition and "    qc_func.x(1)" in definition
    assert [s.element.element_id for s in generator.traced_files["qiskit_circuit.py"].sources] == ["id-circuit"]


def test_the_default_classical_register_added_for_measurements_has_no_region(tmp_path):
    circuit = QuantumCircuit("Circ", qubits=2)
    circuit.add_operation(Measurement(target_qubit=0, output_bit=0))
    circuit.operations[0].id = "id-measure"
    generator, text = _generate(circuit, tmp_path)
    assert "c = ClassicalRegister(2, 'c')" in text
    assert {rule for _, rule in _regions(generator)} == {"operation"}


@pytest.mark.parametrize("backend", ["aer_simulator", "fake_backend", "ibm_quantum"])
def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path, backend):
    circuit, elements = _circuit()
    traced, traced_text = _generate(circuit, tmp_path / "traced", backend_type=backend)
    assert traced.traced_files["qiskit_circuit.py"].regions
    for element in elements.values():
        element.id = None
    untraced, untraced_text = _generate(circuit, tmp_path / "untraced", backend_type=backend)
    assert untraced.traced_files["qiskit_circuit.py"].regions == ()
    assert untraced.traced_files["qiskit_circuit.py"].sources == ()
    assert traced_text == untraced_text
    assert "\x02" not in traced_text and "\x03" not in traced_text


def test_a_partially_identified_circuit_traces_only_the_elements_that_have_ids(tmp_path):
    circuit, elements = _circuit()
    elements["h"].id = None
    elements["anc"].id = None
    generator, _ = _generate(circuit, tmp_path)
    ids = {region.element_id for region in generator.traced_files["qiskit_circuit.py"].regions}
    assert "id-h" not in ids and "id-anc" not in ids and {"id-rx", "id-q", "id-c"} <= ids
