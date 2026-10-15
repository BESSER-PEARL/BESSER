"""Scaffold map, generic family: everything the framework extractors leave unclaimed.

The specialist scaffolds (BESSER agent, Qiskit, PyTorch/TensorFlow, BPMN, the
per-class generators) are generated for real into a temporary workspace, the
way the agent's ``_gen_*`` handlers do, and asserted by anchor id, never by
absolute line number. Hand-written files cover stubs, a TypeScript component,
files that do not parse and the grouping rules around roots other extractors
claimed. Every map is checked for sound anchors and line-free prompt text.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import operator
import re
import shutil
import time
from pathlib import Path

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    EndEvent,
    Gateway,
    GatewayType,
    Process,
    SequenceFlow,
    StartEvent,
    Task,
)
from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen, Text
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.nn import NN, LinearLayer
from besser.BUML.metamodel.quantum import (
    ClassicalRegister,
    HadamardGate,
    Measurement,
    PauliXGate,
    QuantumCircuit,
    RXGate,
)
from besser.BUML.metamodel.quantum.quantum import CustomGate, FunctionGate, InputGate
from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    AgentReply,
    LLMReply,
    WebSocketPlatform,
    WebSocketReplyFile,
)
from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, CustomCodeAction
from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    BooleanType,
    Class,
    DateType,
    DomainModel,
    Generalization,
    IntegerType,
    Method,
    Multiplicity,
    Property,
    StringType,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.planning.scaffold_map import (
    build_workspace_map,
    generic,
    locate,
    scaffold_overview,
)
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_CLASS,
    KIND_FILE,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_STUB,
    ScaffoldMap,
    WorkspaceMap,
)
from besser.spec_driven_agent.planning.scaffold_map.render import render_locate, render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(workspace: Path, tool: str, args: dict | None = None, **models) -> None:
    with contextlib.redirect_stdout(io.StringIO()):      # some generators print their output path
        result = json.loads(ToolExecutor(workspace=str(workspace), **models).execute(tool, args or {}))
    assert result.get("status") == "ok", result


def _extract(workspace: Path, class_names=(), claimed_roots=()) -> list[ScaffoldMap]:
    maps = generic.extract(SourceIndex(str(workspace)), list(class_names), claimed_roots=list(claimed_roots))
    _assert_anchors_sound(workspace, maps)
    _assert_line_free(maps)
    return maps


def _by_id(maps: list[ScaffoldMap]) -> dict:
    return {a.id: a for m in maps for a in m.all_anchors()}


def _map_for(maps: list[ScaffoldMap], root: str) -> ScaffoldMap:
    return next(m for m in maps if m.root == root)


def _write(workspace: Path, rel: str, text: str) -> None:
    path = workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lines(workspace: Path, rel: str) -> list[str]:
    return (workspace / rel).read_text(encoding="utf-8").splitlines()


def _assert_anchors_sound(workspace: Path, maps: list[ScaffoldMap]) -> None:
    """Unique ids, ranges inside the file, every quote verbatim, unique and inside its range."""
    anchors = [a for m in maps for a in m.all_anchors()]
    ids = [a.id for a in anchors]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for anchor in anchors:
        assert anchor.id.startswith("generic:")
        lines = _lines(workspace, anchor.file)
        assert 1 <= anchor.start_line <= anchor.end_line <= max(1, len(lines)), anchor
        if anchor.quote:
            assert lines.count(anchor.quote) == 1, anchor
            # modify_file matches old_text as a substring: unique in the whole text.
            assert "\n".join(lines).count(anchor.quote) == 1, anchor
            assert anchor.start_line <= lines.index(anchor.quote) + 1 <= anchor.end_line, anchor


_LINE_ISH = re.compile(r"\blines? \d|@\d|:\d+\b(?!\])")


def _assert_line_free(maps: list[ScaffoldMap]) -> None:
    """What goes into the system prompt never names a line number (they go stale on the first edit)."""
    for m in maps:
        texts = [m.title, *m.summary, *m.warnings, *(s for r in m.recipes for s in r.steps)]
        texts += [s for e in m.entities.values() for s in e.summary]
        texts += [f"{a.id} {a.label}" for a in m.anchors if a.kind != KIND_FILE]   # the overview's anchor list
        for text in texts:
            assert not _LINE_ISH.search(text), text


# ---------------------------------------------------------------------------
# BESSER agents
# ---------------------------------------------------------------------------

def _support_agent() -> Agent:
    agent = Agent("support_bot")
    agent.platforms.append(WebSocketPlatform())
    agent.new_llm(name="gpt-4o", provider="openai", parameters={})
    city = agent.new_entity("city_entity", entries={"Paris": ["paris"], "Rome": []})
    greet = agent.new_intent("greet_intent", ["hello", "hi"])
    weather = agent.new_intent("weather_intent", ["weather in CITY", "forecast CITY"])
    weather.parameter("city", "CITY", city)
    initial = agent.new_state("initial", initial=True)
    greeting = agent.new_state("greeting")
    weather_state = agent.new_state("weather_state")
    custom_state = agent.new_state("custom_state")
    initial.set_body(Body("initial_body", actions=[AgentReply(message="Welcome!")]))
    initial.set_fallback_body(Body("initial_fallback_body", actions=[AgentReply(message="Sorry?")]))
    greeting.set_body(Body("greeting_body", actions=[LLMReply()]))
    weather_state.set_body(Body("weather_state_body", actions=[AgentReply(message="Sunny")]))
    initial.when_intent_matched(greet).go_to(greeting)
    initial.when_intent_matched(weather).go_to(weather_state)
    greeting.go_to(initial)
    weather_state.go_to(initial)
    condition = Condition(name="condition_1_1", callable=None)
    condition.code = "def condition_1_1(session):\n    return session.get('x') == 1\n"
    initial.when_condition(condition).go_to(custom_state)
    custom_state.when_variable_matches_operation("count", operator.gt, 3).go_to(initial)
    return agent


def _file_agent(name: str) -> Agent:
    """No LLM, no intents, and a body the generator leaves as a TODO (a file reply)."""
    agent = Agent(name)
    agent.platforms.append(WebSocketPlatform())
    start = agent.new_state("initial", initial=True)
    report = agent.new_state("report")
    start.set_body(Body("initial_body", actions=[AgentReply(message="hi")]))
    report.set_body(Body("report_body", actions=[WebSocketReplyFile()]))
    start.go_to(report)
    report.go_to(start)
    return agent


def test_baf_agent_states_intents_and_insertion_points(tmp_path):
    _generate(tmp_path, "generate_baf", agent_model=_support_agent())
    maps = _extract(tmp_path)
    agent_map = _map_for(maps, "baf")
    assert agent_map.family == "generic" and agent_map.title == "BESSER agent (BAF)"
    anchors = _by_id(maps)
    agent = "baf/support_bot.py"

    for state in ("initial", "greeting", "weather_state", "custom_state"):
        block = anchors[f"generic:state:{agent}:{state}"]
        assert block.kind == KIND_BLOCK and block.quote == f"# {state}"
    assert "body greeting_body" in anchors[f"generic:state:{agent}:greeting"].label
    assert "(initial)" in anchors[f"generic:state:{agent}:initial"].label
    assert anchors[f"generic:function:{agent}:greeting_body"].label == "body of state greeting"
    assert anchors[f"generic:function:{agent}:initial_fallback_body"].label == "fallback body of state initial"
    assert "custom condition" in anchors[f"generic:function:{agent}:condition_1_1"].label

    # The body-less state gets an insertion point and a warning, not a guessed body.
    slot = anchors[f"generic:state_body:{agent}:custom_state"]
    assert slot.kind == KIND_INSERT and slot.quote == "# custom_state" and "AFTER" in slot.hint
    assert f"generic:state_body:{agent}:greeting" not in anchors
    assert any("custom_state" in w and "no body" in w for w in agent_map.warnings)
    # The generator quotes the matcher target: the map says so.
    assert any("STRING '3'" in w for w in agent_map.warnings)
    # No Dockerfile next to it: the relative config.yaml load is a pitfall.
    assert any("relative to the working directory" in w for w in agent_map.warnings)

    for anchor_id in (f"generic:intent:{agent}:greet_intent", f"generic:intent:{agent}:weather_intent",
                      f"generic:agent_entity:{agent}:city_entity", f"generic:llms:{agent}",
                      f"generic:section:{agent}:intents", f"generic:section:{agent}:states",
                      f"generic:main:{agent}"):
        assert anchor_id in anchors, anchor_id
    assert "params city" in anchors[f"generic:intent:{agent}:weather_intent"].label
    for anchor_id in (f"generic:intents_end:{agent}", f"generic:state_decls_end:{agent}",
                      f"generic:state_blocks_end:{agent}"):
        assert anchors[anchor_id].kind == KIND_INSERT and anchors[anchor_id].quote, anchor_id
    assert anchors[f"generic:state_blocks_end:{agent}"].quote == "# RUN APPLICATION"
    decls_end = anchors[f"generic:state_decls_end:{agent}"]
    assert decls_end.quote == "custom_state = agent.new_state('custom_state')"
    assert "AFTER this line" in decls_end.hint

    assert any("agent support_bot" in s and "greeting" in s for s in agent_map.summary)
    assert any("config.yaml" in s and "agent config" in s for s in agent_map.summary)
    steps = " ".join(s for r in agent_map.recipes for s in r.steps)
    assert f"generic:state_decls_end:{agent}" in steps and f"generic:intents_end:{agent}" in steps
    assert "use default_llm.predict(...)" in steps        # this agent has an LLM


def test_baf_set_body_of_an_undefined_function_is_reported(tmp_path):
    # A body mixing custom code with another action renders set_body(<def that is never written>).
    agent = Agent("mixed")
    agent.platforms.append(WebSocketPlatform())
    state = agent.new_state("s0", initial=True)
    body = Body("s0_body")
    body.add_action(CustomCodeAction(source="def compute(session: AgentSession):\n    session.set('x', 1)\n"))
    body.add_action(AgentReply(message="done"))
    state.set_body(body)
    _generate(tmp_path, "generate_baf", agent_model=agent)
    source = (tmp_path / "baf" / "mixed.py").read_text(encoding="utf-8")
    warnings = _map_for(_extract(tmp_path), "baf").warnings
    if "def compute" not in source:     # the generator defect is present
        assert any("compute" in w and "not defined" in w for w in warnings), warnings
    else:                                # fixed upstream: nothing to report
        assert not any("not defined" in w for w in warnings)

    # The same shape written by hand, independent of the generator: body and fallback both checked.
    _write(tmp_path, "agent2/bot.py",
           "from baf.core.agent import Agent\nagent = Agent('bot')\n"
           "s0 = agent.new_state('s0', initial=True)\n# s0\ndef ok_body(session):\n    session.reply('hi')\n"
           "s0.set_body(ok_body)\ns0.set_fallback_body(oops)\n")
    warnings = _map_for(_extract(tmp_path), "agent2").warnings
    assert any("s0.set_fallback_body(oops)" in w and "not defined" in w for w in warnings), warnings
    assert not any("ok_body" in w for w in warnings)


def test_several_agents_share_recipes_and_dockerised_agents_raise_no_config_warning(tmp_path):
    """A web app's agents/ folder: one recipe set naming every agent, LLM advice only where an LLM exists."""
    for name, agent in (("alpha", _file_agent("alpha_bot")), ("support", _support_agent())):
        _generate(tmp_path / "gen" / name, "generate_baf", agent_model=agent)
        shutil.copytree(tmp_path / "gen" / name / "baf", tmp_path / "web_app" / "agents" / name)
    shutil.rmtree(tmp_path / "gen")
    # alpha runs in its own container from its folder (WebAppGenerator's agent Dockerfile).
    _write(tmp_path, "web_app/agents/alpha/Dockerfile",
           'FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD ["python", "alpha_bot.py"]\n')
    agents = _map_for(_extract(tmp_path), "web_app")
    alpha, support = "web_app/agents/alpha/alpha_bot.py", "web_app/agents/support/support_bot.py"

    steps = " ".join(s for r in agents.recipes for s in r.steps)
    assert "`generic:state:<agent file>:<state>`" in steps and alpha in steps and support in steps
    assert "`generic:intents_end:<agent file>`" in steps          # support has intents, alpha does not
    assert f"default_llm.predict(...) works only in {support}" in steps
    assert not any(alpha in w and "config.yaml" in w for w in agents.warnings)
    assert any(support in w and "config.yaml" in w for w in agents.warnings)

    # The TODO body the generator leaves (a file reply) reaches the overview through the summary.
    assert any(s.startswith(f"{alpha}: agent alpha_bot") and "no LLM" in s and "stubs report_body" in s
               for s in agents.summary), agents.summary
    assert _by_id([agents])[f"generic:function:{alpha}:report_body"].kind == KIND_STUB


# ---------------------------------------------------------------------------
# Qiskit, neural networks, BPMN
# ---------------------------------------------------------------------------

def _circuit() -> QuantumCircuit:
    circuit = QuantumCircuit("TestCircuit", qubits=3)
    circuit.add_creg(ClassicalRegister("c", 3))
    circuit.add_operation(InputGate(input_type="A", target_qubits=[0, 1], value=3))   # one qc.x line per qubit
    circuit.add_operation(HadamardGate(target_qubit=0))
    circuit.add_operation(PauliXGate(target_qubit=1, control_qubits=[0]))
    circuit.add_operation(RXGate(target_qubit=2, theta=0.785))
    circuit.add_operation(CustomGate(name="Oracle", target_qubits=[0, 1]))
    circuit.add_operation(FunctionGate(name="Grover_step", target_qubits=[0, 1],
                                       gates=[HadamardGate(target_qubit=0)]))
    circuit.add_operation(FunctionGate(name="Grover_step", target_qubits=[1, 2],
                                       gates=[PauliXGate(target_qubit=1)]))
    for qubit in range(3):
        circuit.add_operation(Measurement(target_qubit=qubit, output_bit=qubit))
    return circuit


def test_qiskit_operations_block_placeholder_gate_and_stable_function_gate_id(tmp_path):
    circuit = _circuit()
    _generate(tmp_path, "generate_qiskit", quantum_circuit=circuit)
    maps = _extract(tmp_path)
    circuit_map = _map_for(maps, "qiskit")
    assert circuit_map.title == "Qiskit quantum circuit"
    anchors = _by_id(maps)
    script = "qiskit/qiskit_circuit.py"

    operations = anchors[f"generic:operations:{script}"]
    assert operations.kind == KIND_BLOCK and operations.quote == "# Operations"
    # The two-qubit input is two lines but one operation: the count follows the model.
    assert f"({len(circuit.operations)} operations," in operations.label
    assert "blank-line-separated group" in operations.hint
    end = anchors[f"generic:operations_end:{script}"]
    assert end.kind == KIND_INSERT and end.quote == "# Draw" and "BEFORE" in end.hint
    oracle = anchors[f"generic:placeholder:{script}:Oracle"]
    assert oracle.kind == KIND_STUB and oracle.status and "create_placeholder('Oracle'" in oracle.quote
    assert operations.start_line < oracle.start_line <= operations.end_line < end.start_line
    assert "shots=1024" in anchors[f"generic:execution:{script}"].label
    # The def names carry a per-run id() number; the anchor ids must not, and a repeat is `~2`, not `:2`.
    gate = anchors[f"generic:function:{script}:_function_gate_Grover_step"]
    assert gate.label == "function gate Grover_step"
    assert f"generic:function:{script}:_function_gate_Grover_step~2" in anchors
    assert any(f"{len(circuit.operations)} operations (1 placeholder)" in s for s in circuit_map.summary)


def test_a_qiskit_file_that_does_not_parse_keeps_stable_function_gate_ids(tmp_path):
    _write(tmp_path, "qiskit/qiskit_circuit.py",
           "from qiskit import QuantumCircuit\n\n\ndef _function_gate_G_140240180044976(num_qubits):\n"
           "    return QuantumCircuit(num_qubits).to_instruction()\n\n\nqc = QuantumCircuit(2)\n"
           "qc.append(# PostSelection: expect 1, [0])\n")
    maps = _extract(tmp_path)
    gate = _by_id(maps)["generic:function:qiskit/qiskit_circuit.py:_function_gate_G"]
    assert gate.label == "function gate G (file does not parse)"
    # The parse error (with its line) is listed once, in the broken files, not in the prompt text.
    assert "qiskit/qiskit_circuit.py: does not parse; fix it first" in _map_for(maps, "qiskit").summary
    assert any(b.startswith("qiskit/qiskit_circuit.py: SyntaxError") for b in build_workspace_map(str(tmp_path)).broken)


def _network(sub: bool = False) -> NN:
    network = NN(name="tiny")
    network.add_layer(LinearLayer(name="hidden", actv_func="relu", in_features=4, out_features=8))
    network.add_layer(LinearLayer(name="out", in_features=8, out_features=2))
    if sub:   # each sub-network renders as a multi-line `self.<name> = Sequential(...)` statement
        for name in ("head", "tail"):
            part = NN(name=name)
            part.add_layer(LinearLayer(name=f"{name}1", actv_func="relu", in_features=2, out_features=2))
            part.add_layer(LinearLayer(name=f"{name}2", in_features=2, out_features=2))
            network.add_sub_nn(part)
    return network


def test_pytorch_layers_and_bpmn_gateway_without_conditions(tmp_path):
    _generate(tmp_path, "generate_pytorch", nn_model=_network())
    start, gateway, task_a, task_b = StartEvent(name=""), Gateway(name="?", gateway_type=GatewayType.EXCLUSIVE), \
        Task(name="A"), Task(name="B")
    end_a, end_b = EndEvent(name=""), EndEvent(name="")
    process = Process(name="Gateway", flow_nodes={start, gateway, task_a, task_b, end_a, end_b}, sequence_flows={
        SequenceFlow(start, gateway), SequenceFlow(gateway, task_a, name="happy"),
        SequenceFlow(gateway, task_b, name="default", is_default=True),
        SequenceFlow(task_a, end_a), SequenceFlow(task_b, end_b)})
    _generate(tmp_path, "generate_bpmn", bpmn_model=BPMNModel(name="Gateway", processes={process}))
    maps = _extract(tmp_path)
    anchors = _by_id(maps)

    network_file = next(a.file for a in anchors.values() if a.file.startswith("pytorch/"))
    layers = anchors[f"generic:layers:{network_file}:NeuralNetwork"]
    assert "hidden" in layers.label and "out" in layers.label
    assert anchors[f"generic:layers_end:{network_file}:NeuralNetwork"].kind == KIND_INSERT
    assert "forward pass" in anchors[f"generic:method:{network_file}:NeuralNetwork.forward"].label
    assert _map_for(maps, "pytorch").title == "PyTorch neural network"

    bpmn = _map_for(maps, "bpmn")
    gateways = [a for a in bpmn.anchors if a.label.startswith("exclusiveGateway")]
    assert len(gateways) == 1 and gateways[0].kind == KIND_STUB and "no condition" in gateways[0].status
    assert any(a.label.startswith("process Gateway") for a in bpmn.anchors)
    assert any("not executable" in s and "gateways without conditions: ?" in s for s in bpmn.summary)


@pytest.mark.parametrize("tool", ["generate_pytorch", "generate_tensorflow"])
def test_layers_end_after_a_multi_line_sub_network_never_points_inside_the_call(tmp_path, tool):
    _generate(tmp_path, tool, nn_model=_network(sub=True))
    maps = _extract(tmp_path)
    end = next(a for a in _by_id(maps).values() if a.id.startswith("generic:layers_end:"))
    lines = _lines(tmp_path, end.file)
    assert end.end_line > end.start_line and "tail" in lines[end.start_line - 1]     # the multi-line last layer
    # Both sub-networks close with the same line, so the closing line cannot be quoted: the quote is the
    # opening line and the hint names the whole statement ("after this line" would land inside the call).
    assert lines.count(lines[end.end_line - 1]) > 1
    assert end.quote == lines[end.start_line - 1]
    assert "AFTER this statement, which ends at end_line" in end.hint and "AFTER this line" not in end.hint


def test_bpmn_with_two_pools_names_every_process(tmp_path):
    _write(tmp_path, "bpmn/two.bpmn", """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="d">
  <bpmn:collaboration id="c">
    <bpmn:participant id="p1" name="Buyer" processRef="Process_Buyer" />
    <bpmn:participant id="p2" name="Seller" processRef="Process_Seller" />
  </bpmn:collaboration>
  <bpmn:process id="Process_Buyer" name="Buyer" isExecutable="false">
    <bpmn:task id="t1" name="Order" />
  </bpmn:process>
  <bpmn:process id="Process_Seller" name="Seller" isExecutable="false">
    <bpmn:task id="t2" name="Ship" />
  </bpmn:process>
</bpmn:definitions>
""")
    bpmn = _map_for(_extract(tmp_path), "bpmn")
    assert any("BPMN processes Buyer, Seller" in s for s in bpmn.summary), bpmn.summary


# ---------------------------------------------------------------------------
# Per-class generators (python, java, pydantic, sqlalchemy, sql, supabase, rdf)
# ---------------------------------------------------------------------------

def _library_model() -> DomainModel:
    """Abstract Person, an association class Loan, an N:M and a method stub: what the generators vary on."""
    person = Class(name="Person", is_abstract=True, attributes={Property(name="name", type=StringType)})
    author = Class(name="Author", attributes={Property(name="email", type=StringType)})
    reader = Class(name="Reader", attributes={Property(name="card", type=StringType)})
    book = Class(name="Book", attributes={Property(name="title", type=StringType),
                                          Property(name="pages", type=IntegerType)})
    library = Class(name="Library", attributes={Property(name="name", type=StringType)})
    book.methods = {Method(name="is_long", type=BooleanType)}
    located = BinaryAssociation(name="lib_book", ends={
        Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))})
    written = BinaryAssociation(name="book_author", ends={
        Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*")),
        Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))})
    borrowed = BinaryAssociation(name="loan_assoc", ends={
        Property(name="borrowedBooks", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="borrowers", type=reader, multiplicity=Multiplicity(0, "*"))})
    loan = AssociationClass(name="Loan", attributes={Property(name="due", type=DateType)}, association=borrowed)
    return DomainModel(name="Library", types={person, author, reader, book, library, loan},
                       associations={located, written, borrowed},
                       generalizations={Generalization(general=person, specific=author),
                                        Generalization(general=person, specific=reader)})


MODEL_CLASSES = ["Person", "Author", "Reader", "Book", "Library", "Loan"]
# A modify run whose model gained MORE classes than the scaffold has: each is still reported missing.
NEW_CLASSES = ["Publisher", "Shelf", "Branch", "Fine", "Event", "Member", "Card"]


@pytest.fixture(scope="module")
def per_class_ws(tmp_path_factory) -> Path:
    workspace = tmp_path_factory.mktemp("per_class")
    model = _library_model()
    for tool in ("generate_python_classes", "generate_java_classes", "generate_pydantic", "generate_sqlalchemy",
                 "generate_rdf", "generate_sql", "generate_supabase"):
        _generate(workspace, tool, domain_model=model)
    _generate(workspace / "api", "generate_pydantic", {"backend": True}, domain_model=model)
    return workspace


@pytest.fixture(scope="module")
def per_class_maps(per_class_ws) -> dict[str, ScaffoldMap]:
    return {m.root: m for m in _extract(per_class_ws, MODEL_CLASSES + NEW_CLASSES)}


def test_python_classes_link_model_classes_and_report_missing_ones(per_class_ws, per_class_maps):
    classes_map = per_class_maps["python"]
    assert classes_map.title == "Python domain classes"
    book = classes_map.entities["Book"]
    ids = {a.id: a for a in book.anchors}
    assert ids["generic:class:python/classes.py:Book"].kind == KIND_CLASS
    assert ids["generic:class:python/classes.py:Book"].quote == "class Book:"
    stub = ids["generic:method:python/classes.py:Book.is_long"]
    assert stub.kind == KIND_STUB and "NotImplementedError" in stub.status and stub.entity == "Book"
    assert "generic:properties:python/classes.py:Book" in ids
    assert ids["generic:method:python/classes.py:Book.__init__"].kind == KIND_FUNCTION
    # __init__ is the constructor, not a business method: only the stub counts as a method.
    assert book.summary == ["python/classes.py: class, accessors, 1 constructor, 1 method; stub is_long"]
    assert "method" not in classes_map.entities["Library"].summary[0].replace("1 constructor", "")
    assert book.missing == []
    for name in NEW_CLASSES:     # 7 new classes vs 6 generated ones: provenance comes from the banner
        assert classes_map.entities[name].missing == ["Python class"], name
        assert not classes_map.entities[name].anchors
    steps = " ".join(s for r in classes_map.recipes for s in r.steps)
    assert "generic:properties:python/classes.py:<Class>" in steps

    # Without model classes nothing is linked and nothing is claimed missing; stubs stay anchored.
    bare = _map_for(_extract(per_class_ws), "python")
    assert not bare.entities
    assert "generic:method:python/classes.py:Book.is_long" in _by_id([bare])


def test_java_classes_fold_accessors_and_find_the_todo_stub(per_class_maps):
    java = per_class_maps["java"]
    assert java.title == "Java classes"
    anchors = _by_id([java])
    assert anchors["generic:class:java/Book.java:Book"].entity == "Book"
    assert anchors["generic:method:java/Book.java:Book.is_long"].kind == KIND_STUB
    assert anchors["generic:accessors:java/Book.java:Book"].kind == KIND_BLOCK
    assert "generic:method:java/Book.java:Book.getTitle" not in anchors   # folded into the accessors block
    assert anchors["generic:fields_end:java/Book.java:Book"].kind == KIND_INSERT
    # No package line + the generator's `public <Class> (` constructors: generated, so any new class is missing.
    for name in NEW_CLASSES:
        assert java.entities[name].missing == ["Java class"], name
    assert "no package line" in " ".join(s for r in java.recipes for s in r.steps)


def test_orm_relationship_blocks_never_span_another_classes_lines(per_class_ws, per_class_maps):
    rel = "sqlalchemy/sql_alchemy.py"
    lines = _lines(per_class_ws, rel)
    blocks = sorted((a for a in per_class_maps["sqlalchemy"].all_anchors() if ":relationships:" in a.id),
                    key=lambda a: a.start_line)
    # The generator interleaves an association class's ends with its partners'.
    assert {a.id for a in blocks} >= {f"generic:relationships:{rel}:Book", f"generic:relationships:{rel}:Loan",
                                      f"generic:relationships:{rel}:Book.loans"}
    for previous, current in zip(blocks, blocks[1:]):
        assert previous.end_line < current.start_line, (previous.id, current.id)
    for block in blocks:
        owner = block.label.split()[2].rstrip(":")
        statements = [line for line in lines[block.start_line - 1:block.end_line] if "relationship(" in line]
        assert statements and all(line.startswith(f"{owner}.") for line in statements), block.id
    assert per_class_maps["sqlalchemy"].entities["Publisher"].missing == ["ORM class"]


def test_sql_tables_and_rdf_classes_are_located_by_entity(per_class_ws, per_class_maps):
    sql_map, rdf_map, supabase = per_class_maps["sql"], per_class_maps["rdf"], per_class_maps["supabase"]
    book_table = next(a for a in sql_map.entities["Book"].anchors if a.id.startswith("generic:table:"))
    assert book_table.kind == KIND_BLOCK and book_table.quote.startswith("CREATE TABLE book")
    assert any(a.id == "generic:rdf_class:rdf/vocabulary.ttl:Book" for a in rdf_map.entities["Book"].anchors)
    assert rdf_map.entities["Publisher"].missing == ["RDF class"]      # RDF declares every class
    # SQL never claims a missing table: abstract classes have none, Supabase names Loan's after its association.
    assert not any(e.missing for e in sql_map.entities.values())
    assert not any(e.missing for e in supabase.entities.values())

    answer = locate(str(per_class_ws), entity="Book", domain_model=_library_model(), max_chars=50_000)
    assert "truncated" not in answer
    for anchor_id in ("generic:table:sql/tables_sqlite.sql:book]", "generic:table:supabase/",
                      "generic:rdf_class:rdf/vocabulary.ttl:Book]", "generic:class:python/classes.py:Book]",
                      "generic:class:api/pydantic/pydantic_classes.py:BookCreate]"):
        assert anchor_id in answer["locations"], anchor_id


def test_pydantic_request_schemas_link_to_their_entity(per_class_maps):
    api = per_class_maps["api"]
    assert api.title == "Pydantic models"
    book = api.entities["Book"]
    assert [a.id for a in book.anchors if a.kind == KIND_CLASS] == ["generic:class:api/pydantic/pydantic_classes.py:"
                                                                     "BookCreate"]
    assert api.entities["Publisher"].missing == ["Pydantic model"]
    steps = " ".join(s for r in api.recipes for s in r.steps)
    assert "Only request schemas (`<Entity>Create`" in steps and "no methods or *Create" not in steps
    plain = per_class_maps["pydantic"]
    assert "There are no *Create schemas" in " ".join(s for r in plain.recipes for s in r.steps)


# ---------------------------------------------------------------------------
# Hand-written files
# ---------------------------------------------------------------------------

_SERVICE = '''"""Services."""
from abc import ABC, abstractmethod


class Repo(ABC):
    @abstractmethod
    def load(self):
        ...

    @staticmethod
    def build():
        pass


def compute(x):
    pass


async def fetch(url: str) -> str:
    """Fetch it."""
    raise NotImplementedError


def lookup(order_id: str) -> str:
    return 'TODO'


def done():
    return 1


def dup():
    return 1


def dup():  # noqa: F811
    return 2
'''

_CARD = '''import React from 'react';

interface CardProps {
  title: string;
}

const Card: React.FC<CardProps> = ({ title }) => {
  return <div className="card">{title}</div>;
};

export const useCards = () => {
  // TODO: fetch the cards from the API
  return [];
};

const API_URL = 'http://localhost:8000';

export function Badge({
  label,
}: { label: string }) {
  return <span>{label}</span>;
}

export function formatTitle(title: string) {
  return title.trim();
}

export default Card;
'''


def test_python_stubs_typescript_component_and_a_file_that_does_not_parse(tmp_path):
    _write(tmp_path, "lib/service.py", _SERVICE)
    _write(tmp_path, "lib/broken.py", "def ok():\n    return 1\n\nclass Broken(:\n    pass\n")
    _write(tmp_path, "app/components/Card.tsx", _CARD)
    maps = _extract(tmp_path)
    anchors = _by_id(maps)

    stubs = {a.id: a.status for a in anchors.values() if a.kind == KIND_STUB}
    assert stubs == {
        "generic:method:lib/service.py:Repo.build": "stub: empty body",
        "generic:function:lib/service.py:compute": "stub: empty body",
        "generic:function:lib/service.py:fetch": "stub: NotImplementedError",
        "generic:function:lib/service.py:lookup": "stub: returns a TODO placeholder",
        "generic:function:app/components/Card.tsx:useCards": "TODO: fetch the cards from the API",
    }
    # Plain defs (an abstract method's `...`, an implemented function) are named, not anchored:
    # every app-level anchor goes into the overview's list.
    assert anchors["generic:class:lib/service.py:Repo"].kind == KIND_CLASS
    for plain in ("generic:method:lib/service.py:Repo.load", "generic:function:lib/service.py:done",
                  "generic:function:app/components/Card.tsx:formatTitle"):
        assert plain not in anchors, plain
    assert "done" in anchors["generic:file:lib/service.py"].label
    assert "formatTitle" in anchors["generic:file:app/components/Card.tsx"].label
    lib = _map_for(maps, "lib")
    # A repeated name is listed once, never with an id's repeat suffix.
    assert any(s.startswith("lib/service.py:") and "functions done, dup" in s and "dup~" not in s
               for s in lib.summary), lib.summary

    card = anchors["generic:function:app/components/Card.tsx:Card"]
    assert card.label == "component Card (default export)"
    assert card.quote.startswith("const Card: React.FC<CardProps>") and card.end_line == card.start_line + 2
    assert anchors["generic:class:app/components/Card.tsx:CardProps"].kind == KIND_CLASS
    assert "generic:function:app/components/Card.tsx:API_URL" not in anchors   # a constant, not a component
    # A multi-line destructured signature (`}: Props) {`) does not end the component early.
    badge = anchors["generic:function:app/components/Card.tsx:Badge"]
    lines = _lines(tmp_path, "app/components/Card.tsx")
    assert lines[badge.end_line - 1] == "}" and "return <span>" in lines[badge.end_line - 2]

    # The broken file is still locatable by its top-level defs, and labelled as such.
    broken = anchors["generic:file:lib/broken.py"]
    assert broken.kind == KIND_FILE and "does not parse" in broken.label
    assert anchors["generic:class:lib/broken.py:Broken"].quote == "class Broken(:"
    assert "lib/broken.py: does not parse; fix it first" in lib.summary

    ws = build_workspace_map(str(tmp_path))
    assert any(b.startswith("lib/broken.py: SyntaxError") for b in ws.broken)


def test_a_file_too_deep_to_parse_loses_only_itself(tmp_path):
    _write(tmp_path, "lib/data.py", "X = " + " + ".join(["'a'"] * 20000) + "\n")
    _write(tmp_path, "lib/util.py", "def helper():\n    pass\n")
    _write(tmp_path, "lib/notes.md", "# notes\n")
    lib = _map_for(_extract(tmp_path), "lib")
    anchors = _by_id([lib])
    assert {"generic:file:lib/data.py", "generic:file:lib/notes.md", "generic:function:lib/util.py:helper"} \
        <= set(anchors)
    assert "too complex to parse" in anchors["generic:file:lib/data.py"].label
    assert anchors["generic:file:lib/data.py"].quote == ""      # its only line is far too long to quote


def test_hand_written_models_link_entities_without_claiming_missing_ones(tmp_path):
    _write(tmp_path, "models/schemas.py", "from pydantic import BaseModel\n\n\nclass Book(BaseModel):\n"
                                          "    title: str\n\n\nclass Author(BaseModel):\n    name: str\n")
    # Stdlib dataclasses named like the model are not the Python classes generator's output.
    _write(tmp_path, "app/models.py", "from dataclasses import dataclass, field\n\n\n@dataclass\nclass Book:\n"
                                      "    title: str\n\n\n@dataclass\nclass Author:\n"
                                      "    books: list = field(default_factory=list)\n")
    maps = _extract(tmp_path, ["Book", "Author", "Library"])
    models = _map_for(maps, "models")
    assert models.title == "Pydantic models"
    assert set(models.entities) == {"Book", "Author"}   # linked by name; Library not claimed missing
    assert models.entities["Book"].anchors[0].id == "generic:class:models/schemas.py:Book"
    app = _map_for(maps, "app")
    assert app.title == "Python code" and set(app.entities) == {"Book", "Author"}
    assert not any(e.missing for e in app.entities.values())
    assert not app.recipes      # no __init__ / @property recipe for a dataclass module


def test_hand_written_java_typescript_and_sql_edge_cases(tmp_path):
    _write(tmp_path, "javahand/Service.java", """package com.x;

import java.util.List;

@Service
public class OrderService {
    @Autowired
    private OrderRepo repo;

    @Override
    public List<Order> findAll() {
        throw new UnsupportedOperationException("later");
    }
}
""")
    _write(tmp_path, "tsedge/x.ts", "export default class extends Base {\n  run() {}\n}\n")
    _write(tmp_path, "supabase/0001_init.sql",
           'CREATE TABLE IF NOT EXISTS public."user" (\n'
           '    "id" UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE\n);\n'
           'CREATE TABLE IF NOT EXISTS public."users" (\n    "name" TEXT\n);\n')
    maps = _extract(tmp_path, ["OrderService", "User", "Users", "Order"])
    anchors = _by_id(maps)

    service = anchors["generic:class:javahand/Service.java:OrderService"]
    assert _lines(tmp_path, "javahand/Service.java")[service.start_line - 1] == "@Service"
    assert service.quote == "public class OrderService {"
    # The signature, not the `@Override` that happens to be unique.
    assert anchors["generic:method:javahand/Service.java:OrderService.findAll"].quote \
        == "    public List<Order> findAll() {"
    # Hand-written Java (a package line, no generated constructors) never claims Order missing.
    assert not _map_for(maps, "javahand").entities.get("Order")

    assert "generic:class:tsedge/x.ts:default" in anchors and "generic:class:tsedge/x.ts:extends" not in anchors
    # `REFERENCES auth.users(id)` is Supabase's own table, not the model's public.users.
    supabase = _map_for(maps, "supabase")
    assert not supabase.warnings, supabase.warnings
    assert anchors['generic:table:supabase/0001_init.sql:users'].entity == "Users"


def test_grouping_skips_claimed_roots_and_splits_their_parents(tmp_path):
    _write(tmp_path, "backend/main_api.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    _write(tmp_path, "web_app/backend/routers/book.py", "def get_book():\n    return 1\n")
    _write(tmp_path, "web_app/docker-compose.yml", "services:\n  backend:\n    build: ./backend\n  frontend:\n"
                                                     "    build: ./frontend\n")
    _write(tmp_path, "web_app/agents/bot/bot.py", "def helper():\n    pass\n")
    _write(tmp_path, "README.md", "# Project\n")
    _write(tmp_path, "notes/.env", "SECRET=1\n")
    _write(tmp_path, "notes/package-lock.json", "{}\n")
    _write(tmp_path, "notes/todo.md", "- ship it\n")

    maps = _extract(tmp_path, claimed_roots=["backend", "web_app/backend"])
    by_root = {m.root: m for m in maps}
    assert set(by_root) == {"", "web_app", "web_app/agents", "notes"}
    files = {a.file for m in maps for a in m.all_anchors()}
    assert not any(f.startswith(("backend/", "web_app/backend/")) for f in files)
    assert "notes/.env" not in files and "notes/package-lock.json" not in files
    assert by_root["web_app"].title == "Project files"
    assert any("services backend, frontend" in s for s in by_root["web_app"].summary)
    assert "generic:function:web_app/agents/bot/bot.py:helper" in _by_id(maps)


def _gui(book: Class) -> GUIModel:
    table = Table(name="book_table", title="Books", data_binding=DataBinding(domain_concept=book),
                  component_id="book-table")
    screen = Screen(name="Books", description="", route_path="/book", is_main_page=True,
                    view_elements={Text(name="books_title", content="Books"), table})
    return GUIModel(name="App", package="app", versionCode="1", versionName="1.0", description="",
                    modules={Module(name="Main", screens={screen})})


def test_a_scaffold_at_the_workspace_root_owns_its_whole_tree(tmp_path, library_book_author_model):
    """Backend or frontend generated INTO the workspace root: only specialist folders are mapped next to it."""
    from besser.generators.backend import BackendGenerator
    from besser.generators.react import ReactGenerator

    model = library_book_author_model
    backend, frontend = tmp_path / "backend_ws", tmp_path / "frontend_ws"
    with contextlib.redirect_stdout(io.StringIO()):
        BackendGenerator(model=model, output_dir=str(backend)).generate()
        ReactGenerator(model=model, gui_model=_gui(model.get_class_by_name("Book")),
                       output_dir=str(frontend)).generate()
    _generate(backend, "generate_baf", agent_model=_support_agent())
    _write(backend, "tests/test_api.py", "def test_list():\n    assert True\n")
    _generate(frontend, "generate_qiskit", quantum_circuit=_circuit())

    for workspace, family, specialist in ((backend, "fastapi", "baf"), (frontend, "react", "qiskit")):
        ws = build_workspace_map(str(workspace), model)
        assert any(m.family == family and m.root == "" for m in ws.maps), [(m.family, m.root) for m in ws.maps]
        assert [m.root for m in ws.maps if m.family == "generic"] == [specialist]
        answer = locate(str(workspace), path="routers/book.py" if family == "fastapi" else "src",
                        max_chars=50_000)
        assert f"[{family}:" in answer["locations"] and "[generic:" not in answer["locations"]   # mapped once
        assert len(scaffold_overview(str(workspace), model)) <= 4000


def test_caps_keep_the_map_compact_and_say_so(tmp_path):
    _write(tmp_path, "big/many.py", "".join(f"class C{i}:\n    x = {i}\n\n\n" for i in range(60)))
    big = _map_for(_extract(tmp_path), "big")
    classes = [a for a in big.anchors if a.kind == KIND_CLASS]
    assert len(classes) == generic.MAX_ANCHORS_PER_FILE
    assert any("20 more definitions not mapped" in s for s in big.summary)

    for n in range(11):     # 11 x 40 definitions > the 400-anchor map cap
        _write(tmp_path, f"wide/m{n:02d}.py", "".join(f"class G{i}:\n    x = {i}\n\n\n" for i in range(40)))
    wide = _map_for(_extract(tmp_path), "wide")
    assert len([a for a in wide.anchors if a.kind != KIND_FILE]) <= generic.MAX_ANCHORS_PER_MAP
    assert "generic:file:wide/m10.py" in _by_id([wide])          # still locatable by path
    assert "generic:class:wide/m10.py:G0" not in _by_id([wide])
    assert any("anchor cap was reached" in s for s in wide.summary)


# ---------------------------------------------------------------------------
# Bounded work and partial indexes: the map is rebuilt on every locate_code call
# ---------------------------------------------------------------------------

_PATHOLOGICAL = 20_000
# Long blank runs, blank lines, comment blocks, unclosed brackets, long names, repeated modifiers.
_FILLS = (" " * _PATHOLOGICAL, "\n" * _PATHOLOGICAL, " \n" * (_PATHOLOGICAL // 2), "// note\n" * (_PATHOLOGICAL // 8),
          "{" * _PATHOLOGICAL, "(" * _PATHOLOGICAL, "a" * _PATHOLOGICAL, "a " * (_PATHOLOGICAL // 2),
          "public " * (_PATHOLOGICAL // 7), "/* " * (_PATHOLOGICAL // 3))
# The literal heads the module's patterns start from, so each one is driven into its backtracking.
_HEADS = ("", "export function ", "export default X", "const x: T", "const x = ", "  private ", "    public ",
          "@A ", "<bpmn:task ", "<task id", "CREATE TABLE ", "REFERENCES ", "ex:A ", "# ", "// TODO",
          "throw new Error('")


def test_every_pattern_is_linear_on_pathological_input():
    """Each module pattern finishes in under 0.2 s on 20k characters (``const x: T`` + blanks took seconds)."""
    slow = []
    for name, value in sorted(vars(generic).items()):
        if not isinstance(value, re.Pattern):
            continue
        for head in _HEADS:
            for fill in _FILLS:
                text = head + fill + "x"
                started = time.perf_counter()
                for _ in value.finditer(text):
                    pass
                elapsed = time.perf_counter() - started
                if elapsed > 0.2:
                    slow.append((name, head, fill[:8], round(elapsed, 2)))
    assert not slow, slow


def test_pathological_files_map_in_bounded_time(tmp_path):
    """Lines the old patterns backtracked over for seconds: a declaration or a field followed by a long
    blank run, a long BPMN attribute name, a Dockerfile of blank lines next to an agent."""
    blank = " " * _PATHOLOGICAL
    _write(tmp_path, "ts/decls.ts", f"export function {blank}\nconst x: T{blank}\nexport default X{blank};x\n"
                                    f"function* {blank}\n")
    _write(tmp_path, "javap/Thing.java", f"public class Thing {{\n    private {'a ' * 5000}x\n    private int{blank}x\n}}\n")
    _write(tmp_path, "bpmnp/p.bpmn", f'<definitions>\n<task {"a" * _PATHOLOGICAL} id="T0"/>\n<task id="T1" name="ok"/>\n'
                                     f"</definitions>\n")
    _write(tmp_path, "agentp/bot.py", "from baf.core.agent import Agent\nagent = Agent('bot')\n"
                                      "agent.load_properties('config.yaml')\n")
    _write(tmp_path, "agentp/Dockerfile", "FROM python:3.11\n" + "\n" * (3 * _PATHOLOGICAL) + "CMD python other.py\n")
    started = time.perf_counter()
    maps = _extract(tmp_path)
    assert time.perf_counter() - started < 3.0
    anchors = _by_id(maps)
    assert "generic:class:javap/Thing.java:Thing" in anchors
    assert {"generic:bpmn:bpmnp/p.bpmn:T0", "generic:bpmn:bpmnp/p.bpmn:T1"} <= set(anchors)


def test_a_partial_index_never_claims_that_classes_on_disk_are_missing(tmp_path):
    """SAFE-2: past its cap the index leaves files out; a per-class file on disk is never reported missing."""
    _generate(tmp_path, "generate_java_classes", domain_model=_library_model())
    names = sorted(p.name for p in (tmp_path / "java").iterdir())
    assert names[-1] == "Reader.java"
    index = SourceIndex(str(tmp_path), max_files=len(names) - 1)    # the shallowest-then-sorted cap drops Reader
    assert index.capped and "java/Reader.java" not in index.files
    maps = generic.extract(index, MODEL_CLASSES + ["Publisher"], claimed_roots=[])
    _assert_anchors_sound(tmp_path, maps)
    java = _map_for(maps, "java")
    assert java.entities["Publisher"].missing == ["Java class"]     # really absent: still reported
    assert not java.entities.get("Reader") or not java.entities["Reader"].missing

    # Hand-written code the index did not read entirely proves nothing about what is absent.
    _write(tmp_path, "hand/models.py", "from pydantic import BaseModel\n\n\nclass Book(BaseModel):\n    title: str\n\n\n"
                                       "class Author(BaseModel):\n    name: str\n\n\nclass Library(BaseModel):\n"
                                       "    name: str\n")
    _write(tmp_path, "hand/more/deep/extra.py", "from pydantic import BaseModel\n\n\nclass Reader(BaseModel):\n"
                                                "    card: str\n")
    index = SourceIndex(str(tmp_path))
    index.files.remove("hand/more/deep/extra.py")      # left out, as past the cap
    index.capped = True
    hand = _map_for(generic.extract(index, ["Book", "Author", "Library", "Reader"], claimed_roots=[]), "hand")
    assert not any(e.missing for e in hand.entities.values())


def test_a_dockerfile_left_out_of_the_index_still_silences_the_config_warning(tmp_path):
    _write(tmp_path, "agents/alpha/alpha_bot.py", "from baf.core.agent import Agent\nagent = Agent('alpha')\n"
                                                  "agent.load_properties('config.yaml')\n")
    _write(tmp_path, "agents/alpha/Dockerfile",
           'FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD ["python", "alpha_bot.py"]\n')
    index = SourceIndex(str(tmp_path))
    index.files.remove("agents/alpha/Dockerfile")      # left out, as past the cap
    warnings = _map_for(generic.extract(index, [], claimed_roots=[]), "agents").warnings
    assert not any("relative to the working directory" in w for w in warnings), warnings


def test_json_and_sql_files_get_roles_and_detected_defects(tmp_path):
    _write(tmp_path, "jsonschema/json_schema.json", '{"definitions": {"A": {}}\n "x": 1}\n')
    _write(tmp_path, "supabase/20240101000000_m.sql",
           'CREATE TABLE IF NOT EXISTS public."book" (\n    "title" TEXT\n);\n'
           'CREATE TABLE IF NOT EXISTS public."loan" (\n    "id" UUID PRIMARY KEY,\n'
           '    "book_id" UUID REFERENCES public."book"(id)\n);\n')
    maps = _extract(tmp_path)      # line-free: the JSON error's position is dropped
    schema = _map_for(maps, "jsonschema")
    assert "INVALID JSON (Expecting ',' delimiter)" in schema.anchors[0].label
    assert any("not valid JSON" in w for w in schema.warnings)
    supabase = _map_for(maps, "supabase")
    assert supabase.title == "Supabase SQL migration"
    assert "2 tables" in supabase.anchors[0].label
    assert any("book" in w and "no id column" in w for w in supabase.warnings)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_build_overview_and_locate_end_to_end(tmp_path, library_book_author_model):
    _generate(tmp_path, "generate_baf", agent_model=_support_agent())
    _generate(tmp_path, "generate_python_classes", domain_model=library_book_author_model)
    _write(tmp_path, "tests/test_bot.py", "".join(f"def test_case_{i}():\n    assert {i}\n\n\n" for i in range(30)))

    ws = build_workspace_map(str(tmp_path), library_book_author_model)
    assert {m.root for m in ws.maps if m.family == "generic"} == {"baf", "python", "tests"}
    assert ws.broken == []

    # Deterministic: the same files give the same map.
    assert dataclasses.asdict(ws) == dataclasses.asdict(build_workspace_map(str(tmp_path), library_book_author_model))

    overview = scaffold_overview(str(tmp_path), library_book_author_model)
    assert "### BESSER agent (BAF) — `baf/`" in overview
    assert "### Python domain classes — `python/`" in overview
    assert "read_file offset=" not in overview and len(overview) <= 4000
    # 30 plain test functions cost one summary line, not 30 listed anchors.
    tests_map = next(m for m in ws.maps if m.root == "tests")
    assert [a.kind for a in tests_map.anchors] == [KIND_FILE]
    assert any("test_case_0" in s for s in tests_map.summary)
    assert render_overview(WorkspaceMap(maps=[m for m in ws.maps if m.root == "baf"]))

    answer = locate(str(tmp_path), query="generic:state:baf/support_bot.py:greeting")
    assert answer["count"] == 1
    assert "read_file offset=" in answer["locations"] and "quote: # greeting" in answer["locations"]

    by_entity = locate(str(tmp_path), entity="Book", domain_model=library_book_author_model)
    assert "python/classes.py" in by_entity["locations"]

    # Shift the agent file: the next answer follows the code, the id stays.
    agent_file = tmp_path / "baf" / "support_bot.py"
    first = render_locate(ws, query="generic:state:baf/support_bot.py:greeting")["locations"]
    agent_file.write_text("# a\n# b\n" + agent_file.read_text(encoding="utf-8"), encoding="utf-8")
    second = locate(str(tmp_path), query="generic:state:baf/support_bot.py:greeting")
    assert second["count"] == 1 and second["locations"] != first
