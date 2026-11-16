"""The run's trace on the ``done`` event and in the ZIP.

A real orchestrator runs behind the runner with a scripted client (no network,
no real LLM): the ``done`` event must carry a trace-file fragment that the
backend reads back and that passes ``TraceModel.validate()``.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import zipfile
from types import SimpleNamespace

from besser.BUML.metamodel.traceability import ExecutionStatus
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.trace_model import TRACE_MODEL_FILENAME
from besser.utilities.web_modeling_editor.backend.models import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters import process_trace
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    SmartGenerationRunner,
    _run_status,
    build_verification_report,
)
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_done_verdict import _cleanup
from tests.utilities.web_modeling_editor.backend.spec_driven.test_modify_seed import (
    _build_request,
    _collect_frames,
    _parse,
)


class _Block:
    def __init__(self, block_type, **fields):
        self.type = block_type
        self.__dict__.update(fields)


class _ScriptedClient:
    """Writes one file, then ends; reports usage as a provider would."""

    model = "scripted-model"

    def __init__(self):
        self.usage = UsageTracker("scripted-model")
        self.script = [{"stop_reason": "tool_use", "content": [_Block(
            "tool_use", id="t1", name="write_file", input={"path": "app.py", "content": "print('library')\n"})]}]

    def chat(self, system, messages, tools, **kwargs):
        self.usage.record(SimpleNamespace(input_tokens=200, output_tokens=20))
        if self.script:
            return self.script.pop(0)
        return {"stop_reason": "end_turn", "content": [_Block("text", text="Done")]}


def _run(monkeypatch, generator="generate_sqlalchemy", **request):
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _ScriptedClient())
    runner = SmartGenerationRunner(_build_request(target_generator_override=generator, **request),
                                   runtime=current_runtime())
    frames = [_parse(frame) for frame in asyncio.run(_collect_frames(runner))]
    done = [frame for frame in frames if frame.get("event") == "done"]
    assert done, [frame.get("event") for frame in frames]
    return runner, done[-1]


def test_done_carries_the_trace_of_the_run(monkeypatch):
    try:
        runner, done = _run(monkeypatch, actorName="Ada")
        trace = done["trace"]
        assert trace["schemaVersion"] == 6 and trace["projectId"] is None
        model = process_trace(trace)
        model.validate()
        [operation] = [op for op in trace["operations"] if op["type"] == "CodeGeneration"]
        assert operation["id"] == f"spec-driven:{runner.run_id}"
        tools = {tool["id"]: tool for tool in trace["tools"]}
        called = [tools[call["toolId"]] for call in operation["toolCalls"]]
        assert ("CodeGenerator", "sqlalchemy") in [(t["type"], t["name"]) for t in called]
        llm_calls = [call for call in operation["toolCalls"] if tools[call["toolId"]]["type"] == "LLM"]
        assert llm_calls and llm_calls[0]["inputTokens"] == 200 and llm_calls[0]["outputTokens"] == 20
        assert {tools[c["toolId"]]["provider"] for c in llm_calls} == {"anthropic"}
        assert sorted(a["type"] for a in trace["actors"]) == ["Agent", "Person"]
        roots = [a for a in trace["artifacts"] if a["type"] == "ModelArtifact" and not a.get("parentId")]
        assert [(r["elementId"], r["elementType"]) for r in roots] == [("cd1", "DomainModel")]
        paths = {a["path"] for a in trace["artifacts"] if a["type"] == "CodeArtifact"}
        assert "app.py" in paths and any(p.startswith("sqlalchemy/") for p in paths)
        # The same JSON ships in the ZIP.
        entry = asyncio.run(current_runtime().registry.get(runner.run_id))
        assert entry.is_zip
        with zipfile.ZipFile(entry.file_path) as archive:
            assert json.loads(archive.read(TRACE_MODEL_FILENAME)) == trace
    finally:
        _cleanup()


def test_a_run_without_actor_is_performed_by_the_agent_alone(monkeypatch):
    try:
        _, done = _run(monkeypatch)
        assert [a["type"] for a in done["trace"]["actors"]] == ["Agent"]
    finally:
        _cleanup()


def test_requirement_items_carry_their_trace_ref():
    recipe = {"requirements": [
        {"id": 1, "ref": "call-1:R4", "kind": "data", "text": "Books have titles", "status": "implemented",
         "evidence": "a.py: title"},
        {"id": 2, "kind": "rule", "text": "Loans are limited", "status": "missing", "note": "none"},
        {"id": 3, "kind": "verification", "text": "extraction incomplete", "status": "unverified"},
    ]}
    report = build_verification_report(recipe, "abc")
    items = {item.id: item for item in report.verified + report.shippedUnenforced + report.notVerified}
    assert items["R1"].traceRef == "call-1:R4"
    assert items["R2"].traceRef == "abc:R2"
    assert items["R3"].traceRef is None
    assert build_verification_report(recipe).verified[0].traceRef is None


def test_the_operation_status_follows_how_the_loop_ended():
    assert _run_status(True, "completed") == ExecutionStatus.SUCCEEDED
    assert _run_status(False, "cancelled") == ExecutionStatus.CANCELLED
    assert _run_status(False, "timeout") == ExecutionStatus.TIMED_OUT
    assert _run_status(False, "api_error") == ExecutionStatus.FAILED


_FIXTURES = pathlib.Path(__file__).parents[4]


def _editor_diagram(path: pathlib.Path, diagram_id: str) -> dict:
    return {"id": diagram_id, **json.loads(path.read_text("utf-8"))}


def test_the_run_trace_refers_to_the_elements_of_every_diagram_by_their_editor_ids(monkeypatch):
    """The editor's Library (Full Stack) diagrams through a web app run: each element the trace records is
    an element of the diagram sent, by its editor id, under that diagram's root."""
    class_diagram = _editor_diagram(_FIXTURES / "generators" / "fixtures" / "library_template_class_diagram.json",
                                    "class-diagram")
    agent_diagram = _editor_diagram(_FIXTURES / "generators" / "fixtures" / "library_template_agent_diagram.json",
                                    "agent-diagram")
    gui_diagram = _editor_diagram(_FIXTURES / "utilities" / "web_modeling_editor" / "backend" / "fixtures"
                                  / "library_template_gui_diagram.json", "gui-diagram")
    diagrams = {"class-diagram": class_diagram, "gui-diagram": gui_diagram, "agent-diagram": agent_diagram}
    project = ProjectInput(id="library", type="BesserProject", name="Library", createdAt="2026-04-15T00:00:00Z",
                           diagrams={"ClassDiagram": [class_diagram], "GUINoCodeDiagram": [gui_diagram],
                                     "AgentDiagram": [agent_diagram]})

    def editor_ids(model) -> set:
        ids, pending = set(), [model]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                for key in ("elements", "relationships", "components"):
                    if isinstance(value.get(key), dict):
                        ids.update(value[key])
                if isinstance(value.get("id"), str):
                    ids.add(value["id"])
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        return ids

    try:
        _, done = _run(monkeypatch, generator="generate_web_app", project=project)
        trace = done["trace"]
        process_trace(trace).validate()
        artifacts = {a["id"]: a for a in trace["artifacts"]}

        def root_of(artifact):
            while artifact.get("parentId"):
                artifact = artifacts[artifact["parentId"]]
            return artifact

        roots = {a["elementId"]: a["elementType"] for a in artifacts.values()
                 if a["type"] == "ModelArtifact" and not a.get("parentId")}
        assert roots == {"class-diagram": "DomainModel", "gui-diagram": "GUIModel", "agent-diagram": "Agent"}
        under = {}
        for artifact in artifacts.values():
            if artifact["type"] == "ModelArtifact" and artifact.get("parentId"):
                under.setdefault(root_of(artifact)["elementId"], set()).add(artifact["elementId"])
        assert set(under) == set(diagrams), under.keys()
        for diagram_id, element_ids in under.items():
            assert element_ids <= editor_ids(diagrams[diagram_id]["model"]), diagram_id
    finally:
        _cleanup()
