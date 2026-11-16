"""The run's TraceModel (``state/trace_model.py``), built from what a run persists.

The synthetic tests write a run log, a recipe, files and their line attribution as a run
would, then check the TraceModel built from them. The last tests drive a real orchestrator
with a scripted client (no network) and check the trace of what it did.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.BUML.metamodel.traceability import (
    AIEvidence, CodeArtifact, CodeGeneration, CodeGenerator, ExecutionStatus, LLM, LogType, ModelArtifact,
    PromptArtifact, RequirementArtifact, RequirementExtraction, RuleEvidence,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state import agent_trace
from besser.spec_driven_agent.state.trace_model import (
    MAX_LOGS_PER_CALL, RunModel, build_run_trace_model, generator_output_record, read_run_log,
    requirement_trace_id, run_operation_id, traced_requirement_id,
)
from besser.spec_driven_agent.state.tracing import (
    EVENT_GENERATOR_OUTPUT, EVENT_PHASE_ENTER, EVENT_PHASE_EXIT, EVENT_REQUIREMENTS, EVENT_RUN_END,
    EVENT_RUN_START, EVENT_TOOL_CALL, EVENT_TURN_START, EVENT_VALIDATION_ISSUE, TraceWriter,
)
from besser.utilities.generation_trace import ElementRef, InputModel
from besser.utilities.web_modeling_editor.backend.services.converters import process_trace, trace_buml_to_json

RUN = "run1"
DIAGRAM = InputModel(ElementRef("diagram-1", "DomainModel", "Library"), frozenset({"class-book", "prop-title"}))
MODELS = [RunModel("domain", DIAGRAM)]
GENERATED = "app/models.py"
GENERATED_TEXT = "class Book:\n    title: str\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Log:
    """A run log written as the orchestrator writes it, with a counter the test advances."""

    def __init__(self, workspace, run_id=RUN):
        self.counters = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "cache_creation_tokens": 0, "cache_read_tokens": 0, "model": "gpt-test"}
        self.writer = TraceWriter(str(workspace), run_id=run_id, usage=lambda: dict(self.counters))

    def spend(self, calls=1, tokens_in=0, tokens_out=0, cache_read=0):
        self.counters["api_calls"] += calls
        self.counters["input_tokens"] += tokens_in
        self.counters["output_tokens"] += tokens_out
        self.counters["cache_read_tokens"] += cache_read

    def write(self, event, **payload):
        self.writer.write(event, **payload)


def _write(workspace, rel, text):
    path = os.path.join(workspace, rel)
    os.makedirs(os.path.dirname(path) or str(workspace), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _generator_event(log, workspace, rel=GENERATED, text=GENERATED_TEXT):
    _write(workspace, rel, text)
    agent_trace.record_generated_files(str(workspace), [rel], "app", "1.0")
    log.write(EVENT_GENERATOR_OUTPUT, tool="generate_sqlalchemy", generator="sqlalchemy", reads=["domain"],
              started=0, files=[{
                  "path": rel, "sha256": _sha(text),
                  "regions": [{"element_id": "class-book", "element_type": "Class", "element_name": "Book",
                               "rule": "t.j2#class", "start_line": 1, "end_line": 2,
                               "sha256": _sha(text.rstrip("\n")), "owner": None},
                              {"element_id": "prop-title", "element_type": "Property", "element_name": "title",
                               "rule": "t.j2#attribute", "start_line": 2, "end_line": 2,
                               "sha256": _sha("    title: str"),
                               "owner": {"element_id": "class-book", "element_type": "Class",
                                         "element_name": "Book"}}],
                  "sources": []}])


def _recipe(workspace, **fields):
    with open(os.path.join(workspace, ".besser_recipe.json"), "w", encoding="utf-8") as handle:
        json.dump({"llm_model": "gpt-test", **fields}, handle)


def _reconcile(workspace):
    agent_trace.reconcile_workspace(str(workspace), agent_trace.ai_contributor("openai/gpt-test", RUN))


def _build(workspace, **kwargs):
    kwargs.setdefault("models", MODELS)
    kwargs.setdefault("instructions", "A library app")
    kwargs.setdefault("provider", "openai")
    kwargs.setdefault("agent_version", "9.9.9")
    trace = build_run_trace_model(str(workspace), run_id=RUN, **kwargs)
    assert trace is not None
    trace.validate()
    return trace


def _round_trip(trace):
    data = trace_buml_to_json(trace)
    assert data["projectId"] is None
    restored = process_trace(json.loads(json.dumps(data)))
    restored.validate()
    assert trace_buml_to_json(restored) == data
    return data


def _operation(trace):
    return next(op for op in trace.operations if isinstance(op, CodeGeneration))


def _calls(trace, tool_type):
    return [c for c in _operation(trace).tool_calls if isinstance(c.tool, tool_type)]


def _links_to(trace, target):
    return [link for link in trace.trace_links if link.target is target]


def _file(trace, path, final=True):
    versions = [a for a in trace.artifacts if isinstance(a, CodeArtifact) and a.path == path and a.start_line is None]
    revised = {id(a.revision_of) for a in versions if a.revision_of is not None}
    latest = [a for a in versions if id(a) not in revised]
    return latest[0] if final else versions


def test_a_generator_only_run(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_PHASE_ENTER, phase="phase1")
    _generator_event(log, tmp_path)
    log.write(EVENT_PHASE_EXIT, phase="phase1", generator_used="generate_sqlalchemy")
    log.write(EVENT_RUN_END)
    _recipe(tmp_path)
    _reconcile(tmp_path)

    trace = _build(tmp_path, actor_name="Ada")
    operation = _operation(trace)
    assert operation.id == run_operation_id(RUN) == f"spec-driven:{RUN}"
    assert sorted(type(a).__name__ for a in operation.performed_by) == ["Agent", "Person"]
    assert [type(c.tool) for c in operation.tool_calls] == [CodeGenerator]
    generator = operation.tool_calls[0]
    assert generator.tool.name == "sqlalchemy" and generator.tool.version == "9.9.9"
    assert generator.input_tokens is None

    root = next(a for a in trace.artifacts if isinstance(a, ModelArtifact) and a.parent is None)
    assert (root.element_id, root.element_type) == ("diagram-1", "DomainModel")
    file = _file(trace, GENERATED)
    assert file.content == GENERATED_TEXT and file.revision_of is None
    [model_link] = _links_to(trace, file)
    assert model_link.source is root and model_link.evidence[0].rule == "generator:sqlalchemy"
    title = next(a for a in trace.artifacts if isinstance(a, ModelArtifact) and a.element_id == "prop-title")
    assert title.parent.element_id == "class-book" and title.parent.parent is root
    fragment = next(a for a in trace.artifacts if isinstance(a, CodeArtifact) and a.start_line == 2)
    assert [link.source for link in _links_to(trace, fragment)] == [title]
    # Nothing the LLM wrote: no prompt link, no LLM call.
    assert not any(isinstance(link.source, PromptArtifact) for link in trace.trace_links)
    _round_trip(trace)


def test_a_generated_file_the_llm_edits_is_a_revision(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_PHASE_ENTER, phase="phase1")
    _generator_event(log, tmp_path)
    log.write(EVENT_PHASE_EXIT, phase="phase1")
    log.write(EVENT_PHASE_ENTER, phase="phase2")
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=100, tokens_out=40)
    log.write(EVENT_TOOL_CALL, turn=1, tool="modify_file", success=True, status="ok",
              input={"path": GENERATED, "old_text": "x" * 900, "new_text": "y"})
    edited = GENERATED_TEXT + "    isbn: str\n"
    _write(tmp_path, GENERATED, edited)
    log.write(EVENT_TURN_START, turn=2)
    log.spend(tokens_in=50, tokens_out=10)
    log.write(EVENT_TOOL_CALL, turn=2, tool="write_file", success=True, status="ok",
              input={"path": "app/new.py", "content": "print('hi')\n"})
    _write(tmp_path, "app/new.py", "print('hi')\n")
    log.write(EVENT_PHASE_EXIT, phase="phase2", stop_reason="completed")
    log.write(EVENT_RUN_END)
    _recipe(tmp_path)
    _reconcile(tmp_path)

    trace = _build(tmp_path)
    generated, revised = _file(trace, GENERATED, final=False)
    assert revised.revision_of is generated
    assert generated.content is None and generated.content_hash == _sha(GENERATED_TEXT)
    assert revised.content == edited and revised.content_hash == _sha(edited)
    # The generator's links stay on its version; the edit is linked from the prompt.
    assert all(isinstance(link.evidence[0], RuleEvidence) for link in _links_to(trace, generated))
    prompt = next(a for a in trace.artifacts if isinstance(a, PromptArtifact))
    [edit] = [a for a in trace.artifacts if isinstance(a, CodeArtifact) and a.parent is revised]
    assert (edit.start_line, edit.end_line) == (3, 3)
    [ai_link] = _links_to(trace, edit)
    assert ai_link.source is prompt and isinstance(ai_link.evidence[0], AIEvidence)
    assert ai_link.evidence[0].algorithm == "openai/gpt-test"
    new = _file(trace, "app/new.py")
    assert new.revision_of is None and [a for a in trace.artifacts if a.parent is new]
    turns = _calls(trace, LLM)
    assert [(c.input_tokens, c.output_tokens) for c in turns] == [(100, 40), (50, 10)]
    assert turns[0].tool.name == "gpt-test" and turns[0].tool.provider == "openai"
    # A tool call's log names what was done, without the file bodies.
    [modify] = [r for r in turns[0].logs if r.log_type == LogType.TOOL_CALL]
    assert GENERATED in modify.content and "x" * 50 not in modify.content and "900 characters" in modify.content
    _round_trip(trace)


def test_tokens_are_per_turn_and_calls_outside_turns_are_their_own(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_PHASE_ENTER, phase="phase1")
    log.spend(calls=2, tokens_in=300, tokens_out=20)  # generator selection, planning
    log.write(EVENT_PHASE_EXIT, phase="phase1")
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=1000, tokens_out=100, cache_read=800)
    log.write(EVENT_TOOL_CALL, turn=1, tool="read_file", success=True, input={"path": "a.py"})
    log.write(EVENT_TURN_START, turn=2)  # cut off before its LLM call
    log.write(EVENT_PHASE_EXIT, phase="phase2", stop_reason="cancelled")
    log.write(EVENT_RUN_END)
    _recipe(tmp_path)

    trace = _build(tmp_path, status=ExecutionStatus.CANCELLED)
    assert _operation(trace).status == ExecutionStatus.CANCELLED
    calls = _calls(trace, LLM)
    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(300, 20), (1800, 100)]
    assert "outside the agent's turns" in calls[0].logs[0].content
    assert calls[1].logs[0].content.startswith("Turn 1 (phase1): 1 call(s), 1800 input tokens (cache read 800")
    assert [c.sequence_number for c in _operation(trace).tool_calls] == list(range(len(calls)))


def test_a_resumed_run_counts_from_its_new_client(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=100, tokens_out=10)
    log.write(EVENT_TURN_START, turn=2)
    log.spend(tokens_in=100, tokens_out=10)  # turn 2 crashed: no closing record
    # The resume: a fresh client, so the counters start from zero again.
    log.counters.update(api_calls=0, input_tokens=0, output_tokens=0)
    log.write(EVENT_RUN_START, resumed=True)
    log.write(EVENT_TURN_START, turn=3)
    log.spend(tokens_in=70, tokens_out=7)
    log.write(EVENT_RUN_END, resumed=True)
    # Another run's records in the same log (a seeded workspace) are not this run's.
    other = _Log(tmp_path, run_id="previous")
    other.write(EVENT_TURN_START, turn=1)
    _recipe(tmp_path)

    assert {r["run_id"] for r in read_run_log(str(tmp_path), RUN)} == {RUN}
    calls = _calls(_build(tmp_path), LLM)
    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(100, 10), (70, 7)]


def test_requirements_the_run_extracted(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_PHASE_ENTER, phase="phase2")
    log.spend(calls=2, tokens_in=500, tokens_out=60)
    log.write(EVENT_REQUIREMENTS, source="extracted", model="gpt-test",
              items=[{"id": 1, "text": "Books have a title", "kind": "data"},
                     {"id": 2, "text": "ISBNs are unique", "kind": "uniqueness"},
                     {"id": 3, "text": "(incomplete)", "kind": "verification"}],
              extraction_usage={"api_calls": 1, "input_tokens": 200, "output_tokens": 50,
                                "cache_creation_tokens": 0, "cache_read_tokens": 0})
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=10, tokens_out=1)
    log.write(EVENT_RUN_END)
    _recipe(tmp_path, requirements_source="extracted")

    trace = _build(tmp_path)
    [extraction] = [op for op in trace.operations if isinstance(op, RequirementExtraction)]
    requirements = extraction.writes
    assert [r.id for r in requirements] == [f"{RUN}:R1", f"{RUN}:R2"]
    assert all(isinstance(r, RequirementArtifact) for r in requirements)
    [call] = extraction.tool_calls
    assert (call.input_tokens, call.output_tokens) == (200, 50)
    assert [isinstance(op.reads[0], PromptArtifact) for op in [extraction]] == [True]
    assert {link.target.id for link in trace.trace_links if link.operation is extraction} == {r.id for r in requirements}
    # The extraction's tokens are not counted again in the run's own calls.
    outside = _calls(trace, LLM)[0]
    assert (outside.input_tokens, outside.output_tokens) == (300, 10)
    _round_trip(trace)


def test_requirements_the_editor_sent_are_referenced_by_their_id(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_REQUIREMENTS, source="editor",
              items=[{"id": 1, "text": "Books have a title", "kind": "data", "ref": "call-1:R1"}])
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=10, tokens_out=1)
    log.write(EVENT_RUN_END)
    _recipe(tmp_path, requirements_source="editor")

    trace = _build(tmp_path)
    assert not any(isinstance(op, RequirementExtraction) for op in trace.operations)
    [requirement] = [a for a in trace.artifacts if isinstance(a, RequirementArtifact)]
    assert requirement.id == "call-1:R1" and requirement in _operation(trace).reads
    _round_trip(trace)


def test_the_judge_links_requirements_to_the_code_and_leaves_unmet_ones_unlinked(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_REQUIREMENTS, source="editor", items=[
        {"id": 1, "text": "Books have a title", "kind": "data", "ref": "req-title"},
        {"id": 2, "text": "Books have an ISBN", "kind": "data", "ref": "req-isbn"},
        {"id": 3, "text": "Loans are limited", "kind": "rule", "ref": "req-loans"},
        {"id": 4, "text": "Books are searchable", "kind": "ui", "ref": "req-search"}])
    _generator_event(log, tmp_path)
    log.write(EVENT_RUN_END)
    _recipe(tmp_path, requirements_source="editor", requirements=[
        {"id": 1, "ref": "req-title", "kind": "data", "text": "Books have a title", "status": "implemented",
         "evidence": "models.py: title: str", "note": ""},
        {"id": 2, "ref": "req-isbn", "kind": "data", "text": "Books have an ISBN", "status": "partial",
         "evidence": "app/models.py: class Book:", "note": "no isbn field"},
        {"id": 3, "ref": "req-loans", "kind": "rule", "text": "Loans are limited", "status": "missing",
         "evidence": "", "note": "no loan limit anywhere"},
        {"id": 4, "ref": "req-search", "kind": "ui", "text": "Books are searchable", "status": "unverified",
         "evidence": "", "note": "frontend not shown"}])
    _reconcile(tmp_path)

    trace = _build(tmp_path)
    links = {link.source.id: link for link in trace.trace_links if isinstance(link.source, RequirementArtifact)}
    assert set(links) == {"req-title", "req-isbn"}
    title = links["req-title"]
    assert isinstance(title.target, CodeArtifact) and (title.target.start_line, title.target.path) == (2, GENERATED)
    assert title.confidence.value == 1.0 and title.confidence.explanation.startswith("implemented")
    assert links["req-isbn"].confidence.value == 0.5 and "no isbn field" in links["req-isbn"].confidence.explanation
    assert links["req-isbn"].target.start_line == 1
    [judge] = [c for c in _calls(trace, LLM) if any("verdicts" in r.content for r in c.logs)]
    verdict_records = [r for r in judge.logs if r.log_type == LogType.OUTPUT_RESPONSE]
    assert title.evidence[0].logs == [verdict_records[0]]
    assert any(r.content.startswith("R3 missing") and "no loan limit" in r.content for r in verdict_records)
    assert any(r.content.startswith("R4 unverified") for r in verdict_records)
    _round_trip(trace)


def test_traced_requirement_ids():
    assert requirement_trace_id(RUN, {"id": 2, "ref": "call-1:R2"}) == "call-1:R2"
    assert requirement_trace_id(RUN, {"id": 2}) == f"{RUN}:R2"
    assert traced_requirement_id(RUN, {"id": 3, "kind": "verification"}) is None
    assert traced_requirement_id(RUN, {"id": 3, "kind": "rule"}) == f"{RUN}:R3"


def test_findings_and_gates_are_logged_and_bounded(tmp_path):
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    log.write(EVENT_TURN_START, turn=1)
    log.spend(tokens_in=1, tokens_out=1)
    for n in range(MAX_LOGS_PER_CALL + 5):
        log.write(EVENT_TOOL_CALL, turn=1, tool="read_file", success=True, input={"path": f"f{n}.py"})
    log.write(EVENT_VALIDATION_ISSUE, phase="phase3_runtime_gate",
              message="app did not boot: OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
    log.write(EVENT_RUN_END)
    _recipe(tmp_path, validation_issues=[{"severity": "warning", "message": "style nit"},
                                          {"severity": "blocker", "message": "SyntaxError in a.py"}])

    [turn] = _calls(_build(tmp_path), LLM)
    contents = [r.content for r in turn.logs]
    assert any(c.startswith("... ") and "more message" in c for c in contents)
    finding = next(r for r in turn.logs if "SyntaxError" in r.content)
    assert finding.log_type == LogType.ERROR
    assert contents.index(finding.content) < contents.index(next(c for c in contents if "style nit" in c))
    assert not any("abcdefghijklmnopqrstuvwxyz0123456789" in c for c in contents)


def test_a_run_that_called_nothing_has_no_trace(tmp_path):
    _Log(tmp_path).write(EVENT_RUN_START)
    assert build_run_trace_model(str(tmp_path), run_id=RUN, models=MODELS, instructions="x") is None


def _library_model():
    string = PrimitiveDataType("str")
    title = Property(name="title", type=string)
    title.id = "prop-title"
    book = Class(name="Book", attributes={title})
    book.id = "class-book"
    return DomainModel(name="Library", types={book})


def test_a_generator_tool_logs_its_traced_regions(tmp_path):
    records = []
    executor = ToolExecutor(str(tmp_path), domain_model=_library_model())
    executor.on_generated = records.append
    json.loads(executor.execute("generate_sqlalchemy", {}))
    [record] = records
    assert (record["tool"], record["generator"], record["reads"]) == ("generate_sqlalchemy", "sqlalchemy", ["domain"])
    [entry] = [f for f in record["files"] if f["path"] == "sqlalchemy/sql_alchemy.py"]
    with open(tmp_path / "sqlalchemy" / "sql_alchemy.py", "rb") as handle:
        assert entry["sha256"] == hashlib.sha256(handle.read()).hexdigest()
    book = next(r for r in entry["regions"] if r["element_id"] == "class-book")
    assert book["rule"].endswith("#class") and book["start_line"] <= book["end_line"]
    assert generator_output_record(str(tmp_path), str(tmp_path / "sqlalchemy"), [], {}, "nope", 0)["generator"] == "nope"


# ---------------------------------------------------------------------------------------------
# A real orchestrator, driven by a scripted client
# ---------------------------------------------------------------------------------------------


class _Block:
    def __init__(self, block_type, **fields):
        self.type = block_type
        self.__dict__.update(fields)


class _ScriptedClient:
    """Answers from a script and reports usage as a provider would."""

    model = "scripted-model"

    def __init__(self, script):
        self.script = list(script)
        self.usage = UsageTracker("scripted-model")

    def chat(self, system, messages, tools, **kwargs):
        self.usage.record(SimpleNamespace(input_tokens=120, output_tokens=12))
        if not self.script:
            return {"stop_reason": "end_turn", "content": [_Block("text", text="Done")]}
        return self.script.pop(0)


def _tool_use(tool_id, name, **arguments):
    return {"stop_reason": "tool_use", "content": [_Block("tool_use", id=tool_id, name=name, input=arguments)]}


def test_the_trace_of_an_orchestrated_run(tmp_path):
    client = _ScriptedClient([
        _tool_use("t1", "generate_sqlalchemy"),
        _tool_use("t2", "write_file", path="notes.py", content="NOTES = 1\n"),
    ])
    orchestrator = LLMOrchestrator(llm_client=client, domain_model=_library_model(), output_dir=str(tmp_path),
                                   max_turns=4, use_streaming=False, run_id=RUN, target_generator=None,
                                   target_generator_bound=True)
    orchestrator.run("A library with books")
    _reconcile(tmp_path)

    trace = _build(tmp_path, models=[RunModel("domain", InputModel(ElementRef("diagram-1", "DomainModel", "Library"),
                                                                       frozenset({"class-book", "prop-title"})))],
                   instructions="A library with books")
    turns = _calls(trace, LLM)
    assert len(turns) >= 2 and all(c.input_tokens == 120 and c.output_tokens == 12 for c in turns[:2])
    [generator] = _calls(trace, CodeGenerator)
    assert generator.tool.name == "sqlalchemy"
    generated = _file(trace, "sqlalchemy/sql_alchemy.py")
    book = next(a for a in trace.artifacts if isinstance(a, ModelArtifact) and a.element_id == "class-book")
    assert any(link.source is book and link.target.parent is generated for link in trace.trace_links)
    notes = _file(trace, "notes.py")
    assert notes.content == "NOTES = 1\n"
    assert any(isinstance(link.evidence[0], AIEvidence) and link.target.parent is notes for link in trace.trace_links)
    _round_trip(trace)


def test_an_element_two_models_reach_sits_under_the_model_that_holds_it(tmp_path):
    # A GUI listed first that shows the class reaches it less directly than the domain model holding it.
    gui = InputModel(ElementRef("gui-1", "GUIModel", "Screens"), frozenset({"class-book"}), {"class-book": 2})
    domain = InputModel(ElementRef("diagram-1", "DomainModel", "Library"), frozenset({"class-book", "prop-title"}),
                        {"class-book": 0, "prop-title": 1})
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    _generator_event(log, tmp_path)
    log.write(EVENT_RUN_END)
    _recipe(tmp_path)
    _reconcile(tmp_path)

    trace = _build(tmp_path, models=[RunModel("gui", gui), RunModel("domain", domain)])
    book = next(a for a in trace.artifacts if isinstance(a, ModelArtifact) and a.element_id == "class-book")
    assert book.parent.element_id == "diagram-1"


def test_a_run_trace_keeps_contents_with_their_credentials_redacted(tmp_path):
    settings = 'DEBUG = True\nSECRET_KEY = "django-insecure-0123456789abcdef"\n'
    log = _Log(tmp_path)
    log.write(EVENT_RUN_START)
    _generator_event(log, tmp_path, rel="site/settings.py", text=settings)
    log.write(EVENT_RUN_END)
    _recipe(tmp_path)
    _reconcile(tmp_path)

    file = _file(_build(tmp_path), "site/settings.py")
    assert file.content == 'DEBUG = True\nSECRET_KEY = "[REDACTED]"\n' and file.content_hash == _sha(settings)
