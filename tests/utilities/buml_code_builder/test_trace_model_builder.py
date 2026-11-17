"""``trace_model_to_code`` writes a TraceModel as Python that ``parse_trace_code`` loads back unchanged."""

from datetime import datetime

import pytest

from besser.BUML.metamodel.traceability import (
    LLM, AIEvidence, Agent, ChangeEvent, ChangeKind, CodeArtifact, CodeGenerator, Confidence, Editor,
    LogRecord, LogType, ModelArtifact, ModelGenerator, ModelingSession, CodeGeneration, Conversation,
    ExecutionStatus, Person, PromptArtifact,
    RequirementArtifact, RequirementExtraction, RequirementKind, RuleEvidence, ToolCall, TraceLink,
    TraceModel,
)
from besser.utilities.buml_code_builder.trace_model_builder import trace_model_to_code
from besser.utilities.web_modeling_editor.backend.constants.constants import MAX_BUML_SIZE
from besser.utilities.web_modeling_editor.backend.services.converters import parse_trace_code, process_trace, trace_buml_to_json
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


def _at(minute: int) -> datetime:
    return datetime(2026, 10, 29, 9, minute, 0, 123000)


@pytest.fixture
def trace():
    """A trace with every element kind, a fragment listed before its file and awkward strings."""
    owner = Person('Ada "the first" Lovelace', timestamp=_at(0))
    assistant = Agent("BESSER modeling assistant", version="2.1", timestamp=_at(0))
    editor = Editor("BESSER Web Modeling Editor", timestamp=_at(0))
    generator = CodeGenerator("sqlalchemy", "8.5.0", timestamp=_at(0))
    llm = LLM("gpt-x", provider="openai", timestamp=_at(0))

    model = ModelArtifact("Class Diagram", "diagram-1", "DomainModel", timestamp=_at(1))
    book = ModelArtifact("Book", "class-1", "Class", parent=model, timestamp=_at(1))
    content = 'class Book(Base):\n    title = "it\'s \\ here"\n'
    file = CodeArtifact("sql_alchemy.py", "sql_alchemy.py", "a" * 64, content=content, timestamp=_at(5))
    lines = CodeArtifact("Book (1-2)", "sql_alchemy.py", "b" * 64, start_line=1, end_line=2, parent=file,
                         timestamp=_at(5))

    rename = ChangeEvent(ChangeKind.UPDATED, book, "name", "Libro", "Book", timestamp=_at(2))
    created = ChangeEvent(ChangeKind.CREATED, book, timestamp=_at(1))
    log = LogRecord("rename Libro to Book", 0, LogType.INPUT_REQUEST, timestamp=_at(2))
    modeling = ModelingSession([owner, assistant], changes=[created, rename], started_at=_at(1), ended_at=_at(3),
                         tool_calls=[ToolCall(editor, 0, started_at=_at(1), ended_at=_at(3)),
                                     ToolCall(llm, 1, status=ExecutionStatus.FAILED, started_at=_at(2), logs=[log],
                                              input_tokens=1200, output_tokens=340)],
                         reads=[model])
    generation = CodeGeneration([owner], status=ExecutionStatus.SUCCEEDED, started_at=_at(4),
                                tool_calls=[ToolCall(generator, 0, started_at=_at(4), ended_at=_at(5))],
                                reads=[model, book], writes=[file, lines],
                                confidence=Confidence(0.9, explanation="every element traced", timestamp=_at(5)))

    rule = RuleEvidence("sql_alchemy_template.py.j2#class", timestamp=_at(5))
    file_rule = RuleEvidence("generator:sqlalchemy", logs=[log], timestamp=_at(5))  # rule evidence with a log
    ai = AIEvidence("embedding similarity", logs=[log], timestamp=_at(5))
    links = [
        TraceLink(book, lines, generation, [rule], Confidence(1.0, timestamp=_at(5)), timestamp=_at(5)),
        TraceLink(model, file, generation, [file_rule, ai],
                  Confidence(0.75, explanation="multi\nline", timestamp=_at(5)), timestamp=_at(5)),
    ]
    prompt = PromptArtifact("Library", 'Books have a "title".\nAnd pages.', "c" * 64, timestamp=_at(0))
    answer = LogRecord("Noted: Book(title, pages).", 0, LogType.OUTPUT_RESPONSE, timestamp=_at(0))
    requirement = RequirementArtifact('A book has a "title"', 'Books have a "title".', RequirementKind.DATA,
                                      timestamp=_at(0))
    chat = Conversation([owner, assistant], started_at=_at(0), reads=[prompt],
                        tool_calls=[ToolCall(llm, 0, started_at=_at(0), logs=[answer])])
    extraction = RequirementExtraction([assistant], started_at=_at(0), reads=[prompt], writes=[requirement],
                                       tool_calls=[ToolCall(llm, 0, started_at=_at(0))])
    links.append(TraceLink(prompt, requirement, extraction,
                           [RuleEvidence("quote found in the prompt", timestamp=_at(0))], timestamp=_at(0)))
    links.append(TraceLink(requirement, book, modeling, [AIEvidence("llm", logs=[answer], timestamp=_at(1))],
                           timestamp=_at(1)))
    gui_generator = ModelGenerator("BESSER GUI generator", timestamp=_at(6))
    gui = ModelArtifact("Library GUI", "gui-1", "GUIModel", timestamp=_at(6))
    page = ModelArtifact("Book", "page-1", "Screen", parent=gui, timestamp=_at(6))
    derivation = ModelingSession([owner], started_at=_at(6), reads=[book],
                                 changes=[ChangeEvent(ChangeKind.CREATED, page, timestamp=_at(6))],
                                 tool_calls=[ToolCall(gui_generator, 0, started_at=_at(6))])
    links.append(TraceLink(book, page, derivation, [RuleEvidence("GUI element bound to the model element",
                                                                 timestamp=_at(6))], timestamp=_at(6)))
    # The generated file edited by an LLM: a new version, listed before the one it revises.
    edited = CodeArtifact("sql_alchemy.py", "sql_alchemy.py", "d" * 64, revision_of=file, timestamp=_at(7))
    edit = CodeGeneration([assistant], started_at=_at(7), reads=[file], writes=[edited],
                          tool_calls=[ToolCall(llm, 0, started_at=_at(7))])
    links.append(TraceLink(requirement, edited, edit, [AIEvidence("llm judge", timestamp=_at(7))],
                           Confidence(0.8, explanation="realized by Book", timestamp=_at(7)), timestamp=_at(7)))
    return TraceModel(project_id="project-1", actors=[owner, assistant], tools=[editor, generator, llm, gui_generator],
                      artifacts=[edited, lines, book, model, file, prompt, requirement, gui, page],
                      operations=[chat, extraction, modeling, generation, derivation, edit], trace_links=links)


def _code(trace, tmp_path) -> str:
    path = tmp_path / "trace.py"
    trace_model_to_code(trace, str(path))
    return path.read_text(encoding="utf-8")


def _without_contents(data: dict) -> dict:
    """Trace file JSON as the B-UML form keeps it: a file's content is not written (its hash is)."""
    return {**data, "artifacts": [{k: v for k, v in a.items() if not (k == "content" and a["type"] == "CodeArtifact")}
                                   for a in data["artifacts"]]}


def test_code_loads_back_as_the_same_trace_without_file_contents(trace, tmp_path):
    code = _code(trace, tmp_path)
    assert "class Book(Base)" not in code
    loaded = parse_trace_code(code)
    assert trace_buml_to_json(loaded) == _without_contents(trace_buml_to_json(trace))
    assert loaded.validate(raise_exception=False)["success"]


def test_code_runs_as_a_plain_script(trace, tmp_path):
    namespace = {}
    exec(compile(_code(trace, tmp_path), "trace.py", "exec"), namespace)  # noqa: S102 -- our own builder output
    assert trace_buml_to_json(namespace["trace_model"]) == _without_contents(trace_buml_to_json(trace))


def test_code_with_a_call_outside_the_metamodel_is_refused(trace, tmp_path):
    code = _code(trace, tmp_path) + '\nopen("x", "w")\n'
    with pytest.raises(ConversionError, match="traceability section"):
        parse_trace_code(code)


def test_code_without_a_trace_is_refused():
    with pytest.raises(ConversionError, match="one TraceModel, found 0"):
        parse_trace_code('owner = Person("Ada")\n')


def test_json_form_round_trips_every_element_kind(trace):
    """Tool calls with their logs, and evidence of each kind resting on log records, survive the JSON form."""
    data = trace_buml_to_json(trace)
    assert trace_buml_to_json(process_trace(data)) == data
    generation = next(op for op in data["operations"] if op["type"] == "CodeGeneration")
    assert generation["confidence"]["value"] == 0.9  # an operation's confidence
    assert "confidence" not in next(op for op in data["operations"] if op["type"] == "Conversation")  # not assessed
    rule_with_log = next(e for link in data["traceLinks"] for e in link["evidence"] if e["type"] == "RuleEvidence" and "logIds" in e)
    assert rule_with_log["logIds"] and "logIds" not in data["traceLinks"][0]["evidence"][0]  # left out when empty


def test_a_revision_keeps_the_version_it_revises(trace, tmp_path):
    data = trace_buml_to_json(trace)
    edited = next(a for a in data["artifacts"] if a.get("contentHash") == "d" * 64)
    revised = next(a for a in data["artifacts"] if a["id"] == edited["revisionOfId"])
    assert revised["contentHash"] == "a" * 64
    assert "revisionOfId" not in revised  # left out when unset, as optional fields are
    loaded = parse_trace_code(_code(trace, tmp_path))
    assert next(a for a in loaded.artifacts if a.id == edited["id"]).revision_of.id == revised["id"]


def test_a_trace_section_as_large_as_a_project_file_loads(tmp_path):
    # The section's caps are those of the file it is part of (MAX_BUML_SIZE), not the loader's 1 MB.
    files = [CodeArtifact(f"f{i}.py", f"src/module_{i}/f{i}.py", "a" * 64, timestamp=_at(1)) for i in range(6000)]
    code = _code(TraceModel(artifacts=files), tmp_path)
    assert 1_000_000 < len(code) <= MAX_BUML_SIZE
    assert len(parse_trace_code(code).artifacts) == 6000


def test_a_long_revision_chain_is_written_and_loaded_back(tmp_path):
    # Writing it once recursed per version (RecursionError at 1500); loading set each link with a walk.
    versions = [CodeArtifact("f.py", "f.py", "a" * 64, timestamp=_at(1))]
    for _ in range(3000):
        versions.append(CodeArtifact("f.py", "f.py", "a" * 64, revision_of=versions[-1], timestamp=_at(1)))
    loaded = parse_trace_code(_code(TraceModel(artifacts=list(reversed(versions))), tmp_path))
    assert len(loaded.artifacts) == 3001


@pytest.mark.parametrize("names", [("Person", "Agent"), ("TraceModel", "TracePerson")])
def test_classes_named_like_trace_metaclasses_keep_their_trace(tmp_path, names):
    """A model class named like a traceability metaclass (or like what the trace section imports it as)
    must not shadow it: the project file loads with its trace and runs as a plain script."""
    from besser.BUML.metamodel.project import Metadata, Project
    from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json, project_trace_to_json,
    )

    classes = []
    for index, name in enumerate(names, start=1):
        attribute = Property(name="name", type=StringType)
        attribute.id = f"attr-{index}"
        cls = Class(name=name, attributes={attribute})
        cls.id = f"class-{index}"
        classes.append(cls)
    domain = DomainModel(name="Library", types=set(classes))
    domain.id = "diagram-1"
    project = Project(name="P", models=[domain], owner="me", metadata=Metadata(description="d"))

    owner = Person("BESSER User", timestamp=_at(0))
    generator = CodeGenerator("python", timestamp=_at(0))
    model = ModelArtifact("Library", "diagram-1", "DomainModel", timestamp=_at(0))
    artifacts = [ModelArtifact(cls.name, cls.id, "Class", parent=model, timestamp=_at(0)) for cls in classes]
    file = CodeArtifact("classes.py", "classes.py", "a" * 64, timestamp=_at(0))
    generation = CodeGeneration([owner], started_at=_at(0), tool_calls=[ToolCall(generator, 0, started_at=_at(0))],
                                reads=[model], writes=[file])
    links = [TraceLink(a, file, generation, [RuleEvidence("r", timestamp=_at(0))], timestamp=_at(0))
             for a in artifacts]
    trace = TraceModel(project_id="p1", actors=[owner], tools=[generator], artifacts=[model, *artifacts, file],
                       operations=[generation], trace_links=links)

    path = tmp_path / "project.py"
    project_to_code(project, str(path), emit_ids=True, trace=trace)
    source = path.read_text(encoding="utf-8")

    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 -- our own builder output
    loaded = namespace["trace_model"]
    assert {a.name: type(a.element).__name__ for a in loaded.artifacts
            if isinstance(a, ModelArtifact) and a.parent is not None} == dict.fromkeys(names, "Class")

    imported = project_to_json(source)
    trace_json = project_trace_to_json(source, imported)
    assert {a["name"] for a in trace_json["artifacts"] if a["type"] == "ModelArtifact"} == {"Library", *names}
    assert len(trace_json["traceLinks"]) == 2


def test_a_trace_written_with_the_metaclass_names_still_loads():
    """A trace section exported before the metaclasses were imported under aliases calls them by name."""
    loaded = parse_trace_code(
        "import datetime\n"
        "from besser.BUML.metamodel.traceability import (\n    TraceModel, Person,\n)\n"
        'actor_1 = Person("Ada", id="p1", timestamp=datetime.datetime.fromisoformat("2026-10-29T09:00:00"))\n'
        "trace_model = TraceModel(project_id=None, actors=[actor_1], tools=[], artifacts=[], operations=[],"
        " trace_links=[])\n")
    assert [a.name for a in loaded.actors] == ["Ada"]
