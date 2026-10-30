"""``trace_model_to_code`` writes a TraceModel as Python that ``parse_trace_code`` loads back unchanged."""

from datetime import datetime

import pytest

from besser.BUML.metamodel.traceability import (
    LLM, AIEvidence, Agent, Change, ChangeEvidence, ChangeKind, CodeArtifact, CodeGenerator, Confidence, Editor,
    LogRecord, LogType, ModelArtifact, Modeling, CodeGeneration, Conversation, OperationStatus, Person, PromptArtifact,
    RuleEvidence, TraceLink,
    TraceModel,
)
from besser.utilities.buml_code_builder.trace_model_builder import trace_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import parse_trace_code, trace_buml_to_json
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

    rename = Change(ChangeKind.UPDATED, book, "name", "Libro", "Book", timestamp=_at(2))
    created = Change(ChangeKind.CREATED, book, timestamp=_at(1))
    log = LogRecord("rename Libro to Book", 1, LogType.INPUT_REQUEST, tool=llm, timestamp=_at(2))
    modeling = Modeling([owner, assistant], changes=[created, rename], started_at=_at(1), ended_at=_at(3),
                         uses=[editor, llm], consumes=[model], logs=[log])
    generation = CodeGeneration([owner], status=OperationStatus.SUCCEEDED, started_at=_at(4),
                                uses=[generator], consumes=[model, book], produces=[file, lines])

    rule = RuleEvidence("sql_alchemy_template.py.j2#class", timestamp=_at(5))
    file_rule = RuleEvidence("generator:sqlalchemy", timestamp=_at(5))
    ai = AIEvidence("embedding similarity", logs=[log], timestamp=_at(5))
    from_changes = ChangeEvidence([rename], timestamp=_at(5))
    links = [
        TraceLink(book, lines, generation, [rule], Confidence(1.0, timestamp=_at(5)), timestamp=_at(5)),
        TraceLink(model, file, generation, [file_rule, ai, from_changes],
                  Confidence(0.75, explanation="multi\nline", timestamp=_at(5)), timestamp=_at(5)),
    ]
    prompt = PromptArtifact("Library", 'Books have a "title".\nAnd pages.', "c" * 64, timestamp=_at(0))
    answer = LogRecord("Noted: Book(title, pages).", 2, LogType.OUTPUT_RESPONSE, tool=llm, timestamp=_at(0))
    chat = Conversation([owner, assistant], started_at=_at(0), uses=[llm], consumes=[prompt], logs=[answer])
    links.append(TraceLink(prompt, book, modeling, [AIEvidence("llm", logs=[answer], timestamp=_at(1))],
                           timestamp=_at(1)))
    return TraceModel(project_id="project-1", actors=[owner, assistant], tools=[editor, generator, llm],
                      artifacts=[lines, book, model, file, prompt], operations=[chat, modeling, generation],
                      trace_links=links)


def _code(trace, tmp_path) -> str:
    path = tmp_path / "trace.py"
    trace_model_to_code(trace, str(path))
    return path.read_text(encoding="utf-8")


def test_code_loads_back_as_the_same_trace(trace, tmp_path):
    loaded = parse_trace_code(_code(trace, tmp_path))
    assert trace_buml_to_json(loaded) == trace_buml_to_json(trace)
    assert loaded.validate(raise_exception=False)["success"]


def test_code_runs_as_a_plain_script(trace, tmp_path):
    namespace = {}
    exec(compile(_code(trace, tmp_path), "trace.py", "exec"), namespace)  # noqa: S102 -- our own builder output
    assert trace_buml_to_json(namespace["trace_model"]) == trace_buml_to_json(trace)


def test_code_with_a_call_outside_the_metamodel_is_refused(trace, tmp_path):
    code = _code(trace, tmp_path) + '\nopen("x", "w")\n'
    with pytest.raises(ConversionError, match="traceability section"):
        parse_trace_code(code)


def test_code_without_a_trace_is_refused():
    with pytest.raises(ConversionError, match="one TraceModel, found 0"):
        parse_trace_code('owner = Person("Ada")\n')
