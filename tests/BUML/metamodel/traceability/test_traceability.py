"""Tests for the traceability metamodel (besser.BUML.metamodel.traceability)."""

from datetime import datetime, timedelta, timezone

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.BUML.metamodel.traceability import (
    LLM, AIEvidence, Agent, ChangeEvent, ChangeKind, CodeArtifact, CodeGeneration, CodeGenerator, Confidence,
    Editor, ExecutionStatus, LogRecord, LogType, ModelArtifact, ModelGenerator, Conversation,
    ModelingSession, Operation, OutOfDateReason,
    Person, PromptArtifact, RequirementArtifact, RequirementExtraction, RequirementKind, RuleEvidence, ToolCall,
    TraceLink, TraceModel, Transformation,
)

T0 = datetime(2026, 10, 1, 12, 0, 0)


@pytest.fixture
def generation_trace():
    """A class, the code generated from it and the link between them, made at T0."""
    owner = Person("BESSER User")
    generator = CodeGenerator("sqlalchemy", "8.5.0")
    book = ModelArtifact("Book", "class-1", "Class")
    title = ModelArtifact("title", "attr-1", "Property", parent=book)
    file = CodeArtifact("sql_alchemy.py", "sql_alchemy.py", "h0")
    fragment = CodeArtifact("Book (10-20)", "sql_alchemy.py", "h1", start_line=10, end_line=20, parent=file)
    generation = CodeGeneration([owner], started_at=T0, tool_calls=[ToolCall(generator, 0, started_at=T0)],
                                reads=[book], writes=[file, fragment])
    evidence = RuleEvidence("sql_alchemy_template.py.j2#class", timestamp=T0)
    link = TraceLink(book, fragment, generation, [evidence], Confidence(1.0, timestamp=T0), timestamp=T0)
    trace = TraceModel(project_id="p1", artifacts=[book, title, file, fragment], operations=[generation],
                       trace_links=[link], actors=[owner], tools=[generator])
    return trace, link, owner, book, title


def _edit(trace, owner, artifact, when):
    session = ModelingSession([owner], started_at=when)
    session.add_change(ChangeEvent(ChangeKind.UPDATED, artifact, "name", "old", "new", timestamp=when))
    trace.add_operation(session)


def test_valid_trace_validates(generation_trace):
    trace, *_ = generation_trace
    assert trace.validate() == {"success": True, "errors": [], "warnings": []}


def test_names_are_free_text():
    # NamedElement rejects spaces; a project owner is free text.
    assert Person("BESSER User").name == "BESSER User"


SOURCE, TARGET, UPSTREAM = (OutOfDateReason.SOURCE_CHANGED, OutOfDateReason.TARGET_CHANGED,
                            OutOfDateReason.UPSTREAM_CHANGED)


def test_link_is_out_of_date_only_after_a_later_change_of_its_source(generation_trace):
    trace, link, owner, book, _ = generation_trace
    _edit(trace, owner, book, T0 - timedelta(minutes=5))
    assert trace.out_of_date(link) == set()
    _edit(trace, owner, book, T0 + timedelta(minutes=5))
    assert trace.out_of_date(link) == {SOURCE}


def test_change_of_a_fragment_makes_the_link_out_of_date(generation_trace):
    trace, link, owner, _, title = generation_trace
    _edit(trace, owner, title, T0 + timedelta(seconds=1))
    assert trace.out_of_date(link) == {SOURCE}


def test_an_edited_target_makes_the_link_out_of_date_but_not_what_follows(generation_trace):
    """Prompt -> Book (by the assistant's edits) -> code; the user then edits Book."""
    trace, link, owner, book, title = generation_trace
    assistant = trace.add_actor(Agent("BESSER modeling assistant"))
    earlier = T0 - timedelta(minutes=10)
    prompt = trace.add_artifact(PromptArtifact("Add Book", "Add a Book class.", "h", timestamp=earlier))
    edits = trace.add_operation(ModelingSession([owner, assistant], started_at=earlier))
    derived = trace.add_trace_link(TraceLink(prompt, book, edits, timestamp=earlier))
    # The same edits go on after the link (a later reply of the turn): part of making it.
    edits.add_change(ChangeEvent(ChangeKind.CREATED, title, timestamp=earlier + timedelta(seconds=5)))
    assert trace.out_of_date(derived) == set()
    _edit(trace, owner, book, T0 + timedelta(minutes=1))
    assert trace.out_of_date(derived) == {TARGET}
    assert trace.out_of_date(link) == {SOURCE}  # an edited target is not passed on as upstream


def test_updated_change_needs_a_feature_and_others_have_none():
    book = ModelArtifact("Book", "c1", "Class")
    with pytest.raises(ValueError):
        ChangeEvent(ChangeKind.UPDATED, book)
    with pytest.raises(ValueError):
        ChangeEvent(ChangeKind.CREATED, book, feature="name")
    assert ChangeEvent(ChangeKind.CREATED, book, new_value="Book").feature is None


def test_operation_needs_an_actor_and_ends_after_it_starts():
    with pytest.raises(ValueError):
        ModelingSession([])
    with pytest.raises(ValueError):
        ModelingSession([Person("a")], started_at=T0, ended_at=T0 - timedelta(1))


def test_aware_timestamps_are_stored_as_naive_utc():
    when = datetime(2026, 10, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    operation = ModelingSession([Person("a")], started_at=when, ended_at=when)
    assert operation.started_at == operation.ended_at == T0


def test_default_timestamp_is_utc():
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    assert Person("a").timestamp >= before


def test_code_line_range_is_checked():
    with pytest.raises(ValueError):
        CodeArtifact("f", "f.py", "h", start_line=5)
    with pytest.raises(ValueError):
        CodeArtifact("f", "f.py", "h", start_line=5, end_line=4)
    with pytest.raises(ValueError):
        CodeArtifact("f", "f.py", "h", start_line=0, end_line=4)


def test_artifact_cannot_be_its_own_ancestor():
    a = ModelArtifact("A", "a", "Class")
    b = ModelArtifact("B", "b", "Property", parent=a)
    with pytest.raises(ValueError):
        a.parent = b


def test_link_cannot_join_an_artifact_to_itself():
    a = ModelArtifact("A", "a", "Class")
    op = CodeGeneration([Person("x")])
    with pytest.raises(ValueError):
        TraceLink(a, a, op)


def test_confidence_is_a_probability():
    with pytest.raises(ValueError):
        Confidence(1.5)


def test_any_evidence_can_rest_on_log_records(generation_trace):
    trace, link, owner, _, _ = generation_trace
    generator = trace.tools[0]
    warning = LogRecord("skipped the methods of Book", 0, LogType.INFO, timestamp=T0)
    trace.operations[0].tool_calls = [ToolCall(generator, 0, started_at=T0, logs=[warning])]
    link.evidence = [RuleEvidence("t#class", logs=[warning], timestamp=T0)]
    assert link.evidence[0].logs == [warning]
    assert trace.validate(raise_exception=False)["success"]
    with pytest.raises(TypeError):
        RuleEvidence("t#class", logs=["not a log record"])


def test_validate_reports_references_outside_the_trace(generation_trace):
    trace, link, owner, book, _ = generation_trace
    stray = ModelArtifact("Stray", "s", "Class")
    trace.add_operation(ModelingSession([Person("ghost")], reads=[stray]))
    result = trace.validate(raise_exception=False)
    assert not result["success"]
    assert any("actor" in e for e in result["errors"])
    assert any(stray.id in e for e in result["errors"])
    with pytest.raises(ValueError):
        trace.validate()


def test_validate_reports_an_artifact_written_twice(generation_trace):
    trace, _, owner, book, _ = generation_trace
    trace.add_operation(CodeGeneration([owner], writes=[book]))
    trace.operations[0].writes = trace.operations[0].writes + [book]
    errors = trace.validate(raise_exception=False)["errors"]
    assert any("written by both" in e for e in errors)


def test_validate_reports_duplicate_ids(generation_trace):
    trace, *_ = generation_trace
    trace.add_actor(Person("twin", id=trace.actors[0].id))
    assert any("used by both" in e for e in trace.validate(raise_exception=False)["errors"])


def test_model_artifact_resolves_to_the_live_element():
    title = Property("title", StringType, id="attr-1")
    book = Class("Book", attributes={title}, id="class-1")
    model = DomainModel("Library", types={book})
    assert ModelArtifact("title", "attr-1", "Property").resolve(model) is title
    assert ModelArtifact("Book", "class-1", "Class").resolve(model) is book
    assert ModelArtifact("Gone", "deleted", "Class").resolve(model) is None


def test_model_artifact_for_finds_by_element_id(generation_trace):
    trace, _, _, book, _ = generation_trace
    assert trace.model_artifact_for("class-1") is book
    assert trace.model_artifact_for("nope") is None


def test_operation_and_transformation_are_abstract():
    for abstract in (Operation, Transformation):
        with pytest.raises(TypeError, match="abstract"):
            abstract([Person("a")])


def test_who_did_the_modeling_is_told_by_its_actors():
    session = ModelingSession([Person("Ada"), Agent("BESSER modeling assistant")])
    assert {type(actor).__name__ for actor in session.performed_by} == {"Person", "Agent"}


def test_validate_checks_each_operation_kind(generation_trace):
    trace, link, owner, book, _ = generation_trace
    trace.add_operation(ModelingSession([owner], started_at=T0))
    trace.add_operation(CodeGeneration([owner], started_at=T0))
    trace.add_operation(Conversation([owner], started_at=T0))
    errors = trace.validate(raise_exception=False)["errors"]
    assert any("ModelingSession with no changes" in e for e in errors)
    assert any("calls no CodeGenerator or LLM" in e for e in errors)
    assert any("Conversation with no Agent" in e for e in errors)


def test_a_conversation_may_write_nothing(generation_trace):
    trace, _, owner, _, _ = generation_trace
    assistant = trace.add_actor(Agent("BESSER modeling assistant"))
    trace.add_operation(Conversation([owner, assistant], started_at=T0))  # a clarification question
    assert trace.validate(raise_exception=False)["success"]


def test_a_prompt_is_the_source_of_what_was_derived_from_it(generation_trace):
    """Prompt -> requirement (made by the extraction) -> class (made by the agent's modeling)."""
    trace, link, owner, book, _ = generation_trace
    assistant = trace.add_actor(Agent("BESSER modeling assistant"))
    llm = trace.add_tool(LLM("gpt-6-luna", "openai"))
    earlier = T0 - timedelta(minutes=10)
    prompt = trace.add_artifact(PromptArtifact("Library prompt", "Books have a title.", "h", timestamp=earlier))
    requirement = trace.add_artifact(RequirementArtifact("A book has a title", "Books have a title",
                                                         RequirementKind.DATA, timestamp=earlier))
    trace.add_operation(Conversation([owner, assistant], started_at=earlier, reads=[prompt],
                                     tool_calls=[ToolCall(llm, 0, started_at=earlier)]))
    extraction = trace.add_operation(RequirementExtraction(
        [assistant], started_at=earlier, tool_calls=[ToolCall(llm, 0, started_at=earlier)],
        reads=[prompt], writes=[requirement]))
    assert isinstance(extraction, Transformation)
    trace.add_trace_link(TraceLink(prompt, requirement, extraction, timestamp=earlier))
    edit = trace.add_operation(ModelingSession([assistant], [ChangeEvent(ChangeKind.CREATED, book, timestamp=earlier)],
                                        started_at=earlier))
    derived = trace.add_trace_link(TraceLink(requirement, book, edit, timestamp=earlier))
    assert trace.validate(raise_exception=False)["success"]
    _edit(trace, owner, requirement, T0 + timedelta(minutes=1))
    assert trace.out_of_date(derived) == {SOURCE}
    assert trace.out_of_date(link) == {UPSTREAM}  # the code comes from a changed requirement


def test_a_prompt_has_content():
    with pytest.raises(ValueError):
        PromptArtifact("empty", "  ", "h")


def test_a_requirement_quotes_the_prompt_and_has_a_kind():
    requirement = RequirementArtifact("Room numbers are unique", "each room is identified by its number",
                                      RequirementKind.UNIQUENESS)
    assert (requirement.name, requirement.kind) == ("Room numbers are unique", RequirementKind.UNIQUENESS)
    with pytest.raises(ValueError):
        RequirementArtifact("Room numbers are unique", "  ", RequirementKind.UNIQUENESS)
    with pytest.raises(TypeError):
        RequirementArtifact("Room numbers are unique", "identified by its number", "uniqueness")


def test_a_changed_source_spreads_downstream_as_upstream_changed(generation_trace):
    trace, link, owner, book, _ = generation_trace
    requirement = trace.add_artifact(ModelArtifact("R1", "req-1", "Requirement"))
    derivation = ModelingSession([Agent("assistant")], started_at=T0 - timedelta(minutes=5))
    trace.add_operation(derivation)
    upstream = trace.add_trace_link(TraceLink(requirement, book, derivation, timestamp=T0 - timedelta(minutes=5)))
    assert trace.out_of_date(link) == set()
    _edit(trace, owner, requirement, T0 + timedelta(minutes=1))
    assert trace.out_of_date(upstream) == {SOURCE}
    assert trace.out_of_date(link) == {UPSTREAM}  # Book itself did not change, but what it came from did


def test_out_of_date_terminates_on_a_cycle(generation_trace):
    trace, link, owner, book, _ = generation_trace
    trace.add_trace_link(TraceLink(link.target, book, trace.operations[0], timestamp=T0))
    assert trace.out_of_date(link) == set()


def test_an_operation_may_have_a_confidence(generation_trace):
    trace, link, *_ = generation_trace
    generation = link.operation
    assert generation.confidence is None  # not assessed
    generation.confidence = Confidence(0.9, explanation="every element traced", timestamp=T0)
    assert generation.confidence.value == 0.9
    assert trace.validate(raise_exception=False)["success"]
    with pytest.raises(TypeError):
        generation.confidence = 0.9  # a Confidence, not a bare number


def test_a_model_artifact_references_its_element_or_records_it():
    library = Class("Library", {Property("name", StringType)})
    model = DomainModel("LibraryModel", types={library})
    referenced = ModelArtifact(element=library)
    assert (referenced.name, referenced.element_type, referenced.element_id) == ("Library", "Class", None)
    assert referenced.resolve(model) is library
    library.name = "Catalogue"
    assert referenced.name == "Catalogue"  # the element's own name, not a copy
    referenced.element_id = "imported-id"  # given by the import, which knows the new id
    assert referenced.element_id == "imported-id"
    recorded = ModelArtifact("Ghost", "gone", "Class")  # deleted: only the record remains
    assert recorded.element is None and recorded.resolve(model) is None
    with pytest.raises(ValueError):
        ModelArtifact("Ghost", None, "Class")  # a record needs its id


def test_code_can_be_generated_by_an_llm(generation_trace):
    trace, _, owner, book, _ = generation_trace
    llm = trace.add_tool(LLM("claude", provider="anthropic"))
    code = trace.add_artifact(CodeArtifact("app.py", "app.py", "h2"))
    trace.add_operation(CodeGeneration([owner, trace.add_actor(Agent("assistant"))], started_at=T0,
                                       tool_calls=[ToolCall(llm, 0, started_at=T0)],
                                       reads=[book], writes=[code]))
    assert trace.validate(raise_exception=False)["success"]


def test_an_operation_reaches_its_tools_through_its_calls():
    owner, editor, llm = Person("Ada"), Editor("BESSER Web Modeling Editor"), LLM("claude", provider="anthropic")
    prompt = LogRecord("Add a title to Book", 0, LogType.INPUT_REQUEST, timestamp=T0)
    answer = LogRecord("Done: Book.title added.", 1, LogType.OUTPUT_RESPONSE, timestamp=T0)
    turns = [ToolCall(llm, 0, started_at=T0, logs=[prompt, answer]), ToolCall(editor, 1, started_at=T0),
             ToolCall(llm, 2, status=ExecutionStatus.FAILED, started_at=T0)]
    chat = Conversation([owner, Agent("assistant")], started_at=T0, tool_calls=turns)
    assert chat.tools == [llm, editor]  # derived: distinct, in call order
    assert turns[0].logs == [prompt, answer] and turns[2].status is ExecutionStatus.FAILED
    with pytest.raises(ValueError):
        ToolCall(llm, 0, started_at=T0, ended_at=T0 - timedelta(seconds=1))
    with pytest.raises(TypeError):
        ToolCall("claude", 0)


def test_a_call_to_an_llm_can_record_its_tokens():
    llm, editor = LLM("gpt-6-luna", provider="openai"), Editor("BESSER Web Modeling Editor")
    call = ToolCall(llm, 0, started_at=T0, input_tokens=1200, output_tokens=340)
    assert (call.input_tokens, call.output_tokens) == (1200, 340)
    assert ToolCall(llm, 1, started_at=T0).input_tokens is None  # not reported
    with pytest.raises(ValueError):
        ToolCall(editor, 0, input_tokens=10)  # only an LLM call records tokens
    with pytest.raises(ValueError):
        call.output_tokens = -1
    with pytest.raises(ValueError):
        call.tool = editor  # a call with tokens stays a call to an LLM


def test_validation_checks_tool_calls(generation_trace):
    trace, link, owner, book, _ = generation_trace
    stray = LLM("claude", provider="anthropic")  # never added to the trace's tools
    trace.add_operation(Conversation([owner, trace.add_actor(Agent("assistant"))], started_at=T0,
                                     tool_calls=[ToolCall(stray, 0, started_at=T0), ToolCall(stray, 0, started_at=T0)]))
    elsewhere = LogRecord("not in any call of the trace", 0, LogType.OUTPUT_RESPONSE)
    link.evidence = link.evidence + [AIEvidence("llm", logs=[elsewhere])]
    errors = "\n".join(trace.validate(raise_exception=False)["errors"])
    assert "references tool" in errors
    assert "sharing a sequence number" in errors
    assert "references log record" in errors


def test_a_requirement_extraction_calls_an_llm_and_writes_only_requirements(generation_trace):
    trace, _, owner, book, _ = generation_trace
    editor = trace.add_tool(Editor("BESSER Web Modeling Editor"))
    trace.add_operation(RequirementExtraction([owner], started_at=T0, tool_calls=[ToolCall(editor, 0, started_at=T0)],
                                              writes=[book]))
    errors = "\n".join(trace.validate(raise_exception=False)["errors"])
    assert "RequirementExtraction that calls no LLM" in errors
    assert f"RequirementExtraction that writes ModelArtifact {book.id!r}" in errors


def test_a_gui_derived_from_a_class_diagram_is_a_session_of_the_model_generator(generation_trace):
    trace, _, owner, book, _ = generation_trace
    generator = trace.add_tool(ModelGenerator("BESSER GUI generator"))
    gui = trace.add_artifact(ModelArtifact("Library GUI", "gui-1", "GUIModel", timestamp=T0))
    page = trace.add_artifact(ModelArtifact("Book", "page-1", "Screen", parent=gui, timestamp=T0))
    derivation = trace.add_operation(ModelingSession(
        [owner], started_at=T0, tool_calls=[ToolCall(generator, 0, started_at=T0)], reads=[book],
        changes=[ChangeEvent(ChangeKind.CREATED, page, timestamp=T0)]))
    trace.add_trace_link(TraceLink(book, page, derivation, [RuleEvidence("GUI element bound to the model element",
                                                                         timestamp=T0)], timestamp=T0))
    assert trace.validate(raise_exception=False)["errors"] == []


def test_a_code_generation_calls_a_generator_or_an_llm(generation_trace):
    trace, _, owner, book, _ = generation_trace
    editor = trace.add_tool(Editor("BESSER Web Modeling Editor"))
    trace.add_operation(CodeGeneration([owner], started_at=T0, tool_calls=[ToolCall(editor, 0, started_at=T0)]))
    assert "calls no CodeGenerator or LLM" in "\n".join(trace.validate(raise_exception=False)["errors"])
