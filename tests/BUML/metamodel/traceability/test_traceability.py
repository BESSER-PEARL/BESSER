"""Tests for the traceability metamodel (besser.BUML.metamodel.traceability)."""

from datetime import datetime, timedelta, timezone

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.BUML.metamodel.traceability import (
    LLM, Agent, Change, ChangeEvidence, ChangeKind, CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ModelArtifact,
    Conversation, Modeling, Operation, Person, PromptArtifact, RuleEvidence, TraceLink, TraceModel, Transformation,
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
    generation = CodeGeneration([owner], started_at=T0, uses=[generator], consumes=[book], produces=[file, fragment])
    evidence = RuleEvidence("sql_alchemy_template.py.j2#class", timestamp=T0)
    link = TraceLink(book, fragment, generation, [evidence], Confidence(1.0, timestamp=T0), timestamp=T0)
    trace = TraceModel(project_id="p1", artifacts=[book, title, file, fragment], operations=[generation],
                       trace_links=[link], actors=[owner], tools=[generator])
    return trace, link, owner, book, title


def _edit(trace, owner, artifact, when):
    session = Modeling([owner], started_at=when)
    session.add_change(Change(ChangeKind.UPDATED, artifact, "name", "old", "new", timestamp=when))
    trace.add_operation(session)


def test_valid_trace_validates(generation_trace):
    trace, *_ = generation_trace
    assert trace.validate() == {"success": True, "errors": [], "warnings": []}


def test_names_are_free_text():
    # NamedElement rejects spaces; a project owner is free text.
    assert Person("BESSER User").name == "BESSER User"


def test_link_is_suspect_only_after_a_later_change_of_its_source(generation_trace):
    trace, link, owner, book, _ = generation_trace
    _edit(trace, owner, book, T0 - timedelta(minutes=5))
    assert not trace.is_suspect(link)
    _edit(trace, owner, book, T0 + timedelta(minutes=5))
    assert trace.is_suspect(link)


def test_change_of_a_fragment_makes_the_link_suspect(generation_trace):
    trace, link, owner, _, title = generation_trace
    _edit(trace, owner, title, T0 + timedelta(seconds=1))
    assert trace.is_suspect(link)


def test_updated_change_needs_a_feature_and_others_have_none():
    book = ModelArtifact("Book", "c1", "Class")
    with pytest.raises(ValueError):
        Change(ChangeKind.UPDATED, book)
    with pytest.raises(ValueError):
        Change(ChangeKind.CREATED, book, feature="name")
    assert Change(ChangeKind.CREATED, book, new_value="Book").feature is None


def test_operation_needs_an_actor_and_ends_after_it_starts():
    with pytest.raises(ValueError):
        Modeling([])
    with pytest.raises(ValueError):
        Modeling([Person("a")], started_at=T0, ended_at=T0 - timedelta(1))


def test_aware_timestamps_are_stored_as_naive_utc():
    when = datetime(2026, 10, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    operation = Modeling([Person("a")], started_at=when, ended_at=when)
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


def test_change_evidence_needs_a_change():
    with pytest.raises(ValueError):
        ChangeEvidence([])


def test_validate_reports_references_outside_the_trace(generation_trace):
    trace, link, owner, book, _ = generation_trace
    stray = ModelArtifact("Stray", "s", "Class")
    trace.add_operation(Modeling([Person("ghost")], consumes=[stray]))
    result = trace.validate(raise_exception=False)
    assert not result["success"]
    assert any("actor" in e for e in result["errors"])
    assert any(stray.id in e for e in result["errors"])
    with pytest.raises(ValueError):
        trace.validate()


def test_validate_reports_an_artifact_produced_twice(generation_trace):
    trace, _, owner, book, _ = generation_trace
    trace.add_operation(CodeGeneration([owner], produces=[book]))
    trace.operations[0].produces = trace.operations[0].produces + [book]
    errors = trace.validate(raise_exception=False)["errors"]
    assert any("produced by both" in e for e in errors)


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
    session = Modeling([Person("Ada"), Agent("BESSER modeling assistant")])
    assert {type(actor).__name__ for actor in session.performed_by} == {"Person", "Agent"}


def test_validate_checks_each_operation_kind(generation_trace):
    trace, link, owner, book, _ = generation_trace
    trace.add_operation(Modeling([owner], started_at=T0))
    trace.add_operation(CodeGeneration([owner], started_at=T0))
    trace.add_operation(Conversation([owner], started_at=T0))
    errors = trace.validate(raise_exception=False)["errors"]
    assert any("Modeling with no changes" in e for e in errors)
    assert any("uses no CodeGenerator or LLM" in e for e in errors)
    assert any("Conversation with no Agent" in e for e in errors)


def test_a_conversation_may_produce_nothing(generation_trace):
    trace, _, owner, _, _ = generation_trace
    assistant = trace.add_actor(Agent("BESSER modeling assistant"))
    trace.add_operation(Conversation([owner, assistant], started_at=T0))  # a clarification question
    assert trace.validate(raise_exception=False)["success"]


def test_a_prompt_is_the_source_of_what_was_derived_from_it(generation_trace):
    """Prompt -> requirement (made by the conversation) -> class (made by the agent's modeling)."""
    trace, link, owner, book, _ = generation_trace
    assistant = trace.add_actor(Agent("BESSER modeling assistant"))
    earlier = T0 - timedelta(minutes=10)
    prompt = trace.add_artifact(PromptArtifact("Library prompt", "Books have a title.", "h", timestamp=earlier))
    requirement = trace.add_artifact(ModelArtifact("R1", "req-1", "Requirement", timestamp=earlier))
    chat = trace.add_operation(Conversation([owner, assistant], started_at=earlier, consumes=[prompt],
                                            produces=[requirement]))
    trace.add_trace_link(TraceLink(prompt, requirement, chat, timestamp=earlier))
    edit = trace.add_operation(Modeling([assistant], [Change(ChangeKind.CREATED, book, timestamp=earlier)],
                                        started_at=earlier))
    derived = trace.add_trace_link(TraceLink(requirement, book, edit, timestamp=earlier))
    assert trace.validate(raise_exception=False)["success"]
    _edit(trace, owner, requirement, T0 + timedelta(minutes=1))
    assert trace.is_suspect(derived) and trace.is_stale(link)  # the code comes from a changed requirement


def test_a_prompt_has_content():
    with pytest.raises(ValueError):
        PromptArtifact("empty", "  ", "h")


def test_stale_spreads_downstream_from_a_suspect_link(generation_trace):
    trace, link, owner, book, _ = generation_trace
    requirement = trace.add_artifact(ModelArtifact("R1", "req-1", "Requirement"))
    derivation = Modeling([Agent("assistant")], started_at=T0 - timedelta(minutes=5))
    trace.add_operation(derivation)
    upstream = trace.add_trace_link(TraceLink(requirement, book, derivation, timestamp=T0 - timedelta(minutes=5)))
    assert not trace.is_stale(link)
    _edit(trace, owner, requirement, T0 + timedelta(minutes=1))
    assert trace.is_suspect(upstream) and trace.is_stale(upstream)
    assert not trace.is_suspect(link)  # Book itself did not change...
    assert trace.is_stale(link)  # ...but what it was derived from did


def test_stale_terminates_on_a_cycle(generation_trace):
    trace, link, owner, book, _ = generation_trace
    trace.add_trace_link(TraceLink(link.target, book, trace.operations[0], timestamp=T0))
    assert trace.is_stale(link) is False


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
    trace.add_operation(CodeGeneration([owner, trace.add_actor(Agent("assistant"))], started_at=T0, uses=[llm],
                                       consumes=[book], produces=[code]))
    assert trace.validate(raise_exception=False)["success"]
