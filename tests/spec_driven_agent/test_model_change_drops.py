"""A model rule the user's own model change broke is dropped, not recovered.

#659: the user removed Person.phone but the model kept
``validPhone: self.phone.matches(...)``. The converter rejected the rule, the
ledger turned it into "Recover model constraint 'validPhone' ... resolve
conflicts in favor of the original user specification", and the agent put
``phone`` back in 5 of 6 live runs (once as a required field, which broke
5 of 6 create requests).

A rule is dropped only when the same rule text converted in the previous run
and the element it now misses was removed from the previous model. New or
edited rules, typos and renames keep the normal recovery path.
"""

from __future__ import annotations

import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.agent.prompt_builder import DROPPED_RULES_NOTE, build_system_prompt
from besser.spec_driven_agent.model_serializer import serialize_domain_model
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.gap_analyzer import _note_rejected_constraints
from besser.spec_driven_agent.planning.model_change import model_summary
from besser.spec_driven_agent.planning.model_rules import converted_rule_fingerprints
from besser.utilities.web_modeling_editor.backend.services.converters.parsers.ocl_parser import (
    process_ocl_constraints,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    build_verification_report,
)

PAGES_RULE = "context Book inv positivePages: self.pages > 0"
STR, INT = PrimitiveDataType("str"), PrimitiveDataType("int")


class _Usage:
    estimated_cost = 0.0

    def summary(self):
        return {}


class _Client:
    """No LLM calls happen in these tests; the orchestrator only reads these."""
    model = "test-model"
    usage = _Usage()
    max_tokens = 4096


def _model(book=("title:str", "pages:int"), book_name="Book", author_end="author",
           parent_attrs=None) -> DomainModel:
    """Book (optionally under a Publication parent) and Author, with stable ids."""
    types = {"str": STR, "int": INT}
    cls = Class(name=book_name)
    cls.id = "cls-book"
    cls.attributes = {Property(name=n, type=types[t]) for n, t in (a.split(":") for a in book)}
    for attr in cls.attributes:
        attr.id = f"attr-{attr.name}"
    author = Class(name="Author")
    author.id = "cls-author"
    author.attributes = {Property(name="name", type=STR)}
    classes, generalizations, associations = {cls, author}, set(), set()
    if parent_attrs is not None:
        parent = Class(name="Publication")
        parent.attributes = {Property(name=n, type=types[t]) for n, t in (a.split(":") for a in parent_attrs)}
        classes.add(parent)
        generalizations.add(Generalization(general=parent, specific=cls))
    if author_end:
        association = BinaryAssociation(name="writes", ends={
            Property(name="books", type=cls, multiplicity=Multiplicity(0, "*")),
            Property(name=author_end, type=author, multiplicity=Multiplicity(0, 1)),
        })
        association.id = "assoc-writes"
        associations.add(association)
    return DomainModel(name="Library", types=classes, associations=associations,
                       generalizations=generalizations)


def _with_rules(model, *rules) -> DomainModel:
    """Convert ``rules`` into ``model`` the way the converter does; rejections land in conversion_issues."""
    issues: list[dict] = []
    for index, rule in enumerate(rules, start=1):
        for _, constraint, _, _ in process_ocl_constraints(rule, model, index, issues=issues)[0]:
            model.constraints = set(model.constraints) | {constraint}
    for issue in issues:
        issue.update(id=f"ocl-{issue['name']}", source={"diagram_title": "Class Diagram"})
    model.conversion_issues = issues
    return model


def _seed(tmp_path, previous: DomainModel | None, extra_issues=(), name="app") -> str:
    """A finished run's workspace whose recipe records ``previous`` (None: an older recipe)."""
    ws = tmp_path / name
    ws.mkdir(parents=True)
    recipe = {"generator_used": "generate_fastapi_backend", "output_files": [],
              "model": {"classes": [c.name for c in previous.get_classes()] if previous else ["Book"]}}
    if previous is not None:
        recipe["model"].update(user_model=model_summary(previous),
                               converted_rules=converted_rule_fingerprints(previous))
        recipe["model_conversion_issues"] = list(previous.conversion_issues) + list(extra_issues)
    (ws / ".besser_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return str(ws)


def _modify(ws, model, monkeypatch) -> LLMOrchestrator:
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=model, output_dir=ws,
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    for name in ("_create_snapshot", "_run_phase3_validation", "_remove_snapshot"):
        monkeypatch.setattr(orch, name, lambda *a, **k: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)
    orch.modify("update the app to the new model")
    return orch


def _recover_texts(orch) -> list[str]:
    return [r["text"] for r in orch._requirements_for_validation() if r.get("conversion_issue_id")]


def _run(tmp_path, monkeypatch, previous, current):
    orch = _modify(_seed(tmp_path, previous), current, monkeypatch)
    [issue] = current.conversion_issues
    return orch, issue


# ------------------------------------------------------------- dropped


def test_rule_on_a_removed_attribute_is_dropped(tmp_path, monkeypatch):
    previous = _with_rules(_model(), PAGES_RULE)
    current = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    orch, issue = _run(tmp_path, monkeypatch, previous, current)

    assert issue["dropped_by_model_change"] == ["Book.pages"]
    assert _recover_texts(orch) == []

    # Listed as dropped, without the expression the agent used to re-implement.
    serialized = serialize_domain_model(current)
    assert "conversion_issues" not in serialized
    assert serialized["dropped_by_model_change"] == [
        {"name": "positivePages", "context": "Book", "missing": ["Book.pages"]}]
    prompt = build_system_prompt(current, None, None, "", "update the app", 10, modify_mode=True)
    assert DROPPED_RULES_NOTE in prompt
    assert "self.pages" not in prompt
    assert "Where the model disagrees with your specification" not in prompt

    assert [t for t in _note_rejected_constraints(current, []) if "positivePages" in t] == []

    report = build_verification_report({"model_conversion_issues": current.conversion_issues})
    [item] = report.shippedUnenforced
    assert "dropped by your model change" in item.why and "Book.pages" in item.why


def test_rule_on_a_removed_class_is_dropped(tmp_path, monkeypatch):
    rule = "context Author inv named: self.name <> ''"
    previous = _with_rules(_model(), rule)
    current = _model(author_end=None)
    current.types = {t for t in current.types if getattr(t, "name", "") != "Author"}
    _with_rules(current, rule)
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert issue["dropped_by_model_change"] == ["Author"]
    assert _recover_texts(orch) == []


def test_rule_on_a_removed_association_end_is_dropped(tmp_path, monkeypatch):
    rule = "context Book inv hasAuthor: self.author->notEmpty()"
    previous = _with_rules(_model(), rule)
    current = _with_rules(_model(author_end=None), rule)
    _, issue = _run(tmp_path, monkeypatch, previous, current)
    assert issue["dropped_by_model_change"] == ["Book.author"]


def test_rule_on_an_attribute_removed_from_the_parent_is_dropped(tmp_path, monkeypatch):
    rule = "context Book inv hasLabel: self.label <> ''"
    previous = _with_rules(_model(parent_attrs=("label:str",)), rule)
    current = _with_rules(_model(parent_attrs=()), rule)
    _, issue = _run(tmp_path, monkeypatch, previous, current)
    assert issue["dropped_by_model_change"] == ["Publication.label"]


def test_rule_dropped_last_run_stays_dropped(tmp_path, monkeypatch):
    # Run 2 dropped it (it did not convert there); run 3 must not recover it.
    rule = "context Book inv hasLabel: self.label <> ''"
    run2 = _with_rules(_model(parent_attrs=()), rule)
    run2.conversion_issues[0]["dropped_by_model_change"] = ["Publication.label"]
    current = _with_rules(_model(parent_attrs=()), rule)
    orch, issue = _run(tmp_path, monkeypatch, run2, current)
    assert issue["dropped_by_model_change"] == ["Publication.label"]
    assert _recover_texts(orch) == []


# ------------------------------------------------------------- recovered as today


def test_a_new_rule_with_a_typo_is_recovered(tmp_path, monkeypatch):
    # The previous run had no rule; nothing was removed.
    for rule in ("context Book inv hasTitle: self.ttle <> ''", "context Book inv hasTitle: self.Title <> ''"):
        current = _with_rules(_model(), rule)
        orch, issue = _run(tmp_path / rule[-12:-6], monkeypatch, _with_rules(_model()), current)
        assert "dropped_by_model_change" not in issue
        assert any("'hasTitle'" in t for t in _recover_texts(orch))


def test_a_new_rule_on_an_attribute_not_modeled_yet_is_recovered(tmp_path, monkeypatch):
    current = _with_rules(_model(), "context Book inv validIsbn: self.isbn.size() = 13")
    orch, issue = _run(tmp_path, monkeypatch, _with_rules(_model()), current)
    assert "dropped_by_model_change" not in issue
    assert any("'validIsbn'" in t for t in _recover_texts(orch))


def test_an_edited_rule_is_recovered(tmp_path, monkeypatch):
    # Same name, new text: the edit, not the model change, broke it.
    previous = _with_rules(_model(), PAGES_RULE)
    current = _with_rules(_model(book=("title:str",)), "context Book inv positivePages: self.pages >= 1")
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert "dropped_by_model_change" not in issue
    assert any("'positivePages'" in t for t in _recover_texts(orch))


def test_a_rule_whose_attribute_was_renamed_follows_the_rename(tmp_path, monkeypatch):
    previous = _with_rules(_model(), PAGES_RULE)
    current = _model(book=("title:str", "pageCount:int"))
    next(a for a in next(c for c in current.get_classes() if c.name == "Book").attributes
         if a.name == "pageCount").id = "attr-pages"
    _with_rules(current, PAGES_RULE)
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert "dropped_by_model_change" not in issue
    assert issue["renamed_in_model"] == {"from": "Book.pages", "to": "Book.pageCount"}
    [text] = _recover_texts(orch)
    assert "renamed Book.pages to Book.pageCount" in text and "keep the rule" in text


def test_a_rule_whose_class_was_renamed_follows_the_rename(tmp_path, monkeypatch):
    previous = _with_rules(_model(), PAGES_RULE)
    current = _with_rules(_model(book_name="Volume"), PAGES_RULE)
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert "dropped_by_model_change" not in issue
    assert issue["renamed_in_model"] == {"from": "Book", "to": "Volume"}
    assert any("renamed Book to Volume" in t for t in _recover_texts(orch))


def test_a_class_missing_without_a_recorded_previous_model_is_recovered(tmp_path, monkeypatch):
    current = _with_rules(_model(), "context Person inv validPhone: self.phone <> ''")
    orch = _modify(_seed(tmp_path, None), current, monkeypatch)
    assert "dropped_by_model_change" not in current.conversion_issues[0]
    assert any("'validPhone'" in t for t in _recover_texts(orch))


def test_a_recipe_with_only_class_names_cannot_prove_a_removal(tmp_path, monkeypatch):
    previous = _with_rules(_model(), PAGES_RULE)
    ws = _seed(tmp_path, previous)
    recipe_path = tmp_path / "app" / ".besser_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    del recipe["model"]["user_model"]  # an older recipe: class names only
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    current = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    orch = _modify(ws, current, monkeypatch)
    assert "dropped_by_model_change" not in current.conversion_issues[0]
    assert any("'positivePages'" in t for t in _recover_texts(orch))


def test_a_rule_that_already_failed_last_run_is_recovered(tmp_path, monkeypatch):
    previous = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    current = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert "dropped_by_model_change" not in issue
    assert any("'positivePages'" in t for t in _recover_texts(orch))


def test_other_conversion_failures_and_fresh_runs_are_recovered(tmp_path, monkeypatch):
    previous = _with_rules(_model(), PAGES_RULE)
    current = _with_rules(_model(book=("title:str",)), "context Book inv broken: self.title >")
    orch, issue = _run(tmp_path, monkeypatch, previous, current)
    assert issue["code"] == "parse_error" and "dropped_by_model_change" not in issue
    assert any("'broken'" in t for t in _recover_texts(orch))

    fresh_model = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    fresh = LLMOrchestrator(
        llm_client=_Client(), domain_model=fresh_model, output_dir=str(tmp_path / "fresh"),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )
    assert any("'positivePages'" in t for t in _recover_texts(fresh))


# ------------------------------------------------------------- plumbing


def test_the_converter_names_the_missing_element():
    model = _model()
    issues: list[dict] = []
    for index, rule in enumerate(("context Book inv a: self.author.nme <> ''",
                                  "context Person inv b: self.phone <> ''"), start=1):
        process_ocl_constraints(rule, model, index, issues=issues)
    assert [(i["code"], i["missing_element"]) for i in issues] == [
        ("parse_error", "Author.nme"), ("parse_error", "Person")]


def test_the_recipe_records_converted_rules_whatever_their_name(tmp_path):
    named = _with_rules(_model(), PAGES_RULE)
    unnamed = _with_rules(_model(), "context Book inv: self.pages > 0")
    assert converted_rule_fingerprints(named) == converted_rule_fingerprints(unnamed)

    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=named, output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )
    orch._save_recipe("build it", 1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["model"]["converted_rules"] == converted_rule_fingerprints(named)


def test_phase3_fix_prompt_lists_a_dropped_rule_as_dropped(tmp_path):
    from unittest.mock import patch

    from tests.spec_driven_agent.test_phase3_fix_loop_acts import BLOCKER, _PlainClient, _build

    client = _PlainClient()
    orch = _build(tmp_path, client)
    orch.domain_model = _with_rules(_model(book=("title:str",)), PAGES_RULE)
    orch.domain_model.conversion_issues[0]["dropped_by_model_change"] = ["Book.pages"]
    with patch.object(client, "chat", wraps=client.chat) as spy:
        orch._invoke_phase3_fix_loop([BLOCKER], is_first_attempt=True)
    prompt = spy.call_args_list[0].kwargs["messages"][0]["content"]
    assert DROPPED_RULES_NOTE in prompt
    assert "self.pages" not in prompt and "Recover model constraint" not in prompt
