"""The 'build the frontend' checklist item does not close on the first file written.

Its verifier used to be "any .js/.html file or package.json exists", so in every r9
live run the item auto-closed, "verified", on the first file. With no GUI model the
frontend is LLM-authored, so the item now names the classes with routes and closes once
the app has at least one entity screen (acceptance matrix ``page``). It does not demand
a page per class: the p5 request lists its screens and leaves PaymentMethod out, and a
class without a page stays an acceptance warning. With a GUI model it is unchanged.
"""
from unittest import mock

from besser.BUML.metamodel.structural import AssociationClass, BinaryAssociation, Multiplicity, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

INSTRUCTIONS = "Build a library web app"


class _Client:
    """No LLM call is made: Phase 2 and 3 are stubbed."""

    model = "mock"
    max_tokens = None

    def __init__(self):
        self.usage = UsageTracker("mock")


def _orch(tmp_path, model=None):
    model = model or shelf_model(author=True)
    generate_backend(tmp_path, model)
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=model,
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    orch._generator_used = GENERATOR
    orch.target = Target(kind="web_app", description="library web app", has_user_interface=True)
    return orch


def _task(orch):
    return next(t for t in orch._deterministic_gap_tasks()
                if isinstance(t, dict) and t["text"].startswith("Create the React frontend"))


def _write(tmp_path, rel, text):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _page(name):
    return f"export default function {name}List() {{ return <h1>{name}s</h1>; }}\n"


def test_item_names_the_routed_classes_and_lets_the_request_scope_the_screens(tmp_path):
    text = _task(_orch(tmp_path))["text"]
    assert ("Build a screen for each class with backend routes (Author, Book, Library) unless the request "
            "scopes the UI to fewer screens, in which case the request wins.") in text


def test_first_file_does_not_close_the_item_one_entity_screen_does(tmp_path):
    orch = _orch(tmp_path)
    orch.executor.set_tasks([_task(orch)])
    _write(tmp_path, "frontend/package.json", '{"name": "library"}')
    _write(tmp_path, "frontend/index.html", "<title>Library</title><div id=root></div>")
    assert orch.executor.autoclose_verified_tasks() == []
    assert orch.executor._task_list({"action": "done", "id": 1})["status"] == "refused"
    _write(tmp_path, "frontend/src/pages/BookList.jsx", _page("Book"))
    assert [item["id"] for item in orch.executor.autoclose_verified_tasks()] == [1]


def test_a_class_left_out_is_a_warning_not_a_refusal(tmp_path):
    orch = _orch(tmp_path)
    task = _task(orch)
    _write(tmp_path, "frontend/src/pages/BookList.jsx", _page("Book"))
    assert task["verify"]()
    assert orch._routed_entity_pages() == {"Author": False, "Book": True, "Library": False}
    warnings = orch._acceptance_matrix_issues()
    assert any(issue.startswith("acceptance: entity Author") for issue in warnings)


def test_association_classes_are_not_named(tmp_path):
    model = shelf_model(author=True)
    book = next(c for c in model.get_classes() if c.name == "Book")
    library = next(c for c in model.get_classes() if c.name == "Library")
    loans = BinaryAssociation(name="loans", ends={
        Property(name="lent", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="lenders", type=library, multiplicity=Multiplicity(0, "*"))})
    model.associations = model.associations | {loans}
    model.types = model.types | {AssociationClass(name="Loan", attributes=set(), association=loans)}
    text = _task(_orch(tmp_path, model))["text"]
    assert "(Author, Book, Library)" in text and "Loan" not in text.split("backend routes")[1]


def test_with_a_gui_model_the_item_is_unchanged(tmp_path):
    orch = _orch(tmp_path)
    orch.gui_model = mock.sentinel.gui_model
    task = _task(orch)
    assert task["text"] == orch._FRONTEND_CHECKLIST_TASK
    assert task["verify"] == orch._has_frontend_files


def test_resume_reattaches_the_verifier_to_an_older_item_text(tmp_path, monkeypatch):
    """A checkpoint holding the d1b68618 text (the unchanged constant) lost its
    verifier on resume: ``restore_tasks`` matches verifiers by exact text."""
    crashed = _orch(tmp_path)
    crashed._checkpointing_enabled = True
    crashed._project_fingerprint = crashed._compute_project_fingerprint(INSTRUCTIONS)
    crashed.executor.set_tasks([{"text": crashed._FRONTEND_CHECKLIST_TASK, "verify": crashed._has_frontend_files}])
    _write(tmp_path, "frontend/package.json", '{"name": "library"}')
    crashed._save_checkpoint_for_turn(1, [{"role": "user", "content": INSTRUCTIONS}], INSTRUCTIONS)

    resumed = LLMOrchestrator(llm_client=_Client(), domain_model=shelf_model(author=True),
                              output_dir=str(tmp_path), enable_tracing=False, use_streaming=False)
    for name, stub in {"_run_phase2": lambda instructions, extra_issues=None: None,
                       "_create_snapshot": lambda: None, "_run_phase3_validation": lambda: None,
                       "_remove_snapshot": lambda: None, "_save_recipe": lambda instructions, elapsed: None}.items():
        monkeypatch.setattr(resumed, name, stub)
    resumed.resume(INSTRUCTIONS)
    task = resumed.executor._tasks[0]
    assert task["verify"] is not None and not task["verify"]()
    _write(tmp_path, "frontend/src/pages/AuthorList.jsx", _page("Author"))
    assert task["verify"]()
