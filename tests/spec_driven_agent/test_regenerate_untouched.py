"""A modify run regenerates the generated files nobody edited (#659).

Measured on 44 FastAPI runs: 70% of the generated files were still untouched at
the end of a run, and regenerating them from the changed model finished the
backend change with no LLM edit in about 70% of the runs, every app still
working. Before this, the LLM rewrote every model change by hand.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.pipeline.regeneration import regenerate_untouched
from besser.spec_driven_agent.planning.model_change import ModelChange
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.agent_trace import AttributionState, line_hashes

GENERATOR = "generate_fastapi_backend"


def _model(isbn=False, author=False, library=True) -> DomainModel:
    book = Class(name="Book", attributes={Property(name="title", type=StringType),
                                          Property(name="pages", type=IntegerType)})
    if isbn:
        book.attributes = book.attributes | {Property(name="isbn", type=StringType)}
    types, associations = {book}, set()
    if library:
        lib = Class(name="Library", attributes={Property(name="name", type=StringType)})
        types.add(lib)
        associations.add(BinaryAssociation(name="Has", ends={
            Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
            Property(name="library", type=lib, multiplicity=Multiplicity(0, 1))}))
    if author:
        writer = Class(name="Author", attributes={Property(name="fullName", type=StringType)})
        types.add(writer)
        associations.add(BinaryAssociation(name="Wrote", ends={
            Property(name="works", type=book, multiplicity=Multiplicity(0, "*")),
            Property(name="writer", type=writer, multiplicity=Multiplicity(0, 1))}))
    return DomainModel(name="Shelf", types=types, associations=associations)


def _generate(workspace, model):
    reply = json.loads(ToolExecutor(str(workspace), domain_model=model).execute(GENERATOR, {}))
    assert not reply.get("error"), reply


@pytest.fixture
def app(tmp_path):
    """An app built from the base model, as Phase 1 leaves it (attribution included)."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    _generate(workspace, _model())
    return workspace


def _fresh(tmp_path, model) -> dict[str, bytes]:
    out = tmp_path / "fresh"
    out.mkdir()
    _generate(out, model)
    return {p.relative_to(out).as_posix(): p.read_bytes()
            for p in (out / "backend").rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _append(path, line):
    path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")


def test_untouched_files_become_the_new_models_output(app, tmp_path):
    regeneration, reason = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    assert regeneration is not None, reason
    assert "backend/sql_alchemy.py" in regeneration.replaced
    assert "backend/pydantic_classes.py" in regeneration.replaced
    assert not regeneration.to_update
    # Byte for byte: on Windows the generator writes some files with CRLF and some with LF.
    for rel, content in _fresh(tmp_path, _model(isbn=True)).items():
        assert (app / rel).read_bytes() == content, rel


def test_an_edited_file_is_kept_and_the_agent_gets_its_change(app):
    schemas = app / "backend" / "pydantic_classes.py"
    _append(schemas, "# validated by hand")

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    assert "# validated by hand" in schemas.read_text(encoding="utf-8")
    assert "isbn" not in schemas.read_text(encoding="utf-8")
    update = next(u for u in regeneration.to_update if u.path == "backend/pydantic_classes.py")
    assert any("isbn" in line for line, _ in update.added)
    assert "backend/sql_alchemy.py" in regeneration.replaced
    note = regeneration.note()
    assert "`backend/pydantic_classes.py`" in note and "isbn" in note


def test_a_line_the_editor_changed_is_not_offered_back(app):
    schemas = app / "backend" / "pydantic_classes.py"
    text = schemas.read_text(encoding="utf-8")
    assert "    title: str\n" in text
    schemas.write_text(text.replace("    title: str\n", "    title: str  # shown on the cover\n", 1),
                       encoding="utf-8")

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    update = next(u for u in regeneration.to_update if u.path == "backend/pydantic_classes.py")
    added = [line.strip() for line, _ in update.added]
    assert "title: str" not in added
    assert any("isbn" in line for line in added)


def test_a_new_class_gets_its_files(app):
    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(author=True))

    assert "backend/routers/author.py" in regeneration.added
    assert (app / "backend" / "routers" / "author.py").is_file()


def test_the_router_of_a_removed_class_is_deleted(app):
    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(library=False))

    assert "backend/routers/library.py" in regeneration.removed
    assert not (app / "backend" / "routers" / "library.py").exists()


def test_a_file_an_edited_file_still_uses_is_kept(app):
    _append(app / "backend" / "main_api.py", "# custom startup")

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(library=False))

    assert "backend/routers/library.py" in regeneration.kept_in_use
    assert (app / "backend" / "routers" / "library.py").is_file()
    assert any(u.path == "backend/main_api.py" for u in regeneration.to_update)
    assert "`backend/routers/library.py`" in regeneration.note()


def test_attribution_records_the_regenerated_files(app):
    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True, library=False))

    state = AttributionState.load(str(app))
    generator = {i for i, c in enumerate(state.contributors) if c["kind"] == "generator"}
    for rel in regeneration.written:
        hashes, owners = state.files[rel]
        assert set(owners) <= generator
        assert hashes == line_hashes((app / rel).read_text(encoding="utf-8").splitlines())
    for rel in regeneration.removed:
        assert rel not in state.files


def test_without_attribution_nothing_is_regenerated(app):
    (app / ".besser_attribution.json").unlink()
    before = (app / "backend" / "sql_alchemy.py").read_text(encoding="utf-8")

    regeneration, reason = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    assert regeneration is None
    assert "attribution" in reason
    assert (app / "backend" / "sql_alchemy.py").read_text(encoding="utf-8") == before


def test_only_measured_generators_are_regenerated(app):
    regeneration, reason = regenerate_untouched(str(app), "generate_web_app", _model(isbn=True))

    assert regeneration is None
    assert "generate_web_app" in reason


CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    with TestClient(main_api.app) as client:
        book = client.post("/book/", json={"title": "Dune", "pages": 412, "isbn": "978"})
        writer = client.post("/author/", json={"fullName": "Frank Herbert"})
        print(json.dumps({"book": [book.status_code, book.json()], "author": writer.status_code}))
""")


def test_the_regenerated_app_serves_the_new_model(app):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    regenerate_untouched(str(app), GENERATOR, _model(isbn=True, author=True))

    run = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(app / "backend"),
                         capture_output=True, text=True, timeout=120)

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["book"][0] == 200 and "978" in json.dumps(result["book"][1])
    assert result["author"] == 200


@pytest.fixture
def orchestrator(app):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    built = LLMOrchestrator(llm_client=Client(), domain_model=_model(isbn=True),
                            output_dir=str(app), enable_checkpointing=False)
    built._seed_generator_used = GENERATOR
    return built


def _change(*changes):
    return ModelChange(verdict="small", basis="full_model", gone=0.0, removed=(), added=(),
                       changes=tuple(changes))


def test_a_modify_run_regenerates_when_the_model_changed(orchestrator, app):
    orchestrator._model_change = _change("Book.isbn added (str)")

    orchestrator._regenerate_untouched_files()

    assert "isbn" in (app / "backend" / "sql_alchemy.py").read_text(encoding="utf-8")
    assert "backend/sql_alchemy.py" in orchestrator._regeneration.replaced


def test_a_modify_run_with_the_same_model_leaves_the_files(orchestrator, app):
    """Without a model change, regeneration would only bring in a newer generator's output."""
    orchestrator._model_change = _change()

    orchestrator._regenerate_untouched_files()

    assert "isbn" not in (app / "backend" / "sql_alchemy.py").read_text(encoding="utf-8")
    assert orchestrator._regeneration is None


# Review findings, each reproduced on the first version of this change.

def _recipe(app, *paths):
    (app / ".besser_recipe.json").write_text(
        json.dumps({"output_files": [{"path": p, "source": "generator"} for p in paths]}), encoding="utf-8")


def test_lines_of_a_packaging_step_are_not_the_generators(app):
    """The secret scrub is recorded as a generator contributor too. An AI file whose
    only line it rewrote looked untouched and no longer generated, so it was deleted."""
    from besser.spec_driven_agent.state.agent_trace import ai_contributor, generator_contributor
    from besser.spec_driven_agent.state.agent_trace import reconcile_workspace
    keys = app / "backend" / "keys.py"
    keys.write_text("STRIPE_KEY = 'sk_live_x'\n", encoding="utf-8")
    reconcile_workspace(str(app), ai_contributor("openai/gpt"), paths=["backend/keys.py"])
    keys.write_text("STRIPE_KEY = '[REDACTED]'\n", encoding="utf-8")
    reconcile_workspace(str(app), generator_contributor("secret_redaction"), paths=["backend/keys.py"])

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    assert keys.is_file()
    assert "backend/keys.py" not in regeneration.removed
    assert all(u.path != "backend/keys.py" for u in regeneration.to_update)


def test_a_changed_line_is_not_offered_back_after_seeding(app):
    """In production the runner reconciles the seed first, so the stored hashes already
    match the edited file and the generator's original line is gone from them."""
    from besser.spec_driven_agent.state.agent_trace import seed_attribution
    schemas = app / "backend" / "pydantic_classes.py"
    schemas.write_text(schemas.read_text(encoding="utf-8").replace(
        "    title: str\n", "    title: str  # shown on the cover\n", 1), encoding="utf-8")
    assert seed_attribution(str(app))

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True))

    update = next(u for u in regeneration.to_update if u.path == "backend/pydantic_classes.py")
    added = [line.strip() for line, _ in update.added]
    assert "title: str" not in added
    assert any("isbn" in line for line in added)


def test_a_new_class_keeps_its_repeated_template_lines(app):
    """Every mapped class has the same primary-key line: a set diff dropped it,
    so the agent was told to add a class with no primary key."""
    _append(app / "backend" / "sql_alchemy.py", "# indexed by hand")

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(author=True))

    update = next(u for u in regeneration.to_update if u.path == "backend/sql_alchemy.py")
    added = [line.strip() for line, _ in update.added]
    start = added.index("class Author(Base):")
    assert any("primary_key=True" in line for line in added[start:start + 4])


def test_a_generated_file_deleted_on_purpose_is_not_brought_back(app):
    router = app / "backend" / "routers" / "library.py"
    _recipe(app, "backend/routers/library.py", "backend/routers/book.py")
    router.unlink()

    regeneration, _ = regenerate_untouched(str(app), GENERATOR, _model(isbn=True, author=True))

    assert not router.exists()
    assert "backend/routers/library.py" in regeneration.not_restored
    assert "backend/routers/author.py" in regeneration.added


def test_a_failed_regeneration_leaves_the_run_going(orchestrator, app, monkeypatch):
    import besser.spec_driven_agent.pipeline.modify_run as modify_run

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(modify_run, "regenerate_untouched", broken)
    orchestrator._model_change = _change("Book.isbn added (str)")

    orchestrator._regenerate_untouched_files()

    assert orchestrator._regeneration is None
