"""A modify run merges a model change into the generated files someone edited (#659, stage 3).

Stage 2 regenerated only the untouched files and left every edited one to the LLM.
Measured offline on 42 working FastAPI runs and 7 kinds of model change, a
three-way merge against the generator's stored output finished the change with
no hand edit in 78% of the cases (stage 2: 31%), with no clean merge failing the
boot, create or conformance checks. After a BESSER upgrade, merging against the
stored output finished none; merging against the new generator's output for the
old model finished 78% again.
"""
import json
import subprocess
import sys
import textwrap

import pytest

pygit2 = pytest.importorskip("pygit2")

import besser.spec_driven_agent.pipeline.regeneration as regeneration_module  # noqa: E402
from besser.BUML.metamodel.structural import (  # noqa: E402
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor  # noqa: E402
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator  # noqa: E402
from besser.spec_driven_agent.pipeline.regeneration import regenerate  # noqa: E402
from besser.spec_driven_agent.planning.model_change import ModelChange  # noqa: E402
from besser.spec_driven_agent.providers.llm_client import UsageTracker  # noqa: E402
from besser.spec_driven_agent.state.agent_trace import (  # noqa: E402
    AttributionState,
    ai_contributor,
    reconcile_workspace,
)
from besser.spec_driven_agent.state.generator_output import (  # noqa: E402
    GENERATOR_OUTPUT_FILENAME,
    attach_model,
    load_generator_output,
)
from besser.utilities.provenance import get_besser_version  # noqa: E402

GENERATOR = "generate_fastapi_backend"
SCHEMAS = "backend/pydantic_classes.py"


def _model(isbn=False, author=False, library="Library") -> DomainModel:
    book = Class(name="Book", attributes={Property(name="title", type=StringType),
                                          Property(name="pages", type=IntegerType)})
    if isbn:
        book.attributes = book.attributes | {Property(name="isbn", type=StringType)}
    types, associations = {book}, set()
    if library:
        lib = Class(name=library, attributes={Property(name="name", type=StringType)})
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


def _fresh(tmp_path, model) -> dict[str, bytes]:
    out = tmp_path / "fresh"
    out.mkdir()
    _generate(out, model)
    return {p.relative_to(out).as_posix(): p.read_bytes()
            for p in (out / "backend").rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _regenerate(app, model, renamed=(), previous_model=None):
    regeneration, reason = regenerate(str(app), GENERATOR, model, set(), renamed, previous_model)
    assert regeneration is not None, reason
    return regeneration


def _newline(data: bytes) -> str:
    """On Windows the generator writes some files with CRLF; an edit keeps the file's own."""
    return "\r\n" if b"\r\n" in data else "\n"


def _append(path, line):
    data = path.read_bytes()
    path.write_bytes(data + (line + _newline(data)).encode("utf-8"))


def _replace(path, old, new):
    """Replace ``old`` with ``new``, a ``\\n`` in them standing for the file's own line ending."""
    data = path.read_bytes()
    nl = _newline(data)
    text = data.decode("utf-8")
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert old in text, old
    path.write_bytes(text.replace(old, new, 1).encode("utf-8"))


@pytest.fixture
def app(tmp_path):
    """An app as Phase 1 leaves it: the generator's output stored next to it."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    _generate(workspace, _model())
    return workspace


# -- the stored output ------------------------------------------------------------


def test_phase1_stores_the_generator_output_as_written(app):
    stored = load_generator_output(str(app), "backend")

    assert stored is not None and stored.besser_version == get_besser_version()
    assert stored.files == {rel: (app / rel).read_bytes() for rel in stored.files}
    assert "backend/main_api.py" in stored.files and SCHEMAS in stored.files


def test_the_new_output_becomes_the_next_base(app, tmp_path):
    attach_model(str(app), {"title": "Shelf"})

    _regenerate(app, _model(isbn=True))

    stored = load_generator_output(str(app), "backend")
    assert stored.files == _fresh(tmp_path, _model(isbn=True))
    # The attached model described the old output.
    assert stored.model is None


# -- the merge ----------------------------------------------------------------------


def test_an_edited_file_gets_the_model_change_and_keeps_the_edit(app, tmp_path):
    schemas = app / SCHEMAS
    _append(schemas, "# validated by hand")

    regeneration = _regenerate(app, _model(isbn=True))

    text = schemas.read_text(encoding="utf-8")
    assert "    isbn: str" in text and "# validated by hand" in text
    assert regeneration.merged == [SCHEMAS]
    assert not regeneration.to_update
    # The untouched files are the new output, byte for byte.
    fresh = _fresh(tmp_path, _model(isbn=True))
    for rel in regeneration.replaced:
        assert (app / rel).read_bytes() == fresh[rel], rel
    assert "merged the model change" in regeneration.note()


def _authors(app, rel) -> dict[str, str]:
    state = AttributionState.load(str(app))
    lines = (app / rel).read_text(encoding="utf-8").splitlines()
    return {line.strip(): state.contributors[o]["kind"] if o >= 0 else "unrecorded"
            for line, o in zip(lines, state.owners_of(rel, lines))}


def test_the_merged_in_lines_are_the_generators_and_the_edit_keeps_its_author(app):
    """Packaging gives every line that differs from the record to the LLM: without
    this, the model change the merge wrote would read as AI-written."""
    _append(app / SCHEMAS, "# validated by hand")
    reconcile_workspace(str(app), ai_contributor("openai/gpt"), paths=[SCHEMAS])

    _regenerate(app, _model(isbn=True))

    authors = _authors(app, SCHEMAS)
    assert authors["isbn: str"] == "generator"
    assert authors["# validated by hand"] == "ai"


def test_a_conflicting_edit_is_kept_and_the_agent_gets_the_change(app):
    """The edit sits next to where the model change lands: no half-merged file."""
    schemas = app / SCHEMAS
    _replace(schemas, "    pages: int\n", "    pages: int = 0\n")
    before = schemas.read_bytes()

    regeneration = _regenerate(app, _model(isbn=True))

    assert schemas.read_bytes() == before
    update = next(u for u in regeneration.to_update if u.path == SCHEMAS)
    assert any("isbn" in line for line, _ in update.added)
    assert "backend/sql_alchemy.py" in regeneration.replaced


def test_an_edit_that_also_converted_the_line_endings_still_merges(app):
    """An editor on another platform rewrote every line ending; only the added line is the edit."""
    schemas = app / SCHEMAS
    data = schemas.read_bytes()
    other = b"\n" if b"\r\n" in data else b"\r\n"
    schemas.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", other) + b"# validated by hand" + other)

    regeneration = _regenerate(app, _model(isbn=True))

    data = schemas.read_bytes()
    assert regeneration.merged == [SCHEMAS]
    assert b"    isbn: str" + other in data and b"# validated by hand" + other in data
    # The file keeps the endings it has now.
    assert data.replace(b"\r\n", b"\n").replace(b"\n", other) == data


def test_a_renamed_class_moves_its_router_with_the_edits(app):
    router = app / "backend" / "routers" / "library.py"
    _append(router, "# opening hours are checked by hand")
    # As the seed records a person's edits before the run.
    reconcile_workspace(str(app), ai_contributor("openai/gpt"), paths=["backend/routers/library.py"])

    regeneration = _regenerate(app, _model(library="Branch"), renamed=(("Library", "Branch"),))

    moved = app / "backend" / "routers" / "branch.py"
    assert not router.exists()
    assert regeneration.moved == [("backend/routers/library.py", "backend/routers/branch.py")]
    text = moved.read_text(encoding="utf-8")
    assert "# opening hours are checked by hand" in text and "Branch" in text and "Library" not in text
    # The moved file keeps its record: the edit stays the AI's, the renamed lines are the generator's.
    authors = _authors(app, "backend/routers/branch.py")
    assert authors["# opening hours are checked by hand"] == "ai"
    assert {kind for line, kind in authors.items() if "Branch" in line} == {"generator"}


def test_a_generated_file_deleted_on_purpose_stays_deleted(app):
    """The stored output says it was generated, so no recipe is needed to know."""
    router = app / "backend" / "routers" / "library.py"
    router.unlink()

    regeneration = _regenerate(app, _model(isbn=True, author=True))

    assert not router.exists()
    assert "backend/routers/author.py" in regeneration.added


def test_the_router_of_a_removed_class_is_deleted(app):
    regeneration = _regenerate(app, _model(library=None))

    assert "backend/routers/library.py" in regeneration.removed


# -- a BESSER upgrade in between -------------------------------------------------------


GENERATED = "    name: Mapped_[str] = mapped_column(String_(100))"
OLDER = "    name: Mapped_[str] = mapped_column(String_(80))"
EDIT = "    # name is shown on the badge"


def _as_older_besser(app):
    """The app as an older BESSER generated it (one line differs), edited next to that line."""
    path = app / "backend" / "sql_alchemy.py"
    store = json.loads((app / GENERATOR_OUTPUT_FILENAME).read_text(encoding="utf-8"))
    store["besser_version"] = "0.0.1"
    assert GENERATED in store["files"]["backend/sql_alchemy.py"]
    store["files"]["backend/sql_alchemy.py"] = store["files"]["backend/sql_alchemy.py"].replace(
        GENERATED, OLDER, 1)
    (app / GENERATOR_OUTPUT_FILENAME).write_text(json.dumps(store), encoding="utf-8")
    _replace(path, GENERATED + "\n", OLDER + "\n" + EDIT + "\n")
    return path


def test_after_an_upgrade_the_base_is_rebuilt_from_the_old_model(app):
    path = _as_older_besser(app)

    regeneration = _regenerate(app, _model(isbn=True), previous_model=_model())

    text = path.read_text(encoding="utf-8")
    assert regeneration.upgraded_from == "0.0.1" and not regeneration.merge_warning
    assert "backend/sql_alchemy.py" in regeneration.merged
    assert "isbn" in text and EDIT in text
    # Only the model change reaches an edited file, not the newer generator's own changes.
    assert OLDER in text


def test_after_an_upgrade_without_the_old_model_the_agent_is_told(app):
    path = _as_older_besser(app)
    before = path.read_bytes()

    regeneration = _regenerate(app, _model(isbn=True))

    assert "no model to rebuild the merge base" in regeneration.merge_warning
    assert path.read_bytes() == before
    assert any(u.path == "backend/sql_alchemy.py" for u in regeneration.to_update)


# -- what the merge must not trust ----------------------------------------------------


def test_a_store_naming_a_path_outside_the_output_folder_is_not_used(app):
    victim = app / "victim.txt"
    victim.write_text("keep me\n", encoding="utf-8")
    store = json.loads((app / GENERATOR_OUTPUT_FILENAME).read_text(encoding="utf-8"))
    store["files"]["backend/../victim.txt"] = "keep me\n"
    (app / GENERATOR_OUTPUT_FILENAME).write_text(json.dumps(store), encoding="utf-8")

    regeneration = _regenerate(app, _model(isbn=True))

    assert victim.read_text(encoding="utf-8") == "keep me\n"
    assert regeneration.method == "attribution"
    assert "outside backend/" in regeneration.merge_warning


def test_without_pygit2_edited_files_are_left_to_the_agent(app, monkeypatch):
    monkeypatch.setattr(regeneration_module, "pygit2", None)
    _append(app / SCHEMAS, "# validated by hand")

    regeneration = _regenerate(app, _model(isbn=True))

    assert regeneration.method == "attribution"
    assert "pygit2 is not installed" in regeneration.merge_warning
    assert any(u.path == SCHEMAS for u in regeneration.to_update)


# -- the app it leaves -------------------------------------------------------------------

CUSTOM_ENDPOINT = textwrap.dedent('''

    @router.get("/shelf-count/", tags=["Book"])
    def shelf_count(database: Session = Depends(get_db)) -> dict:
        return {"books": database.query(Book).count()}
''')

CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    with TestClient(main_api.app) as client:
        book = client.post("/book/", json={"title": "Dune", "pages": 412, "isbn": "978"})
        count = client.get("/shelf-count/")
        print(json.dumps({"book": [book.status_code, book.json()], "count": [count.status_code, count.json()]}))
""")


def test_the_merged_app_serves_the_new_model_and_the_edit(app):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    _append(app / "backend" / "routers" / "book.py", CUSTOM_ENDPOINT)

    regeneration = _regenerate(app, _model(isbn=True))

    assert "backend/routers/book.py" in regeneration.merged
    run = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(app / "backend"),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["book"][0] == 200 and "978" in json.dumps(result["book"][1])
    assert result["count"] == [200, {"books": 1}]


# -- the modify run ----------------------------------------------------------------------


class _Client:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def test_a_modify_run_merges_a_class_rename_and_traces_it(app):
    _append(app / "backend" / "routers" / "library.py", "# opening hours are checked by hand")
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=_model(library="Branch"),
                                   output_dir=str(app), enable_checkpointing=False,
                                   previous_domain_model=_model())
    orchestrator._seed_generator_used = GENERATOR
    orchestrator._seed_output_files = set()
    orchestrator._seed_llm_modified = {"backend/routers/library.py"}
    orchestrator._model_change = ModelChange(
        verdict="small", basis="full_model", gone=0.0, removed=(), added=(),
        renamed=(("Library", "Branch"), ("Library.name", "Branch.name")), changes=("Library renamed to Branch",))

    orchestrator._regenerate_generated_files()

    assert orchestrator._regeneration.moved == [("backend/routers/library.py", "backend/routers/branch.py")]
    assert "backend/routers/branch.py" in orchestrator.executor._generator_files
    assert orchestrator._seed_llm_modified == {"backend/routers/branch.py"}
    events = [json.loads(line) for line in (app / ".besser_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    traced = next(e.get("payload", e) for e in events if e.get("event") == "regeneration")
    assert traced["method"] == "merge"
    assert traced["moved"] == [["backend/routers/library.py", "backend/routers/branch.py"]]
