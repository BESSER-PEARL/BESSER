"""A kept 'implemented' verdict must also notice edits to the code its citation runs.

Review (2026-11, PR 788): a verdict was kept while every file it cited was
byte-identical. Gutting a helper the cited router imports
(``services/availability.py`` -> ``return False``) kept the verdict and the
judge was never asked. The kept verdict's fingerprint now also covers the cited
files' direct local imports and the entry file that imports them.
"""

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

UNIQUE = {"id": 1, "text": "A room cannot be double-booked", "kind": "rule"}
OTHER = {"id": 2, "text": "Guests have an e-mail", "kind": "rule"}

ROOM_PY = (
    "from fastapi import APIRouter, HTTPException\n"
    "from services.availability import is_free\n\n"
    "router = APIRouter()\n\n"
    "def book(room, day):\n"
    "    if not is_free(room, day):\n"
    "        raise HTTPException(409)\n"
)
AVAILABILITY_PY = "def is_free(room, day):\n    return (room, day) not in BOOKED\n"
MAIN_API_PY = (
    "from fastapi import FastAPI\n"
    "from routers import room, guest\n\n"
    "app = FastAPI()\n"
    "app.include_router(room.router)\n"
    "app.include_router(guest.router)\n"
)
GUEST_PY = "from fastapi import APIRouter\n\nrouter = APIRouter()\n\ndef add_guest():\n    return 1\n"
EVIDENCE = "routers/room.py: if not is_free(room, day):"


def _model():
    room = Class(name="Room")
    room.attributes = {Property(name="number", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="Hotel", types={room})


class _Client:
    model = "mock-model"
    planning_model = None

    def __init__(self):
        self._client = object()
        self.usage = UsageTracker("mock-model")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """An orchestrator over a FastAPI backend in ``backend/``, and the ids each judge call was asked."""
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._instructions = "Rooms cannot be double-booked. Guests have an e-mail."
    orch._generator_used = "generate_fastapi_backend"
    orch._requirements = [dict(UNIQUE), dict(OTHER)]
    _write(tmp_path, "backend/main_api.py", MAIN_API_PY)
    _write(tmp_path, "backend/routers/room.py", ROOM_PY)
    _write(tmp_path, "backend/routers/guest.py", GUEST_PY)
    _write(tmp_path, "backend/services/availability.py", AVAILABILITY_PY)
    answers = {1: {"status": "implemented", "evidence": EVIDENCE},
               2: {"status": "missing", "evidence": ""}}
    calls = []

    def judge(requirements, digest, client, **kwargs):
        calls.append([item["id"] for item in requirements])
        return [{**item, "note": "", **answers[item["id"]]} for item in requirements]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return orch, tmp_path, answers, calls


def test_gutting_a_helper_the_cited_router_imports_judges_the_verdict_again(run):
    orch, root, answers, calls = run
    orch._collect_requirement_issues()
    _write(root, "backend/services/availability.py", "def is_free(room, day):\n    return False\n")
    answers[1] = {"status": "missing", "evidence": ""}
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]], "the verdict was kept although the helper it calls changed"


def test_a_relative_import_is_followed(run):
    orch, root, _, calls = run
    _write(root, "backend/routers/room.py", ROOM_PY.replace(
        "from services.availability import is_free", "from ..services.availability import is_free"))
    orch._collect_requirement_issues()
    assert set(orch._requirement_verdicts[0]["dependency_file_hashes"]) >= {"backend/services/availability.py"}
    _write(root, "backend/services/availability.py", "def is_free(room, day):\n    return False\n")
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]]


def test_dropping_the_router_from_the_entry_file_judges_the_verdict_again(run):
    orch, root, _, calls = run
    orch._collect_requirement_issues()
    _write(root, "backend/main_api.py", MAIN_API_PY.replace("app.include_router(room.router)\n", ""))
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]], "the verdict was kept although its router is no longer mounted"


def test_an_edit_to_a_file_the_citation_does_not_reach_keeps_the_verdict(run):
    """The entry file imports guest.py too, but guest.py is not on the cited router's path."""
    orch, root, _, calls = run
    orch._collect_requirement_issues()
    _write(root, "backend/routers/guest.py", GUEST_PY.replace("return 1", "return 2"))
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [2]]
    assert orch._requirement_verdicts[0]["status"] == "implemented"


def test_the_fingerprint_names_the_imports_and_the_entry_file(run):
    orch, _, _, _ = run
    orch._collect_requirement_issues()

    verdict = orch._requirement_verdicts[0]
    assert set(verdict["cited_file_hashes"]) == {"backend/routers/room.py"}
    assert set(verdict["dependency_file_hashes"]) == {"backend/services/availability.py", "backend/main_api.py"}


def test_a_relative_js_import_is_followed(run):
    orch, root, answers, calls = run
    orch._requirements[0] = {"id": 1, "text": "A guest can book a room from the rooms page", "kind": "action"}
    _write(root, "frontend/src/pages/Rooms.tsx",
           "import { bookRoom } from '../api/rooms';\n\nexport function Rooms() {\n"
           "  return <button onClick={() => bookRoom(1)}>Book</button>;\n}\n")
    _write(root, "frontend/src/api/rooms.ts",
           "export const bookRoom = (id: number) => fetch(`/rooms/${id}/book`, { method: 'POST' });\n")
    answers[1] = {"status": "implemented", "evidence": [
        {"path": "frontend/src/pages/Rooms.tsx", "line": "return <button onClick={() => bookRoom(1)}>Book</button>;"}]}
    orch._collect_requirement_issues()
    assert "frontend/src/api/rooms.ts" in orch._requirement_verdicts[0]["dependency_file_hashes"]
    _write(root, "frontend/src/api/rooms.ts", "export const bookRoom = (id: number) => undefined;\n")
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]]


def test_a_deleted_dependency_counts_as_changed(run):
    orch, root, _, calls = run
    orch._collect_requirement_issues()
    (root / "backend/services/availability.py").unlink()
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]]


def test_a_recorded_empty_hash_is_not_a_match(run):
    """``cited_file_hash`` answers "" for an unreadable file: "" must never read as unchanged."""
    orch, root, _, calls = run
    orch._collect_requirement_issues()
    orch._requirement_verdicts[0]["dependency_file_hashes"]["backend/gone.py"] = ""
    _write(root, "backend/routers/guest.py", GUEST_PY.replace("return 1", "return 2"))  # not the cache
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [1, 2]]


def test_a_verdict_with_only_cited_hashes_is_kept_as_before(run):
    """A verdict recorded before dependency hashes existed keeps the old rule."""
    orch, root, _, calls = run
    orch._collect_requirement_issues()
    orch._requirement_verdicts[0].pop("dependency_file_hashes")
    _write(root, "backend/routers/guest.py", GUEST_PY.replace("return 1", "return 2"))
    orch._collect_requirement_issues()

    assert calls == [[1, 2], [2]]


def test_the_recipe_carries_no_file_hashes(run):
    """Resume never reads verdict hashes from the recipe; the download need not carry them."""
    orch, root, _, _ = run
    orch._collect_requirement_issues()
    assert "cited_file_hashes" in orch._requirement_verdicts[0]
    orch._save_recipe("build it", elapsed=1.0)

    recipe = json.loads((root / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["requirements"][0]["status"] == "implemented"
    assert not {"cited_file_hashes", "dependency_file_hashes"} & set(recipe["requirements"][0])


# Review (2026-11, round 6): the import graph was rebuilt on every verify_evidence
# call and parsed every judged file, an agent-made ``env/`` virtualenv included
# (0.50 s -> 3.01 s on one call over a 620-file venv).


def test_the_imports_are_parsed_once_per_judged_tree(run, monkeypatch):
    orch, root, _, _ = run
    parsed = []
    real = ledger._local_imports
    monkeypatch.setattr(ledger, "_local_imports", lambda rel, *args: parsed.append(rel) or real(rel, *args))
    orch._collect_requirement_issues()
    first = len(parsed)
    for _ in range(2):
        _write(root, "README.md", f"# Hotel {len(parsed)}\n")  # not judged source
        orch._collect_requirement_issues()

    assert first and len(parsed) == first, "an unchanged tree was parsed again"
    _write(root, "backend/routers/guest.py", GUEST_PY.replace("return 1", "return 2"))
    orch._collect_requirement_issues()
    assert len(parsed) > first, "an edited tree reused the old imports"


def test_only_the_files_an_import_can_name_are_parsed(run, monkeypatch):
    orch, root, _, _ = run
    for i in range(20):
        _write(root, f"backend/models/m{i}.py", "import os\n")
    parsed = []
    real = ledger._local_imports
    monkeypatch.setattr(ledger, "_local_imports", lambda rel, *args: parsed.append(rel) or real(rel, *args))
    orch._collect_requirement_issues()

    assert not any(rel.startswith("backend/models/") for rel in parsed)


def test_a_virtualenv_is_not_read_for_imports(run, monkeypatch):
    """Found by its ``pyvenv.cfg`` under any name, as ``ToolExecutor._list_dir`` does."""
    orch, root, _, _ = run
    _write(root, "backend/env/pyvenv.cfg", "home = x\n")
    # A venv module that imports the cited router would otherwise be its entry file.
    _write(root, "backend/env/Lib/site-packages/room_plugin.py", "from routers import room\n")
    read = []
    real = ledger.ImportIndex._text
    monkeypatch.setattr(ledger.ImportIndex, "_text", lambda self, rel: read.append(rel) or real(self, rel))
    orch._collect_requirement_issues()

    assert not any("/env/" in rel for rel in read)
    assert set(orch._requirement_verdicts[0]["dependency_file_hashes"]) == {
        "backend/services/availability.py", "backend/main_api.py"}
