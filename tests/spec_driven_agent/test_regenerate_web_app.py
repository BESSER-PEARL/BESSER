"""A web-app modify run brings the generated backend and typed client up to a model change.

Only the FastAPI backend generator's output was regenerated, so after a model change
a web app kept its old backend, api_contract.json and src/api/generated client, and
the checks then read the stale contract. The parts of a web app derived from the
domain model alone (web_app/backend/ and the typed client) are now regenerated and
merged like the FastAPI backend; the GUI-derived files (pages, components) are not
touched, as they need the GUI model.
"""
import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.pipeline.regeneration import regenerate
from besser.spec_driven_agent.planning.model_change import ModelChange
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.generator_output import GENERATOR_OUTPUT_FILENAME, load_generator_output
from besser.spec_driven_agent.validation.endpoint_coherence import collect_endpoint_coherence_issues
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)
from tests.generators.react.conftest import class_diagram, gui_json

GENERATOR = "generate_web_app"
BASE = {"Guest": [("full_name", "str")]}
CHANGED = {"Guest": [("full_name", "str"), ("phone", "str")], "Room": [("number", "int")]}
STAYS = [("stays", ("Room", "rooms", "0..*"), ("Guest", "guest", "0..1"))]
CLIENT = "web_app/frontend/src/api/generated/client.ts"
PAGE = "web_app/frontend/src/pages/Rooms.tsx"


def _domain(classes=CHANGED, associations=STAYS):
    return process_class_diagram({"title": "Domain", "model": class_diagram(classes, associations)})


def _build(workspace, classes=BASE, associations=(), app_name="Front Desk"):
    """Phase 1 of a web app: the web app generator with a table screen of each class."""
    class_json = class_diagram(classes, associations)
    domain = process_class_diagram({"title": "Domain", "model": class_json})
    pages = {name: [{"type": "table", "attributes": {"id": f"{name.lower()}-table", "chart-title": name,
                                                     "data-source": f"cls-{name}"}}] for name in classes}
    gui = process_gui_diagram(gui_json(pages), class_json, domain)
    reply = json.loads(ToolExecutor(str(workspace), domain_model=domain, gui_model=gui, app_name=app_name)
                       .execute(GENERATOR, {}))
    assert not reply.get("error"), reply
    return domain


def _files(root, prefix="web_app/"):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).as_posix().startswith(prefix)}


@pytest.fixture
def app(tmp_path):
    workspace = tmp_path / "app"
    workspace.mkdir()
    base = _build(workspace)
    # A page the agent wrote, calling a relation the model does not have yet.
    page = workspace / PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text('import { guestApi } from "../api/generated/client";\n'
                    "export const load = (id: number) => guestApi.relations.rooms.list(id);\n", encoding="utf-8")
    return workspace, base


def test_phase_1_stores_only_the_model_derived_part(app):
    workspace, _ = app

    stored = load_generator_output(str(workspace), "web_app")

    assert stored is not None
    assert "web_app/backend/api_contract.json" in stored.files and CLIENT in stored.files
    assert all(rel.startswith(("web_app/backend/", "web_app/frontend/src/api/generated/")) for rel in stored.files)


def test_a_model_change_regenerates_the_backend_and_the_client(app, tmp_path):
    workspace, base = app
    gui_files = {rel: content for rel, content in _files(workspace).items()
                 if not rel.startswith(("web_app/backend/", "web_app/frontend/src/api/generated/"))}
    stale = collect_endpoint_coherence_issues(str(workspace))
    assert any("guestApi.relations.rooms.list(...), which the generated API client does not have" in i
               for i in stale), stale

    regeneration, reason = regenerate(str(workspace), GENERATOR, _domain(), None, (), base, "Front Desk")

    assert regeneration is not None, reason
    assert not regeneration.merge_warning
    assert CLIENT in regeneration.replaced
    assert "web_app/backend/api_contract.json" in regeneration.replaced
    assert "phone" in (workspace / "web_app" / "backend" / "pydantic_classes.py").read_text(encoding="utf-8")
    # Byte for byte what a fresh web app of the new model has, for the regenerated part.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _build(fresh, CHANGED, STAYS)
    for rel in ("web_app/backend/api_contract.json", "web_app/backend/sql_alchemy.py", CLIENT,
                "web_app/frontend/src/api/generated/types.ts"):
        assert (workspace / rel).read_bytes() == (fresh / rel).read_bytes(), rel
    # The GUI-derived files, and the agent's page, are left as they were.
    for rel, content in gui_files.items():
        assert (workspace / rel).read_bytes() == content, rel
    # The agent is told the screens are its part of the model change.
    assert "`web_app/backend/`, `web_app/frontend/src/api/generated/` follow the model" in regeneration.note()
    # The checks read the new contract: the page's call to the new relation is no longer unknown.
    assert collect_endpoint_coherence_issues(str(workspace)) == []


def test_an_edited_backend_file_is_merged(app):
    workspace, base = app
    router = workspace / "web_app" / "backend" / "routers" / "guest.py"
    router.write_text(router.read_text(encoding="utf-8") + "# checked by hand\n", encoding="utf-8")

    regeneration, reason = regenerate(str(workspace), GENERATOR, _domain(), None, (), base, "Front Desk")

    assert regeneration is not None, reason
    assert "web_app/backend/routers/guest.py" in regeneration.merged
    assert router.read_text(encoding="utf-8").endswith("# checked by hand\n")


def test_without_a_stored_output_the_gui_files_are_neither_replaced_nor_deleted(app):
    """An app built before the web app was regenerated: the line attribution decides, for the same part only."""
    workspace, base = app
    (workspace / GENERATOR_OUTPUT_FILENAME).unlink()
    before = _files(workspace, "web_app/frontend/src/")

    regeneration, reason = regenerate(str(workspace), GENERATOR, _domain(), None, (), base, "Front Desk")

    assert regeneration is not None, reason
    assert regeneration.method == "attribution"
    changed = set(regeneration.written + regeneration.removed + regeneration.merged
                  + [u.path for u in regeneration.to_update])
    assert changed and all(rel.startswith(("web_app/backend/", "web_app/frontend/src/api/generated/"))
                           for rel in changed), changed
    after = _files(workspace, "web_app/frontend/src/")
    assert {rel for rel in before if rel not in after} == set()
    assert all(after[rel] == content for rel, content in before.items() if "/api/generated/" not in rel)


class _Client:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def test_a_web_app_modify_run_regenerates_with_the_apps_name(app):
    """The backend's title carries the app's name: the run passes it, so the stored output checks out."""
    workspace, base = app
    orchestrator = LLMOrchestrator(llm_client=_Client(), domain_model=_domain(), output_dir=str(workspace),
                                   enable_checkpointing=False, source_project_export={"name": "Front Desk"})
    orchestrator._seed_generator_used = GENERATOR
    orchestrator._seed_output_files = set()
    orchestrator._previous_domain_model = base
    orchestrator._model_change = ModelChange(verdict="small", basis="full_model", gone=0.0, removed=(), added=(),
                                             changes=("Guest.phone added (str)",))

    orchestrator._regenerate_generated_files()

    assert orchestrator._regeneration is not None
    assert orchestrator._regeneration.method == "merge" and not orchestrator._regeneration.merge_warning
    assert CLIENT in orchestrator._regeneration.replaced
