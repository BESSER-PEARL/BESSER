"""A change's life through the service, on a fake GitHub: pull requests, partly merged, merged and closed (D2
decision 16), closed unmerged on a click, update from the base with and without a pull request; and the backend's
lifespan starting and stopping the lab."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from fastapi import FastAPI

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND
from besser.utilities.web_modeling_editor.backend import backend
from besser.utilities.web_modeling_editor.backend.services.lab.access import LabUser
from besser.utilities.web_modeling_editor.backend.services.lab.service import REFUSAL_STATUS, LabRefusal, LabService
from besser.utilities.web_modeling_editor.backend.settings import get_settings
from tests.utilities.web_modeling_editor.backend.lab.conftest import BRANCH, LOGIN, TOKEN
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import (  # noqa: F401 (lab_env: a fixture)
    lab_env,
    make_service,
    record,
)

JORDI = LabUser(login=LOGIN, session_id="session-jordi", access_token=TOKEN)


def _refused(code: str, coroutine) -> LabRefusal:
    with pytest.raises(LabRefusal) as caught:
        asyncio.run(coroutine)
    assert (caught.value.code, caught.value.status) == (code, REFUSAL_STATUS[code])
    return caught.value


def _open(service: LabService, change_id: str):
    return asyncio.run(service.open_pull_requests(JORDI, change_id, title=None, body=None, closes_issues=[]))


def _audit(service: LabService) -> list[tuple[str, str]]:
    return [(entry.action, entry.detail) for entry in service.audit.entries(service.audit.current_month())]


def test_lifecycle_partly_merged_then_merged_and_closable(lab, lab_env):  # noqa: F811 - lab_env is imported
    service = make_service(lab.workspace)
    change_id = lab.change.id
    lab.write(REPO_FRONTEND, "packages/webapp/src/note.ts", "export const note = 1;\n")
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")

    opened, _ = _open(service, change_id)

    assert [(view.repo, view.number, view.branch) for view in opened] == [(REPO_FRONTEND, 1, BRANCH),
                                                                         (REPO_BESSER, 1, BRANCH)]
    detail = service.change_detail(change_id)
    assert detail.phase == "in_review" and detail.pullRequests == opened
    assert [item.type for item in detail.thread] == ["pull_requests"]
    assert ("pull_requests_opened", "frontend #1, besser #1") in _audit(service)

    lab.fake.merge_pull(lab.remote(REPO_FRONTEND), 1)
    partly = asyncio.run(service.review(JORDI, change_id))
    assert (partly.lifecycle, partly.closable) == ("partly_merged", False)
    refusal = _refused("state", service.close(JORDI, change_id))
    assert "still open" in refusal.message
    assert service.change_detail(change_id).status == "open"

    lab.fake.merge_pull(lab.remote(REPO_BESSER), 1)
    closed = asyncio.run(service.close(JORDI, change_id))

    assert (closed.status, closed.outcome, closed.phase) == ("closed", "merged", "merged")
    assert closed.closedAt and closed.pullRequests == opened
    assert not Path(lab.layout.root, "changes", change_id).exists()
    assert Path(lab.layout.state_dir, "changes", change_id, "state.json").is_file()
    notice = closed.thread[-1]
    assert notice.type == "notice" and notice.text == 'Change "Add a note element" was merged; the lab closed it.'
    assert ("change_closed", "merged") in _audit(service)
    _refused("state", service.close(JORDI, change_id))  # once is enough


def test_closed_unmerged_pull_requests_close_the_change_on_a_click(lab, lab_env):  # noqa: F811
    service = make_service(lab.workspace)
    change_id = lab.change.id
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    _open(service, change_id)
    lab.fake.close_pull(lab.remote(REPO_BESSER), 1)
    ticket = service.begin_run(change_id, "r1", "ana")
    assert "@ana" in _refused("change_busy", service.close(JORDI, change_id)).message
    service.end_run(ticket, status="cancelled", record=record("r1", "ana", usd=0.1), done=None, error=None)

    closed = asyncio.run(service.close(JORDI, change_id))

    assert (closed.status, closed.outcome, closed.phase) == ("closed", "closed", "closed")
    assert closed.thread[-1].text.endswith("was closed without being merged; the lab closed it.")


def test_update_from_base_without_a_pull_request_merges_locally_and_says_so(lab, lab_env):  # noqa: F811
    service = make_service(lab.workspace)
    change_id = lab.change.id
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    lab.workspace.commit_all(lab.change, "work", login=LOGIN)
    upstream = lab.fake.commit_files(lab.remote(REPO_BESSER), {"besser/upstream.py": "UP = 1\n"}, "upstream work",
                                     branch="development")

    result = asyncio.run(service.update_from_base(JORDI, change_id, REPO_BESSER))

    assert (result.outcome, result.conflicts) == ("updated", [])
    detail = service.change_detail(change_id)
    assert (detail.thread[-1].type, detail.thread[-1].text) == ("notice", result.message)
    besser = next(base for base in detail.bases if base.repo == REPO_BESSER)
    assert (besser.sha, besser.latest, besser.behind) == (upstream, upstream, 0)
    assert ("base_updated", "besser: updated") in _audit(service)


def test_update_from_base_with_a_pull_request_merges_on_github(lab, lab_env):  # noqa: F811
    service = make_service(lab.workspace)
    change_id = lab.change.id
    lab.write(REPO_BESSER, "besser/note.py", "NOTE = 1\n")
    _open(service, change_id)
    besser = lab.remote(REPO_BESSER)
    upstream = lab.fake.commit_files(besser, {"besser/upstream.py": "UP = 1\n"}, "upstream work",
                                     branch="development")

    result = asyncio.run(service.update_from_base(JORDI, change_id, REPO_BESSER))

    assert result.outcome == "updated" and "GitHub merged" in result.message
    assert lab.fake.is_ancestor(besser, upstream, lab.fake.ref(besser, BRANCH))
    assert service.change_detail(change_id).thread[-1].text == result.message


def test_update_from_base_needs_an_open_change(lab_env):  # noqa: F811
    service = make_service(lab_env.workspace())
    draft = service.workspace.create_draft(owner="ana")

    assert "draft" in _refused("state", service.update_from_base(JORDI, draft.id, REPO_BESSER)).message
    _refused("state", service.open_pull_requests(JORDI, draft.id, title=None, body=None, closes_issues=[]))


# -- Lifespan ------------------------------------------------------------------------------------------------

@pytest.fixture
def quiet_lifespan(monkeypatch):
    """The backend's lifespan without its hourly temp cleanup; the settings read again from the test's env."""
    monkeypatch.setattr(backend, "cleanup_old_temp_files", lambda: None)
    monkeypatch.setattr(backend, "schedule_cleanup", lambda: asyncio.ensure_future(asyncio.sleep(3600)))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_the_lifespan_starts_the_lab_with_its_tasks_and_shuts_it_down(lab_env, quiet_lifespan,  # noqa: F811
                                                                      monkeypatch):
    monkeypatch.setenv("BESSER_LAB_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("BESSER_LAB_PREVIEW_URLS", "https://preview.lab.example.org")
    monkeypatch.setenv("BESSER_LAB_PREVIEW_OPENAI_API_KEY", "preview-test-key")
    monkeypatch.setenv("BESSER_LAB_BASE_REFRESH_MINUTES", "60")
    proxy_ports = []

    async def fake_proxy(settings):
        proxy_ports.append(settings.lab_preview_llm_proxy_port)
        await asyncio.sleep(3600)

    monkeypatch.setattr(backend, "run_preview_llm_proxy", fake_proxy)

    async def main() -> None:
        app = FastAPI()
        async with backend.lifespan(app):
            lab = app.state.lab
            assert isinstance(lab, LabService) and app.state.lab_unavailable == ""
            await asyncio.sleep(0)
            running = {task.get_name(): task for task in asyncio.all_tasks()}
            assert {"lab-base-refresh", "lab-preview-llm-proxy"} <= set(running)
        assert running["lab-base-refresh"].cancelled() and running["lab-preview-llm-proxy"].cancelled()
        assert lab._closed

    asyncio.run(main())
    assert proxy_ports == [9300]


def test_a_lab_that_cannot_start_keeps_the_reason_for_its_routes(lab_env, quiet_lifespan, monkeypatch,  # noqa: F811
                                                                  tmp_path):
    monkeypatch.setenv("BESSER_LAB_WORKSPACE_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.delenv("BESSER_LAB_GIT_ROOT")
    monkeypatch.delenv("BESSER_LAB_STATE_DIR")

    async def main() -> None:
        app = FastAPI()
        async with backend.lifespan(app):
            assert app.state.lab is None
            assert "setup_workspace.sh" in app.state.lab_unavailable
            assert not {"lab-base-refresh", "lab-preview-llm-proxy"} & {t.get_name() for t in asyncio.all_tasks()}

    asyncio.run(main())
    assert not os.path.exists(tmp_path / "nowhere")
