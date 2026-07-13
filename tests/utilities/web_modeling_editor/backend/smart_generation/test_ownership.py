"""Authentication and owner-isolation tests for SmartGen resources."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import (
    smart_generation_router as router_module,
)
from besser.utilities.web_modeling_editor.backend.services import (
    principal as principal_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation import (
    runner as runner_module,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
    SmartRunEntry,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_import_github_run import (
    _FakeGitHubService,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _FakeClient,
    _FakeOrchestrator,
    _clear_registry,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_smart_generation_router import (
    _build_project_body,
    _parse_sse_stream,
)


BASE_URL = "http://testserver"
ALICE_OWNER = "github:101"
BOB_OWNER = "github:202"


@pytest.fixture(autouse=True)
def reset_run_state():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def authenticated_sessions(monkeypatch):
    sessions = {
        "alice-session": {
            "access_token": "alice-token",
            "username": "alice",
            "github_user_id": 101,
        },
        "bob-session": {
            "access_token": "bob-token",
            "username": "bob",
            "github_user_id": 202,
        },
    }
    monkeypatch.setenv("BESSER_SMART_GEN_AUTH_REQUIRED", "true")
    monkeypatch.setattr(
        principal_module,
        "get_user_session",
        lambda session_id: sessions.get(session_id),
    )
    return sessions


def _headers(session: str) -> dict[str, str]:
    return {"X-GitHub-Session": session}


async def _request(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    session: str | None = None,
) -> httpx.Response:
    transport = ASGITransport(app=app)
    headers = _headers(session) if session else {}
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        return await client.request(method, path, json=body, headers=headers)


def _preview_body() -> dict:
    generated = _build_project_body()
    return {
        "project": generated["project"],
        "instructions": generated["instructions"],
    }


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/besser_api/smart-generate", _build_project_body()),
        ("POST", "/besser_api/smart-preview", _preview_body()),
        ("POST", "/besser_api/resume-smart-gen/" + "a" * 32, _build_project_body()),
        ("POST", "/besser_api/cancel-smart-gen/" + "a" * 32, None),
        ("GET", "/besser_api/download-smart/" + "a" * 32, None),
        (
            "POST",
            "/besser_api/push-smart-to-github",
            {"run_id": "a" * 32, "deploy_config": {"repo_name": "example"}},
        ),
        (
            "POST",
            "/besser_api/import-github-run",
            {"owner": "example", "repo": "example"},
        ),
    ],
)
def test_required_auth_rejects_every_smartgen_operation(
    monkeypatch,
    method,
    path,
    body,
):
    monkeypatch.setenv("BESSER_SMART_GEN_AUTH_REQUIRED", "true")
    response = asyncio.run(_request(method, path, body=body))
    assert response.status_code == 401


def test_production_cannot_disable_auth(monkeypatch):
    monkeypatch.setenv("BESSER_ENV", "production")
    monkeypatch.setenv("BESSER_SMART_GEN_AUTH_REQUIRED", "false")
    assert principal_module.smart_gen_auth_required() is True


def test_github_principal_prefers_stable_numeric_id(monkeypatch):
    monkeypatch.setattr(
        principal_module,
        "get_user_session",
        lambda _session_id: {
            "username": "RenameableLogin",
            "github_user_id": 987654,
        },
    )
    principal = principal_module.principal_from_github_session("session")
    assert principal is not None
    assert principal.subject == "github:987654"
    assert principal.provider == "github"


def test_completed_run_download_is_owner_scoped(
    monkeypatch,
    authenticated_sessions,
):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        runner_module,
        "create_llm_client",
        lambda **_kwargs: _FakeClient(),
    )

    async def _exercise() -> None:
        response = await _request(
            "POST",
            "/besser_api/smart-generate",
            body=_build_project_body(),
            session="alice-session",
        )
        assert response.status_code == 200
        done = [
            event
            for event in _parse_sse_stream(response.content)
            if event["event"] == "done"
        ][0]
        run_id = done["runId"]
        entry = await SMART_RUN_REGISTRY.get(run_id)
        assert entry is not None
        assert entry.owner_id == ALICE_OWNER

        denied = await _request(
            "GET",
            f"/besser_api/download-smart/{run_id}",
            session="bob-session",
        )
        assert denied.status_code == 404

        allowed = await _request(
            "GET",
            f"/besser_api/download-smart/{run_id}",
            session="alice-session",
        )
        assert allowed.status_code == 200

    asyncio.run(_exercise())


def test_cancel_only_signals_the_owner(authenticated_sessions):
    run_id = "c" * 32

    async def _exercise() -> None:
        event = await runner_module.reserve_active_run(run_id, ALICE_OWNER)
        assert event is not None
        try:
            denied = await _request(
                "POST",
                f"/besser_api/cancel-smart-gen/{run_id}",
                session="bob-session",
            )
            assert denied.status_code == 200
            assert denied.json()["status"] == "not_found"
            assert event.is_set() is False

            allowed = await _request(
                "POST",
                f"/besser_api/cancel-smart-gen/{run_id}",
                session="alice-session",
            )
            assert allowed.json()["status"] == "cancelled"
            assert event.is_set() is True
        finally:
            await runner_module.release_active_run(run_id, event)

    asyncio.run(_exercise())


def test_request_logs_redact_run_ids(caplog, authenticated_sessions):
    run_id = "f" * 32
    with caplog.at_level(logging.INFO, logger="besser.backend.requests"):
        response = asyncio.run(_request(
            "GET",
            f"/besser_api/download-smart/{run_id}",
            session="alice-session",
        ))
    assert response.status_code == 404
    assert run_id not in caplog.text
    assert "/besser_api/download-smart/{run_id}" in caplog.text


def test_resume_hides_another_owners_checkpoint(
    monkeypatch,
    authenticated_sessions,
):
    run_id = "d" * 32
    monkeypatch.setattr(router_module, "_locate_run_temp_dir", lambda _run_id: "run-dir")
    monkeypatch.setattr(
        router_module,
        "load_checkpoint",
        lambda _directory: type(
            "Checkpoint",
            (),
            {"run_id": run_id, "owner_id": ALICE_OWNER},
        )(),
    )

    async def _exercise() -> None:
        event = await runner_module.reserve_active_run(run_id, ALICE_OWNER)
        assert event is not None
        try:
            denied = await _request(
                "POST",
                f"/besser_api/resume-smart-gen/{run_id}",
                body=_build_project_body(),
                session="bob-session",
            )
            assert denied.status_code == 404

            allowed_owner = await _request(
                "POST",
                f"/besser_api/resume-smart-gen/{run_id}",
                body=_build_project_body(),
                session="alice-session",
            )
            assert allowed_owner.status_code == 409
        finally:
            await runner_module.release_active_run(run_id, event)

    asyncio.run(_exercise())


def test_modify_and_push_cannot_use_another_owners_run(
    monkeypatch,
    authenticated_sessions,
):
    run_id = "e" * 32
    temp_dir = tempfile.mkdtemp(prefix="smartgen_owner_test_")
    artifact = os.path.join(temp_dir, "main.py")
    with open(artifact, "w", encoding="utf-8") as handle:
        handle.write("print('owned')\n")
    asyncio.run(SMART_RUN_REGISTRY.put(
        run_id,
        SmartRunEntry(
            file_path=artifact,
            file_name="main.py",
            is_zip=False,
            temp_dir=temp_dir,
            created_at=time.time(),
            owner_id=ALICE_OWNER,
        ),
    ))
    monkeypatch.setattr(router_module, "get_user_token", lambda _session: "token")

    modify_body = _build_project_body(mode="modify", base_run_id=run_id)
    modify = asyncio.run(_request(
        "POST",
        "/besser_api/smart-generate",
        body=modify_body,
        session="bob-session",
    ))
    assert modify.status_code == 404
    assert modify.json()["detail"] == "base_run_expired"

    preview = asyncio.run(_request(
        "POST",
        "/besser_api/smart-preview",
        body={
            "project": modify_body["project"],
            "instructions": modify_body["instructions"],
            "mode": "modify",
            "base_run_id": run_id,
        },
        session="bob-session",
    ))
    assert preview.status_code == 404
    assert preview.json()["detail"] == "base_run_expired"

    push = asyncio.run(_request(
        "POST",
        "/besser_api/push-smart-to-github",
        body={"run_id": run_id, "deploy_config": {"repo_name": "owned-app"}},
        session="bob-session",
    ))
    assert push.status_code == 404
    assert push.json()["detail"] == "run_expired"


def test_imported_run_is_bound_to_importing_principal(
    monkeypatch,
    authenticated_sessions,
):
    fake_github = _FakeGitHubService(with_model=False)
    monkeypatch.setattr(router_module, "get_user_token", lambda _session: "token")
    monkeypatch.setattr(
        router_module,
        "create_github_service",
        lambda _token: fake_github,
    )

    response = asyncio.run(_request(
        "POST",
        "/besser_api/import-github-run",
        body={"owner": "example", "repo": "example"},
        session="bob-session",
    ))
    assert response.status_code == 200
    entry = asyncio.run(SMART_RUN_REGISTRY.get(response.json()["run_id"]))
    assert entry is not None
    assert entry.owner_id == BOB_OWNER
