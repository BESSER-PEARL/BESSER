"""HTTP-level tests for /besser_api/smart-generate and /download-smart."""

import asyncio
import io
import json
import os
import zipfile
from typing import Iterable

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.smart_generation import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    SMART_RUN_REGISTRY,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _FakeOrchestrator,
    _FakeClient,
    _FailingOrchestrator,
    _clear_registry,
)


BASE_URL = "http://testserver"


@pytest.fixture(autouse=True)
def reset_registry():
    asyncio.run(_clear_registry())
    yield
    asyncio.run(_clear_registry())


@pytest.fixture
def stub_backend(monkeypatch):
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _FakeOrchestrator)

    def _fake_client(**kwargs):
        key = kwargs.get("api_key", "")
        if key.startswith("bad"):
            raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY.")
        return _FakeClient()

    monkeypatch.setattr(runner_module, "create_llm_client", _fake_client)


def _build_project_body(**overrides) -> dict:
    body = {
        "project": {
            "id": "test-project",
            "type": "BesserProject",
            "name": "TestProject",
            "createdAt": "2026-04-15T00:00:00Z",
            "diagrams": {
                "ClassDiagram": [
                    {
                        "id": "cd1",
                        "title": "Library",
                        "model": CLASS_DIAGRAM_MODEL,
                    }
                ]
            },
            "currentDiagramIndices": {"ClassDiagram": 0},
        },
        "instructions": "Build a simple backend for the library",
        "api_key": "sk-ant-happy-path-NEVER-LEAK",
        "provider": "anthropic",
        "llm_model": "claude-sonnet-4-5",
        "max_cost_usd": 1.0,
        "max_runtime_seconds": 60,
    }
    body.update(overrides)
    return body


async def _post_sse(body: dict) -> tuple[int, list[dict]]:
    """POST to /smart-generate and parse the SSE event stream."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        async with ac.stream(
            "POST",
            "/besser_api/smart-generate",
            json=body,
        ) as response:
            status = response.status_code
            raw = b""
            async for chunk in response.aiter_bytes():
                raw += chunk
    return status, _parse_sse_stream(raw)


def _parse_sse_stream(raw: bytes) -> list[dict]:
    events: list[dict] = []
    text = raw.decode("utf-8")
    # SSE frames are separated by blank lines
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        data_lines = [
            line[len("data: "):]
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        events.append(json.loads(payload))
    return events


class TestSmartGenerateEndpoint:
    def test_happy_path_event_ordering(self, stub_backend):
        status, events = asyncio.run(_post_sse(_build_project_body()))
        assert status == 200
        event_names = [e["event"] for e in events]
        assert event_names[0] == "start"
        assert event_names[-1] == "done"
        assert "phase" in event_names
        assert "text" in event_names

    def test_api_key_never_in_response_bytes(self, stub_backend):
        body = _build_project_body(api_key="sk-ant-happy-path-NEVER-LEAK-abcdef123")
        transport = ASGITransport(app=app)

        async def _run():
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                async with ac.stream(
                    "POST",
                    "/besser_api/smart-generate",
                    json=body,
                ) as response:
                    data = b""
                    async for chunk in response.aiter_bytes():
                        data += chunk
                    return data

        raw = asyncio.run(_run())
        assert b"sk-ant-happy-path-NEVER-LEAK" not in raw

    def test_invalid_key_yields_error_event(self, stub_backend):
        body = _build_project_body(api_key="bad-key-value")
        status, events = asyncio.run(_post_sse(body))
        assert status == 200  # SSE is always 200; errors flow as events
        errors = [e for e in events if e["event"] == "error"]
        assert len(errors) >= 1
        assert errors[-1]["code"] == "INVALID_KEY"

    def test_upstream_failure_yields_error_event(self, monkeypatch):
        monkeypatch.setattr(runner_module, "LLMOrchestrator", _FailingOrchestrator)
        monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())
        status, events = asyncio.run(_post_sse(_build_project_body()))
        assert status == 200
        errors = [e for e in events if e["event"] == "error"]
        assert any(e["code"] == "UPSTREAM_LLM" for e in errors)
        assert not any(e["event"] == "done" for e in events)

    def test_missing_class_diagram_yields_bad_request_event(self, stub_backend):
        body = _build_project_body()
        body["project"]["diagrams"] = {}
        body["project"]["currentDiagramIndices"] = {}
        status, events = asyncio.run(_post_sse(body))
        assert status == 200
        errors = [e for e in events if e["event"] == "error"]
        assert any(e["code"] == "BAD_REQUEST" for e in errors)


class TestDownloadEndpoint:
    def test_download_returns_file_then_404_on_second_call(self, stub_backend):
        status, events = asyncio.run(_post_sse(_build_project_body()))
        assert status == 200
        done = [e for e in events if e["event"] == "done"][0]
        run_id = done["downloadUrl"].rsplit("/", 1)[-1]

        transport = ASGITransport(app=app)

        async def _get(path: str) -> httpx.Response:
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get(path)

        # First GET succeeds
        r1 = asyncio.run(_get(f"/besser_api/download-smart/{run_id}"))
        assert r1.status_code == 200
        assert r1.content.startswith(b"# fake generated content")

        # Second GET is 404 (single-use)
        r2 = asyncio.run(_get(f"/besser_api/download-smart/{run_id}"))
        assert r2.status_code == 404

    def test_unknown_run_id_returns_404(self, stub_backend):
        transport = ASGITransport(app=app)

        async def _get() -> httpx.Response:
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get(
                    "/besser_api/download-smart/" + "0" * 32
                )

        r = asyncio.run(_get())
        assert r.status_code == 404

    def test_malformed_run_id_returns_422(self, stub_backend):
        """Path regex must reject non-hex IDs."""
        transport = ASGITransport(app=app)

        async def _get() -> httpx.Response:
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get("/besser_api/download-smart/not-a-hex-id")

        r = asyncio.run(_get())
        assert r.status_code == 422


class TestApiInfo:
    def test_root_endpoints_advertise_smart_generate(self):
        transport = ASGITransport(app=app)

        async def _get() -> httpx.Response:
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get("/besser_api/")

        r = asyncio.run(_get())
        assert r.status_code == 200
        endpoints = r.json()["endpoints"]
        assert "smart_generate" in endpoints
        assert "download_smart" in endpoints
