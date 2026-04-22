"""HTTP-level tests for POST /besser_api/smart-preview.

The preview endpoint is pure computation — no LLM call, no API key, no
worker thread — so these tests exercise the full route with a real
ASGITransport and assert on the response body shape rather than SSE
frames.
"""

from __future__ import annotations

import asyncio

import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    CLASS_DIAGRAM_MODEL,
)


BASE_URL = "http://testserver"


def _project(diagrams: dict) -> dict:
    return {
        "id": "test-project",
        "type": "BesserProject",
        "name": "TestProject",
        "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": diagrams,
        "currentDiagramIndices": {k: 0 for k in diagrams},
    }


async def _preview_async(body: dict) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.post("/besser_api/smart-preview", json=body)


def _preview(body: dict) -> httpx.Response:
    """Sync wrapper — pytest-asyncio isn't configured here, so we drive
    the async ASGI transport via ``asyncio.run`` the same way the rest
    of the smart-generation HTTP tests do."""
    return asyncio.run(_preview_async(body))


def test_preview_class_diagram_only_picks_fastapi_backend():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "Build a FastAPI backend with JWT auth",
    }
    resp = _preview(body)

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["primary_kind"] == "class"
    assert payload["target_generator"] == "generate_fastapi_backend"
    # Explicit framework mention should push confidence high
    assert payload["target_generator_confidence"] >= 0.8
    # Turn and cost estimates must be positive integers / floats
    assert payload["estimated_turns"] > 0
    assert payload["estimated_cost_usd"] > 0
    # Model summary reports the primary alongside what's present
    assert payload["model_summary"]["primary"] == "class"


def test_preview_empty_project_returns_400():
    body = {
        "project": _project({}),
        "instructions": "Do something",
    }
    resp = _preview(body)

    assert resp.status_code == 400
    # The error message should point the user at what's missing rather
    # than claim ClassDiagram specifically — agent / sm / gui-only
    # projects are valid now.
    assert "at least one modeling artifact" in resp.json()["detail"]


def test_preview_respects_max_cost_cap():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "Build a huge thing with auth and docker and tests and stripe payments",
        "max_cost_usd": 0.05,          # artificially tight
        "max_runtime_seconds": 60,
    }
    resp = _preview(body)

    assert resp.status_code == 200
    payload = resp.json()
    # Cost is clamped to the request's max_cost_usd so the UI never
    # shows a number the user hasn't authorised.
    assert payload["estimated_cost_usd"] <= 0.05
    # Duration is clamped to the runtime cap for the same reason.
    assert payload["estimated_duration_seconds"] <= 60


def test_preview_nestjs_drops_generator():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "Generate a NestJS backend from the model",
    }
    resp = _preview(body)

    assert resp.status_code == 200
    payload = resp.json()
    # NestJS isn't in BESSER, so Phase 1 should be skipped — the LLM
    # writes from scratch. That's a valid, expected outcome.
    assert payload["target_generator"] is None


def test_preview_whitespace_instructions_rejected():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "   ",
    }
    resp = _preview(body)

    # Pydantic validator rejects whitespace-only instructions with 422
    assert resp.status_code == 422
