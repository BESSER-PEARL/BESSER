"""HTTP-level tests for POST /besser_api/smart-preview.

The preview endpoint is pure computation — no LLM call, no API key, no
worker thread — so these tests exercise the full route with a real
ASGITransport and assert on the response body shape rather than SSE
frames.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    AssembledModels,
)
from besser.utilities.web_modeling_editor.backend.services.smart_generation.preview import (
    _predict_target_generator,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_model_assembly import (
    BPMN_DIAGRAM_MODEL,
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


def test_preview_accepts_real_wme_bpmn_bucket():
    body = {
        "project": _project({
            "BPMN": [{
                "id": "bpmn-1",
                "title": "Approval",
                "model": BPMN_DIAGRAM_MODEL,
            }],
        }),
        "instructions": "Generate an executable BPMN workflow",
    }

    resp = _preview(body)

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["primary_kind"] == "bpmn"
    assert payload["target_generator"] == "generate_bpmn"


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
    assert any("cost limit" in note for note in payload["notes"])
    assert any("runtime limit" in note for note in payload["notes"])


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


def test_preview_modify_mode_matches_incremental_execution():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "Add filtering to the existing book list",
        "mode": "modify",
        "base_run_id": "a" * 32,
    }
    resp = _preview(body)

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["execution_mode"] == "modify"
    assert payload["target_generator"] is None
    assert "Modify the previous generated app" in payload["summary"]
    assert any("Incremental mode" in note for note in payload["notes"])


def test_preview_modify_mode_requires_base_run_id():
    body = {
        "project": _project({
            "ClassDiagram": [
                {"id": "cd1", "title": "Library", "model": CLASS_DIAGRAM_MODEL}
            ]
        }),
        "instructions": "Add filtering",
        "mode": "modify",
    }
    resp = _preview(body)

    assert resp.status_code == 422


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


@pytest.mark.parametrize(
    ("assembled", "instructions", "expected"),
    [
        (
            AssembledModels(primary_kind="class", domain_model=object()),
            "Generate a Supabase project",
            "generate_supabase",
        ),
        (
            AssembledModels(primary_kind="object", object_model=object()),
            "Export JSON fixtures",
            "generate_json_object",
        ),
        (
            AssembledModels(primary_kind="agent", agent_model=object()),
            "Generate a BAF chatbot",
            "generate_baf",
        ),
        (
            AssembledModels(primary_kind="bpmn", bpmn_model=object()),
            "Generate BPMN process XML",
            "generate_bpmn",
        ),
        (
            AssembledModels(primary_kind="nn", nn_model=object()),
            "Generate a PyTorch model",
            "generate_pytorch",
        ),
        (
            AssembledModels(primary_kind="nn", nn_model=object()),
            "Generate a TensorFlow Keras model",
            "generate_tensorflow",
        ),
    ],
)
def test_preview_supports_new_registered_generators(
    assembled, instructions, expected,
):
    generator, confidence = _predict_target_generator(assembled, instructions)
    assert generator == expected
    assert confidence >= 0.8
