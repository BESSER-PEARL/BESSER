"""
Route-level tests for Supabase generation from v4 ClassDiagram payloads.

POSTs the canonical v4 wire shape ({nodes, edges}) to /besser_api/generate-output
with generator='supabase' and exercises the user_root config contract:

- valid user_root -> 200, stable HTTP filename supabase_output.sql, auth.users mirror
- empty user_root -> 200, auth integration skipped (no mirror, no RLS)
- invalid user_root -> 400 with the exact router detail message
"""

import asyncio

import pytest
import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app

BASE_URL = "http://testserver"

_INVALID_USER_ROOT_DETAIL = (
    "Invalid user_root: must be a class name matching "
    "[A-Za-z_][A-Za-z0-9_]{0,62} (or empty to skip auth integration)."
)


def _post(url: str, **kwargs) -> httpx.Response:
    async def _request():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
            return await ac.request("POST", url, **kwargs)

    return asyncio.run(_request())


@pytest.fixture(autouse=True)
def isolate_backend_test_artifacts(tmp_path, monkeypatch):
    """Keep generated test artifacts out of the repository root."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def user_rooted_class_diagram_model():
    """v4 ClassDiagram: User (isId attr) 1 --< 0..* Note."""
    return {
        "version": "4.0.0",
        "type": "ClassDiagram",
        "nodes": [
            {
                "id": "class-user", "type": "class",
                "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                "measured": {"width": 160, "height": 100},
                "data": {
                    "name": "User", "stereotype": None,
                    "attributes": [
                        {"id": "attr-user-id", "name": "id", "attributeType": "str",
                         "visibility": "public", "isId": True},
                    ],
                    "methods": [],
                },
            },
            {
                "id": "class-note", "type": "class",
                "position": {"x": 300, "y": 0}, "width": 160, "height": 100,
                "measured": {"width": 160, "height": 100},
                "data": {
                    "name": "Note", "stereotype": None,
                    "attributes": [
                        {"id": "attr-note-id", "name": "id", "attributeType": "str",
                         "visibility": "public", "isId": True},
                        {"id": "attr-note-title", "name": "title", "attributeType": "str",
                         "visibility": "public"},
                    ],
                    "methods": [],
                },
            },
        ],
        "edges": [
            {
                "id": "rel-user-note",
                "source": "class-user", "target": "class-note",
                "type": "ClassBidirectional",
                "sourceHandle": "Right", "targetHandle": "Left",
                "data": {
                    "name": "user_note",
                    "sourceRole": "user", "sourceMultiplicity": "1",
                    "targetRole": "notes", "targetMultiplicity": "0..*",
                    "points": [],
                },
            },
        ],
    }


@pytest.fixture
def supabase_input(user_rooted_class_diagram_model):
    """DiagramInput payload targeting the supabase generator."""
    return {
        "title": "SupabaseDemo",
        "model": user_rooted_class_diagram_model,
        "generator": "supabase",
    }


class TestSupabaseGenerateOutputV4:
    """POST /besser_api/generate-output with generator='supabase' (v4 model)."""

    def test_supabase_registered_in_supported_generators(self):
        async def _request():
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
                return await ac.get("/besser_api/")

        response = asyncio.run(_request())
        assert response.status_code == 200
        assert "supabase" in response.json()["supported_generators"]

    def test_generate_supabase_with_user_root(self, supabase_input):
        """Valid user_root yields auth.users mirroring + RLS, stable filename."""
        payload = {**supabase_input, "config": {"user_root": "User"}}
        response = _post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200, response.text

        content_disp = response.headers.get("content-disposition", "")
        assert 'filename="supabase_output.sql"' in content_disp

        body = response.text
        assert "auth.users" in body
        assert "handle_new_user" in body
        assert 'CREATE TABLE IF NOT EXISTS public."user"' in body
        assert 'CREATE TABLE IF NOT EXISTS public."note"' in body
        assert "ENABLE ROW LEVEL SECURITY" in body

    def test_generate_supabase_empty_user_root_skips_auth(self, supabase_input):
        """Empty user_root means 'skip auth integration' (no mirror, no RLS)."""
        payload = {**supabase_input, "config": {"user_root": ""}}
        response = _post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200, response.text

        body = response.text
        assert "auth.users" not in body
        assert "handle_new_user" not in body
        assert "ENABLE ROW LEVEL SECURITY" not in body
        # The tables themselves are still emitted.
        assert 'CREATE TABLE IF NOT EXISTS public."user"' in body
        assert 'CREATE TABLE IF NOT EXISTS public."note"' in body

    def test_generate_supabase_invalid_user_root_returns_400(self, supabase_input):
        """An injection-shaped user_root is rejected at the HTTP boundary."""
        payload = {**supabase_input, "config": {"user_root": 'Robert"; DROP'}}
        response = _post("/besser_api/generate-output", json=payload)
        assert response.status_code == 400
        assert response.json()["detail"] == _INVALID_USER_ROOT_DETAIL

    def test_generate_supabase_missing_user_root_defaults_to_user(self, supabase_input):
        """Omitting user_root entirely falls back to DEFAULT_SUPABASE_USER_ROOT."""
        response = _post("/besser_api/generate-output", json=supabase_input)
        assert response.status_code == 200, response.text
        body = response.text
        # Default 'User' matches the diagram's User class -> auth integration on.
        assert "auth.users" in body
        assert "handle_new_user" in body
