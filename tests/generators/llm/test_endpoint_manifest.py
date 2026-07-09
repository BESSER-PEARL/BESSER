"""Tests for build_endpoint_manifest — the exact backend route list handed to
the Phase-2 agent so the LLM-authored frontend stops guessing URLs (the #1
"builds but 404s" failure).

Correctness is about producing the EXACT served path: right prefix, right
trailing slash, no invented plurals. These pin the parser against the real
generator shape (full path in the decorator, no prefix) plus the defensive
prefix cases.
"""
import os

from besser.generators.llm.prompt_builder import build_endpoint_manifest


def _mk(tmp_path, files: dict) -> str:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return str(tmp_path)


def test_full_path_in_decorator_no_prefix(tmp_path):
    # The real BackendGenerator shape: APIRouter() with no prefix, full path
    # (singular, trailing slash) baked into each decorator.
    d = _mk(tmp_path, {
        "backend/main_api.py": (
            "from fastapi import FastAPI\n"
            "from routers import book as book_router\n"
            "app = FastAPI()\n"
            "app.include_router(book_router.router)\n"
            'if __name__ == "__main__":\n'
            "    import uvicorn\n"
            '    uvicorn.run(app, host="0.0.0.0", port=8000)\n'
        ),
        "backend/routers/book.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/book/")\n'
            "def list_book(): ...\n"
            '@router.post("/book/")\n'
            "def create_book(): ...\n"
            '@router.get("/book/{book_id}/")\n'
            "def get_book(): ...\n"
            '@router.put("/book/{book_id}/")\n'
            "def update_book(): ...\n"
            '@router.delete("/book/{book_id}/")\n'
            "def delete_book(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "http://localhost:8000" in m
    # exact served paths, singular, trailing slash
    assert "/book/" in m
    assert "/book/{book_id}/" in m
    # methods merged per path
    assert "GET, POST" in m
    assert "GET, PUT, DELETE" in m
    # never invents a plural or an /api prefix as an actual route line
    assert "/books/" not in m
    assert "\n  GET                  /api" not in m


def test_apirouter_prefix_is_prepended(tmp_path):
    d = _mk(tmp_path, {
        "app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "app/routers/item.py": (
            "from fastapi import APIRouter\n"
            'router = APIRouter(prefix="/items")\n'
            '@router.get("/")\n'
            "def list_items(): ...\n"
            '@router.get("/{item_id}")\n'
            "def get_item(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/items/" in m
    assert "/items/{item_id}" in m


def test_include_router_prefix_via_import_alias(tmp_path):
    d = _mk(tmp_path, {
        "main.py": (
            "from fastapi import FastAPI\n"
            "from routers import user as user_router\n"
            "app = FastAPI()\n"
            'app.include_router(user_router.router, prefix="/api/v1")\n'
        ),
        "routers/user.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/user/")\n'
            "def list_user(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/api/v1/user/" in m


def test_no_routes_returns_empty(tmp_path):
    d = _mk(tmp_path, {"backend/database.py": "engine = None\n"})
    assert build_endpoint_manifest(d) == ""


def test_default_port_when_no_uvicorn(tmp_path):
    d = _mk(tmp_path, {
        "routers/x.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/x/")\n'
            "def x(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "http://localhost:8000" in m


def test_skips_node_modules(tmp_path):
    d = _mk(tmp_path, {
        "routers/real.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/real/")\n'
            "def r(): ...\n"
        ),
        "frontend/node_modules/pkg/decoy.py": (
            'router.get("/should_not_appear/")\n'
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/real/" in m
    assert "should_not_appear" not in m


def test_no_double_slashes(tmp_path):
    d = _mk(tmp_path, {
        "main.py": (
            "from routers import a as a_router\n"
            'app.include_router(a_router.router, prefix="/api/")\n'
        ),
        "routers/a.py": (
            "from fastapi import APIRouter\n"
            'router = APIRouter(prefix="/a/")\n'
            '@router.get("/")\n'
            "def a(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    # inspect only the route lines (indented "METHOD  /path"), not the http:// header
    route_lines = [ln for ln in m.splitlines() if ln.startswith("  ") and "/" in ln]
    assert route_lines
    assert all("//" not in ln for ln in route_lines)
    assert "/api/a/" in m
