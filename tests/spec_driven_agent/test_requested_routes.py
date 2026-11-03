"""A path the user wrote must exist verbatim before Phase 2 may be skipped.

Live run, deterministic FastAPI generator on: "A Python FastAPI REST API with
CRUD for books and members, SQLite, plus a GET /stats endpoint returning the
number of books and members, and request logging middleware." The scaffold has
the logging middleware and ``GET /statistics`` (book_count, member_count). The
gap planner returned ``[]``, Phase 2 was skipped ("scaffold already covers the
request"), the requirements judge accepted ``/statistics`` as ``/stats``, and
the delivered app had no ``/stats``.
"""
from __future__ import annotations

from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.requested_routes import (
    requested_route_tasks,
    requested_routes,
    route_declared,
)
from besser.spec_driven_agent.providers.llm_client import UsageTracker

LIVE_REQUEST = (
    "A Python FastAPI REST API with CRUD for books and members, SQLite, plus a GET /stats "
    "endpoint returning the number of books and members, and request logging middleware."
)
PURE_CRUD_REQUEST = "A Python FastAPI REST API with CRUD for books and members, SQLite."

# The routes of the live run's generated scaffold (backend/main_api.py, routers/).
_MAIN_API = '''
from fastapi import FastAPI, Request
from routers.book import router as book_router
from routers.member import router as member_router

app = FastAPI()
app.include_router(book_router)
app.include_router(member_router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    return await call_next(request)

@app.get("/", tags=["System"])
def root():
    return {}

@app.get("/health", tags=["System"])
def health_check():
    return {}

@app.get("/statistics", tags=["System"])
def get_statistics():
    return {"member_count": 0, "book_count": 0}
'''
_BOOK_ROUTER = '''
from fastapi import APIRouter
router = APIRouter()

@router.get("/book/", response_model=None, tags=["Book"])
async def get_all_book():
    return []

@router.get("/book/{book_id}/", response_model=None, tags=["Book"])
async def get_book(book_id: int):
    return {}

@router.post("/book/", response_model=None, tags=["Book"])
async def create_book():
    return {}
'''


def _scaffold(root):
    backend = root / "backend"
    (backend / "routers").mkdir(parents=True)
    (backend / "main_api.py").write_text(_MAIN_API, encoding="utf-8")
    (backend / "routers" / "book.py").write_text(_BOOK_ROUTER, encoding="utf-8")
    (backend / "routers" / "member.py").write_text(
        _BOOK_ROUTER.replace("book", "member"), encoding="utf-8")
    return backend


class TestRequestedRoutes:

    def test_live_request_names_get_stats(self):
        assert requested_routes(LIVE_REQUEST) == [("GET", "/stats")]

    def test_pure_crud_request_names_no_path(self):
        assert requested_routes(PURE_CRUD_REQUEST) == []

    def test_quoted_param_and_noun_forms(self):
        text = ("Add `POST /orders/{id}/cancel`, a /health/live endpoint, "
                "and GET /reports/:year.")
        assert requested_routes(text) == [
            ("POST", "/orders/{id}/cancel"), (None, "/health/live"), ("GET", "/reports/:year"),
        ]

    def test_negated_and_non_route_slashes_are_ignored(self):
        text = ("Use SQLite/PostgreSQL, input/output logs, docs at http://localhost:8000/docs, "
                "and do not add DELETE /books, without a /admin endpoint.")
        assert requested_routes(text) == []


class TestRouteDeclared:

    def test_statistics_does_not_serve_stats(self, tmp_path):
        _scaffold(tmp_path)
        assert not route_declared(tmp_path, "GET", "/stats")
        assert route_declared(tmp_path, "GET", "/statistics")

    def test_trailing_slash_params_and_method(self, tmp_path):
        _scaffold(tmp_path)
        assert route_declared(tmp_path, "GET", "/book/{id}")
        assert route_declared(tmp_path, None, "/book")
        assert not route_declared(tmp_path, "DELETE", "/book/{id}")

    def test_router_prefix_and_api_route(self, tmp_path):
        (tmp_path / "app.py").write_text(
            'from fastapi import APIRouter, FastAPI\n'
            'app = FastAPI()\n'
            'router = APIRouter(prefix="/api")\n'
            '@router.get("/stats")\n'
            'def stats():\n    return {}\n'
            '@app.api_route("/ping", methods=["GET", "HEAD"])\n'
            'def ping():\n    return {}\n',
            encoding="utf-8")
        assert route_declared(tmp_path, "GET", "/api/stats")
        assert route_declared(tmp_path, "GET", "/ping")
        assert not route_declared(tmp_path, "POST", "/ping")

    def test_task_verifies_once_the_route_exists(self, tmp_path):
        backend = _scaffold(tmp_path)
        [task] = requested_route_tasks(tmp_path, LIVE_REQUEST)
        assert "GET /stats" in task["text"]
        assert task["verify"]() is False
        with open(backend / "main_api.py", "a", encoding="utf-8") as fh:
            fh.write('\n@app.get("/stats")\ndef stats():\n    return {}\n')
        assert task["verify"]() is True
        assert requested_route_tasks(tmp_path, LIVE_REQUEST) == []


# ----------------------------------------------------------------------
# The Phase 2 skip, with the planner's wrong "covered" verdict
# ----------------------------------------------------------------------

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")
        self.chat_calls = 0

    def chat(self, system, messages, tools):
        self.chat_calls += 1
        return {"stop_reason": "end_turn", "content": [_Block("Done")]}


def _orchestrator(tmp_path, monkeypatch, instructions):
    class _SM:
        name = "DummySM"

    _scaffold(tmp_path)
    orch = LLMOrchestrator(llm_client=_Client(), state_machines=[_SM()], output_dir=str(tmp_path),
                           enable_requirements_ledger=False)
    orch._generator_used = "generate_fastapi_backend"
    orch._instructions = instructions
    orch.max_turns = 2
    # The planner's live verdict: the scaffold covers the request.
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm", lambda **kw: [])
    progress: list[tuple] = []
    orch.on_progress = lambda *a: progress.append(a)
    return orch, progress


def test_missing_requested_endpoint_runs_phase2_with_a_task(tmp_path, monkeypatch):
    orch, progress = _orchestrator(tmp_path, monkeypatch, LIVE_REQUEST)

    orch._run_phase2(LIVE_REQUEST, extra_issues=[])

    assert not any(a[1] == "__customize_skipped__" for a in progress)
    assert orch.client.chat_calls >= 1
    texts = [task["text"] for task in orch.executor._tasks]
    assert any("GET /stats" in text for text in texts)


def test_pure_crud_request_still_skips_phase2(tmp_path, monkeypatch):
    orch, progress = _orchestrator(tmp_path, monkeypatch, PURE_CRUD_REQUEST)

    orch._run_phase2(PURE_CRUD_REQUEST, extra_issues=[])

    assert any(a[1] == "__customize_skipped__" for a in progress)
    assert orch.client.chat_calls == 0
