"""The real editor backend with GitHub replaced by the git-backed stand-in, for browser checks.

Every BESSER endpoint, the OAuth callback and ``GitHubService`` run their real
code; only github.com and api.github.com answer, from local git repositories
(``fake_github.FakeGitHub``). Two LLM-backed services are stood in for, so a
whole GitHub round trip runs without a GitHub account or an API key:

- ``POST /besser_api/spec-driven/generate`` runs a short scripted modify run:
  it seeds a real run from its base run, edits code and adds a ``Review``
  class to the model, then streams the real event types.
- a modeling-agent WebSocket on port 8766 answers every chat message with a
  Spec-Driven modify request.

Run from the BESSER root (git on PATH)::

    python -m tests.utilities.web_modeling_editor.backend.github_harness

then the editor with ``UML_BOT_WS_URL=ws://localhost:8766`` on port 5173
(``npx vite --port 5173`` in ``frontend/packages/webapp``). "Connect GitHub"
skips GitHub's consent page and lands on the real callback as user ``ana``.

``/mock`` endpoints act on GitHub as another person would:
``POST /mock/teammate-class?repo=library&name=Member``,
``POST /mock/teammate-rename?repo=library&element=<id>&name=Writer``,
``POST /mock/teammate-code?repo=…&path=…&content=…``,
``POST /mock/merge-pull?repo=…&number=1``, and to inspect:
``GET /mock/files?repo=…[&ref=…]``, ``GET /mock/state?repo=…``.
"""

import asyncio
import copy
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

WORKDIR = tempfile.mkdtemp(prefix="besser_github_harness_")
os.environ.update(
    GITHUB_CLIENT_ID="fake-client", GITHUB_CLIENT_SECRET="fake-secret",
    DEPLOYMENT_URL=os.environ.get("HARNESS_EDITOR_URL", "http://localhost:5173"),
    BESSER_LLM_RUN_WORKSPACE_ROOT=os.path.join(WORKDIR, "runs"),
    # Offers the keyless tier in the editor; the stand-in run below never calls it.
    BESSER_FREE_LLM_BASE_URL="http://127.0.0.1:9/v1", BESSER_FREE_LLM_MODEL="stand-in",
)

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import RedirectResponse, StreamingResponse  # noqa: E402

from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub, github_client_class  # noqa: E402

fake = FakeGitHub(login="ana", token="gho_fake_token", root=os.path.join(WORKDIR, "github"))
# Before the backend is imported, so every GitHub request it makes reaches the fake.
httpx.AsyncClient = github_client_class(fake)

from besser.utilities.web_modeling_editor.backend.backend import app  # noqa: E402
from besser.utilities.web_modeling_editor.backend.services.deployment import github_oauth, repo_model  # noqa: E402
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (  # noqa: E402
    SMART_RUN_REGISTRY, SmartRunEntry, _seed_workspace_from_base, create_run_workspace, run_changes,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (  # noqa: E402
    DoneEvent, PhaseEvent, StartEvent, ToolCallEvent, format_sse,
)

BESSER_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = (BESSER_ROOT / "besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main"
            / "templates/pattern/structural/Library.json")
AGENT_PORT = 8766
BACKEND_PORT = 9000
# Empty repos beyond GitHub's first page, so the repo list's paging is exercised.
FILLER_REPOS = 120


def envelope(model: dict, name: str, project_id: str) -> dict:
    """A V2 project export holding one class diagram."""
    return {
        "project": {
            "id": project_id, "type": "Project", "schemaVersion": 3, "name": name, "description": f"{name} project",
            "owner": "ana", "createdAt": "2026-10-01T09:00:00Z", "currentDiagramType": "ClassDiagram",
            "currentDiagramIndices": {"ClassDiagram": 0},
            "diagrams": {"ClassDiagram": [{"id": f"{project_id}-cd", "title": name, "model": model,
                                           "lastUpdate": "2026-10-01T09:00:00Z"}]},
            "settings": {},
        },
        "exportedAt": "2026-10-01T09:00:00Z", "version": "2.0.0",
    }


def with_class(model: dict, name: str, dx: int = 420) -> dict:
    """``model`` plus class ``name`` with one attribute, placed beside an existing class (Apollon format)."""
    model = copy.deepcopy(model)
    template = next(e for e in model["elements"].values() if e.get("type") == "Class")
    class_id, attr_id = f"cls-{name.lower()}", f"attr-{name.lower()}"
    bounds = template["bounds"]
    model["elements"][class_id] = {**template, "id": class_id, "name": name, "attributes": [attr_id], "methods": [],
                                   "bounds": {**bounds, "x": bounds["x"] + dx}}
    model["elements"][attr_id] = {"id": attr_id, "name": "+ id: int", "type": "ClassAttribute", "owner": class_id,
                                  "bounds": {**bounds, "x": bounds["x"] + dx, "y": bounds["y"] + 40, "height": 30}}
    return model


BOOKSHOP_MODELS = '''from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    books = relationship("Book", back_populates="author")


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    price = Column(Float, nullable=False)
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False)
    author = relationship("Author", back_populates="books")
'''


def seed() -> None:
    """A BESSER repo, a plain code repo with build output, organization repos and filler."""
    library_model = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    library = fake.create_repo("library", topics=["besser"], files={
        **repo_model.model_files(envelope(library_model, "Library", "proj-library")), "README.md": "# Library\n"})
    fake.commit_files(library, {"app/main.py": "print('library app')\n"},
                      repo_model.with_origin("Build the app", repo_model.ORIGIN_AGENT))
    fake.create_repo("bookshop-api", private=False, files={
        "app/models.py": BOOKSHOP_MODELS,
        "app/main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n",
        "requirements.txt": "fastapi\nsqlalchemy\n",
        "scripts/start.sh": ("#!/bin/sh\nuvicorn app.main:app\n", "100755"),
        "current": ("app/main.py", "120000"),
        "build/bundle.js": "console.log('built');\n",
        "data/shop.db": "SQLite format 3\0",
        "README.md": "# bookshop-api\n",
    })
    fake.create_repo("crm", owner="acme", topics=["besser"], files=repo_model.model_files(
        envelope(with_class(library_model, "Customer"), "CRM", "proj-crm")))
    fake.create_repo("handbook", owner="acme", can_push=False, files={"README.md": "# handbook\n"})
    for index in range(FILLER_REPOS):
        fake.create_repo(f"sandbox-{index:03d}")


def _repo(owner: str, repo: str):
    return fake.repos[f"{owner}/{repo}".lower()]


def _model_of(target) -> tuple[dict, dict]:
    current = json.loads(fake.files(target)[repo_model.MODEL_FILE][1])
    return current, current["project"]["diagrams"]["ClassDiagram"][0]["model"]


mock = FastAPI()


@mock.post("/teammate-class")
def teammate_class(repo: str = "library", name: str = "Member", owner: str = "ana"):
    """Another person adds a class on GitHub, writing the model files as BESSER does."""
    target = _repo(owner, repo)
    current, model = _model_of(target)
    project = current["project"]
    files = repo_model.model_files(envelope(with_class(model, name, dx=-420), project["name"], project["id"]))
    return {"sha": fake.commit_files(target, files, repo_model.with_origin(f"Add {name}", repo_model.ORIGIN_EDITOR),
                                     author="ben")}


@mock.post("/teammate-rename")
def teammate_rename(repo: str = "library", element: str = "", name: str = "", owner: str = "ana"):
    target = _repo(owner, repo)
    current, model = _model_of(target)
    element = element or next(k for k, e in model["elements"].items() if e.get("type") == "Class")
    model["elements"][element]["name"] = name
    files = repo_model.model_files(envelope(model, current["project"]["name"], current["project"]["id"]))
    return {"sha": fake.commit_files(target, files, "Rename on github.com", author="ben"), "element": element}


@mock.post("/teammate-code")
def teammate_code(repo: str, path: str, content: str, owner: str = "ana"):
    return {"sha": fake.commit_files(_repo(owner, repo), {path: content}, f"Edit {path}", author="ben")}


@mock.post("/merge-pull")
def merge_pull(repo: str, number: int, owner: str = "ana"):
    return {"sha": fake.merge_pull(_repo(owner, repo), number)}


@mock.get("/files")
def files(repo: str, owner: str = "ana", ref: str = ""):
    return {path: {"mode": mode, "size": len(data), "text": data.decode("utf-8", "replace")[:400]}
            for path, (mode, data) in fake.files(_repo(owner, repo), ref or None).items()}


@mock.get("/state")
def state(repo: str, owner: str = "ana"):
    target = _repo(owner, repo)
    branches = fake.out(target, "for-each-ref", "--format=%(refname:lstrip=2) %(objectname:short)").splitlines()
    log = fake.out(target, "log", "--format=%h %an: %s", "-15", "--all", "--graph")
    return {"branches": branches, "log": log.splitlines(), "pulls": target.pulls, "topics": target.topics,
            "revoked": fake.revoked}


async def login(request: Request):
    """"Connect GitHub": skip github.com's consent page, keep the real callback."""
    state_token = uuid.uuid4().hex
    github_oauth._oauth_sessions.set(state_token, {"client_ip": "harness"})
    return RedirectResponse(f"/besser_api/github/auth/callback?code=fake&state={state_token}")


REVIEW_MODEL = '''

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    stars = Column(Integer, nullable=False)
'''


async def generate(request: Request):
    """A real modify run without an LLM: seeded from its base, code edited, a class added."""
    body = await request.json()
    base = await SMART_RUN_REGISTRY.get(body.get("base_run_id") or "")
    run_id = uuid.uuid4().hex
    workdir = create_run_workspace(run_id)
    if base is not None:
        _seed_workspace_from_base(base.temp_dir, workdir)
    target = os.path.join(workdir, "app", "models.py")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(REVIEW_MODEL)
    with open(os.path.join(workdir, "app", "reviews.py"), "w", encoding="utf-8") as fh:
        fh.write("def list_reviews():\n    return []\n")
    project = copy.deepcopy(body["project"])
    index = project.get("currentDiagramIndices", {}).get("ClassDiagram", 0)
    diagram = project["diagrams"]["ClassDiagram"][index]
    entry = SmartRunEntry(file_path=target, file_name="app.zip", is_zip=True, temp_dir=workdir,
                          created_at=time.time())
    changes = []
    if any(e.get("type") == "Class" for e in diagram["model"].get("elements", {}).values()):
        diagram["model"] = with_class(diagram["model"], "Review", dx=840)
        entry.updated_project_export = {"project": project, "exportedAt": "2026-10-29T12:00:00Z", "version": "2.0.0"}
        changes = ["class Review added"]
    await SMART_RUN_REGISTRY.put(run_id, entry)
    recipe = {"history": [{"mode": "modify"}],
              "tool_calls": [{"tool": "modify_file", "input": {"path": "app/models.py"}},
                             {"tool": "write_file", "input": {"path": "app/reviews.py"}}],
              "model_change": {"changes": changes}}
    with open(os.path.join(workdir, ".besser_recipe.json"), "w", encoding="utf-8") as fh:
        json.dump(recipe, fh)

    async def stream():
        for event in (
            StartEvent(runId=run_id, provider="free", llmModel="stand-in", maxCost=1.0, maxRuntime=600),
            PhaseEvent(phase="customize", message="Updating the app"),
            ToolCallEvent(turn=1, tool="modify_file", detail="path=app/models.py"),
            ToolCallEvent(turn=2, tool="write_file", detail="path=app/reviews.py"),
            PhaseEvent(phase="validate", message="0 blockers"),
            DoneEvent(runId=run_id, downloadUrl=f"/besser_api/spec-driven/download/{run_id}", fileName="app.zip",
                      isZip=True, recipe=recipe, blockerCount=0, **run_changes(recipe)),
        ):
            yield format_sse(event)
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream")


async def agent(ws):
    """Every chat message becomes a Spec-Driven modify request (the client wraps messages twice)."""
    async for raw in ws:
        frame = json.loads(raw)
        if frame.get("action") != "user_message":
            continue
        turn = json.loads(frame["message"])
        if turn.get("action") != "user_message":
            continue
        await ws.send(json.dumps({
            "action": "trigger_smart_generator", "mode": "modify", "turnId": turn.get("turnId"),
            "instructions": turn.get("message", ""), "message": "Updating the app on top of its code.",
        }))


def main() -> None:
    seed()
    app.mount("/mock", mock)
    route = type(app.router.routes[0])
    # Inserted first, so they win over the real routes of the same path.
    app.router.routes.insert(0, route("/besser_api/github/auth/login", login, methods=["GET"]))
    app.router.routes.insert(0, route("/besser_api/spec-driven/generate", generate, methods=["POST"]))

    async def serve_agent():
        async with websockets.serve(agent, "127.0.0.1", AGENT_PORT):
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(serve_agent()), daemon=True).start()
    print(f"GitHub stand-in seeded in {WORKDIR}; backend on :{BACKEND_PORT}, chat stand-in on :{AGENT_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=BACKEND_PORT, log_level="warning")


if __name__ == "__main__":
    main()
