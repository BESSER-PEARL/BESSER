"""`.besser_probe.py routes` shows the body each route reads and what it returns.

The agent wrote a page that posted a method's values flat although the route reads
``{"params": {...}}``, and treated a relation route's object as an array. ``routes``
listed ``params:object`` and nothing about the response, which the app's OpenAPI
does not describe for a route declared with ``response_model=None``.
"""
import contextlib
import importlib.util
import io

from fastapi import FastAPI

from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, install_probe

ROUTER = '''from typing import Optional
from fastapi import APIRouter, Body
from pydantic import BaseModel

router = APIRouter()


class TaskCreate(BaseModel):
    title: str
    estimate: Optional[int] = None
    project: str


@router.get("/project/{project_id}/tasks/", response_model=None)
async def get_tasks_of_project(project_id: str):
    tasks_list = []
    return {"project_id": project_id, "tasks_count": len(tasks_list), "tasks": tasks_list}


@router.get("/task/", response_model=None)
def get_all_task():
    rows = []
    return rows


@router.post("/task/", response_model=None)
async def create_task(task_data: TaskCreate):
    return task_data


@router.post("/task/{task_id}/methods/close/", response_model=None)
async def execute_task_close(task_id: str, params: dict = Body(default=None, embed=True)):
    return {"task": task_id}
'''


def _routes_output(tmp_path) -> list[str]:
    backend = tmp_path / "backend"
    (backend / "routers").mkdir(parents=True)
    (backend / "main_api.py").write_text("", encoding="utf-8")
    (backend / "sql_alchemy.py").write_text("", encoding="utf-8")
    (backend / "routers" / "task.py").write_text(ROUTER, encoding="utf-8")
    assert install_probe(str(tmp_path))
    spec = importlib.util.spec_from_file_location("installed_probe", tmp_path / PROBE_FILENAME)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    namespace: dict = {}
    exec(compile(ROUTER, "task.py", "exec"), namespace)
    app = FastAPI()
    app.include_router(namespace["router"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        probe.print_routes(app.openapi(), probe.source_shapes(probe.find_backend()))
    return [line.strip() for line in out.getvalue().splitlines()]


def test_routes_show_the_body_wrapper_and_the_response_shape(tmp_path):
    lines = _routes_output(tmp_path)

    assert 'POST   /task/{task_id}/methods/close/  <- {"params": {...}}  -> object {task}' in lines
    assert "GET    /project/{project_id}/tasks/  -> object {project_id, tasks, tasks_count}" in lines
    assert "GET    /task/  -> list" in lines
    # A handler returning its argument says nothing certain about the answer.
    assert 'POST   /task/  <- {"title": string*, "estimate": integer|null, "project": string*}' in lines
    assert any(line.startswith('NOTE: a body shown as {"params": {...}} is sent with that key') for line in lines)


def test_an_install_without_python_sources_still_lists_routes(tmp_path, monkeypatch):
    """A bytecode-only install has no source to embed; importing the runbook must not fail."""
    from besser.spec_driven_agent.agent import runbook

    def no_source(_module):
        raise OSError("could not get source code")

    monkeypatch.setattr(runbook.inspect, "getsource", no_source)
    script = tmp_path / "probe.py"
    script.write_text(runbook._with_route_returns(runbook._PROBE_TEMPLATE), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bare_probe", script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    namespace: dict = {}
    exec(compile(ROUTER, "task.py", "exec"), namespace)
    app = FastAPI()
    app.include_router(namespace["router"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        probe.print_routes(app.openapi(), probe.source_shapes(str(tmp_path)))
    lines = [line.strip() for line in out.getvalue().splitlines()]

    assert 'POST   /task/{task_id}/methods/close/  <- {"params": {...}}' in lines
    assert "GET    /task/" in lines
