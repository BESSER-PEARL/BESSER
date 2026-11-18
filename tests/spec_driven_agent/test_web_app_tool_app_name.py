"""The web app scaffold carries the project's name, not the class diagram's title."""

from __future__ import annotations

import os
from types import SimpleNamespace

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


def _models():
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    domain = DomainModel(name="Class_Diagram", types={book})
    screen = Screen(name="Home", description="Home", is_main_page=True,
                    view_elements={Text(name="welcome", content="Welcome")})
    gui = GUIModel(name="GUI", package="", versionCode="1", versionName="1.0",
                   modules={Module(name="M", screens={screen})}, description="")
    return domain, gui


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return handle.read()


def test_the_web_app_tool_names_the_app_after_the_project(tmp_path):
    domain, gui = _models()
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=domain, gui_model=gui, app_name="Book_Shelf")

    assert executor._gen_web_app({}).get("status") == "ok"

    out = os.path.join(str(tmp_path), "web_app")
    assert _read(out, "README.md").startswith("# Book Shelf\n")
    assert "title='Book Shelf API'," in _read(out, "backend", "main_api.py")
    assert "<title>Book Shelf</title>" in _read(out, "frontend", "index.html")


def test_the_react_tool_names_the_app_after_the_project(tmp_path):
    domain, gui = _models()
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=domain, gui_model=gui, app_name="Book_Shelf")

    assert executor._gen_react({}).get("status") == "ok"

    assert "<title>Book Shelf</title>" in _read(str(tmp_path), "react", "index.html")


def test_the_run_hands_the_project_name_to_the_tools(tmp_path):
    domain, gui = _models()
    client = SimpleNamespace(model="test-model", planning_model=None, max_tokens=4096)

    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=domain, gui_model=gui, output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
        source_project_export={"name": "Book_Shelf"},
    )

    assert orchestrator.executor.app_name == "Book_Shelf"
