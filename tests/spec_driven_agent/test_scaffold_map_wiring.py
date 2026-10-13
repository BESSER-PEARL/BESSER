"""The scaffold map reaches the agent: the locate_code tool, the prompt and the planner.

The extractors have their own tests (test_scaffold_map_<family>.py); these cover
how the map is wired into a run:

* ``locate_code`` is a registered, read-only tool offered on every run, and
  answers from the files as they are when it is called;
* the system prompt carries the line-free overview without switching on the
  frontend rule for a backend-only scaffold;
* the gap analyzer gets the overview, and its prompt is unchanged without it;
* a generator re-run in Phase 2 attributes only the files it wrote.
"""
from __future__ import annotations

import json
import os
import re

from besser.spec_driven_agent.agent import prompt_builder
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import FILE_TOOLS, _TOOL_MODEL_REQUIREMENTS, get_tools_for
from besser.spec_driven_agent.pipeline.constants import _READONLY_TOOLS, _SCAFFOLD_MAP_CHARS_SMALL
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.gap_analyzer import _build_user_prompt
from besser.spec_driven_agent.planning.scaffold_map import scaffold_overview
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.agent_trace import AttributionState
from tests.spec_driven_agent.test_app_request_gets_frontend import REST_API, RULE_15, _todo_model


def _run(executor: ToolExecutor, tool: str, args: dict) -> dict:
    return json.loads(executor.execute(tool, args))


def _backend(tmp_path, model) -> ToolExecutor:
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=model)
    assert _run(executor, "generate_fastapi_backend", {})["status"] == "ok"
    return executor


class _Client:
    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.usage = UsageTracker(model)

    def chat(self, **kwargs):  # pragma: no cover
        raise AssertionError("no LLM call expected")


def test_locate_code_is_a_registered_read_only_tool_on_every_run():
    assert "locate_code" in {tool["name"] for tool in FILE_TOOLS}
    assert _TOOL_MODEL_REQUIREMENTS["locate_code"] == frozenset()
    assert "locate_code" in ToolExecutor._handlers
    assert "locate_code" in _READONLY_TOOLS
    # Offered without a domain model too (agent / BPMN / NN runs).
    assert "locate_code" in {t["name"] for t in get_tools_for(has_domain_model=False, allow_shell=False)}


def test_locate_code_answers_from_the_files_as_they_are_now(tmp_path, library_book_author_model):
    executor = _backend(tmp_path, library_book_author_model)

    first = _run(executor, "locate_code", {"entity": "Book"})
    assert "backend/routers/book.py" in first["locations"]
    assert "read_file offset=" in first["locations"]

    # Shift every line of the router: the next answer must follow.
    router = tmp_path / "backend" / "routers" / "book.py"
    router.write_text("# a\n# b\n# c\n" + router.read_text(encoding="utf-8"), encoding="utf-8")
    second = _run(executor, "locate_code", {"entity": "Book"})
    assert second["locations"] != first["locations"]


def test_locate_code_without_arguments_lists_the_entities(tmp_path, library_book_author_model):
    executor = _backend(tmp_path, library_book_author_model)
    result = _run(executor, "locate_code", {})
    assert {"Library", "Book", "Author"} <= set(result["entities"])


def test_locate_code_rejects_a_non_string_filter(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    assert "must be a string" in _run(executor, "locate_code", {"entity": 3})["error"]


def test_a_generator_rerun_attributes_only_the_files_it_wrote(tmp_path, library_book_author_model):
    executor = _backend(tmp_path, library_book_author_model)
    state = AttributionState.load(str(tmp_path))
    assert "backend/routers/book.py" in state.files

    # The model adds its own module inside the generator's folder, then the
    # generator runs again: the model's file must not become generator code.
    (tmp_path / "backend" / "auth.py").write_text("def current_user():\n    return None\n", encoding="utf-8")
    assert _run(executor, "generate_fastapi_backend", {})["status"] == "ok"
    state = AttributionState.load(str(tmp_path))
    assert "backend/auth.py" not in state.files
    assert "backend/routers/book.py" in state.files


def test_inventory_symbols_include_react_pages_and_app(tmp_path):
    src = tmp_path / "src"
    (src / "pages").mkdir(parents=True)
    (src / "App.tsx").write_text("function App() {\n  return null;\n}\n\nexport default App;\n", encoding="utf-8")
    (src / "pages" / "Main.tsx").write_text(
        "import React from 'react';\n\nconst Main: React.FC = () => {\n  return <div />;\n};\n"
        "\nexport default Main;\n", encoding="utf-8")
    assert prompt_builder._symbol_map(str(src / "App.tsx")) == "App@1"
    assert prompt_builder._symbol_map(str(src / "pages" / "Main.tsx")) == "Main@3"


def test_prompt_carries_the_map_without_switching_the_frontend_rule_on(tmp_path, library_book_author_model):
    _backend(tmp_path, library_book_author_model)
    overview = scaffold_overview(str(tmp_path), library_book_author_model)
    assert overview and "backend/routers/book.py" in overview

    prompt = build_system_prompt(_todo_model(), None, None, "", REST_API, 10, scaffold_map=overview)
    assert "## Where to add code (scaffold map)" in prompt
    assert overview in prompt
    assert RULE_15 not in prompt
    # No map, no section.
    assert "## Where to add code" not in build_system_prompt(_todo_model(), None, None, "", REST_API, 10)


def test_overview_is_line_free(tmp_path, library_book_author_model):
    _backend(tmp_path, library_book_author_model)
    overview = scaffold_overview(str(tmp_path), library_book_author_model)
    # No line numbers in any shape: "line 12", "lines 3-9", "book.py:41", "get_book@40".
    assert not re.search(r"\blines? \d|\.\w+:\d|@\d", overview)
    assert "offset=" not in overview


def test_gap_analyzer_gets_the_map_and_is_unchanged_without_it():
    without = _build_user_prompt("req", "generate_fastapi_backend", "{}", "inv", "act")
    assert without == _build_user_prompt("req", "generate_fastapi_backend", "{}", "inv", "act", scaffold_map="")
    assert "SCAFFOLD MAP" not in without

    with_map = _build_user_prompt("req", "generate_fastapi_backend", "{}", "inv", "act",
                                  scaffold_map="- Book: backend/routers/book.py")
    assert "SCAFFOLD MAP (where each entity's code lives and where new code goes):" in with_map
    assert "- Book: backend/routers/book.py" in with_map
    assert "name the file (and the anchor id" in with_map


def _orchestrator(tmp_path, model, client_model="mock-model") -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(client_model), domain_model=model, output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False,
    )


def test_orchestrator_overview_honours_the_switch_and_small_windows(tmp_path, monkeypatch, library_book_author_model):
    _backend(tmp_path, library_book_author_model)
    on = _orchestrator(tmp_path, library_book_author_model)
    assert on._scaffold_map_overview()
    assert "locate_code" in {t["name"] for t in on.tools}

    small = _orchestrator(tmp_path, library_book_author_model, client_model="devstral:24b")
    assert 0 < len(small._scaffold_map_overview()) <= _SCAFFOLD_MAP_CHARS_SMALL

    monkeypatch.setenv("BESSER_LLM_SCAFFOLD_MAP", "0")
    off = _orchestrator(tmp_path, library_book_author_model)
    assert off._scaffold_map_overview() == ""
    assert "locate_code" not in {t["name"] for t in off.tools}     # off = the pre-map baseline


def test_an_empty_workspace_has_no_map(tmp_path):
    assert scaffold_overview(str(tmp_path)) == ""
    assert not os.listdir(tmp_path)
