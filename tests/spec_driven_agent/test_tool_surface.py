"""A tool is offered only when it can return something the context lacks.

Every tool schema rides on every Phase 2 and repair turn. Over the recorded
production runs ``query_class``, ``list_classes_with``, ``validate_model`` and
``check_syntax`` got no call, and ``get_constraints_for`` got eight, all
answering ``count: 0``. The model-query tools return what the prompt's
serialized domain model already holds (in full, with inherited members and
constraints), ``validate_model`` re-checks an input the run cannot change, and
``check_syntax`` repeats the write-time parse check and ``validate_app``.
"""
import json

import pytest

from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators, get_tools_for

DROPPED = ("query_class", "list_classes_with", "get_constraints_for", "validate_model", "check_syntax")


def test_the_redundant_tools_are_not_offered():
    offered = {tool["name"] for tool in get_tools_for(has_domain_model=True, has_gui_model=True,
                                                      allow_shell=True)}
    assert not offered & set(DROPPED)
    assert not {tool["name"] for tool in get_all_tools_including_generators()} & set(DROPPED)


@pytest.mark.parametrize("name", DROPPED)
def test_a_call_to_a_dropped_tool_is_an_unknown_tool(tmp_path, name):
    payload = ToolExecutor(workspace=str(tmp_path)).execute_typed(name, {}).payload
    assert payload == {"error": f"Unknown tool: {name}"}


def test_the_prompt_does_not_advertise_them(simple_library_book_model):
    prompt = build_system_prompt(simple_library_book_model, None, None, "", "Build it", 10)
    for name in DROPPED:
        assert name not in prompt
    assert "## Domain Model" in prompt and json.dumps("Library")[1:-1] in prompt


def test_installs_go_through_run_command(tmp_path):
    """install_dependencies duplicated run_command (same shell, same serial group);
    the pins it applied now apply wherever a requirements.txt is written."""
    assert "install_dependencies" not in {tool["name"] for tool in get_tools_for(allow_shell=True)}
    payload = ToolExecutor(workspace=str(tmp_path), allow_shell=True).execute_typed(
        "install_dependencies", {}).payload
    assert payload == {"error": "Unknown tool: install_dependencies"}


def test_the_install_hints_use_the_probes_own_interpreter(tmp_path, monkeypatch):
    """A bare ``pip`` on PATH can belong to another Python than the ``python`` the probe boots with."""
    from besser.spec_driven_agent.agent import runbook

    monkeypatch.setattr(runbook, "runbook_enabled", lambda: True)
    for name in ("main_api.py", "sql_alchemy.py"):
        (tmp_path / name).write_text("app = None\n", encoding="utf-8")
    section = runbook.runbook_section(str(tmp_path))
    assert section, "the runbook section should render for a backend"
    for text in (section, runbook._PROBE_TEMPLATE):
        assert "python -m pip install -r requirements.txt" in text
        assert "'pip install" not in text
