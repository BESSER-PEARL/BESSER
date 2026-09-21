"""The arbitrary-shell agent tools (run_command / install_dependencies) are a
hosted-RCE surface: on a shared BYOK box they run user/LLM-authored commands in
the backend process. They must be droppable, and the web runner must drop them.

These tests pin: (1) get_tools_for(allow_shell=False) removes exactly the shell
tools and keeps every static tool; (2) the orchestrator threads the flag into
its live tool list; (3) the hosted default is OFF.
"""
from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.tools import get_tools_for, _SHELL_TOOLS


def _simple_model():
    StringType = PrimitiveDataType("str")
    IntegerType = PrimitiveDataType("int")
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    }
    return DomainModel(name="TestModel", types={user})


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        return {"stop_reason": "end_turn", "content": []}


# --------------------------------------------------------------------------- #
# get_tools_for gate
# --------------------------------------------------------------------------- #
def test_shell_tools_present_when_allowed():
    names = {t["name"] for t in get_tools_for(allow_shell=True)}
    assert "run_command" in names
    assert "install_dependencies" in names


def test_shell_tools_dropped_when_disallowed():
    names = {t["name"] for t in get_tools_for(allow_shell=False)}
    assert "run_command" not in names
    assert "install_dependencies" not in names
    assert _SHELL_TOOLS.isdisjoint(names)


def test_only_shell_tools_are_dropped():
    on = {t["name"] for t in get_tools_for(allow_shell=True)}
    off = {t["name"] for t in get_tools_for(allow_shell=False)}
    # Exactly the shell tools differ; every static/generator tool is retained.
    assert on - off == set(_SHELL_TOOLS)
    for essential in ("write_file", "read_file", "modify_file", "check_syntax", "list_files"):
        assert essential in off


def test_default_allows_shell_for_library_callers():
    # Library/CLI/bench default keeps self-verification shell tools.
    names = {t["name"] for t in get_tools_for()}
    assert "run_command" in names


# --------------------------------------------------------------------------- #
# Orchestrator threads the flag into its live tool list
# --------------------------------------------------------------------------- #
def test_orchestrator_disables_shell_tools(tmp_path):
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
        allow_shell_tools=False,
    )
    names = {t["name"] for t in orch.tools}
    assert "run_command" not in names
    assert "install_dependencies" not in names
    assert orch.allow_shell_tools is False


def test_orchestrator_default_drops_shell_tools(tmp_path):
    """The default is OFF as of 2026-09-14.

    It used to be ON, and LLMGenerator never passed the argument, so every
    library run silently got run_command / install_dependencies — the exact
    capability the hosted gate exists to withhold. Opting in is now explicit.
    """
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
    )
    names = {t["name"] for t in orch.tools}
    assert "run_command" not in names
    assert "install_dependencies" not in names


def test_orchestrator_grants_shell_tools_when_asked(tmp_path):
    """The other direction: this is a default change, not a removal. The flag
    must still thread into the live tool list."""
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
        allow_shell_tools=True,
    )
    names = {t["name"] for t in orch.tools}
    assert "run_command" in names


def test_hosted_validation_never_invokes_pip(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("example-package==1.0\n")
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
        allow_shell_tools=False,
        enable_toolchain_validation=False,
    )
    monkeypatch.setattr(orch, "_collect_frontend_contract_issues", lambda: [])
    monkeypatch.setattr(orch, "_collect_ruff_issues", lambda: [])

    import subprocess
    calls = []

    def _record_run(command, *args, **kwargs):
        calls.append(command)
        raise AssertionError("hosted validation must not launch subprocesses")

    monkeypatch.setattr(subprocess, "run", _record_run)
    orch._collect_validation_issues()

    assert calls == []


# --------------------------------------------------------------------------- #
# The hosted backend defaults the flag OFF
# --------------------------------------------------------------------------- #
def test_hosted_constant_defaults_off(monkeypatch):
    # With the env var unset, the backend constant must resolve to False so a
    # fresh hosted deploy is secure without any extra configuration.
    monkeypatch.delenv("BESSER_LLM_ENABLE_SHELL_TOOLS", raising=False)
    import importlib
    from besser.utilities.web_modeling_editor.backend.constants import constants
    importlib.reload(constants)
    assert constants.LLM_ENABLE_SHELL_TOOLS is False


# ---------------------------------------------------------------------------
# The gate must refuse the CALL, not merely hide the tool.
#
# Every test above this block asserts a tool is absent from the advertised
# list. None asserted that executing it is refused — and it was not: the
# handler table always held run_command/install_dependencies and
# execute_typed did no membership check, so a model that named a tool it was
# never offered got it. Verified live on 2026-09-14:
#
#     ToolExecutor(workspace=tmp).execute("run_command", {"command": "echo X"})
#     -> {"exit_code": 0, "stdout": "X\n", "success": true}
#
# That matters because on an OpenAI-compatible endpoint the tool name is taken
# verbatim from the model's own output, and the Phase-3 fix prompt names
# run_command unconditionally.
# ---------------------------------------------------------------------------

import tempfile

import pytest

from besser.spec_driven_agent.tool_executor import ToolExecutor


@pytest.fixture
def workspace():
    return tempfile.mkdtemp()


@pytest.mark.parametrize("tool", ["run_command", "install_dependencies"])
def test_executing_a_shell_tool_is_refused_by_default(tool, workspace):
    ex = ToolExecutor(workspace=workspace)
    result = ex.execute_typed(tool, {"command": "echo MARKER", "packages": ["x"]})
    assert result.status == "error"
    assert "not available" in result.payload["error"]
    assert "MARKER" not in str(result.payload), "the command must not have run"


@pytest.mark.parametrize("tool", ["run_command", "install_dependencies"])
def test_the_refusal_does_not_depend_on_the_advertised_list(tool, workspace):
    """A model can name a tool it was never offered — that is the whole point."""
    ex = ToolExecutor(workspace=workspace)
    assert tool not in {t["name"] for t in get_tools_for(allow_shell=False)}
    assert ex.execute_typed(tool, {"command": "echo X"}).status == "error"


def test_an_explicit_opt_in_still_executes(workspace):
    ex = ToolExecutor(workspace=workspace, allow_shell=True)
    result = ex.execute_typed("run_command", {"command": "echo ALLOWED"})
    assert result.status == "ok"
    assert "ALLOWED" in result.payload["stdout"]


def test_non_shell_tools_are_untouched_by_the_gate(workspace):
    ex = ToolExecutor(workspace=workspace)
    assert ex.execute_typed("list_files", {}).status == "ok"


def test_the_orchestrator_threads_its_flag_into_the_executor(tmp_path):
    orch = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=_simple_model(),
        output_dir=str(tmp_path),
    )
    assert orch.executor.allow_shell is False
    opted_in = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=_simple_model(),
        output_dir=str(tmp_path), allow_shell_tools=True,
    )
    assert opted_in.executor.allow_shell is True
