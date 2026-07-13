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
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.tools import get_tools_for, _SHELL_TOOLS
from types import SimpleNamespace


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


def test_default_denies_shell_for_library_callers():
    names = {t["name"] for t in get_tools_for()}
    assert "run_command" not in names
    assert "install_dependencies" not in names


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
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
    )
    names = {t["name"] for t in orch.tools}
    assert "run_command" not in names
    assert orch.allow_shell_tools is False


def test_shell_execution_waits_for_owner_approval(tmp_path, monkeypatch):
    decisions = []
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
        request_tool_approval=lambda turn, tool, arguments: decisions.append(
            (turn, tool, arguments)
        ) or False,
    )
    executed = []
    monkeypatch.setattr(
        orch.executor,
        "execute",
        lambda tool, arguments: executed.append((tool, arguments)) or "{}",
    )
    block = SimpleNamespace(
        name="run_command",
        input={"command": "npm test"},
        id="tool-1",
    )

    result = orch._execute_single_tool(block, 0)

    assert decisions == [(1, "run_command", {"command": "npm test"})]
    assert executed == []
    assert "approval was not granted" in result["content"]


def test_workspace_file_edits_do_not_prompt(tmp_path, monkeypatch):
    decisions = []
    orch = LLMOrchestrator(
        llm_client=_MockClient(),
        domain_model=_simple_model(),
        output_dir=str(tmp_path),
        request_tool_approval=lambda *args: decisions.append(args) or False,
    )
    monkeypatch.setattr(orch.executor, "execute", lambda *_: "{}")
    block = SimpleNamespace(
        name="write_file",
        input={"path": "README.md", "content": "ok"},
        id="tool-2",
    )

    result = orch._execute_single_tool(block, 0)

    assert decisions == []
    assert result["content"] == "{}"


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
