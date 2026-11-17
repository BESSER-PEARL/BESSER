"""The gap planner is told what the user wants built, a web app included.

Analyst finding (r9-web1, "a todo app to manage my tasks"): the planner, shown a
backend-only scaffold, returned no task, because it was given the target only
for non-web runs and so was never told the app needs screens. In r9-p51, r10-p51,
r11-p51, r9-p111 and r11-p111 it then added its own "create a frontend" task
beside the harness's, so the checklist named the same work twice.
"""

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import gap_analyzer
from besser.spec_driven_agent.planning.target import Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

WEB = Target("web_app", "todo app with lists and tags", True)


class _Planner:
    model = "mock"
    _client = object()
    planning_model = None

    def __init__(self):
        self.usage = UsageTracker("mock")
        self.prompts = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.prompts.append(messages[0]["content"])
        block = type("B", (), {"type": "tool_use", "name": force_tool, "input": {"tasks": []}})()
        return {"stop_reason": "tool_use", "content": [block]}


class _EndsTurn:
    model = "mock"
    max_tokens = None

    def __init__(self):
        self.usage = UsageTracker("mock")

    def chat(self, system, messages, tools, **_):
        return {"stop_reason": "end_turn", "content": [type("T", (), {"type": "text", "text": "done"})()]}


def _prompt(target):
    planner = _Planner()
    gap_analyzer.analyze_gaps_via_llm(
        instructions="I want a todo app to manage my tasks, with lists and tags.",
        generator_used="generate_fastapi_backend", domain_model=None, inventory="backend/main_api.py",
        llm_client=planner, workspace_files=[], action_endpoints=[], target=target,
    )
    return planner.prompts[0]


def test_a_web_target_is_named_with_its_screens_and_the_harness_frontend_item():
    prompt = _prompt(WEB)
    assert "WHAT THE USER WANTS BUILT: todo app with lists and tags (a web app used through screens)" in prompt
    assert "already has a task to build it" in prompt


def test_an_api_target_says_it_has_no_user_interface():
    prompt = _prompt(Target("backend", "REST API", False))
    assert "WHAT THE USER WANTS BUILT: REST API (no user interface)" in prompt
    assert "already has a task to build it" not in prompt


def test_no_target_no_line():
    assert "WHAT THE USER WANTS BUILT" not in _prompt(None)


def test_the_orchestrator_passes_a_web_target(tmp_path, monkeypatch):
    generate_backend(tmp_path, shelf_model(author=True))
    seen = {}

    def planner(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", planner)
    orch = LLMOrchestrator(llm_client=_EndsTurn(), domain_model=shelf_model(author=True), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_import_smoke_check=False,
                           enable_requirements_ledger=False, use_streaming=False, max_turns=1)
    orch._generator_used = GENERATOR
    orch.target = WEB
    orch._instructions = "I want a todo app"
    orch._run_phase2(orch._instructions)
    assert seen["target"] == WEB
