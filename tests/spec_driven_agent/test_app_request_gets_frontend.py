"""An "app" request delivers a frontend, whatever models the project has.

Live evidence: "I want a todo app" on a class-diagram-only project selected
generate_fastapi_backend (right: no generator builds a UI without a GUI
model) and then shipped no UI at all. Every gate that orders the
LLM-authored frontend - prompt Rule 15, the harness checklist task, the
Phase 3 missing-frontend blocker - keyed on "web app / frontend / UI"
vocabulary, and a bare "app" matched none of them. Explicitly headless
requests ("build a REST API for todos") stay backend-only.
"""

import json

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.agent.prompt_builder import (
    build_system_prompt, requests_frontend,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

TODO_APP = "I want a todo app"
REST_API = "build a REST API for todos"
RULE_15 = "COMPLETE, NAVIGABLE CRUD frontend"


class _MockClient:
    """No ``_client``: the LLM selector is skipped, the fallback decides."""

    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("never called")


class _SelectorClient(_MockClient):
    """Structured selector that records the prompt it was shown."""

    _client = object()
    planning_model = None

    def __init__(self):
        self.prompts = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.prompts.append(messages[0]["content"])
        return {"content": [{"type": "tool_use",
                             "input": {"generator": "generate_fastapi_backend"}}]}


def _todo_model():
    todo = Class(name="Todo")
    todo.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="title", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="Todos", types={todo})


def _orch(tmp_path, instructions, client=None, **kwargs):
    orch = LLMOrchestrator(llm_client=client or _MockClient(),
                           domain_model=_todo_model(),
                           output_dir=str(tmp_path), **kwargs)
    orch._instructions = instructions
    return orch


def _frontend_task(orch):
    return [t for t in orch._deterministic_gap_tasks()
            if isinstance(t, dict) and t["text"] == orch._FRONTEND_CHECKLIST_TASK]


@pytest.mark.parametrize("text", [
    TODO_APP,
    "Build me an inventory application",
    "a todo app with a FastAPI backend",
    "a web app with a REST API",
    "an admin dashboard",
])
def test_app_requests_ask_for_a_frontend(text):
    assert requests_frontend(text), text


@pytest.mark.parametrize("text", [
    REST_API,
    "a todo app, API only",
    "a FastAPI app for todos",
    "a todo app exposing a REST API",
    "a CLI app for todos",
    "backend only - no UI",
    "a REST API with no frontend",
    "the hotel has a spa, a gym and a restaurant",
])
def test_headless_requests_do_not(text):
    assert not requests_frontend(text), text


def test_a_modify_request_naming_the_app_is_not_a_frontend_ask():
    assert not requests_frontend("add a due date to the app", bare_app=False)
    assert requests_frontend("add a dashboard to the app", bare_app=False)


def test_todo_app_on_class_only_project_takes_the_frontend_path(tmp_path):
    orch = _orch(tmp_path, TODO_APP)
    assert orch._select_generator(TODO_APP) == "generate_fastapi_backend"
    assert _frontend_task(orch), "no harness task orders the frontend"
    assert orch._collect_missing_frontend_issue(), "a UI-less tree would pass"


def test_rest_api_stays_backend_only(tmp_path):
    orch = _orch(tmp_path, REST_API)
    assert orch._select_generator(REST_API) == "generate_fastapi_backend"
    assert not _frontend_task(orch)
    assert not orch._collect_missing_frontend_issue()


def test_domainless_app_request_also_orders_the_frontend(tmp_path):
    class _SM:
        name = "TodoLifecycle"

    orch = LLMOrchestrator(llm_client=_MockClient(), state_machines=[_SM()],
                           output_dir=str(tmp_path))
    orch._instructions = TODO_APP
    assert _frontend_task(orch)


def test_modify_run_does_not_grow_a_frontend_from_the_word_app(tmp_path):
    orch = _orch(tmp_path, "add a due date to the app")
    orch._modify_mode = True
    assert not _frontend_task(orch)
    assert not orch._collect_missing_frontend_issue()


def test_selector_is_told_a_class_only_app_gets_an_authored_frontend(tmp_path):
    client = _SelectorClient()
    orch = _orch(tmp_path, TODO_APP, client=client)
    assert orch._select_generator(TODO_APP) == "generate_fastapi_backend"
    assert "React frontend is written in the customization phase" in client.prompts[0]


def test_phase2_prompt_orders_the_frontend_for_an_app():
    def prompt(instructions, **kwargs):
        return build_system_prompt(_todo_model(), None, None, "", instructions, 10, **kwargs)

    assert RULE_15 in prompt(TODO_APP)
    assert RULE_15 not in prompt(REST_API)
    assert RULE_15 not in prompt("add a due date to the app", modify_mode=True)


def test_phase1_says_the_frontend_will_be_authored(tmp_path, monkeypatch):
    details = []
    orch = _orch(tmp_path, TODO_APP,
                 on_phase_details=lambda phase, text: details.append((phase, text)))
    monkeypatch.setattr(orch.executor, "execute",
                        lambda name, args: json.dumps({"status": "ok", "files": []}))
    monkeypatch.setattr(orch, "_install_scaffold_frontend_dependencies", lambda: None)
    orch._run_phase1(TODO_APP)
    assert orch._generator_used == "generate_fastapi_backend"
    assert any(phase == "generate" and "frontend" in text.lower()
               for phase, text in details), details
