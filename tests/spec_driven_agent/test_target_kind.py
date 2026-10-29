"""A run builds what the user asked for, not always a web app.

Live evidence (prod, 2026-10-29): "quiero una app móvil para controlar mis
gastos" and "una app de windows 11 local" reached the Spec-Driven Agent as
"Generate a mobile application for both Android and iOS ..." / "Generate a
Windows 11 local application ...". Phase 1 picked generate_fastapi_backend
(the selector's rule sent every app with no GUI model there; NONE was
reserved for named web frameworks) and every web gate then ordered a React
frontend: the user got a web app. The Phase 1 selector now decides the target
(``planning/target.py``) in the same call, and the web gates key on it.
"""

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.agent.tools import (
    GENERATOR_TOOLS, get_available_generator_names,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import Target, named_target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, Checkpoint, load_checkpoint, save_checkpoint,
)
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.toolchain import (
    _collect_kotlinc_issues, _collect_uncompiled_source_issues,
)

MOBILE = ("Generate a mobile application for both Android and iOS that tracks "
          "workouts, with data synchronization across devices through a REST API.")
WINDOWS = "Generate a Windows 11 local application for managing inventory."
TODO_APP = "I want a todo app"
RULE_15_WEB = "COMPLETE, NAVIGABLE CRUD frontend"
STACK_LOCK = "If the user's wording seems to imply a different"

NATIVE = {"target_kind": "other", "target_description": "native Android/iOS app",
          "has_user_interface": True}
WEB = {"target_kind": "web_app", "target_description": "todo web app",
       "has_user_interface": True}
API = {"target_kind": "backend", "target_description": "REST API",
       "has_user_interface": False}


class _Client:
    """A structured selector that answers with ``answer`` and records prompts."""

    model = "mock-model"
    usage = UsageTracker("mock-model")
    _client = object()
    planning_model = None

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append({"prompt": messages[0]["content"], "tools": tools})
        return {"content": [{"type": "tool_use", "input": dict(self.answer)}]}


def _model():
    workout = Class(name="Workout")
    workout.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="title", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="Workouts", types={workout})


def _orch(tmp_path, instructions, answer=None, **kwargs):
    orch = LLMOrchestrator(llm_client=_Client(answer or {**NATIVE, "generator": "NONE"}),
                           domain_model=_model(), output_dir=str(tmp_path), **kwargs)
    orch._instructions = instructions
    return orch


def _frontend_task(orch):
    return [t for t in orch._deterministic_gap_tasks()
            if isinstance(t, dict) and t["text"] == orch._FRONTEND_CHECKLIST_TASK]


def _prompt(instructions, **kwargs):
    return build_system_prompt(_model(), None, None, "", instructions, 10, **kwargs)


# -- Phase 1: the selector decides the target ------------------------------

def test_selector_is_asked_for_the_target(tmp_path):
    orch = _orch(tmp_path, MOBILE)
    orch._select_generator(MOBILE)
    call = orch.client.calls[0]
    schema = call["tools"][0]["input_schema"]
    assert {"target_kind", "target_description", "has_user_interface"} <= set(schema["required"])
    assert "native mobile app" in call["prompt"] and "Answer NONE" in call["prompt"]
    assert orch.target == Target("other", "native Android/iOS app", True)


def test_keywords_never_overrule_a_decided_none(tmp_path):
    # "REST API" is a FastAPI keyword; before, it turned the NONE back into FastAPI.
    orch = _orch(tmp_path, MOBILE)
    assert orch._select_generator(MOBILE) is None


@pytest.mark.parametrize("answer, instructions", [
    (WEB, "Create a web app for a library with a REST API backend"),
    (API, "Build a FastAPI backend for a library"),
])
def test_a_web_or_backend_none_keeps_the_keyword_fallback(tmp_path, answer, instructions):
    # Only an ``other`` NONE stands; a web or backend one was a from-scratch build.
    orch = _orch(tmp_path, instructions, {**answer, "generator": "NONE"})
    assert orch._select_generator(instructions) == "generate_fastapi_backend"


def test_an_approved_web_app_generator_settles_the_target_without_a_call(tmp_path):
    orch = _orch(tmp_path, TODO_APP, target_generator="generate_web_app")
    assert orch._select_generator(TODO_APP) == "generate_web_app"
    assert orch.client.calls == []
    assert orch.target.is_web_app


def test_an_approved_fastapi_generator_still_decides_the_target(tmp_path):
    # FastAPI is also the scaffold of a mobile app's API.
    orch = _orch(tmp_path, MOBILE, target_generator="generate_fastapi_backend")
    assert orch._select_generator(MOBILE) == "generate_fastapi_backend"
    assert len(orch.client.calls) == 1
    assert orch.target.kind == "other"


def test_web_app_target_still_gets_the_fastapi_scaffold(tmp_path):
    orch = _orch(tmp_path, TODO_APP, {**WEB, "generator": "generate_fastapi_backend"})
    assert orch._select_generator(TODO_APP) == "generate_fastapi_backend"
    assert orch.target.is_web_app
    assert _frontend_task(orch)


def test_a_named_language_still_decides_the_target(tmp_path):
    # The Swift override skips the selector; the target must not fall back to
    # the "bare app = web frontend" regex.
    orch = _orch(tmp_path, "Build a Swift iOS app for workouts")
    assert orch._select_generator(orch._instructions) is None
    assert orch.target.kind == "other"
    assert not _frontend_task(orch)


def test_domainless_run_decides_the_target(tmp_path):
    class _SM:
        name = "Lifecycle"

    orch = LLMOrchestrator(llm_client=_Client({**NATIVE, "generator": "NONE"}),
                           state_machines=[_SM()], output_dir=str(tmp_path))
    orch._instructions = MOBILE
    orch._run_phase1(MOBILE)
    assert orch.target.kind == "other"


def test_the_old_flutter_generator_is_not_offered():
    assert "generate_flutter" not in {t["name"] for t in GENERATOR_TOOLS}
    assert "generate_flutter" not in get_available_generator_names(
        has_domain_model=True, has_gui_model=True)


# -- The web gates honour the target ----------------------------------------

def test_a_native_target_gets_no_web_frontend_demands(tmp_path):
    orch = _orch(tmp_path, MOBILE)
    orch._select_generator(MOBILE)
    assert not _frontend_task(orch)
    assert not orch._collect_missing_frontend_issue()
    assert not orch._collect_frontend_contract_issues()


def test_phase2_prompt_for_a_native_target():
    prompt = _prompt(MOBILE, target=Target("other", "native Android/iOS app", True))
    assert RULE_15_WEB not in prompt
    assert "Build exactly the requested target: native Android/iOS app" in prompt
    assert "Do NOT add a web frontend" in prompt
    # No scaffold: nothing to keep, so no "ignore the user's stack".
    assert STACK_LOCK not in prompt
    assert "The generator handled the base app" not in prompt


def test_phase2_prompt_for_a_web_target_is_unchanged():
    web = _prompt(TODO_APP, target=Target("web_app", "todo web app", True))
    assert RULE_15_WEB in web
    assert web == _prompt(TODO_APP, target=Target("web_app", "todo web app", True))
    scaffolded = _prompt(TODO_APP, scaffold_snapshot="backend/main_api.py ...")
    assert STACK_LOCK in scaffolded and "The generator handled the base app" in scaffolded


def test_a_native_target_keeps_its_ui_requirements_in_scope(tmp_path):
    orch = _orch(tmp_path, MOBILE)
    orch.target = Target("other", "native Android/iOS app", True)
    assert orch._ui_in_scope()
    orch.target = Target("other", "Python CLI", False)
    assert not orch._ui_in_scope()


def test_phase0_5_writes_no_server_template_for_a_kotlin_android_app(tmp_path):
    orch = _orch(tmp_path, "Build a Kotlin Android app for workouts")
    orch.target = Target("other", "Kotlin Android app", True)
    orch._run_phase0_5_metadata(orch._instructions)
    assert not os.path.exists(tmp_path / "build.gradle.kts")
    assert orch._phase0_5_stack is None


# -- Phase 3 reports what it could not compile ------------------------------

def test_uncompiled_languages_are_reported_as_skipped(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "main.dart").write_text("void main() {}")
    (tmp_path / "App.xaml.cs").write_text("class App {}")
    issues = _collect_uncompiled_source_issues(str(tmp_path), frozenset())
    assert len(issues) == 2
    assert all("did not run" in i and "not compiled" in i for i in issues)
    assert {_classify_issue(i).severity for i in issues} == {"warning"}


def test_web_output_gets_no_uncompiled_note(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "main_api.py").write_text("x = 1")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "App.tsx").write_text("export {}")
    assert _collect_uncompiled_source_issues(str(tmp_path), frozenset()) == []


def test_android_kotlin_module_is_not_compiled_with_bare_kotlinc(tmp_path, monkeypatch):
    app = tmp_path / "app"
    (app / "src" / "main" / "kotlin").mkdir(parents=True)
    (app / "src" / "main" / "kotlin" / "Main.kt").write_text("fun main() {}")
    (app / "build.gradle.kts").write_text('plugins { id("com.android.application") }')
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/kotlinc")
    from besser.spec_driven_agent.validation import toolchain
    monkeypatch.setattr(toolchain, "run_bounded",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("kotlinc ran")))
    issues = _collect_kotlinc_issues(str(tmp_path))
    assert len(issues) == 1 and "Android" in issues[0]
    assert _classify_issue(issues[0]).severity == "warning"


# -- The target survives resume and modify runs -----------------------------

def test_target_survives_a_checkpoint_round_trip(tmp_path):
    target = Target("other", "native Android/iOS app", True)
    checkpoint = Checkpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION, run_id="r", instructions=MOBILE,
        primary_kind="class", turn=1, total_turns=1, messages=[], tool_calls_log=[],
        validation_issues=[], inventory="", generator_used=None, estimated_cost_usd=0.0,
        compaction_count=0, project_fingerprint="", saved_at=0.0, target=target.to_dict())
    assert save_checkpoint(str(tmp_path), checkpoint)
    assert Target.from_payload(load_checkpoint(str(tmp_path)).target) == target


def test_target_is_recorded_in_the_recipe_and_read_by_a_modify_run(tmp_path):
    orch = _orch(tmp_path, MOBILE)
    orch.target = Target("other", "native Android/iOS app", True)
    orch._save_recipe(MOBILE, 1.0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["target"]["kind"] == "other"

    follow_up = _orch(tmp_path, "add a settings screen")
    follow_up._seed_generator_files_from_recipe()
    assert follow_up.target == orch.target
    assert not _frontend_task(follow_up)


def _modify(tmp_path, instructions, seed, answer):
    """A modify run's orchestrator, seeded with a recipe built for ``seed``."""
    (tmp_path / ".besser_recipe.json").write_text(
        json.dumps({"target": seed.to_dict()}), encoding="utf-8")
    orch = _orch(tmp_path, instructions, answer)
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    return orch


def test_modify_on_a_backend_adds_the_frontend_it_asks_for(tmp_path):
    instructions = "Add a React frontend with pages for every entity"
    orch = _modify(tmp_path, instructions, Target("backend", "REST API", False),
                   {**WEB, "generator": "NONE"})
    assert orch.target.is_web_app
    assert _frontend_task(orch)
    orch._inventory = "backend/main_api.py (generate_fastapi_backend)"
    assert RULE_15_WEB in orch._build_system_prompt(instructions)


@pytest.mark.parametrize("instructions", [
    "No, make it a native desktop app instead.",
    "Target: native app (not a web app).\nBuild the workouts tracker.",
])
def test_modify_turns_a_web_app_into_the_target_the_user_now_names(tmp_path, instructions):
    orch = _modify(tmp_path, instructions, Target("web_app", "workouts web app", True),
                   {**NATIVE, "generator": "NONE"})
    assert orch.target.kind == "other"
    assert "The existing app was built as: workouts web app" in orch.client.calls[0]["prompt"]
    assert not _frontend_task(orch)
    orch._inventory = "backend/main_api.py, frontend/src/App.tsx (generate_fastapi_backend)"
    prompt = orch._build_system_prompt(instructions)
    assert RULE_15_WEB not in prompt
    assert "Build exactly the requested target: native Android/iOS app" in prompt
    # The keep-the-stack constraint would contradict the new target.
    assert STACK_LOCK not in prompt


@pytest.mark.parametrize("instructions, seed", [
    ("add a settings screen", Target("other", "native Android/iOS app", True)),
    ("Target: native Android/iOS app (not a web app).\nAdd a settings screen.",
     Target("other", "native Android/iOS app", True)),
    ("add a search box to the book list", Target("web_app", "library web app", True)),
    ("add an endpoint that exports loans as CSV", Target("backend", "REST API", False)),
])
def test_modify_keeps_the_seed_target_without_a_call(tmp_path, instructions, seed):
    orch = _modify(tmp_path, instructions, seed, {**WEB, "generator": "NONE"})
    assert orch.target == seed
    assert orch.client.calls == []
    assert orch._previous_target is None


def test_a_modify_that_keeps_the_kind_keeps_the_stack(tmp_path):
    # "desktop" is only a cue; the selector keeps the web app.
    instructions = "Make the dashboard easier to use on desktop screens"
    orch = _modify(tmp_path, instructions, Target("web_app", "library web app", True),
                   {**WEB, "generator": "NONE"})
    assert len(orch.client.calls) == 1
    assert orch.target.is_web_app and orch._previous_target is None
    orch._inventory = "backend/main_api.py, frontend/src/App.tsx (generate_fastapi_backend)"
    assert STACK_LOCK in orch._build_system_prompt(instructions)


def test_a_failed_modify_decision_keeps_the_seed_target(tmp_path):
    seed = Target("backend", "REST API", False)
    orch = _modify(tmp_path, "Add a React frontend", seed, {"generator": "NONE"})
    assert len(orch.client.calls) == 1
    assert orch.target == seed


# -- The cross-repo contract ------------------------------------------------

def test_the_target_line_modeling_agent_writes_is_parsed():
    # modeling-agent's generation_handler._instructions_naming_target writes
    # f"Target: {description} (not a web app)." on a line of its own;
    # smart_generation_handler then nests it under the assistant's notes.
    description = "Windows 11 desktop app"
    line = f"Target: {description} (not a web app)."
    assert named_target(f"{line}\nBuild the inventory application.") == description
    assert named_target(f"{line}\r\nArchitecture: MVC.\r\nBuild it.") == description
    payload = (
        "## Assistant's notes (may be inaccurate)\n\nUse these only for ...\n\n"
        f"{line}\nBuild the inventory application.\n\n"
        "## The user's original request, verbatim\n\nThis is the authority.\n\n"
        "una app de windows 11 local para inventario")
    assert named_target(payload) == description
    assert named_target("Build the inventory application.") is None
    assert named_target("Build it. Target: Windows 11 desktop app (not a web app).") is None


def test_without_a_target_line_the_request_wording_decides(tmp_path):
    # An oversized spec reaches BESSER as the original request alone.
    request = "una app de windows 11 local para inventario"
    orch = _orch(tmp_path, request, {**NATIVE, "target_description": "Windows 11 desktop app",
                                     "generator": "NONE"})
    assert named_target(request) is None
    assert orch._select_generator(request) is None
    assert orch.target.kind == "other"
    assert not _frontend_task(orch)
