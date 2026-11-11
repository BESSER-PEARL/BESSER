"""What a run builds is decided by the forced ``choose_generator`` call, not regexes.

Review 2026-10-30 (``reviews/2026-10-30-spec-driven-bugs-and-test-gaps.md``),
each case reproduced by a probe before the fix:

- H1: ``_names_unsupported_stack`` read "express delivery", "ruby rings",
  "spring semester", "C# minor" or "Swift Logistics" as a stack, so a plain web
  app lost its scaffold; the preview's substring copy read "trails" as Rails.
- H2: keywords overruled a decided NONE, so "a REST API in Go" got FastAPI.
- B5: when the selector call failed, the target was never decided and an
  Android request got FastAPI plus a React frontend task.
- B6: a modify run asked the selector again only on a keyword, so "Turn it into
  a Flutter app" kept the web target.
- B1: a reworded ``Target:`` line of the same kind counted as a switch and
  dropped the keep-the-stack constraint.
- H3: about 25 regexes decided whether a message reports a failure; they missed
  "When I delete an author, it is still in the list" and flagged "Map
  TimeoutError to a 504".
- B7: ``api``, ``v1`` and ``id`` from ``/api/v1/books/{id}`` became entities.
- B4: ``_DELETE_SCAFFOLD_RE`` dropped "Add a Delete button ... React frontend"
  on every run, and on a create run the emptied list (``[]``) skipped Phase 2.
- H6: the ``_markers`` substring list overruled the planner's empty checklist.
"""

import json
import logging

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property, StringType,
)
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.fix_target import parse_reported_target
from besser.spec_driven_agent.planning.gap_analyzer import analyze_gaps_via_llm
from besser.spec_driven_agent.planning.target import Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.utilities.web_modeling_editor.backend.services.spec_driven.model_assembly import (
    AssembledModels,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.preview import (
    _predict_target_generator,
)

STACK_LOCK = "If the user's wording seems to imply a different"
WEB = {"target_kind": "web_app", "target_description": "web app", "has_user_interface": True}
API = {"target_kind": "backend", "target_description": "REST API", "has_user_interface": False}
NATIVE = {"target_kind": "other", "target_description": "Flutter mobile app",
          "has_user_interface": True}
KEEP = {"changes_target": False, "is_failure_report": False, "reported_entity": ""}


class _Client:
    """A structured selector: fails the first ``fail_first`` calls, then answers."""

    model = "mock-model"
    usage = UsageTracker("mock-model")
    _client = object()
    planning_model = None

    def __init__(self, answer, fail_first=0):
        self.answer = answer
        self.fail_first = fail_first
        self.calls = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append({"prompt": messages[0]["content"], "tools": tools})
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("gateway 502")
        return {"content": [{"type": "tool_use", "input": dict(self.answer)}]}


def _model():
    book = Class(name="Book")
    book.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True),
                       Property(name="title", type=PrimitiveDataType("str"))}
    author = Class(name="Author")
    author.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    return DomainModel(name="Library", types={book, author})


def _orch(tmp_path, instructions, answer, **kwargs):
    orch = LLMOrchestrator(llm_client=_Client(answer, **kwargs), domain_model=_model(),
                           output_dir=str(tmp_path))
    orch._instructions = instructions
    return orch


def _frontend_task(orch):
    return [t for t in orch._deterministic_gap_tasks()
            if isinstance(t, dict) and t["text"] == orch._FRONTEND_CHECKLIST_TASK]


def _modify(tmp_path, instructions, seed, answer):
    """A modify run routed against a recipe built for ``seed``."""
    (tmp_path / ".besser_recipe.json").write_text(
        json.dumps({"target": seed.to_dict()}), encoding="utf-8")
    orch = _orch(tmp_path, instructions, answer)
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    orch._route_modify_request()
    return orch


# -- H1: domain words are not stacks --------------------------------------

@pytest.mark.parametrize("instructions", [
    "A web app for a courier company: customers pick standard or express delivery.",
    "A web app for a jewellery shop selling ruby, emerald and sapphire rings.",
    "A web app for our club; the contact form emails info@ourclub.net.",
    "A web app to plan the spring semester courses.",
    "A web app for music students to practise scales like C# minor.",
    "A school web app listing pupils in grade C classes.",
    "A web app to score dart tournaments.",
    "A web app for Swift Logistics to track parcels.",
])
def test_unsupported_stack_ignores_domain_words(tmp_path, instructions):
    orch = _orch(tmp_path, instructions,
                 {**WEB, "unsupported_stack": "", "generator": "generate_fastapi_backend"})
    assert orch._select_generator(instructions) == "generate_fastapi_backend"
    assert orch.target.is_web_app


def test_the_selector_is_asked_for_the_unsupported_stack(tmp_path):
    orch = _orch(tmp_path, "a web app", {**WEB, "unsupported_stack": "", "generator": "generate_fastapi_backend"})
    orch._select_generator("a web app")
    schema = orch.client.calls[0]["tools"][0]["input_schema"]
    assert "unsupported_stack" in schema["required"]
    # A create run has no existing app to change or report on.
    assert "changes_target" not in schema["properties"]


@pytest.mark.parametrize("instructions", [
    "REST API for a regular expression library",
    "backend with guardrails on order totals",
    "FastAPI API with a triangular pricing grid",
])
def test_preview_routes_domain_words_like_the_orchestrator(instructions):
    # Substrings: "express" in "expression", "rails" in "guardrails", "angular" in "triangular".
    assembled = AssembledModels(primary_kind="class", domain_model=_model())
    generator, _confidence = _predict_target_generator(assembled, instructions)
    assert generator == "generate_fastapi_backend"


# -- H2: a decided NONE for a named stack stands ---------------------------

@pytest.mark.parametrize("instructions, stack", [
    ("a REST API in Go for the inventory", "Go"),
    ("a Kotlin Ktor backend for the library", "Kotlin + Ktor"),
])
def test_a_none_for_an_unsupported_stack_is_not_overruled_by_keywords(tmp_path, instructions, stack):
    orch = _orch(tmp_path, instructions, {**API, "unsupported_stack": stack, "generator": "NONE"})
    assert orch._select_generator(instructions) is None


def test_a_web_none_without_an_unsupported_stack_keeps_the_scaffold(tmp_path):
    # The rules give every Python web app a generator: a bare NONE is the gap
    # the keyword fallback fills, as before.
    instructions = "Create a web app for a library with a REST API backend"
    orch = _orch(tmp_path, instructions, {**WEB, "unsupported_stack": "", "generator": "NONE"})
    assert orch._select_generator(instructions) == "generate_fastapi_backend"


# -- B5: a failed selector call still decides the target --------------------

def test_selector_failure_still_decides_the_target(tmp_path, caplog):
    instructions = "Generate an Android app to track my reading."
    orch = _orch(tmp_path, instructions, {**NATIVE, "unsupported_stack": "", "generator": "NONE"},
                 fail_first=1)
    with caplog.at_level(logging.WARNING):
        generator = orch._select_generator(instructions)
    assert orch.target is not None and orch.target.kind == "other"
    assert generator is None
    assert not _frontend_task(orch)
    assert any("selector" in r.getMessage().lower() for r in caplog.records
               if r.levelno >= logging.WARNING)


# -- B6 / B1: every modify run asks; changes_target decides -----------------

@pytest.mark.parametrize("instructions, answer", [
    ("Turn it into a Flutter app for my phone.", NATIVE),
    ("Rewrite it as a Tkinter GUI I can run offline.", NATIVE),
    ("Make it a Telegram chatbot.", NATIVE),
    ("Drop the frontend, I only need the REST API now.", API),
])
def test_modify_retargets_without_known_keywords(tmp_path, instructions, answer):
    seed = Target("web_app", "library web app", True)
    orch = _modify(tmp_path, instructions, seed,
                   {**answer, **KEEP, "changes_target": True, "unsupported_stack": "", "generator": "NONE"})
    assert len(orch.client.calls) == 1
    assert orch.target.kind == answer["target_kind"]
    assert orch._previous_target == seed


def test_reworded_target_line_on_same_kind_keeps_the_stack(tmp_path):
    instructions = "Target: Android and iOS mobile app (not a web app).\nAdd a settings screen."
    seed = Target("other", "native Android/iOS app", True)
    orch = _modify(tmp_path, instructions, seed,
                   {**NATIVE, **KEEP, "target_description": "Android and iOS mobile app",
                    "unsupported_stack": "", "generator": "NONE"})
    assert orch.target == seed
    assert orch._previous_target is None
    orch._inventory = "app/build.gradle.kts, app/src/main/java/MainActivity.kt"
    assert STACK_LOCK in orch._build_system_prompt(instructions)


def _agent_modify_instructions(target_line_description, original, later):
    """What modeling-agent sends for a modify run of a non-web app.

    ``generation_handler._instructions_naming_target`` puts
    ``Target: {description} (not a web app).`` (its own classifier's words) and an
    ``Architecture:`` line first; ``smart_generation_handler`` nests that under the
    assistant's notes, then the verbatim original request and the later one.
    """
    return (
        "## Assistant's notes (may be inaccurate)\n\nUse these only for ...\n\n"
        f"Target: {target_line_description} (not a web app).\nArchitecture: MVC.\n"
        "Add a settings screen to the expense tracker.\n\n"
        "## The user's original request, verbatim\n\nThis is the authority.\n\n"
        f"{original}\n\n## Subsequent user request, verbatim\n\n"
        "This later request takes precedence where it changes earlier requirements.\n\n"
        f"{later}"
    )


def test_the_modeling_agents_target_line_does_not_switch_a_native_app(tmp_path):
    # The seed's description came from BESSER's selector, the Target line from the
    # modeling agent's classifier: they almost never match, and every chat-driven
    # modify of a native app dropped the stack lock.
    instructions = _agent_modify_instructions(
        "native mobile app for Android and iOS", "quiero una app móvil para controlar mis gastos",
        "añade una pantalla de ajustes")
    seed = Target("other", "Flutter mobile app", True)
    orch = _modify(tmp_path, instructions, seed,
                   {**NATIVE, **KEEP, "target_description": "native mobile app for Android and iOS",
                    "unsupported_stack": "", "generator": "NONE"})
    assert orch.target == seed and orch._previous_target is None
    orch._inventory = "pubspec.yaml, lib/main.dart"
    assert STACK_LOCK in orch._build_system_prompt(instructions)
    prompt = orch.client.calls[0]["prompt"]
    assert "Target: native mobile app for Android and iOS (not a web app)." in prompt
    assert "Newest user message: añade una pantalla de ajustes" in prompt


def test_modify_keeps_the_seed_target_unless_the_request_changes_it(tmp_path):
    # Even a different kind in the answer: changes_target=False is the decision.
    seed = Target("web_app", "library web app", True)
    orch = _modify(tmp_path, "Make the dashboard easier to use on desktop screens", seed,
                   {**NATIVE, **KEEP, "unsupported_stack": "", "generator": "NONE"})
    assert orch.target == seed and orch._previous_target is None


def test_the_modify_call_names_the_existing_app_and_asks_for_the_new_fields(tmp_path):
    seed = Target("web_app", "library web app", True)
    orch = _modify(tmp_path, "add a search box", seed,
                   {**WEB, **KEEP, "unsupported_stack": "", "generator": "NONE"})
    call = orch.client.calls[0]
    schema = call["tools"][0]["input_schema"]
    assert {"changes_target", "is_failure_report", "reported_entity"} <= set(schema["required"])
    assert "The existing app was built as: library web app" in call["prompt"]
    assert "Author" in call["prompt"] and "Book" in call["prompt"]


def test_the_modify_call_runs_on_the_primary_model(tmp_path):
    # Live eval 2026-10-31: on the cheap planning model, 6 of 222 modify answers
    # turned a web app into a backend ("Add an API endpoint the mobile team can
    # call"); on the primary model, none did.
    seed = Target("web_app", "library web app", True)
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"target": seed.to_dict()}), encoding="utf-8")
    overrides = []

    class _Recording(_Client):
        planning_model = "cheap-sibling"

        def chat(self, system, messages, tools, force_tool=None, model_override=None):
            overrides.append(model_override)
            return super().chat(system, messages, tools, force_tool, model_override)

    orch = LLMOrchestrator(llm_client=_Recording({**WEB, **KEEP, "unsupported_stack": "", "generator": "NONE"}),
                           domain_model=_model(), output_dir=str(tmp_path))
    orch._instructions = "add a search box"
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    orch._route_modify_request()
    assert overrides == [None]
    orch.target = None
    orch._select_generator("a library web app")
    assert overrides[1] == "cheap-sibling"  # the create selector keeps the cheap model


def test_a_modify_without_a_seed_target_is_decided_by_the_call(tmp_path):
    # An old recipe with no target used to fall back to the regexes (S6).
    (tmp_path / ".besser_recipe.json").write_text("{}", encoding="utf-8")
    orch = _orch(tmp_path, "make it a Flutter app", {**NATIVE, **KEEP, "changes_target": True,
                                                     "unsupported_stack": "Flutter", "generator": "NONE"})
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    orch._route_modify_request()
    assert orch.target.kind == "other" and orch._previous_target is None


def test_a_failed_modify_call_keeps_the_seed_and_is_no_fix_run(tmp_path, caplog):
    seed = Target("backend", "REST API", False)
    with caplog.at_level(logging.WARNING):
        orch = _modify(tmp_path, "Add a React frontend", seed, {"generator": "NONE"})
        orch._detect_fix_target(orch._instructions)
    assert orch.target == seed
    assert orch._is_fix_run is False
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# -- H3 / B7: the call decides a failure report; entities are model classes --

@pytest.mark.parametrize("instructions, entity, expected", [
    ("When I delete an author, it is still in the list after refresh.", "Author", ("Author",)),
    ("The total on the order page is wrong: 3 x 10 shows 13.", "", ()),
    ("Editing an author opens the create form instead of the edit form.", "Author", ("Author",)),
])
def test_a_failure_report_the_regexes_missed_is_a_fix_run(tmp_path, instructions, entity, expected):
    seed = Target("web_app", "library web app", True)
    orch = _modify(tmp_path, instructions, seed,
                   {**WEB, **KEEP, "is_failure_report": True, "reported_entity": entity,
                    "unsupported_stack": "", "generator": "NONE"})
    orch._detect_fix_target(instructions)
    assert orch._is_fix_run
    assert orch._fix_target.entities == expected


@pytest.mark.parametrize("instructions", [
    "Map TimeoutError to a 504 response with a retry hint.",
    "Translate the error: Invalid input message into French.",
])
def test_a_feature_request_naming_errors_is_not_a_fix_run(tmp_path, instructions):
    seed = Target("web_app", "library web app", True)
    orch = _modify(tmp_path, instructions, seed,
                   {**WEB, **KEEP, "unsupported_stack": "", "generator": "NONE"})
    orch._detect_fix_target(instructions)
    assert not orch._is_fix_run and orch._fix_target is None


def test_path_segments_api_v1_id_are_not_entities():
    target = parse_reported_target("POST /api/v1/books/{id} returns 500", _model())
    assert target.entities == ("Book",)
    assert target.method == "POST" and target.path == "/api/v1/books/{id}" and target.status == "500"


def test_a_reported_entity_outside_the_model_is_kept():
    target = parse_reported_target("the checkout total is wrong", _model(), reported_entity="Checkout")
    assert target.entities == ("Checkout",)


# -- B4 / H6: an empty checklist only ever comes from the planner -----------

class _PlannerClient:
    _client = object()
    planning_model = None

    def __init__(self, tasks):
        self.tasks = tasks

    def chat(self, system, messages, tools=None, force_tool=None, model_override=None, **kw):
        return {"content": [{"type": "tool_use", "name": "submit_tasks",
                             "input": {"tasks": self.tasks}}]}


def _library():
    book = Class("Book", attributes={Property("title", StringType)})
    return DomainModel("Lib", types={book})


@pytest.mark.parametrize("tasks", [
    ["Add a Delete button to each row of the Book table in the React frontend."],
    [{"task": "Add a Delete button to each row of the Book table in the React frontend.", "kind": "change"},
     {"task": "Replace the default favicon of the frontend with the library logo.", "kind": "change"}],
])
def test_sanitizer_keeps_ui_tasks_that_mention_delete(tasks):
    out = analyze_gaps_via_llm("Add a delete button to each book row", "generate_web_app",
                               _library(), "", _PlannerClient(tasks))
    assert out == [t["task"] if isinstance(t, dict) else t for t in tasks]


def test_all_tasks_sanitized_never_skips_phase2(caplog):
    tasks = [{"task": "Delete the generated React frontend and rebuild it in Vue.", "kind": "removes_scaffold"}]
    with caplog.at_level(logging.WARNING):
        out = analyze_gaps_via_llm("Rebuild the UI in Vue", "generate_web_app",
                                   _library(), "", _PlannerClient(tasks))
    assert out is None  # "no usable checklist", never "the scaffold covers it"
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_an_empty_planner_answer_is_still_trusted():
    out = analyze_gaps_via_llm("a library api", "generate_fastapi_backend", _library(), "",
                               _PlannerClient([]))
    assert out == []


class _NoChatClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):  # pragma: no cover - guard
        raise AssertionError("Phase 2 was not skipped")


def test_request_words_do_not_overrule_an_empty_checklist(tmp_path, monkeypatch):
    # "theme" (a theme park) is not custom styling; the planner judged the scaffold enough.
    class _SM:
        name = "DummySM"

    orch = LLMOrchestrator(llm_client=_NoChatClient(), state_machines=[_SM()],
                           output_dir=str(tmp_path))
    orch._generator_used = "generate_fastapi_backend"
    monkeypatch.setattr(
        "besser.spec_driven_agent.pipeline.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: [],
    )
    orch._run_phase2("a ticket api for a theme park, with an email field on Visitor", extra_issues=[])
    assert orch.total_turns == 0 and orch._phase2_exited_cleanly


# -- Degraded paths (review of 3ee9a794): failing calls, gateways without forced tools --

EXPRESS = "A web app for a courier company: customers pick standard or express delivery."
RUST = "Build it in Rust: a REST API for the library."


class _TextGateway(_Client):
    """A gateway that ignores tool_choice: every reply is text."""

    def __init__(self, text, fail_first=0):
        super().__init__(None, fail_first)
        self.text = text

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append({"prompt": messages[0]["content"], "tools": tools})
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("gateway 502")
        return {"content": [{"type": "text", "text": self.text}]}


class _ProseClient:
    """A client with no force_tool / model_override parameters."""

    model = "mock-model"
    usage = UsageTracker("mock-model")
    _client = object()

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat(self, system, messages, tools):
        self.calls += 1
        return {"content": [{"type": "text", "text": self.text}]}


def _plain(client, tmp_path, instructions):
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(), output_dir=str(tmp_path))
    orch._instructions = instructions
    return orch


def test_a_failed_selector_then_a_stack_without_generator_builds_from_scratch(tmp_path):
    orch = _orch(tmp_path, RUST, {**API, "unsupported_stack": "Rust", "generator": "NONE"}, fail_first=1)
    assert orch._select_generator(RUST) is None
    assert orch.target.kind == "backend"


def test_a_domain_word_keeps_its_scaffold_when_every_call_fails(tmp_path):
    orch = _orch(tmp_path, EXPRESS, WEB, fail_first=99)
    assert orch._select_generator(EXPRESS) == "generate_fastapi_backend"
    assert len(orch.client.calls) == 3  # an outage adds no JSON retries


def test_a_routed_none_without_a_stack_uses_the_models_generator_not_keywords(tmp_path):
    orch = _orch(tmp_path, EXPRESS, {**WEB, "unsupported_stack": "", "generator": "NONE"})
    assert orch._select_generator(EXPRESS) == "generate_fastapi_backend"


@pytest.mark.parametrize("instructions", [RUST, "A Kotlin backend API for the library"])
def test_a_prose_none_stands(tmp_path, instructions):
    assert _plain(_ProseClient("NONE"), tmp_path, instructions)._select_generator(instructions) is None


def test_a_stack_without_generator_overrules_the_answers_generator(tmp_path, caplog):
    orch = _orch(tmp_path, RUST, {**API, "unsupported_stack": "Rust", "generator": "generate_fastapi_backend"})
    with caplog.at_level(logging.WARNING):
        assert orch._select_generator(RUST) is None
    assert any("no generator builds" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("instructions, expected", [
    (RUST, None),
    ("A Kotlin Ktor backend API for the library", None),
    ("A Flutter mobile app with a REST API for the library", None),
    (EXPRESS, "generate_web_app"),
])
def test_the_preview_estimate_reads_stacks_narrowly(instructions, expected):
    assembled = AssembledModels(primary_kind="class", domain_model=_model(), gui_model=object())
    assert _predict_target_generator(assembled, instructions)[0] == expected


def test_a_gateway_that_ignores_tool_choice_still_routes_a_create_run(tmp_path):
    answer = json.dumps({**NATIVE, "unsupported_stack": "Flutter", "generator": "NONE"})
    orch = _plain(_TextGateway(f"Here is my answer: {answer}"), tmp_path, "a Flutter app for the gym")
    assert orch._select_generator("a Flutter app for the gym") is None
    assert orch.target.kind == "other"


def test_a_prose_only_client_routes_through_json(tmp_path):
    answer = json.dumps({**NATIVE, "unsupported_stack": "", "generator": "NONE"})
    orch = _plain(_ProseClient(answer), tmp_path, "an Android app for the gym")
    orch._select_generator("an Android app for the gym")
    assert orch.target is not None and orch.target.kind == "other"


def test_a_gateway_that_ignores_tool_choice_still_detects_a_failure_report(tmp_path):
    text = "I get a 500 when I save a book"
    seed = Target("web_app", "library web app", True)
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"target": seed.to_dict()}), encoding="utf-8")
    answer = json.dumps({**WEB, **KEEP, "is_failure_report": True, "reported_entity": "Book",
                         "unsupported_stack": "", "generator": "NONE"})
    orch = _plain(_TextGateway(answer), tmp_path, text)
    orch._modify_mode = True
    orch._seed_generator_files_from_recipe()
    orch._route_modify_request()
    orch._detect_fix_target(text)
    assert orch._is_fix_run and orch._fix_target.entities == ("Book",)


def test_a_garbled_task_kind_still_drops_the_demolition_task(caplog):
    tasks = [{"task": "Delete the generated React frontend and rebuild it in Vue.", "kind": "removes-scaffold"},
             {"task": "Add a search box to the Book list.", "kind": "change"},
             {"task": "Add a footer.", "kind": "tweak"}]
    with caplog.at_level(logging.WARNING):
        out = analyze_gaps_via_llm("Rebuild the UI in Vue", "generate_web_app", _library(), "",
                                   _PlannerClient(tasks))
    assert out == ["Add a search box to the Book list.", "Add a footer."]
    assert any("without a valid kind" in r.getMessage() for r in caplog.records)
