"""Frontend guidance follows the routed stack and the scaffold, not the request's words.

Every web_app target was told to build a React/Vite app: the checklist item and
Rule 15 ignored the routing answer, so a request routed to Vue got a React task
beside a role line saying "in the stack the user asked for", and a Django
scaffold was ordered a React app instead of its templates. With no target
decided, an English word list chose whether a frontend was due at all, so the
same request got a frontend in English and none in German or Spanish; the
models decide it now. The idiom block came from request words too, not from
the routing answer.

React stays the default whenever the routed stack names no other frontend
framework, so the React guards (``plugins: [react()]``) are never stripped by a
missing or unknown value.
"""

from types import SimpleNamespace

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.stack_metadata import idiom_guidance_section
from besser.spec_driven_agent.planning.target import WEB_APP, Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker

WEB = Target(WEB_APP, "web app", True)
REACT_GUARD = "plugins: [react()]"


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("never called")


def _model():
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=PrimitiveDataType("str"))}
    return DomainModel(name="Library", types={book})


def _orch(tmp_path, instructions="Build a library web app", target=WEB, routed_stack=None, generator=None):
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=_model(), output_dir=str(tmp_path))
    orch._instructions = instructions
    orch.target = target
    if routed_stack is not None:
        orch._routed, orch._unsupported_stack = True, routed_stack
    orch._generator_used = generator
    return orch


def _frontend_tasks(orch):
    return [t["text"] for t in orch._deterministic_gap_tasks()
            if isinstance(t, dict) and t["text"].startswith("Create the ")]


def _rule_15(**kwargs):
    prompt = build_system_prompt(_model(), None, None, "", "Build a library web app", 10, target=WEB, **kwargs)
    start = prompt.index("15. **")
    return prompt[start:prompt.index("\n\n", start)]


@pytest.mark.parametrize("routed_stack, framework", [
    ("Vue 3 + Pinia", "Vue"), ("Angular", "Angular"), ("SvelteKit", "SvelteKit"), ("Next.js", "Next.js"),
    ("Vue.js + Express", "Vue"), ("Spring Boot + Angular", "Angular"),
])
def test_a_routed_frontend_framework_gets_no_react_text(tmp_path, routed_stack, framework):
    task, = _frontend_tasks(_orch(tmp_path, routed_stack=routed_stack))
    assert "React" not in task and "vite" not in task
    # The framework, not the whole stack: no "Spring Boot + Angular frontend".
    assert task.startswith(f"Create the {framework} frontend ")
    assert routed_stack == framework or routed_stack not in task
    rule = _rule_15(routed_stack=routed_stack)
    assert f"For the {framework} frontend" in rule
    assert "React" not in rule


@pytest.mark.parametrize("routed_stack", [None, "", "Spring Boot", "Express", "a shop for vintage records"])
def test_no_frontend_framework_named_keeps_the_react_guards(tmp_path, routed_stack):
    orch = _orch(tmp_path, routed_stack=routed_stack)
    assert _frontend_tasks(orch) == [orch._FRONTEND_CHECKLIST_TASK]
    rule = _rule_15(routed_stack=routed_stack)
    assert "For the React frontend" in rule and REACT_GUARD in rule


def test_a_django_scaffold_without_pages_is_ordered_django_pages_not_a_react_app(tmp_path):
    # Without a GUI model the Django generator writes models and admin only: the
    # item and the blocker are the only things that ask for the pages.
    orch = _orch(tmp_path, generator="generate_django")
    task, = _frontend_tasks(orch)
    assert "Django" in task and "template" in task
    assert "React" not in task and "vite" not in task
    assert orch._collect_missing_frontend_issue()
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "book_list.html").write_text("<table></table>", encoding="utf-8")
    assert not _frontend_tasks(orch) and not orch._collect_missing_frontend_issue()


def test_a_django_crud_rule_carries_no_single_page_app_bullets():
    rule = _rule_15(scaffold_family="django")
    assert "For the Django frontend" in rule and "React" not in rule
    for spa_only in ("${base}${path}", "raw-JSON textarea", "Create MUST POST to the collection route",
                     "`package.json` dependencies", "POST / PUT / DELETE endpoints", "through the API layer",
                     "onSubmit", "Router/Routes", "on every data fetch", "the router does not declare"):
        assert spa_only not in rule
        assert spa_only in _rule_15()
    # The server-rendered wiring instead: a view saves the POSTed form, a delete view, a "/" pattern.
    for server_rendered in ("saves the POSTed", "a delete view", 'no pattern for "/"'):
        assert server_rendered in rule and server_rendered not in _rule_15()


SAME_REQUEST = [
    "I want a todo app with a web interface",
    "Ich möchte eine Webanwendung für meine Aufgaben",
    "Quiero una aplicación web para gestionar mis tareas",
]


@pytest.mark.parametrize("gui", [False, True])
def test_without_a_target_the_models_decide_in_any_language(tmp_path, gui):
    verdicts = set()
    for n, instructions in enumerate(SAME_REQUEST):
        orch = _orch(tmp_path / str(n), instructions=instructions, target=None)
        orch.gui_model = SimpleNamespace(name="GUI", modules=set()) if gui else None
        prompt = build_system_prompt(_model(), orch.gui_model, None, "", instructions, 10)
        verdicts.add((bool(_frontend_tasks(orch)), bool(orch._collect_missing_frontend_issue()),
                      "NAVIGABLE CRUD frontend" in prompt))
    assert verdicts == {(gui, gui, gui)}


@pytest.mark.parametrize("routed_stack, label", [
    ("Spring Boot", "Spring Boot"), ("Rust + Axum", "Rust (axum)"), ("Flask", "Flask / FastAPI (Python)"),
    ("NestJS", "this stack"),
])
def test_the_idiom_block_comes_from_the_routed_stack(routed_stack, label):
    assert idiom_guidance_section(routed_stack).startswith(f"## Idiomatic conventions for {label}\n")


@pytest.mark.parametrize("routed_stack", [None, ""])
def test_a_generated_or_unrouted_stack_gets_no_idiom_block_whatever_the_request_says(routed_stack):
    for request in ("A FastAPI shop app with orders", "A Django shop app", "Build a REST API in Rust with axum"):
        prompt = build_system_prompt(_model(), None, None, "", request, 10, routed_stack=routed_stack)
        assert "Idiomatic conventions" not in prompt
