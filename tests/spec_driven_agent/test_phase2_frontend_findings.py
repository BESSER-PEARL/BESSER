"""How Phase 2 uses the static frontend findings validate_app now reports.

Two regressions of the round that added them, found by a scripted Phase 2 on r9-p51:
- the opening prompt listed an advisory acceptance warning ("entity PaymentMethod - no
  frontend page") under "Issues to fix from Phase 1 validation";
- a frontend blocker at end_turn ended Phase 2 at once, before the open-checklist nudge
  and without the model ever seeing the blocker.
Now only blockers open the prompt, and a static frontend blocker is sent back once
(``_MAX_FRONTEND_BLOCKER_NUDGES``, inside the end-turn nudge cap) before Phase 2 ends.
"""
import pytest

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.target import Target
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model

DEAD_FORM = ("import React from 'react';\nexport default function Feedback() {\n"
             "  return (<form onSubmit={(e) => e.preventDefault()}><input name='msg'/></form>);\n}\n")
BOOK_PAGE = "import React from 'react';\nexport default function BookList() { return <h1>Books</h1>; }\n"


class _Block:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _EndsEveryTurn:
    """Ends every turn without doing anything; records what it was shown."""

    model = "mock"
    max_tokens = None

    def __init__(self):
        self.usage = UsageTracker("mock")
        self.systems, self.last_messages = [], []

    def chat(self, system, messages, tools, **_):
        self.systems.append(system)
        self.last_messages.append(messages[-1]["content"])
        return {"stop_reason": "end_turn", "content": [_Block(type="text", text="done")]}


def _run(tmp_path, monkeypatch, *, dead_form=True, planner=("Add a monthly overview endpoint",),
         backend_blocker=False):
    generate_backend(tmp_path, shelf_model(author=True))
    pages = tmp_path / "frontend" / "src" / "pages"
    pages.mkdir(parents=True)
    (tmp_path / "frontend" / "package.json").write_text('{"name": "shelf"}', encoding="utf-8")
    (pages / "BookList.jsx").write_text(BOOK_PAGE, encoding="utf-8")
    if dead_form:
        (pages / "Feedback.jsx").write_text(DEAD_FORM, encoding="utf-8")
    client = _EndsEveryTurn()
    orch = LLMOrchestrator(llm_client=client, domain_model=shelf_model(author=True), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_import_smoke_check=False,
                           enable_requirements_ledger=False, use_streaming=False)
    orch._generator_used = GENERATOR
    orch.target = Target(kind="web_app", description="shelf web app", has_user_interface=True)
    orch._instructions = "Build the shelf web app"
    execution = ["python contract: backend/main_api.py line 1: undefined name 'x'"] if backend_blocker else []
    monkeypatch.setattr(orch, "_collect_execution_issues", lambda: list(execution))
    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", lambda **_: list(planner))
    orch._run_phase2(orch._instructions)
    return orch, client


def _text(content) -> str:
    return content if isinstance(content, str) else " ".join(block.get("text", "") for block in content)


def test_opening_prompt_lists_blockers_only(tmp_path, monkeypatch):
    _, client = _run(tmp_path, monkeypatch)
    issues = client.systems[0].split("## Issues to fix from Phase 1 validation", 1)[1].split("When done", 1)[0]
    assert "Feedback.jsx has a form whose onSubmit does nothing" in issues
    assert "acceptance:" not in issues


def test_no_blocker_no_issues_section(tmp_path, monkeypatch):
    _, client = _run(tmp_path, monkeypatch, dead_form=False)
    assert "## Issues to fix" not in client.systems[0]


def test_frontend_blocker_is_sent_back_once_and_open_items_keep_their_nudges(tmp_path, monkeypatch):
    orch, client = _run(tmp_path, monkeypatch)
    nudges = [_text(content) for content in client.last_messages[1:]]
    assert len(client.systems) == 3
    assert "the frontend still has these blockers" in nudges[0] and "still OPEN" in nudges[0]
    assert "the frontend still has these blockers" not in nudges[1] and "still OPEN" in nudges[1]
    assert (orch._phase2_stop_reason, orch._phase2_stop_detail) == (
        "validation_required", "end_turn frontend blocker gate")


def test_frontend_blocker_alone_gets_one_nudge_then_phase2_ends(tmp_path, monkeypatch):
    orch, client = _run(tmp_path, monkeypatch, planner=())
    assert len(client.systems) == 2
    assert orch._phase2_stop_reason == "validation_required"


@pytest.mark.parametrize("dead_form", [True, False])
def test_a_backend_blocker_still_ends_phase2_at_once(tmp_path, monkeypatch, dead_form):
    orch, client = _run(tmp_path, monkeypatch, dead_form=dead_form, backend_blocker=True)
    assert len(client.systems) == 1
    assert (orch._phase2_stop_reason, orch._phase2_stop_detail) == ("validation_required", "end_turn blocker gate")
