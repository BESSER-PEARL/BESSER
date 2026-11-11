"""The requirements judge is the largest planning call of a run; it must not be paid for nothing.

Its prompt carries a digest of the generated source: ~12k tokens on a 3-class
app, ~40k on a 13-class one, per call. A review (2026-11) measured three ways
a run re-sent it without any chance of a different answer:

- the cached judgment was keyed on the whole workspace, so an edit to a file
  the judge never reads (README, a requirements pin, a test) re-sent a
  byte-identical prompt and re-rolled the verdicts of an unchanged app;
- re-judging ONE requirement with a bad citation re-sent the whole app
  (39,955 tokens against 3,032 for the cited file);
- a judge that returned nothing usable was asked again, twice, on every
  validation pass, even with nothing changed: three passes, six calls.
"""

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SPEC = "Room numbers are unique."
REQUIREMENT = {"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"}


def _model():
    room = Class(name="Room")
    room.attributes = {Property(name="number", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="Hotel", types={room})


class _ProseClient:
    """A real provider whose judge answers in prose, never with the tool."""

    model = "mock-model"
    planning_model = "cheap-model"

    def __init__(self):
        self._client = object()
        self.usage = UsageTracker("mock-model")
        self.calls = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append(model_override)
        block = type("T", (), {"type": "text", "text": "All requirements look implemented."})()
        return {"stop_reason": "end_turn", "content": [block]}


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def orch(tmp_path):
    orchestrator = LLMOrchestrator(
        llm_client=_ProseClient(), domain_model=_model(), output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False,
    )
    orchestrator._instructions = SPEC
    orchestrator._generator_used = "generate_fastapi_backend"
    orchestrator._requirements = [dict(REQUIREMENT)]
    _write(tmp_path, "routers/room.py", "def create_room():\n    return 1\n")
    return orchestrator


@pytest.fixture
def judge_calls(monkeypatch):
    calls = []

    def judge(requirements, digest, client, **kwargs):
        calls.append(digest)
        return [dict(REQUIREMENT, status="missing", evidence="", note="no unique check")]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return calls


# ------------------------------------------------- what the judgment is keyed on


def test_edits_to_files_the_judge_never_reads_reuse_the_judgment(orch, tmp_path, judge_calls):
    first = orch._collect_requirement_issues()
    _write(tmp_path, "README.md", "# Hotel\n")
    _write(tmp_path, "requirements.txt", "fastapi==0.115.0\n")
    _write(tmp_path, "tests/test_room.py", "def test_room():\n    assert True\n")

    assert orch._collect_requirement_issues() == first
    assert len(judge_calls) == 1, "an unchanged app was judged again"


def test_an_edit_to_judged_source_is_judged_again(orch, tmp_path, judge_calls):
    """Raw bytes, so even a comment-only edit re-judges: the digest's comment
    stripping is a prompt-size heuristic, not a promise the code is the same."""
    orch._collect_requirement_issues()
    _write(tmp_path, "routers/room.py", "def create_room():\n    return 1\n# reviewed\n")
    orch._collect_requirement_issues()
    _write(tmp_path, "routers/booking.py", "def create_booking():\n    return 2\n")
    orch._collect_requirement_issues()

    assert len(judge_calls) == 3


def test_a_changed_specification_is_judged_again(orch, judge_calls):
    """The original specification is the judge's scope authority, so it is
    part of what a cached judgment answered."""
    orch._collect_requirement_issues()
    orch._instructions = "Room numbers are unique. A room holds at most four guests."
    orch._collect_requirement_issues()

    assert len(judge_calls) == 2
