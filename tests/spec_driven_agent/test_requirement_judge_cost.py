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

import logging

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.issues import _classify_issue, is_completion_issue

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


# ------------------------------------------------------- a judge that fails


# A judge outage is an unknown, not a defect: no code edit brings the judge
# back, so it must not drive the blocker-only fix loop - but it still refuses
# verified completion.
_JUDGE_FAILED = "validation unverified: the requirements judge did not run completely (it returned no verdicts"


def _is_unverified_warning(message):
    issue = _classify_issue(message)
    return issue.severity == "warning" and is_completion_issue(issue)


def test_a_failed_judgment_is_not_asked_again_for_the_same_source(orch, monkeypatch):
    monkeypatch.setattr(ledger, "_JUDGE_RETRY_BACKOFF_SECONDS", 0)

    for _ in range(3):
        issues = orch._collect_requirement_issues()
        assert len(issues) == 1 and issues[0].startswith(_JUDGE_FAILED)
        assert _is_unverified_warning(issues[0])

    # One round: the planning model, then one retry on the primary model.
    assert orch.client.calls == ["cheap-model", None]


def test_failed_judgments_stop_after_the_per_run_cap(orch, tmp_path, monkeypatch, caplog):
    """No code edit can fix the failure finding; the fix loop's edits must not
    buy a fresh full-price round on every pass."""
    monkeypatch.setattr(ledger, "_JUDGE_RETRY_BACKOFF_SECONDS", 0)
    rounds = ledger.JUDGE_FAILED_ROUNDS_PER_RUN

    for n in range(rounds + 2):
        _write(tmp_path, "routers/room.py", f"def create_room():\n    return {n}\n")
        with caplog.at_level(logging.WARNING):
            issues = orch._collect_requirement_issues()
        assert len(issues) == 1 and issues[0].startswith(_JUDGE_FAILED)
        assert _is_unverified_warning(issues[0])

    assert len(orch.client.calls) == rounds * ledger._JUDGE_MAX_ATTEMPTS
    assert "not asked again this run" in caplog.text


# --------------------------------------------------- citation repair digest


@pytest.fixture
def app(tmp_path):
    _write(tmp_path, "backend/pydantic_classes.py", "class RoomCreate:\n    number: int\n")
    _write(tmp_path, "backend/sql_alchemy.py", "number = mapped_column(Integer, unique=True)\n")
    _write(tmp_path, "backend/routers/room.py", "def create_room():\n    raise HTTPException(409)\n")
    _write(tmp_path, "backend/routers/booking.py", "def create_booking():\n    return 'booking'\n")
    _write(tmp_path, "frontend/src/pages/Rooms.tsx", "export const Rooms = () => null;\n")
    return tmp_path


def test_the_repair_digest_holds_the_cited_and_schema_files_only(app):
    digest = ledger.build_app_digest(str(app), focus_paths=["routers/room.py"])
    source, coverage = digest.split("## Digest coverage")

    assert source.startswith("### backend/routers/room.py")
    assert "### backend/pydantic_classes.py" in source
    assert "### backend/sql_alchemy.py" in source
    assert "routers/booking.py" not in source and "Rooms.tsx" not in source
    # The judge is told which files it was not shown, so it cannot read their
    # absence as missing behaviour.
    assert "backend/routers/booking.py" in coverage and "frontend/src/pages/Rooms.tsx" in coverage
    assert "not evidence of absent behavior" in coverage


def test_a_repair_with_no_resolvable_focus_reads_the_whole_app(app, caplog):
    with caplog.at_level(logging.WARNING):
        digest = ledger.build_app_digest(str(app), focus_paths=["nowhere.py"])

    assert digest == ledger.build_app_digest(str(app))
    assert "no cited path resolved" in caplog.text


def test_the_citation_repair_call_sends_the_focused_digest(orch, tmp_path, monkeypatch):
    """End to end through the orchestrator: one bad citation re-judges one
    requirement against the file it cited, not against the whole app."""
    _write(tmp_path, "routers/booking.py", "def create_booking():\n    return 'booking'\n")
    digests = []

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        return [dict(REQUIREMENT, status="implemented", evidence="routers/room.py: raise HTTPException(409)")]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    orch._collect_requirement_issues()  # the citation is not in room.py: unverified
    orch._collect_requirement_issues()  # same source: the one citation repair

    assert len(digests) == 2
    assert "### routers/booking.py" in digests[0]
    assert digests[1].startswith("### routers/room.py")
    assert "### routers/booking.py" not in digests[1]


# ------------------------------------------------------------ the spend cap


class _CostlyClient(_ProseClient):
    """Every call costs $0.30; the planning model raises, the primary answers in prose."""

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append(model_override)
        self.usage._extra_cost_usd += 0.30
        if model_override:
            raise RuntimeError("planning model not reachable")
        return {"stop_reason": "end_turn", "content": []}


def test_the_spend_cap_is_checked_before_every_paid_judge_call(tmp_path, monkeypatch):
    """Checked once before the judge, three full-digest calls ran from $0.95
    to $1.85 against a $1.00 cap: planning model, its primary fallback, retry."""
    monkeypatch.setattr(ledger, "_JUDGE_RETRY_BACKOFF_SECONDS", 0)
    client = _CostlyClient()
    orch = LLMOrchestrator(
        llm_client=client, domain_model=_model(), output_dir=str(tmp_path),
        enable_tracing=False, enable_checkpointing=False, max_cost_usd=1.0,
    )
    orch._instructions = SPEC
    orch._generator_used = "generate_fastapi_backend"
    orch._requirements = [dict(REQUIREMENT)]
    _write(tmp_path, "routers/room.py", "def create_room():\n    return 1\n")
    client.usage._extra_cost_usd = 0.95

    issues = orch._collect_requirement_issues()

    assert client.calls == ["cheap-model"]
    assert issues == ["requirement unverified: verification stopped or budget exhausted; "
                      "original-specification coverage is not verified."]
    assert not orch._requirement_judge_failures, "a budget stop is not a judge failure"
