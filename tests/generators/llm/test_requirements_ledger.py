"""The run is verified against what the user asked for, not only against the model.

Run 19h35 (2026-09-18, hotel, Qwen3-30B): with the verbatim spec in hand the
gap analyser planned the guest-capacity rule twice (tasks 7 and 13) and the app
still shipped without it; the unique room number and the extra charges were
never planned at all. Every static gate reported success. An opencode build of
the same spec with the same model had all three, because it read the spec and
implemented what it read. The ledger does the same in two bounded calls: extract
the atomic requirements once, judge each against the generated code on every
Phase 3 pass, re-check every citation, and feed the misses to the fix loop.
"""

import os

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.generators.llm import requirements_ledger as ledger
from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import LLMOrchestrator, _classify_issue

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "run_0c537a4e")

SPEC = (
    "Every room is identified by its room number. The total number of guests "
    "listed on a booking may never be greater than the combined capacity of "
    "all the rooms it covers. The total price is worked out from the agreed "
    "prices of the rooms involved, the length of the stay and any extra "
    "charges recorded against those rooms."
)
INSTRUCTIONS = (
    "Generate a web app for a hotel booking system.\n\n"
    "## The user's original request, verbatim\n\n"
    "This is the authority. Where the summary above is shorter or differs, "
    "this text wins.\n\n" + SPEC
)


class _ToolClient:
    """Answers each forced tool call with the next canned payload."""

    model = "tool-model"

    def __init__(self, payloads):
        self.usage = UsageTracker("tool-model")
        self._client = object()          # a real provider, to the ledger
        self.payloads = list(payloads)
        self.calls = []

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        self.calls.append((force_tool, messages[-1]["content"]))
        block = type("B", (), {"type": "tool_use", "name": force_tool,
                               "input": self.payloads.pop(0)})()
        return {"stop_reason": "tool_use", "content": [block]}


class _MockClient:
    """No ``_client``: the test double the harness must never call out from."""

    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        raise AssertionError("a mock client must never be asked to plan")


# ------------------------------------------------------------- the spec text


def test_the_verbatim_request_is_what_gets_extracted():
    text = ledger.original_request(INSTRUCTIONS)
    assert text == SPEC
    assert "This is the authority" not in text


def test_without_the_marker_the_whole_instructions_are_the_request():
    assert ledger.original_request("  build a todo app  ") == "build a todo app"


# -------------------------------------------------------------- extraction


def test_requirements_are_numbered_in_order():
    client = _ToolClient([{"requirements": [
        {"text": "Room numbers are unique", "kind": "uniqueness"},
        {"text": "Guests never exceed the rooms' combined capacity", "kind": "rule"},
        {"text": ""},
    ]}])
    reqs = ledger.extract_requirements(INSTRUCTIONS, client)
    assert [r["id"] for r in reqs] == [1, 2]
    assert reqs[0]["kind"] == "uniqueness"
    assert client.calls[0][0] == "submit_requirements"
    assert SPEC in client.calls[0][1]
    assert "This is the authority" not in client.calls[0][1]


def test_a_mock_client_is_never_called():
    assert ledger.extract_requirements(INSTRUCTIONS, _MockClient()) is None


# ------------------------------------------------------------------ digest


def test_the_digest_reads_schemas_then_routers_and_skips_the_stdlib():
    """Validators and unique columns first, then every router: the hotel
    scaffold's eight routers alone are ~55k chars, so the old 60k budget
    dropped pydantic_classes.py, the one file that holds the validators."""
    digest = ledger.build_app_digest(FIXTURE)
    schemas_at = digest.index("### web_app/backend/pydantic_classes.py")
    orm_at = digest.index("### web_app/backend/sql_alchemy.py")
    routers_at = digest.index("### web_app/backend/routers/booking.py")
    assert schemas_at < routers_at and orm_at < routers_at
    assert "### web_app/backend/routers/room.py" in digest
    assert "bal_stdlib.py" not in digest
    assert "\n\n\n" not in digest, "blank lines are dropped"
    assert len(digest) <= ledger._DIGEST_MAX_TOTAL_CHARS


# ---------------------------------------------------------- evidence check


def _verdict(status, evidence="", kind="rule",
             text="A booking must not be saved without an employee"):
    return {"id": 1, "text": text, "kind": kind,
            "status": status, "evidence": evidence, "note": ""}


# A real enforcing line from the fixture router, and a real declaration.
_ENFORCING_LINE = 'raise HTTPException(status_code=400, detail="Employee ID is required")'
_DECLARING_LINE = 'Employee.handledBy: Mapped_[List_["Booking"]] = relationship("Booking", back_populates="employee", foreign_keys=[Booking.employee_id])'


def test_a_quoted_enforcing_line_that_exists_stays_implemented():
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/routers/booking.py: {_ENFORCING_LINE}")],
        FIXTURE,
    )
    assert out[0]["status"] == "implemented"


def test_a_path_suffix_and_whitespace_differences_are_accepted():
    spaced = 'routers/booking.py:   raise  HTTPException(status_code=400,   detail="Employee ID is required")'
    out = ledger.verify_evidence([_verdict("implemented", spaced)], FIXTURE)
    assert out[0]["status"] == "implemented"


def test_a_rule_backed_only_by_a_declaration_is_missing():
    """The live judge marked 'guests never exceed the rooms' capacity'
    implemented by citing ``Booking.reservedRooms``. A relationship existing
    enforces nothing; that citation is what a missing rule looks like."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}", kind="rule")],
        FIXTURE,
    )
    assert out[0]["status"] == "missing"
    assert "declares data" in out[0]["note"]


def test_a_non_rule_kind_may_cite_a_declaration():
    """'A booking is handled by an employee' IS the relationship."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}", kind="action")],
        FIXTURE,
    )
    assert out[0]["status"] == "implemented"


def test_a_data_statement_labelled_rule_keeps_its_column():
    """The extractor pads the ledger with 'a person must have a first name'
    and may call it a rule; the column is that requirement, not a gap."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}",
                  kind="rule", text="Each booking must have an employee handling it")],
        FIXTURE,
    )
    assert out[0]["status"] == "implemented"


def test_an_at_least_rule_needs_more_than_a_relationship():
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}",
                  kind="rule", text="A booking must cover at least one room")],
        FIXTURE,
    )
    assert out[0]["status"] == "missing"


def test_a_must_have_field_labelled_validation_keeps_its_column():
    """The extractor labels 'each person must have a first name' a validation;
    with no constraint in the text, the column is the implementation."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}",
                  kind="validation", text="Each booking must have an employee")],
        FIXTURE,
    )
    assert out[0]["status"] == "implemented"


def test_escape_sequences_quote_style_and_later_lines_are_normalised():
    """Live 2026-09-18: quotes arrived with literal backslash-n sequences,
    single quotes where the file has double quotes, and abridged continuation
    lines; ten real citations were reported unverified."""
    quoted = (
        "web_app/backend/sql_alchemy.py: "
        + _DECLARING_LINE.replace('"', "'")
        + "\\n    # abridged continuation that is not in the file"
    )
    out = ledger.verify_evidence([_verdict("implemented", quoted, kind="action")], FIXTURE)
    assert out[0]["status"] == "implemented"


def test_a_verbatim_line_counts_even_when_the_path_is_the_word_path():
    """Live 2026-09-18: the judge wrote the literal word 'path' as the file
    for all 40 citations. The quote is the evidence; it is searched for."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"path: {_ENFORCING_LINE}")], FIXTURE,
    )
    assert out[0]["status"] == "implemented"


@pytest.mark.parametrize("evidence", [
    "web_app/backend/routers/nowhere.py: raise HTTPException(status_code=418, detail='no such line')",
    "web_app/backend/routers/booking.py: raise HTTPException(status_code=418, detail='no such line')",
    "web_app/backend/routers/booking.py",
    "",
])
def test_a_quote_that_is_not_in_the_file_is_unverified(evidence):
    out = ledger.verify_evidence([_verdict("implemented", evidence)], FIXTURE)
    assert out[0]["status"] == "unverified"


def test_missing_and_partial_are_not_touched_by_the_check():
    out = ledger.verify_evidence(
        [_verdict("missing"), _verdict("partial")], FIXTURE,
    )
    assert [v["status"] for v in out] == ["missing", "partial"]


# ------------------------------------------------------------------- judge


def test_a_requirement_the_judge_skipped_is_missing_not_implemented():
    reqs = [{"id": 1, "text": "a", "kind": ""}, {"id": 2, "text": "b", "kind": ""}]
    client = _ToolClient([{"verdicts": [
        {"id": 1, "status": "implemented", "evidence": "x.py"},
    ]}])
    verdicts = ledger.judge_coverage(reqs, "### x.py\ncode", client)
    assert [v["status"] for v in verdicts] == ["implemented", "missing"]
    assert client.calls[0][0] == "submit_verdicts"
    assert "R2. b" in client.calls[0][1]


# --------------------------------------------------------- issues + severity


def test_only_a_missing_requirement_is_a_blocker():
    issues = ledger.ledger_issues([
        {"id": 1, "text": "Room numbers are unique", "status": "missing", "note": "no unique constraint"},
        {"id": 2, "text": "Bills can be paid", "status": "partial", "note": "no refusal when paid"},
        {"id": 3, "text": "Bookings can be cancelled", "status": "unverified", "evidence": "x.py:cancel"},
        {"id": 4, "text": "Bookings are created", "status": "implemented", "evidence": "y.py"},
    ])
    assert len(issues) == 3
    assert issues[0].startswith("requirement: R1 — Room numbers are unique is not implemented")
    assert _classify_issue(issues[0]).severity == "blocker"
    assert _classify_issue(issues[1]).severity == "warning"
    assert _classify_issue(issues[2]).severity == "warning"


# ---------------------------------------------------------------- the run


@pytest.fixture
def simple_model():
    room = Class(name="Room")
    room.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="roomNumber", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="Hotel", types={room})


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_phase3_feeds_missing_requirements_to_the_fix_loop(simple_model, tmp_path, monkeypatch):
    _write(tmp_path, "backend/routers/room.py", "def create_room():\n    pass\n")
    monkeypatch.setattr(ledger, "extract_requirements", lambda instr, client: [
        {"id": 1, "text": "Rooms can be created", "kind": "action"},
        {"id": 2, "text": "Room numbers are unique", "kind": "uniqueness"},
    ])
    monkeypatch.setattr(ledger, "judge_coverage", lambda reqs, digest, client: [
        {"id": 1, "text": "Rooms can be created", "status": "implemented",
         "evidence": "backend/routers/room.py:create_room", "note": ""},
        {"id": 2, "text": "Room numbers are unique", "status": "missing",
         "evidence": "", "note": "no unique constraint anywhere"},
    ])
    orch = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=simple_model, output_dir=str(tmp_path),
    )
    orch._instructions = INSTRUCTIONS
    issues = orch._collect_validation_issues()
    blockers = [i.message for i in issues if i.severity == "blocker"]
    assert any(m.startswith("requirement: R2 — Room numbers are unique") for m in blockers), blockers
    assert [v["status"] for v in orch._requirement_verdicts] == ["implemented", "missing"]


def test_the_ledger_can_be_switched_off(simple_model, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "extract_requirements",
                        lambda *_: (_ for _ in ()).throw(AssertionError("must not run")))
    orch = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=simple_model, output_dir=str(tmp_path),
        enable_requirements_ledger=False,
    )
    orch._instructions = INSTRUCTIONS
    assert not [i for i in orch._collect_validation_issues()
                if i.message.startswith("requirement")]


def _run_phase3_with(orch, passes, monkeypatch):
    """Drive the Phase 3 loop with scripted validation passes and record
    whether the pre-Phase-3 snapshot was restored."""
    calls = {"n": 0, "restored": False}

    def scripted():
        idx = min(calls["n"], len(passes) - 1)
        calls["n"] += 1
        return list(passes[idx])

    monkeypatch.setattr(orch, "_collect_validation_issues", scripted)
    monkeypatch.setattr(orch, "_invoke_phase3_fix_loop", lambda blockers, is_first_attempt: None)
    monkeypatch.setattr(orch, "_restore_snapshot", lambda: calls.__setitem__("restored", True) or True)
    orch._run_phase3_validation()
    return calls["restored"]


def _req(n):
    from besser.generators.llm.orchestrator import ValidationIssue
    return ValidationIssue("blocker", f"requirement: R{n} — rule {n} is not implemented")


def test_a_flapping_requirement_verdict_never_rolls_back_a_fix(simple_model, tmp_path, monkeypatch):
    """Two judge calls on the same 19h35 app returned 12 and 22 missing
    requirements. A verdict that appears between passes is judge variance,
    not a regression the fix caused; rolling back would discard real work."""
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path), auto_fix_issues=True)
    restored = _run_phase3_with(orch, [[_req(1)], [_req(1), _req(2)]], monkeypatch)
    assert restored is False


def test_a_new_hard_blocker_after_a_fix_still_rolls_back(simple_model, tmp_path, monkeypatch):
    from besser.generators.llm.orchestrator import ValidationIssue
    syntax = ValidationIssue("blocker", "Syntax error in backend/main_api.py line 3: invalid syntax")
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path), auto_fix_issues=True)
    restored = _run_phase3_with(orch, [[_req(1)], [_req(1), syntax]], monkeypatch)
    assert restored is True


def test_the_verdicts_are_written_to_the_recipe(simple_model, tmp_path):
    orch = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=simple_model, output_dir=str(tmp_path),
    )
    orch._requirement_verdicts = [{"id": 1, "text": "x", "status": "missing",
                                   "evidence": "", "note": ""}]
    orch._save_recipe("build it", elapsed=1.0)
    import json
    recipe = json.load(open(tmp_path / ".besser_recipe.json", encoding="utf-8"))
    assert recipe["requirements"][0]["status"] == "missing"
