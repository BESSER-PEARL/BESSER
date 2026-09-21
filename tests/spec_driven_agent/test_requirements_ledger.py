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
from besser.spec_driven_agent import requirements_ledger as ledger
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator, _classify_issue

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
    # Long accepted specifications retain their last clause in extraction too.
    from besser.spec_driven_agent.specification import MAX_SPECIFICATION_CHARS
    client = _ToolClient([{"requirements": [{"text": "Never charge twice", "kind": "rule"}]}])
    long_spec = "x" * (MAX_SPECIFICATION_CHARS - len(SPEC)) + SPEC
    assert ledger.extract_requirements(long_spec, client)
    assert client.calls[0][1].endswith(long_spec)
    with pytest.raises(ValueError, match="not truncated"):
        ledger.extract_requirements(long_spec + "!", client)
    assert len(client.calls) == 1, "oversize requests must not make a provider call"


def test_a_mock_client_is_never_called():
    assert ledger.extract_requirements(INSTRUCTIONS, _MockClient()) is None


def test_extraction_keeps_the_billing_tail_and_saturation_cannot_look_complete(tmp_path):
    requirements = [{"text": f"Rule {i}", "kind": "rule"} for i in range(40)]
    requirements.append({"text": "Settling a bill confirms its booking", "kind": "transition"})
    client = _ToolClient([{"requirements": requirements}])
    extracted = ledger.extract_requirements(INSTRUCTIONS, client)
    assert len(extracted) == 41
    assert extracted[-1]["text"] == requirements[-1]["text"]

    # A noncompliant provider returning over maxItems must not lose any item.
    for count in (ledger._MAX_REQUIREMENTS, ledger._MAX_REQUIREMENTS + 1):
        response = [{"text": f"Rule {i}", "kind": "rule"} for i in range(count)]
        extracted = ledger.extract_requirements(INSTRUCTIONS, _ToolClient([{"requirements": response}]))
        assert len(extracted) == count + 1
        assert extracted[-2]["text"] == response[-1]["text"]
        gate = extracted[-1]
        assert gate["kind"] == "verification"
        client = _ToolClient([{"verdicts": [
            {"id": gate["id"], "status": "implemented", "evidence": "app.py: return True"},
        ]}])
        judged = ledger.judge_coverage(extracted, "", client)
        assert judged[-1]["status"] == "unverified"
        assert gate["text"] not in client.calls[0][1], "code judgment cannot repair incomplete extraction"
        checked = ledger.verify_evidence([dict(gate, status="implemented", evidence="app.py: return True")], str(tmp_path))
        assert checked[0]["status"] == "unverified"
        assert _classify_issue(ledger.ledger_issues(checked)[0]).severity == "blocker"
        assert ledger.judge_coverage([gate], "", _ToolClient([]))[0]["status"] == "unverified"
    assert ledger.extract_requirements(INSTRUCTIONS, _ToolClient([{"requirements": {"text": "bad shape"}}])) is None


# ------------------------------------------------------------------ digest


def test_the_digest_reads_schemas_then_routers_and_skips_the_stdlib(tmp_path, monkeypatch):
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
    _write(tmp_path, "backend/routers/booking.py", "x = 1\n" * 200 + "def tail_rule(): return True\n")
    _write(tmp_path, "backend/models.py", "y = 2\n" * 200)
    monkeypatch.setattr(ledger, "_DIGEST_MAX_FILE_CHARS", 100)
    ordinary = ledger.build_app_digest(str(tmp_path))
    assert "DIGEST TRUNCATED: backend/routers/booking.py" in ordinary
    assert "def tail_rule" not in ordinary
    focused = ledger.build_app_digest(str(tmp_path), focus_paths=["routers/booking.py"])
    assert focused.startswith("### backend/routers/booking.py")
    assert "def tail_rule(): return True" in focused
    assert len(focused) <= ledger._DIGEST_MAX_TOTAL_CHARS
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", 200)
    bounded = ledger.build_app_digest(str(tmp_path))
    assert len(bounded) <= 200
    assert "omitted/unreadable files: 2" in bounded


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


def test_a_rule_backed_only_by_a_declaration_is_unverified():
    """The live judge marked 'guests never exceed the rooms' capacity'
    implemented by citing ``Booking.reservedRooms``. A relationship existing
    enforces nothing. That citation cannot prove either implementation or
    absence of an implementation elsewhere."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}", kind="rule")],
        FIXTURE,
    )
    assert out[0]["status"] == "unverified"
    assert "declares data" in out[0]["note"]


def test_an_action_cannot_be_proven_by_a_relationship_declaration():
    out = ledger.verify_evidence(
        [_verdict("implemented", f"web_app/backend/sql_alchemy.py: {_DECLARING_LINE}", kind="action")],
        FIXTURE,
    )
    assert out[0]["status"] == "unverified"


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
    assert out[0]["status"] == "unverified"


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
    out = ledger.verify_evidence([
        _verdict("implemented", quoted, kind="rule", text="Each booking has an employee"),
    ], FIXTURE)
    assert out[0]["status"] == "implemented"


def test_a_verbatim_line_needs_a_real_citation_path():
    """Live 2026-09-18: the judge wrote the literal word 'path' as the file
    for all 40 citations. Searching all files then accepted the run's own
    recipe as evidence. A bad citation must be corrected, not rescued."""
    out = ledger.verify_evidence(
        [_verdict("implemented", f"path: {_ENFORCING_LINE}")], FIXTURE,
    )
    assert out[0]["status"] == "unverified"


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


def test_evidence_stays_inside_unambiguous_application_source(tmp_path):
    _write(tmp_path, "backend/app.py", "def pay():\n    return True\n")
    _write(tmp_path, "other/app.py", "def pay():\n    return True\n")
    artifacts = (".besser_recipe.json", "logs/app.py", "reports/app.py", "tests/app.py",
                 "frontend/src/App.test.tsx", "frontend/src/pages/Booking.spec.js", "backend/test_api.py")
    for artifact in artifacts:
        _write(tmp_path, artifact, "def pay():\n    return True\n")
    _write(tmp_path, "web_app/frontend/src/components/PaymentButton.tsx",
           "export const PaymentButton = () => <button>Pay</button>;\n")
    # A reports/logs application module is not the workspace's artifact folder.
    # Windows-authored Python with a UTF-8 BOM remains valid source evidence.
    _write(tmp_path, "backend/reports/service.py", "\ufeffdef report():\n    return True\n")
    _write(tmp_path, "backend/logs/events.py", "def record_event():\n    return True\n")
    citations = ["missing.py", "app.py", "../backend/app.py", "/backend/app.py", *artifacts]
    verdicts = [_verdict("implemented", f"{p}: return True", kind="action") for p in citations]
    assert {v["status"] for v in ledger.verify_evidence(verdicts, str(tmp_path))} == {"unverified"}
    digest = ledger.build_app_digest(str(tmp_path))
    assert "components/PaymentButton.tsx" in digest
    assert not any(f"### {p}" in digest for p in citations)
    app_modules = ["backend/reports/service.py", "backend/logs/events.py"]
    assert all(f"### {p}" in digest for p in app_modules)
    assert [v["status"] for v in ledger.verify_evidence([
        _verdict("implemented", f"{p}: return True", kind="action") for p in app_modules
    ], str(tmp_path))] == ["implemented", "implemented"]
    # An incidental identifier substring is not a copied executable line.
    assert ledger.verify_evidence([
        _verdict("implemented", "backend/app.py: pay", kind="action"),
    ], str(tmp_path))[0]["status"] == "unverified"


def test_scaffolds_and_prose_do_not_prove_behaviour_but_real_code_can(tmp_path):
    _write(tmp_path, "app.py", '''from fastapi import HTTPException
class Status:
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
class Booking:
    totalPrice: float
    physicalStatus: Status
    status = mapped_column(default="AWAITING_PAYMENT")
    amount = mapped_column(Computed("price * nights"))
def scaffold():
    """if too_many_guests: raise ValueError"""
    # if too_many_guests: raise ValueError
    booking = load_booking()
    try:
        raise HTTPException(status_code=501, detail="not implemented")
    except HTTPException:
        raise
def pay():
    if unsupported:
        raise HTTPException(status_code=501, detail="unsupported payment method")
    booking.paid = True
    return True
def validate():
    if guest_count > capacity:
        raise ValueError("Too many guests")
def calculate():
    return sum(room.price for room in rooms)
def create_link():
    return Link(
        **link_attrs(link, ('agreedPrice', 'extraCharges'))
    )
def whole_nights():
    return (
        hours
        // 24
    )

class BookingCreate:
    @model_validator(mode="after")
    def validate_dates(self):
        if self.arrivalDate > self.departureDate:
            raise ValueError("Invalid dates")
        return self

    @model_validator(mode="after")
    def validate_capacity(self):
        """Capacity is enforced elsewhere."""
        return self

    @field_validator("phone")
    def validate_phone(cls, value):
        return value

    @field_validator("email")
    def validate_email(cls, value):
        return validate_email_format(value)
''')
    invalid = [
        ("action", "def scaffold():"),
        ("action", 'raise HTTPException(status_code=501, detail="not implemented")'),
        ("transition", 'AWAITING_PAYMENT = "AWAITING_PAYMENT"'),
        ("computed", "totalPrice: float"),
        ("rule", '# if too_many_guests: raise ValueError'),
        ("rule", '"""if too_many_guests: raise ValueError"""'),
        ("rule", 'def validate_capacity(self):'),
        ("rule", '@model_validator(mode="after")\n    def validate_capacity(self):'),
        ("rule", '@model_validator(mode="after")'),
        ("validation", 'def validate_phone(cls, value):'),
    ]
    valid = [("action", "def pay():"), ("transition", "booking.paid = True"),
             ("action", "**link_attrs(link, ('agreedPrice', 'extraCharges'))"),
             ("computed", "// 24"),
             ("rule", "if guest_count > capacity:"),
             ("computed", "return sum(room.price for room in rooms)"),
             ("rule", '@model_validator(mode="after")\n    def validate_dates(self):'),
             ("validation", 'def validate_email(cls, value):')]
    verdicts = [_verdict("implemented", f"app.py: {line}", kind=kind)
                for kind, line in invalid + valid]
    statuses = [v["status"] for v in ledger.verify_evidence(verdicts, str(tmp_path))]
    assert statuses == ["unverified"] * len(invalid) + ["implemented"] * len(valid)
    declarative = [
        _verdict("implemented", 'app.py: status = mapped_column(default="AWAITING_PAYMENT")',
                 kind="transition", text="A booking starts awaiting payment"),
        _verdict("implemented", 'app.py: amount = mapped_column(Computed("price * nights"))',
                 kind="computed", text="The amount is computed from the price and nights"),
    ]
    assert [v["status"] for v in ledger.verify_evidence(declarative, str(tmp_path))] == [
        "implemented", "implemented",
    ]
    _write(tmp_path, "app.js", '''/* implementation example:
    payEverything();
 * still only a comment
 */
class Items {
    *values() { yield 1; }
}
let remaining = 2;
--remaining;
// ignored();
''')
    js_quotes = ["payEverything();", "* still only a comment", "// ignored();",
                 "*values() { yield 1; }", "--remaining;"]
    assert [v["status"] for v in ledger.verify_evidence([
        _verdict("implemented", f"app.js: {quote}", kind="action") for quote in js_quotes
    ], str(tmp_path))] == ["unverified"] * 3 + ["implemented"] * 2


def test_invalid_declarations_do_not_prove_rules_but_valid_constraints_still_do(tmp_path):
    _write(tmp_path, "database.py", 'DATABASE_URL = "sqlite:///hotel.db"\n')
    _write(tmp_path, "models.py", '''from enum import Enum
class Status(Enum):
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
class Booking:
    __tablename__ = "booking"
    id = mapped_column(Integer, primary_key=True)
    arrival = mapped_column(Date)
    departure = mapped_column(Date)
    status = mapped_column(default=Status.awaiting_payment)
    multiline_status = mapped_column(
        default=Status.not_a_member,
    )
    __table_args__ = (CheckConstraint("arrival <= departure", name="shadowed_date_rule"),)
    __table_args__ = ()
    @field_validator("email")
    def validate_contact(cls, value):
        raise ValueError("shadowed email validator")
    @field_validator("phone")
    def validate_contact(cls, value):
        return validate_phone_format(value)
class InvalidRoom:
    __tablename__ = "invalid_room"
    capacity = mapped_column(Integer)
    __table_args__ = (
        CheckConstraint("agreedPrice > 0", name="wrong_table"),
        CheckConstraint("capacity >= (SELECT COUNT(*) FROM guests)", name="subquery"),
    )
class Room:
    __tablename__ = "room"
    capacity = mapped_column(Integer)
    status = mapped_column(default=Status.AWAITING_PAYMENT)
    __table_args__ = (CheckConstraint("capacity > 0", name="positive_capacity"),)
''')
    invalid = [
        ("transition", "status = mapped_column(default=Status.awaiting_payment)"),
        ("transition", "multiline_status = mapped_column("),
        ("rule", '__table_args__ = (CheckConstraint("arrival <= departure", name="shadowed_date_rule"),)'),
        ("validation", 'raise ValueError("shadowed email validator")'),
        ("rule", 'CheckConstraint("agreedPrice > 0", name="wrong_table"),'),
        ("rule", 'CheckConstraint("capacity >= (SELECT COUNT(*) FROM guests)", name="subquery"),'),
    ]
    valid = [
        ("transition", "status = mapped_column(default=Status.AWAITING_PAYMENT)"),
        ("rule", '__table_args__ = (CheckConstraint("capacity > 0", name="positive_capacity"),)'),
    ]
    verdicts = [_verdict("implemented", f"models.py: {line}", kind=kind,
                         text="A record starts awaiting payment" if kind == "transition" else "The value must be valid")
                for kind, line in invalid + valid]
    checked = ledger.verify_evidence(verdicts, str(tmp_path))
    assert [v["status"] for v in checked] == ["unverified"] * len(invalid) + ["implemented"] * len(valid)
    assert all("invalid" in v["note"] for v in checked[:len(invalid)])


# ------------------------------------------------------------------- judge


def test_a_requirement_the_judge_skipped_is_unverified_not_missing():
    reqs = [{"id": 1, "text": "a", "kind": ""}, {"id": 2, "text": "b", "kind": ""}]
    client = _ToolClient([{"verdicts": [
        {"id": 1, "status": "implemented", "evidence": "x.py",
         "inspection_paths": ["frontend/Table.tsx", 123]},
    ]}])
    verdicts = ledger.judge_coverage(
        reqs, "### x.py\ncode", client, original_spec=INSTRUCTIONS,
        previous_verdicts=[{"id": 1, "evidence": "x.py:@router.post('/')",
                            "note": "decorator does not implement the action"},
                           {"id": 999, "note": "unrelated-requirement-secret"}],
    )
    assert [v["status"] for v in verdicts] == ["implemented", "unverified"]
    assert client.calls[0][0] == "submit_verdicts"
    assert "R2. b" in client.calls[0][1]
    assert SPEC in client.calls[0][1]
    assert "This is the authority" not in client.calls[0][1]
    assert "decorator does not implement the action" in client.calls[0][1]
    assert "x.py:@router.post('/')" in client.calls[0][1]
    assert "unrelated-requirement-secret" not in client.calls[0][1]
    assert verdicts[0]["inspection_paths"] == ["frontend/Table.tsx"]
    assert "unverified" in ledger._SUBMIT_VERDICTS_TOOL["input_schema"]["properties"]["verdicts"]["items"]["properties"]["status"]["enum"]


# --------------------------------------------------------- issues + severity


def test_unresolved_requirements_block_completion_without_inventing_missing_code():
    issues = ledger.ledger_issues([
        {"id": 1, "text": "Room numbers are unique", "status": "missing", "note": "no unique constraint"},
        {"id": 2, "text": "Bills can be paid", "status": "partial", "note": "no refusal when paid"},
        {"id": 3, "text": "Bookings can be cancelled", "status": "unverified", "evidence": "x.py:cancel"},
        {"id": 4, "text": "Bookings are created", "status": "implemented", "evidence": "y.py"},
    ])
    assert len(issues) == 3
    assert issues[0].startswith("requirement: R1 — Room numbers are unique is not implemented")
    assert _classify_issue(issues[0]).severity == "blocker"
    assert _classify_issue(issues[1]).severity == "blocker"
    assert _classify_issue(issues[2]).severity == "blocker"
    assert issues[1].startswith("requirement partial:")
    assert issues[2].startswith("requirement unverified:")
    assert "implementing behaviour only if it is absent" in issues[2]


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
    _write(tmp_path, "backend/routers/room.py", "def create_room():\n    return True\n")
    monkeypatch.setattr(ledger, "extract_requirements", lambda instr, client: [
        {"id": 1, "text": "Rooms can be created", "kind": "action"},
        {"id": 2, "text": "Room numbers are unique", "kind": "uniqueness"},
    ])
    monkeypatch.setattr(ledger, "judge_coverage", lambda reqs, digest, client, **kwargs: [
        {"id": 1, "text": "Rooms can be created", "status": "implemented",
         "evidence": "backend/routers/room.py:return True", "note": ""},
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
    from besser.spec_driven_agent.validation.issues import is_completion_issue

    monkeypatch.setattr(ledger, "extract_requirements",
                        lambda *_: (_ for _ in ()).throw(AssertionError("must not run")))
    orch = LLMOrchestrator(
        llm_client=_MockClient(), domain_model=simple_model, output_dir=str(tmp_path),
        enable_requirements_ledger=False,
    )
    orch._instructions = INSTRUCTIONS
    issues = orch._collect_validation_issues()
    assert not [i for i in issues if i.message.startswith("requirement")]
    findings = [i for i in issues if i.message.startswith("validation unverified: original-specification coverage")]
    assert len(findings) == 1 and findings[0].severity == "warning" and is_completion_issue(findings[0])
    orch.auto_fix_issues = True
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: findings)
    monkeypatch.setattr(orch.client, "chat", lambda *a, **kw: pytest.fail("disabled checking must not request code repairs"))
    orch._run_phase3_validation()
    assert orch._validation_issues == findings
    orch._instructions = ""
    assert orch._collect_requirement_issues() == []


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
    from besser.spec_driven_agent.orchestrator import ValidationIssue
    return ValidationIssue("blocker", f"requirement: R{n} — rule {n} is not implemented")


def test_a_flapping_requirement_verdict_never_rolls_back_a_fix(simple_model, tmp_path, monkeypatch):
    """Two judge calls on the same 19h35 app returned 12 and 22 missing
    requirements. A verdict that appears between passes is judge variance,
    not a regression the fix caused; rolling back would discard real work."""
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path), auto_fix_issues=True)
    restored = _run_phase3_with(orch, [[_req(1)], [_req(1), _req(2)]], monkeypatch)
    assert restored is False


def test_a_new_hard_blocker_is_reported_without_count_based_rollback(simple_model, tmp_path, monkeypatch):
    from besser.spec_driven_agent.orchestrator import ValidationIssue
    syntax = ValidationIssue("blocker", "Syntax error in backend/main_api.py line 3: invalid syntax")
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path), auto_fix_issues=True)
    restored = _run_phase3_with(orch, [[_req(1)], [_req(1), syntax]], monkeypatch)
    assert restored is False
    assert orch._validation_issues == [_req(1), syntax]


# ------------------------------------------- the ledger reaches Phase 2 too


_LEDGER = [
    {"id": 1, "text": "Room numbers are unique", "kind": "uniqueness"},
    {"id": 2, "text": "Guests never exceed the rooms' combined capacity", "kind": "rule"},
]


def test_requirements_render_one_numbered_line_each():
    assert ledger.render_requirements(_LEDGER) == (
        "R1. Room numbers are unique\n"
        "R2. Guests never exceed the rooms' combined capacity"
    )
    assert ledger.render_requirements(None) == ""


def test_the_planner_is_handed_the_ledger_under_the_request(simple_model, tmp_path, monkeypatch):
    """The gap analyser reads the spec too, but on the 19h35 model it still
    skipped the unique room number and the extra charges. Numbered
    requirements are a list to diff against, not prose to skim."""
    seen = {}

    def fake_planner(**kwargs):
        seen["instructions"] = kwargs["instructions"]
        return []

    monkeypatch.setattr(ledger, "extract_requirements", lambda instr, client: list(_LEDGER))
    monkeypatch.setattr("besser.spec_driven_agent.orchestrator.analyze_gaps_via_llm", fake_planner)
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path))
    orch._generator_used = "generate_web_app"
    planner_text = orch._planner_instructions(INSTRUCTIONS)
    assert planner_text.startswith(INSTRUCTIONS), "the verbatim request stays first"
    assert "R1. Room numbers are unique" in planner_text
    assert orch._requirements == _LEDGER


def test_the_phase2_prompt_names_the_requirements_it_will_be_judged_on(simple_model, tmp_path):
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path))
    orch._requirements = list(_LEDGER)
    prompt = orch._build_system_prompt(instructions=INSTRUCTIONS)
    assert "R2. Guests never exceed the rooms' combined capacity" in prompt
    assert "verified" in prompt.split("R1. Room numbers are unique")[0][-600:].lower()


def test_without_a_ledger_the_prompt_is_unchanged(simple_model, tmp_path):
    orch = LLMOrchestrator(llm_client=_MockClient(), domain_model=simple_model,
                           output_dir=str(tmp_path))
    prompt = orch._build_system_prompt(instructions=INSTRUCTIONS)
    assert "Requirements the user stated" not in prompt


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
