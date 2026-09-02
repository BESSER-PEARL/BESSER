"""Tests for the model-derived data-contract checks.

Covers the three consumers of ``contract_checks``:

1. the pure lint (``build_data_contract`` / ``lint_file``),
2. the per-write feedback in ``ToolExecutor`` (violations appear in the
   write_file/modify_file result the model reads next turn),
3. the Phase-3 sweep (``_collect_data_contract_issues``) and its
   blocker/advisory classification,
4. the prompt section (``## Data contract``) in ``build_system_prompt``.

Precision matters more than recall (a false blocker burns billable
auto-fix turns), so the no-false-positive cases are pinned as tightly
as the true positives.
"""

import json
import types

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.contract_checks import (
    build_data_contract,
    lint_file,
)
from besser.generators.llm.orchestrator import LLMOrchestrator, _classify_issue
from besser.generators.llm.prompt_builder import build_system_prompt
from besser.generators.llm.tool_executor import ToolExecutor

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")


def _string_id_model() -> DomainModel:
    reservation = Class(name="Reservation")
    reservation.attributes = {
        Property(name="id", type=StringType, is_id=True),
        Property(name="date", type=StringType),
    }
    guest = Class(name="Guest")
    guest.attributes = {
        Property(name="id", type=StringType, is_id=True),
    }
    return DomainModel(name="Hotel", types={reservation, guest})


def _mixed_id_model() -> DomainModel:
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=StringType, is_id=True)}
    item = Class(name="Item")
    item.attributes = {Property(name="id", type=IntegerType, is_id=True)}
    return DomainModel(name="Shop", types={order, item})


# --------------------------------------------------------------------------- #
# Contract extraction
# --------------------------------------------------------------------------- #
def test_contract_from_none_model_is_none():
    assert build_data_contract(None) is None


def test_contract_extracts_string_ids():
    contract = build_data_contract(_string_id_model())
    assert contract.string_id_classes == ["Guest", "Reservation"]
    assert contract.pk_types["Reservation"] == ("id", "str")
    assert not contract.has_int_ids


def test_contract_mixed_ids():
    contract = build_data_contract(_mixed_id_model())
    assert contract.string_id_classes == ["Order"]
    assert contract.has_int_ids


# --------------------------------------------------------------------------- #
# Frontend lint
# --------------------------------------------------------------------------- #
def test_parseint_on_id_is_blocker_when_all_ids_are_strings():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "frontend/src/pages/Reservations.tsx",
        "const rid = parseInt(reservationId);\nconst n = Number(params.id);\n",
        contract,
    )
    assert len(findings) == 2
    assert all(f.blocker for f in findings)


def test_parseint_on_id_is_advisory_when_model_also_has_int_ids():
    contract = build_data_contract(_mixed_id_model())
    findings = lint_file(
        "frontend/src/api.ts", "const x = parseInt(itemId);", contract,
    )
    assert len(findings) == 1
    assert not findings[0].blocker


def test_parseint_on_non_id_is_not_flagged():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "frontend/src/api.ts",
        "const total = parseInt(countStr); const p = Number(price);",
        contract,
    )
    assert findings == []


def test_post_with_id_in_payload_is_advisory():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "frontend/src/api.ts",
        "await api.post('/reservations', { id: newId, date });",
        contract,
    )
    assert len(findings) == 1
    assert not findings[0].blocker


# --------------------------------------------------------------------------- #
# Python lint
# --------------------------------------------------------------------------- #
def test_fake_executed_is_blocker():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "backend/routers/reservation.py",
        'return {"status": "executed"}\n',
        contract,
    )
    assert len(findings) == 1
    assert findings[0].blocker
    assert "501" in findings[0].message


def test_create_schema_with_server_owned_fields_is_blocker():
    contract = build_data_contract(_string_id_model())
    content = (
        "class ReservationCreate(BaseModel):\n"
        "    id: str\n"
        "    created_at: datetime\n"
        "    date: str\n"
    )
    findings = lint_file("backend/pydantic_classes.py", content, contract)
    flagged = sorted(f.message.split("`")[1] for f in findings)
    assert flagged == ["created_at", "id"]
    assert all(f.blocker for f in findings)


def test_create_schema_with_natural_key_is_not_flagged():
    contract = build_data_contract(_string_id_model())
    content = (
        "class BookCreate(BaseModel):\n"
        "    isbn: str\n"
        "    title: str\n"
    )
    assert lint_file("backend/pydantic_classes.py", content, contract) == []


def test_int_typed_fk_param_for_string_id_class_is_blocker():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "backend/sql_alchemy.py",
        'reservation_id: int\n'
        'guest_id = Column(Integer, ForeignKey("guest.id"))\n',
        contract,
    )
    assert len(findings) == 2
    assert all(f.blocker for f in findings)


def test_string_fk_column_is_not_flagged():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "backend/sql_alchemy.py",
        'guest_id = Column(String, ForeignKey("guest.id"))\n',
        contract,
    )
    assert findings == []


def test_id_int_in_entity_named_file_is_blocker():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "backend/routers/reservation.py",
        'def get_reservation(id: int):\n    pass\n',
        contract,
    )
    assert len(findings) == 1
    assert findings[0].blocker


def test_id_int_in_unrelated_file_is_not_flagged():
    contract = build_data_contract(_string_id_model())
    findings = lint_file(
        "backend/utils.py", "def helper(id: int):\n    pass\n", contract,
    )
    assert findings == []


# --------------------------------------------------------------------------- #
# Per-write feedback in ToolExecutor
# --------------------------------------------------------------------------- #
def test_write_file_result_carries_contract_warnings(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=_string_id_model())
    result = json.loads(executor.execute("write_file", {
        "path": "frontend/src/api.ts",
        "content": "export const load = (rid) => parseInt(rid.id);",
    }))
    assert result["status"] == "written"
    assert "DATA-CONTRACT VIOLATIONS" in result["contract_warnings"]


def test_clean_write_has_no_warnings(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=_string_id_model())
    result = json.loads(executor.execute("write_file", {
        "path": "frontend/src/api.ts",
        "content": "export const ok = true;",
    }))
    assert "contract_warnings" not in result


def test_modify_file_result_carries_contract_warnings(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=_string_id_model())
    (tmp_path / "api.ts").write_text("const x = 1;", encoding="utf-8")
    result = json.loads(executor.execute("modify_file", {
        "path": "api.ts",
        "old_text": "const x = 1;",
        "new_text": "const x = parseInt(reservationId);",
    }))
    assert result["status"] == "modified"
    assert "parseInt" in result["contract_warnings"]


# --------------------------------------------------------------------------- #
# Phase-3 sweep + classification
# --------------------------------------------------------------------------- #
def test_phase3_sweep_finds_and_classifies(tmp_path):
    (tmp_path / "backend" / "routers").mkdir(parents=True)
    (tmp_path / "backend" / "routers" / "reservation.py").write_text(
        'def run(id: int):\n    return {"status": "executed"}\n',
        encoding="utf-8",
    )
    shim = types.SimpleNamespace(
        output_dir=str(tmp_path), domain_model=_string_id_model(),
    )
    issues = LLMOrchestrator._collect_data_contract_issues(shim)
    assert len(issues) == 2
    assert all(i.startswith("data contract:") for i in issues)
    assert all(_classify_issue(i).severity == "blocker" for i in issues)


def test_advisory_findings_classify_as_warning():
    issue = _classify_issue(
        "data contract (advisory): frontend/src/api.ts line 3: create "
        "request appears to send an `id`"
    )
    assert issue.severity == "warning"


def test_sweep_without_domain_model_is_empty(tmp_path):
    shim = types.SimpleNamespace(output_dir=str(tmp_path), domain_model=None)
    assert LLMOrchestrator._collect_data_contract_issues(shim) == []


# --------------------------------------------------------------------------- #
# Prompt section
# --------------------------------------------------------------------------- #
def test_prompt_contains_data_contract_section():
    prompt = build_system_prompt(
        domain_model=_string_id_model(),
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        max_turns=20,
    )
    assert "## Data contract" in prompt
    assert "`Reservation.id`: **str** (TypeScript: `string`)" in prompt
    assert "HTTP 501" in prompt


def test_prompt_has_no_contract_section_without_domain_model():
    prompt = build_system_prompt(
        domain_model=None,
        gui_model=None,
        agent_model=None,
        inventory="",
        instructions="Build the app",
        max_turns=20,
    )
    assert "## Data contract" not in prompt


def test_executed_with_real_impl_in_file_is_not_flagged():
    """The generated method endpoint answers "executed" after actually
    running the modeled body via a _impl function — that's honest."""
    contract = build_data_contract(_string_id_model())
    content = (
        "def _check_in_impl(database):\n"
        "    return 1\n"
        "result = _check_in_impl(database)\n"
        'return {"status": "executed", "result": result}\n'
    )
    assert lint_file("backend/routers/reservation.py", content, contract) == []


def test_string_pk_scaffold_mints_ids_and_types_fks(tmp_path):
    """Audit defects #2/#3 (2026-09-02): a string-PK scaffold must mint the
    id server-side (uuid default) and type relationship fields str."""
    import json as _json
    guest = Class(name="Guest")
    guest.attributes = {
        Property(name="id", type=StringType, is_id=True),
        Property(name="name", type=StringType),
    }
    reservation = Class(name="Reservation")
    reservation.attributes = {
        Property(name="id", type=StringType, is_id=True),
        Property(name="date", type=StringType),
    }
    from besser.BUML.metamodel.structural import BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY
    assoc = BinaryAssociation(name="Guest_Reservation", ends={
        Property(name="guest", type=guest, multiplicity=Multiplicity(1, 1)),
        Property(name="reservations", type=reservation,
                 multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
    })
    model = DomainModel(name="Hotel", types={guest, reservation}, associations={assoc})
    executor = ToolExecutor(workspace=str(tmp_path), domain_model=model)
    result = _json.loads(executor.execute("generate_fastapi_backend", {}))
    assert result["status"] == "ok"

    sqla = (tmp_path / "backend" / "sql_alchemy.py").read_text(encoding="utf-8")
    assert "default=_new_str_id" in sqla          # server mints string ids
    assert "def _new_str_id" in sqla
    pyd = (tmp_path / "backend" / "pydantic_classes.py").read_text(encoding="utf-8")
    assert "guest: str" in pyd                     # FK field typed str, not int
    assert "reservations: Optional[List[str]]" in pyd
