"""Reads on items of a ``list[XCreate]`` payload were invisible.

``bulk_create_bill(items: list[BillCreate])`` iterates
``for idx, item_data in enumerate(items)`` and reads ``item_data.<field>``.
The annotation is a subscript, so no schema was resolved, and the ``_data``
suffix guessed ``ItemCreate``, which does not exist. Seven recorded
gpt-5.6-terra runs (61nry6wd, dp3trml9, jffm1tid, jowqmmxr, kau7c07h,
h60ln0ky, vf6rtqk9) shipped a bulk endpoint that fails on every request with
0 blockers: the model removed server-owned fields from the Create schema and
the generated bulk handler kept reading them.
"""

import json
import shutil
from pathlib import Path

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.validation.python_source import (
    _create_schema_router_mismatches,
)


FIXTURE = Path(__file__).parent / "fixtures" / "run_61nry6wd_bulk_create"

_SCHEMAS = (
    "from pydantic import BaseModel\n"
    "\n"
    "class BillCreate(BaseModel):\n"
    "    billNumber: str\n"
    "    booking: int\n"
)


def _app(tmp_path, handler: str):
    (tmp_path / "backend" / "routers").mkdir(parents=True)
    (tmp_path / "backend" / "pydantic_classes.py").write_text(_SCHEMAS, encoding="utf-8")
    (tmp_path / "backend" / "routers" / "bill.py").write_text(handler, encoding="utf-8")
    return _create_schema_router_mismatches(str(tmp_path))


def test_the_recorded_61nry6wd_bulk_create_is_caught():
    issues = _create_schema_router_mismatches(str(FIXTURE))

    missing = {"totalAmountDue", "settled", "issuedDate"}
    assert len(issues) == len(missing), issues
    for field in missing:
        assert any(f"`item_data.{field}`" in issue
                   and f"BillCreate does not define `{field}`" in issue
                   and "web_app/backend/routers/bill.py" in issue
                   for issue in issues), (field, issues)


def test_the_schema_write_reports_the_bulk_consumer_same_turn(tmp_path):
    """Per-write diagnostics use the same check, so the model hears it at write time."""
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    schemas = tmp_path / "web_app" / "backend" / "pydantic_classes.py"
    content = schemas.read_text(encoding="utf-8")
    schemas.unlink()
    executor = ToolExecutor(workspace=str(tmp_path))

    result = json.loads(executor.execute("write_file", {
        "path": "web_app/backend/pydantic_classes.py", "content": content,
    }))

    mismatches = [item for item in result.get("diagnostics", [])
                  if item["code"] == "schema-consumer-mismatch"]
    assert {m["message"].split("`")[1] for m in mismatches} == {
        "item_data.totalAmountDue", "item_data.settled", "item_data.issuedDate",
    }, result


def test_a_plain_for_loop_over_the_payload_is_caught(tmp_path):
    issues = _app(tmp_path, (
        "async def bulk_create_bill(items: List[BillCreate]):\n"
        "    for item in items:\n"
        "        Bill(total=item.totalAmountDue)\n"
    ))

    assert len(issues) == 1, issues
    assert "`item.totalAmountDue`" in issues[0]


def test_an_optional_payload_is_resolved(tmp_path):
    issues = _app(tmp_path, (
        "async def create_bill(payload: Optional[BillCreate] = None):\n"
        "    return Bill(total=payload.totalAmountDue)\n"
    ))

    assert len(issues) == 1, issues
    assert "`payload.totalAmountDue`" in issues[0]


def test_a_correct_bulk_handler_is_clean(tmp_path):
    assert _app(tmp_path, (
        "async def bulk_create_bill(items: list[BillCreate]):\n"
        "    for idx, item_data in enumerate(items):\n"
        "        Bill(billNumber=item_data.billNumber, booking_id=item_data.booking)\n"
    )) == []


def test_a_guarded_read_on_the_loop_item_is_skipped(tmp_path):
    assert _app(tmp_path, (
        "async def bulk_create_bill(items: list[BillCreate]):\n"
        "    for idx, item_data in enumerate(items):\n"
        "        total = item_data.totalAmountDue"
        " if hasattr(item_data, 'totalAmountDue') else 0.0\n"
        "        Bill(total=total, settled=getattr(item_data, 'settled', False))\n"
    )) == []


def test_a_loop_over_a_non_schema_list_is_ignored(tmp_path):
    assert _app(tmp_path, (
        "async def bulk_delete_bill(ids: list[int], rows: list[Bill]):\n"
        "    for item in ids:\n"
        "        item.bit_length()\n"
        "    for idx, row in enumerate(rows):\n"
        "        row.totalAmountDue\n"
    )) == []


def test_the_element_type_is_bound_to_its_own_loop_only(tmp_path):
    """The same name iterating something else is not a BillCreate."""
    assert _app(tmp_path, (
        "async def bulk_create_bill(items: list[BillCreate], database):\n"
        "    for item in items:\n"
        "        Bill(billNumber=item.billNumber)\n"
        "    for item in database.query(Bill).all():\n"
        "        item.totalAmountDue\n"
    )) == []
