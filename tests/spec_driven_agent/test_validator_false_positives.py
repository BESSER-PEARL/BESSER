"""False blockers reproduced against apps that demonstrably work.

Every case here was replayed over the 74 delivered trees in
``verification/spec-iterations`` whose ``workflow_ok`` is true in
``verification/batch_results.jsonl`` - the app boots, creates every entity
and runs the workflow. A validator that reports a ``blocker`` on one of
those is wrong by construction, and the run pays for it: a false blocker
buys fix turns and can drive a rollback of a genuine repair.

Each false positive is paired with the true positive it must keep
catching, taken from the same corpus wherever one exists.
"""

from __future__ import annotations

import re
import shutil

import pytest

from besser.spec_driven_agent.validation.acceptance import (
    _post_helpers, build_acceptance_matrix, matrix_issues)
from besser.spec_driven_agent.planning.action_inventory import (
    action_body_present, action_implementation_issues, collect_action_endpoints)
from besser.spec_driven_agent.validation.endpoint_coherence import collect_endpoint_coherence_issues
from besser.spec_driven_agent.planning.fix_target import finding_matches_target, parse_reported_target
from besser.spec_driven_agent.validation.frontend_bindings import literal_component_props
from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence
from besser.spec_driven_agent.validation.frontend_contract import _method_button_source_issues
from besser.spec_driven_agent.validation.frontend_schema import collect_frontend_schema_issues
from besser.spec_driven_agent.validation.issues import _classify_issue
from besser.spec_driven_agent.validation.python_source import _create_schema_router_mismatches
from besser.spec_driven_agent.validation.write_diagnostics import diagnose_written_content


def _write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# validation/python_source.py - `<entity>_data.<field>` reads
# ---------------------------------------------------------------------------

_BOOKING_CREATE = (
    "from pydantic import BaseModel\n"
    "\n"
    "class BookingCreate(BaseModel):\n"
    "    bookingNumber: str\n"
    "    arrivalDate: date\n"
    "    departureDate: date\n"
)


def test_a_hasattr_guarded_payload_read_is_not_a_500_blocker(tmp_path):
    """Live tree ``...-d71pocck`` (hotel, 11/11 workflow checks passed).

    ``booking_data.id`` sits in the true branch of a conditional whose test
    is ``hasattr(booking_data, 'id')``. ``BookingCreate`` has no ``id``, so
    the guard is False and the attribute is never evaluated - the endpoint
    creates bookings on every request. Claiming it "returns 500 on every
    request" is a false blocker on a shipped, working app.
    """
    _write(tmp_path, "web_app/backend/pydantic_classes.py", _BOOKING_CREATE)
    _write(tmp_path, "web_app/backend/routers/booking.py", (
        "from pydantic_classes import BookingCreate\n"
        "\n"
        "@router.post('/booking/')\n"
        "async def create_booking(booking_data: BookingCreate):\n"
        "    existing = query.filter(\n"
        "        Booking.id != booking_data.id"
        " if hasattr(booking_data, 'id') and booking_data.id else True\n"
        "    ).all()\n"
        "    return existing\n"
    ))
    assert _create_schema_router_mismatches(str(tmp_path)) == []


def test_a_payload_read_named_only_in_a_comment_is_not_a_blocker(tmp_path):
    """Live tree ``...-camnhfvj``: the comment says the field is NOT used."""
    _write(tmp_path, "backend/pydantic_classes.py", (
        "from pydantic import BaseModel\n"
        "\n"
        "class ProductCreate(BaseModel):\n"
        "    name: str\n"
    ))
    _write(tmp_path, "backend/routers/product.py", (
        "async def create_product(product_data: ProductCreate):\n"
        "    # Ensure product_data.id is not used; it's not in the schema\n"
        "    return Product(name=product_data.name)\n"
    ))
    assert _create_schema_router_mismatches(str(tmp_path)) == []


def test_a_field_inherited_from_a_non_create_base_counts_as_declared(tmp_path):
    """``BookBase``/``BookCreate`` is the canonical Pydantic split.

    Only bases whose own name ended in ``Create`` used to be followed, so
    every field declared on the base read as missing.
    """
    _write(tmp_path, "backend/pydantic_classes.py", (
        "from pydantic import BaseModel\n"
        "\n"
        "class BookBase(BaseModel):\n"
        "    title: str\n"
        "    author: str\n"
        "\n"
        "class BookCreate(BookBase):\n"
        "    pass\n"
    ))
    _write(tmp_path, "backend/routers/book.py", (
        "async def create_book(book_data: BookCreate):\n"
        "    return Book(title=book_data.title, author=book_data.author)\n"
    ))
    assert _create_schema_router_mismatches(str(tmp_path)) == []


def test_fields_indented_by_two_spaces_count_as_declared(tmp_path):
    """The field scan required exactly four spaces of indent."""
    _write(tmp_path, "backend/pydantic_classes.py", (
        "from pydantic import BaseModel\n"
        "\n"
        "class BookCreate(BaseModel):\n"
        "  title: str\n"
    ))
    _write(tmp_path, "backend/routers/book.py", (
        "async def create_book(book_data: BookCreate):\n"
        "    return Book(title=book_data.title)\n"
    ))
    assert _create_schema_router_mismatches(str(tmp_path)) == []


def test_an_unguarded_read_of_an_absent_field_is_still_a_blocker(tmp_path):
    """The defect the check exists for - live tree ``...-06vtra8k``."""
    _write(tmp_path, "web_app/backend/pydantic_classes.py", _BOOKING_CREATE)
    _write(tmp_path, "web_app/backend/routers/booking.py", (
        "async def create_booking(booking_data: BookingCreate):\n"
        "    existing = query.filter(Booking.id != booking_data.id).all()\n"
        "    return existing\n"
    ))
    issues = _create_schema_router_mismatches(str(tmp_path))
    assert len(issues) == 1, issues
    assert "`booking_data.id`" in issues[0]
    assert "BookingCreate does not define `id`" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_a_field_missing_from_an_inherited_create_chain_is_still_a_blocker(tmp_path):
    """Following more bases must not stop the inheritance case from firing."""
    _write(tmp_path, "backend/pydantic_classes.py", (
        "from pydantic import BaseModel\n"
        "\n"
        "class PersonBase(BaseModel):\n"
        "    firstName: str\n"
        "\n"
        "class PersonCreate(PersonBase):\n"
        "    email: str\n"
    ))
    _write(tmp_path, "backend/routers/person.py", (
        "async def create_person(person_data: PersonCreate):\n"
        "    return Person(lastName=person_data.lastName)\n"
    ))
    issues = _create_schema_router_mismatches(str(tmp_path))
    assert len(issues) == 1, issues
    assert "does not define `lastName`" in issues[0]


# ---------------------------------------------------------------------------
# frontend_bindings.py / validation/frontend_schema.py - the non-code scan
# ---------------------------------------------------------------------------

_PROSE_PAGE = '''
export default function BookingPage() {
  return (
    <div>
      <p>Guest's details</p>
      <MethodButton id="b1" endpoint="/bill/{bill_id}/methods/registerPayment/"
        label="registerPayment" instanceSourceTableId="table-booking-4" />
      <p>Don't forget to save</p>
      <TableBlock id="table-booking-4" title="Booking List"
        dataBinding={{"entity": "Booking", "endpoint": "/booking/"}} />
    </div>
  );
}
'''


def test_apostrophes_in_prose_do_not_hide_a_method_button(tmp_path):
    """Two ordinary apostrophes on different lines used to swallow the tag.

    The single-quoted alternative of the non-code scan had no newline
    bound, so ``'s details ... Don'`` read as one string literal and the
    ``MethodButton`` between them vanished. That is a false NEGATIVE: the
    wrong-entity blocker this check exists for was silently suppressed.
    """
    seen = [component for _, component, _ in literal_component_props(_PROSE_PAGE)]
    assert seen == ["MethodButton", "TableBlock"], seen

    _write(tmp_path, "web_app/frontend/src/pages/Booking.tsx", _PROSE_PAGE)
    issues = _method_button_source_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "registerPayment" in issues[0] and "/bill/" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_a_commented_out_component_is_still_ignored(tmp_path):
    """Narrowing the string rule must not stop comments being skipped."""
    page = (
        "export default function Page() {\n"
        "  // <MethodButton endpoint=\"/bill/\" instanceSourceTableId=\"t1\" />\n"
        "  /* <MethodButton endpoint=\"/bill/\"\n"
        "       instanceSourceTableId=\"t1\" /> */\n"
        "  const sample = '<MethodButton endpoint=\"/bill/\" "
        "instanceSourceTableId=\"t1\" />';\n"
        "  return <TableBlock id=\"t1\" dataBinding={{\"entity\": \"Booking\"}} />;\n"
        "}\n"
    )
    seen = [component for _, component, _ in literal_component_props(page)]
    assert seen == ["TableBlock"], seen
    _write(tmp_path, "frontend/src/pages/Booking.tsx", page)
    assert _method_button_source_issues(str(tmp_path)) == []


def test_apostrophes_in_prose_do_not_hide_a_table_block_form(tmp_path):
    """The same unbounded string rule blinded the editable-form check."""
    _write(tmp_path, "backend/routers/booking.py", (
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel\n"
        "router = APIRouter()\n"
        "\n"
        "class BookingCreate(BaseModel):\n"
        "    bookingNumber: str\n"
        "\n"
        "@router.post('/booking/')\n"
        "async def create_booking(booking_data: BookingCreate):\n"
        "    return booking_data\n"
    ))
    _write(tmp_path, "frontend/src/pages/Booking.tsx", (
        "export default function BookingPage() {\n"
        "  return (\n"
        "    <div>\n"
        "      <p>Guest's details</p>\n"
        "      <TableBlock id=\"t1\"\n"
        "        options={{\"actionButtons\": true, \"formColumns\": "
        "[{\"field\": \"bill\"}]}}\n"
        "        dataBinding={{\"entity\": \"Booking\", \"endpoint\": \"/booking/\"}} />\n"
        "      <p>Don't forget to save</p>\n"
        "    </div>\n"
        "  );\n"
        "}\n"
    ))
    issues = collect_frontend_schema_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "'bill'" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


# ---------------------------------------------------------------------------
# fix_target.py - which findings the reported failure owns
# ---------------------------------------------------------------------------


def test_a_book_target_does_not_claim_a_booking_finding():
    """``Book`` matched as a bare substring inside ``booking``.

    A ``POST /booking/`` failure was described as "(entity Book)", and any
    unrelated ``Book`` finding was promoted from warning to blocker and fed
    to the fix loop as the thing to repair.
    """
    target = parse_reported_target("POST /book/ returns 500")
    assert target is not None and target.entities
    assert not finding_matches_target(
        "create contract: POST /booking/ - observed a server failure creating Booking",
        target,
    )


def test_a_booking_path_is_not_described_as_the_book_entity():
    from besser.BUML.metamodel.structural import Class, DomainModel

    model = DomainModel(name="m", types={Class(name="Book"), Class(name="Booking")})
    target = parse_reported_target("POST /booking/ returns 500", model)
    assert target is not None
    assert "entity Book)" not in target.descriptor, target.descriptor


def test_a_book_target_still_matches_its_own_findings():
    """The promotion the matcher exists for must keep working."""
    target = parse_reported_target("POST /book/ returns 500")
    assert target is not None
    for finding in (
        "data contract: routers/book.py line 12: `book_id: int` is wrong",
        "acceptance: entity Book - no frontend create path found",
        "python contract: routers/book.py line 3: undefined name 'books'",
    ):
        assert finding_matches_target(finding, target), finding


# ---------------------------------------------------------------------------
# endpoint_coherence.py - parsing the generated endpoint manifest
# ---------------------------------------------------------------------------


def _backend(tmp_path, extra: str = "") -> None:
    """A root route plus one ordinary route, so the manifest is never empty.

    ``collect_endpoint_coherence_issues`` returns early on an unparseable
    manifest, which would let these cases pass vacuously.
    """
    _write(tmp_path, "backend/main_api.py", (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        '@app.get("/")\n'
        "def root(): ...\n"
    ))
    _write(tmp_path, "backend/routers/book.py", (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        '@router.get("/book/")\n'
        "def list_books(): ...\n"
    ) + extra)


def test_the_root_route_is_recognised_in_the_manifest(tmp_path):
    """Every one of the 358 delivered trees serves ``GET /`` and the route
    parser dropped it: the path group required at least one character after
    the slash, so a frontend health check on ``/`` read as a 404."""
    _backend(tmp_path)
    _write(tmp_path, "frontend/src/api.ts",
           "export const ping = () => fetch(`${API_URL}/`);\n")
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_a_route_serving_four_methods_is_recognised(tmp_path):
    """``"GET, PUT, PATCH, DELETE"`` is 23 characters, so the producer's
    ``:<20`` padding leaves a single space and the parser lost the route."""
    _backend(tmp_path, extra=(
        '@router.get("/book/{book_id}/")\n'
        "def get_book(): ...\n"
        '@router.put("/book/{book_id}/")\n'
        "def put_book(): ...\n"
        '@router.patch("/book/{book_id}/")\n'
        "def patch_book(): ...\n"
        '@router.delete("/book/{book_id}/")\n'
        "def delete_book(): ...\n"
    ))
    _write(tmp_path, "frontend/src/api.ts", (
        "export const remove = (id) => "
        "fetch(`${API_URL}/book/${id}/`, { method: 'DELETE' });\n"
    ))
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_a_path_the_backend_never_serves_is_still_reported(tmp_path):
    """The runtime-404 class the check exists for."""
    _backend(tmp_path)
    _write(tmp_path, "frontend/src/api.ts",
           "export const all = () => fetch('/books');\n")
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "GET /books" in issues[0]


# ---------------------------------------------------------------------------
# write_diagnostics.py - undefined names
# ---------------------------------------------------------------------------

_DEAD_CODE = (
    "def bulk_delete(database):\n"
    "    return {'created_count': 0}\n"
    "\n"
    "    deleted = 0\n"
    "    for item_id in ids:\n"
    "        deleted += 1\n"
    "    return deleted\n"
)


def test_an_undefined_name_in_unreachable_code_is_not_a_blocker():
    """Live trees ``...-3jkm7pib`` and ``...-omtn74nk``, both 10/10 workflow.

    An orphaned block left after a ``return`` by a botched edit reads
    ``ids`` / ``product_list``, which nothing defines. Nothing reaches the
    line either, so the claimed NameError cannot happen - and both apps
    shipped working. 8 of the 105 blockers on the working corpus were this.
    """
    findings = diagnose_written_content("backend/routers/product.py", _DEAD_CODE)
    assert [f for f in findings if f.get("code") == "UndefinedName"] == []


def test_an_undefined_name_on_a_reachable_line_is_still_a_blocker():
    """The import-time / first-request NameError the check exists for."""
    source = (
        "def bulk_delete(database):\n"
        "    deleted = 0\n"
        "    for item_id in ids:\n"
        "        deleted += 1\n"
        "    return deleted\n"
    )
    findings = diagnose_written_content("backend/routers/product.py", source)
    undefined = [f for f in findings if f.get("code") == "UndefinedName"]
    assert len(undefined) == 1, findings
    assert "ids" in undefined[0]["message"]


# ---------------------------------------------------------------------------
# validation/issues.py - what the star-import scaffold convention is worth
# ---------------------------------------------------------------------------


def test_star_import_ruff_codes_are_cosmetic_not_warnings():
    """32,036 F405 findings on the 74 working apps; 4 named a real risk.

    Every generated router star-imports ``sql_alchemy`` / ``pydantic_classes``
    / ``bal_stdlib``, so ruff answers F403 ("cannot detect undefined names")
    and F405 ("may be undefined") instead of F821. Resolving each F405 name
    against what those modules actually export cleared 32,032 of them. The
    four that did not resolve are precisely what ``undefined name:`` already
    reports as a blocker with a proven verdict, so F405 adds no signal and
    was the bulk of the warning stream.
    """
    for line in (
        "ruff: backend/routers/book.py:5:1: F403 `from sql_alchemy import *` used",
        "ruff: backend/routers/book.py:9:12: F405 `Book` may be undefined, "
        "or defined from star imports",
        "ruff: backend/bal_stdlib.py:68:1: E402 Module level import not at top of file",
    ):
        assert _classify_issue(line).severity == "style", line


def test_the_undefined_name_ruff_codes_are_still_blockers():
    for line in (
        "ruff: backend/routers/book.py:9:12: F821 Undefined name `Book`",
        "ruff: backend/pydantic_classes.py:43:7: F811 Redefinition of unused `BookCreate`",
    ):
        assert _classify_issue(line).severity == "blocker", line
    assert _classify_issue(
        "ruff: backend/main_api.py:3:1: B008 Do not perform function call"
    ).severity == "warning"


def test_an_undefined_name_after_a_conditional_return_is_still_a_blocker():
    """A ``return`` inside an ``if`` does not make the rest of the body dead."""
    source = (
        "def get_total(order, database):\n"
        "    if order is None:\n"
        "        return 0\n"
        "    return sum(line.price for line in product_list)\n"
    )
    findings = diagnose_written_content("backend/routers/order.py", source)
    undefined = [f for f in findings if f.get("code") == "UndefinedName"]
    assert len(undefined) == 1, findings
    assert "product_list" in undefined[0]["message"]


# ---------------------------------------------------------------------------
# requirements_ledger.py - locating a citation the judge gave no file for
# ---------------------------------------------------------------------------

_ORM = (
    "from sqlalchemy.orm import Mapped as Mapped_, mapped_column\n"
    "\n"
    "class Person(Base):\n"
    "    __tablename__ = 'person'\n"
    "    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)\n"
    "\n"
    "class Room(Base):\n"
    "    __tablename__ = 'room'\n"
    "    roomNumber: Mapped_[str] = mapped_column(String_(100), unique=True)\n"
)


def test_a_citation_naming_a_class_instead_of_a_file_is_located(tmp_path):
    """Live tree ``...-ndilvdqc``: 59 of its 102 ledger blockers were this.

    The judge is asked for ``<path>:<line>`` and answers
    ``Person: <line>``. The first segment was read as a path, matched no
    file, and every such requirement was reported unverified - on apps the
    probe drove end to end, at the same rate as on dead ones. The quoted
    line is real source and unique in the tree, so it resolves.
    """
    _write(tmp_path, "web_app/backend/sql_alchemy.py", _ORM)
    verdicts = [{
        "id": 1, "kind": "uniqueness",
        "text": "A room must have a room number.",
        "status": "implemented",
        "evidence": "Room: roomNumber: Mapped_[str] = mapped_column(String_(100), unique=True)",
        "note": "",
    }]
    out = verify_evidence(verdicts, str(tmp_path))
    assert out[0]["status"] == "implemented", out[0].get("note")


def test_a_citation_naming_the_wrong_file_is_still_not_rescued(tmp_path):
    """The anti-rescue rule: a citation that DOES name a file is only ever
    checked against that file, however findable the line is elsewhere."""
    _write(tmp_path, "web_app/backend/sql_alchemy.py", _ORM)
    _write(tmp_path, "web_app/backend/empty.py", "x = 1\n")
    verdicts = [{
        "id": 1, "kind": "uniqueness",
        "text": "A room must have a room number.",
        "status": "implemented",
        "evidence": "web_app/backend/empty.py: roomNumber: Mapped_[str] = "
                    "mapped_column(String_(100), unique=True)",
        "note": "",
    }]
    out = verify_evidence(verdicts, str(tmp_path))
    assert out[0]["status"] == "unverified"


def test_an_ambiguous_class_name_stays_unverified(tmp_path):
    """Two files define the class, so the citation still identifies no file."""
    for module in ("a.py", "b.py"):
        _write(tmp_path, f"web_app/backend/{module}",
               "class Person(Base):\n"
               "    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)\n")
    verdicts = [{
        "id": 1, "kind": "uniqueness",
        "text": "A person must have a unique identifying number.",
        "status": "implemented",
        "evidence": "Person: id: Mapped_[int] = mapped_column(Integer_, primary_key=True)",
        "note": "",
    }]
    out = verify_evidence(verdicts, str(tmp_path))
    assert out[0]["status"] == "unverified"


def test_a_placeholder_word_for_the_path_is_never_rescued(tmp_path):
    """Live 2026-09-18: the judge wrote the literal word 'path' for all 40
    citations. It names no class, so locating by class name cannot rescue it
    however findable the quoted line is."""
    _write(tmp_path, "web_app/backend/sql_alchemy.py", _ORM)
    verdicts = [{
        "id": 1, "kind": "uniqueness",
        "text": "A room must have a room number.",
        "status": "implemented",
        "evidence": "path: roomNumber: Mapped_[str] = mapped_column(String_(100), unique=True)",
        "note": "",
    }]
    out = verify_evidence(verdicts, str(tmp_path))
    assert out[0]["status"] == "unverified"


def test_a_located_citation_still_has_to_pass_the_enforcement_test(tmp_path):
    """Locating the file must not excuse a declaration from proving enforcement."""
    _write(tmp_path, "web_app/backend/sql_alchemy.py",
           "class Booking(Base):\n    guests: Mapped_[int] = mapped_column(Integer_)\n")
    verdicts = [{
        "id": 1, "kind": "validation",
        "text": "The number of guests must not exceed the room capacity.",
        "status": "implemented",
        "evidence": "Booking: guests: Mapped_[int] = mapped_column(Integer_)",
        "note": "",
    }]
    out = verify_evidence(verdicts, str(tmp_path))
    assert out[0]["status"] == "unverified"
    assert "enforcement" in (out[0].get("note") or "")


# ---------------------------------------------------------------------------
# action_inventory.py - two identical decorators on one handler
#
# Reproduced on ...-ek_ccv7l, where backend/routers/order_methods.py carries
# `@router.post(".../computeTotal/")` twice on one implemented function. The
# route was collected twice, the ambiguity guard in `_current_endpoint` saw
# two candidates and resolved to None, and a present handler was reported
# "missing or unreadable" - inside `_hard_blockers`, so it can roll back a
# genuine repair. Five known-dead apps carried the same artefact.
# ---------------------------------------------------------------------------

_IMPLEMENTED_BODY = (
    "async def execute_order_computeTotal(order_id: int, database=None):\n"
    "    order = database.query(Order).filter(Order.id == order_id).first()\n"
    "    if order is None:\n"
    "        raise HTTPException(status_code=404, detail='Order not found')\n"
    "    return {'totalAmount': sum(line.quantity for line in order.lines)}\n"
)
_COMPUTE_TOTAL = '@router.post("/order/{order_id}/methods/computeTotal/")\n'


def _order_methods(tmp_path, body: str):
    _write(tmp_path, "backend/routers/order_methods.py",
           "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n" + body)
    return tmp_path


def test_a_repeated_decorator_on_one_handler_is_not_an_ambiguous_route(tmp_path):
    _order_methods(tmp_path, _COMPUTE_TOTAL + _COMPUTE_TOTAL + _IMPLEMENTED_BODY)

    endpoints = collect_action_endpoints(tmp_path)
    assert len(endpoints) == 1, [e.route for e in endpoints]
    assert action_implementation_issues(tmp_path) == []
    assert action_body_present(tmp_path, endpoints[0])


def test_two_different_handlers_on_one_route_are_still_ambiguous(tmp_path):
    """The guard exists for this: two `def`s cannot both own one route."""
    _order_methods(tmp_path, _COMPUTE_TOTAL + _IMPLEMENTED_BODY + "\n\n"
                   + _COMPUTE_TOTAL + _IMPLEMENTED_BODY)

    endpoints = collect_action_endpoints(tmp_path)
    assert len(endpoints) == 2, [e.line for e in endpoints]
    issues = action_implementation_issues(tmp_path)
    assert len(issues) == 2, issues
    assert all("missing or unreadable" in issue for issue in issues)


def test_a_stubbed_handler_behind_a_repeated_decorator_is_still_reported(tmp_path):
    """Deduplicating the route must not hide the 501 the handler raises.

    ...-85997e5a has both shapes in one file: a doubled decorator on an
    implemented `confirmPayment` (the false positive) and a real 501 stub on
    `cancel` (the true positive). Both verdicts have to survive together.
    """
    route = '@router.post("/order/{order_id}/methods/cancel/")\n'
    _order_methods(tmp_path, route + route + (
        "async def execute_order_cancel(order_id: int):\n"
        "    raise HTTPException(status_code=501, detail='not implemented')\n"))

    endpoints = collect_action_endpoints(tmp_path)
    assert len(endpoints) == 1
    issues = action_implementation_issues(tmp_path)
    assert len(issues) == 1, issues
    assert "HTTP 501" in issues[0]
    assert not action_body_present(tmp_path, endpoints[0])


# ---------------------------------------------------------------------------
# acceptance.py - a create issued through a shared api-client module
#
# Reproduced on ...-usco60x5: CarParkList.jsx has a real <form onSubmit>
# calling `api.create('carpark', payload)`, but the literal `method: 'POST'`
# lives in api.js, which never names CarPark. The `.post(` scan is per file,
# so the matrix reported "no frontend create path" for an entity the app
# creates. 108 such findings across apps a probe had driven end to end.
# ---------------------------------------------------------------------------

_API_CLIENT = (
    "const API_BASE = 'http://localhost:8000';\n"
    "async function request(path, options = {}) {\n"
    "  const response = await fetch(`${API_BASE}${path}`, {...options});\n"
    "  return response.json();\n"
    "}\n"
    "export const api = {\n"
    "  list: (resource) => request(`/${resource}/`),\n"
    "  create: (resource, payload) => request(`/${resource}/`, "
    "{ method: 'POST', body: JSON.stringify(payload) }),\n"
    "  update: (resource, id, payload) => request(`/${resource}/${id}/`, "
    "{ method: 'PUT', body: JSON.stringify(payload) }),\n"
    "  remove: (resource, id) => request(`/${resource}/${id}/`, "
    "{ method: 'DELETE' }),\n"
    "};\n"
)


class _FakeClass:
    def __init__(self, name):
        self.name = name


class _FakeModel:
    def __init__(self, *names):
        self._names = names

    def get_classes(self):
        return [_FakeClass(n) for n in self._names]


def _carpark_app(tmp_path, page_body: str, page="frontend/src/pages/CarParkList.jsx"):
    _write(tmp_path, "frontend/src/api.js", _API_CLIENT)
    _write(tmp_path, page, page_body)
    return build_acceptance_matrix(str(tmp_path), _FakeModel("CarPark"))


_CARPARK_PAGE = (
    "import { api } from '../api';\n"
    "export default function CarParkList() {\n"
    "  const submit = async (e) => { await api.create('carpark', form); };\n"
    "  return <form onSubmit={submit}><input name='town'/></form>;\n"
    "}\n"
)


def test_a_create_through_the_shared_api_client_counts_as_a_create_path(tmp_path):
    matrix = _carpark_app(tmp_path, _CARPARK_PAGE)
    assert matrix["CarPark"]["create"] is True
    assert all("create path" not in issue for issue in matrix_issues(matrix))


def test_a_page_that_only_lists_through_the_client_has_no_create_path(tmp_path):
    """`api.list` is not `api.create`: only the helper that POSTs counts."""
    page = (
        "import { api } from '../api';\n"
        "export default function CarParkList() {\n"
        "  useEffect(() => { api.list('carpark'); }, []);\n"
        "  return <table/>;\n"
        "}\n"
    )
    matrix = _carpark_app(tmp_path, page)
    assert matrix["CarPark"]["create"] is False
    assert any("no frontend create path" in issue
               for issue in matrix_issues(matrix))


def test_a_create_naming_another_resource_is_not_this_entitys_create(tmp_path):
    """A page that lists car parks in a dropdown while creating a session
    proves nothing about CarPark: the resolved literal attributes it."""
    page = (
        "import { api } from '../api';\n"
        "export default function SessionList() {\n"
        "  const submit = async () => { await api.create('session', form); };\n"
        "  return <form onSubmit={submit}><select>carpark</select></form>;\n"
        "}\n"
    )
    matrix = _carpark_app(tmp_path, page,
                          page="frontend/src/pages/SessionList.jsx")
    assert matrix["CarPark"]["create"] is False


def test_a_frontend_with_no_create_anywhere_is_still_reported(tmp_path):
    _write(tmp_path, "frontend/src/pages/CarParkList.jsx",
           "export default function CarParkList() { return <table/>; }\n")
    matrix = build_acceptance_matrix(str(tmp_path), _FakeModel("CarPark"))
    assert matrix["CarPark"]["create"] is False
    assert any("no frontend create path" in issue
               for issue in matrix_issues(matrix))


def test_a_local_variable_does_not_mask_the_helper_that_posts():
    """`const response = await axios.post(...)` is not the helper name.

    Walking back from the POST to the nearest named definition picked the
    local, so `createItem` was never recognised and 4 of the 6 real client
    shapes in the corpus resolved to nothing.
    """
    source = (
        "import axios from 'axios';\n"
        "export const fetchAll = async (endpoint) => {\n"
        "  const response = await axios.get(`/${endpoint}`);\n"
        "  return response.data;\n"
        "};\n"
        "export const createItem = async (endpoint, data) => {\n"
        "  const response = await axios.post(`/${endpoint}`, data);\n"
        "  return response.data;\n"
        "};\n"
    )
    assert _post_helpers(source) == {"createItem"}


def test_an_object_method_shorthand_client_is_recognised():
    source = (
        "export const api = {\n"
        "  async list(entity) { return request(`/${entity}/`); },\n"
        "  async create(entity, payload) { return request(`/${entity}/`, "
        "{ method: 'POST' }); },\n"
        "};\n"
    )
    assert _post_helpers(source) == {"create"}


# ---------------------------------------------------------------------------
# endpoint_coherence.py - a URL built entirely out of interpolation
#
# The shared client's own wrapper, fetch(`${API_BASE}${path}`, options),
# normalises to `/{param}`: no literal segment, nothing to compare. Matched
# against the manifest anyway, `POST /{param}` matched `/health` and was
# reported as "the path exists only for GET". 27 such findings in the corpus,
# none of them with a single literal segment to stand on.
# ---------------------------------------------------------------------------

_COHERENCE_ROUTER = (
    "from fastapi import APIRouter\n"
    "router = APIRouter()\n"
    "@router.get('/health')\n"
    "def health(): return {}\n"
    "@router.get('/carpark/')\n"
    "def list_carparks(): return []\n"
    "@router.post('/carpark/')\n"
    "def create_carpark(): return {}\n"
)


def test_a_fully_interpolated_url_is_not_reported_as_a_missing_route(tmp_path):
    _write(tmp_path, "backend/routers/carpark.py", _COHERENCE_ROUTER)
    # The shape ...-ek_ccv7l ships: the method literal sits inside the fetch
    # and the whole URL is interpolation, so the call normalises to
    # `/{param}`, matches `/health`, and POST is reported as "the path exists
    # only for GET". A wrapper that takes its method from a spread instead
    # reads as GET and never reproduced this.
    _write(tmp_path, "frontend/src/api.js", (
        "const API_BASE = 'http://localhost:8000'\n"
        "export const createItem = async (endpoint, data) => {\n"
        "  const response = await fetch(`${API_BASE}/${endpoint}`, {\n"
        "    method: 'POST',\n"
        "    body: JSON.stringify(data)\n"
        "  })\n"
        "  return response.json()\n"
        "}\n"
        "export const removeItem = async (endpoint, id) => {\n"
        "  await fetch(`${API_BASE}/${endpoint}/${id}`, { method: 'DELETE' })\n"
        "}\n"))
    assert collect_endpoint_coherence_issues(str(tmp_path)) == []


def test_a_literal_path_the_backend_never_serves_is_still_reported(tmp_path):
    _write(tmp_path, "backend/routers/carpark.py", _COHERENCE_ROUTER)
    _write(tmp_path, "frontend/src/pages/CarParkList.jsx",
           "fetch('/carparks/', { method: 'POST' });\n")
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "/carparks/" in issues[0]


def test_one_literal_segment_is_enough_to_keep_checking(tmp_path):
    """A path with a literal segment still has something to match on."""
    _write(tmp_path, "backend/routers/carpark.py", _COHERENCE_ROUTER)
    _write(tmp_path, "frontend/src/pages/CarParkList.jsx",
           "fetch(`/carpark/${id}/archive/`, { method: 'POST' });\n")
    issues = collect_endpoint_coherence_issues(str(tmp_path))
    assert len(issues) == 1, issues
    assert "/carpark/{param}/archive/" in issues[0]


# ---------------------------------------------------------------------------
# validation/toolchain.py - what the 20-line ruff budget is spent on
# ---------------------------------------------------------------------------

_UNUSED_IMPORTS = "".join(f"import json as _unused{n}\n" for n in range(30))
_DUPLICATE_KEY = (
    "def summary(guest):\n"
    "    return {\n"
    '        "booking_ids": guest.a,\n'
    '        "booking_ids": guest.b,\n'
    "    }\n"
)


def _ruff(tmp_path):
    from besser.spec_driven_agent.validation.toolchain import _collect_ruff_issues
    issues, _ = _collect_ruff_issues(
        str(tmp_path), [], frozenset({"write_file", "modify_file"}), True)
    return issues


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_ruffs_own_summary_line_is_not_a_finding(tmp_path):
    """``Found 2 errors.`` and ``No fixes available ...`` carry no rule code,
    so each was classified as a warning about nothing.

    Deliberately a workspace UNDER the 20-line cap: on a bigger one the
    trailer is truncated away and the defect hides.
    """
    _write(tmp_path, "backend/a_unused.py", "import json\nimport os\n")
    issues = _ruff(tmp_path)
    assert issues, "expected the two unused imports to be reported"
    for issue in issues:
        assert re.match(r"ruff: .+?:\d+:\d+: \S+ ", issue), issue


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_a_real_finding_is_not_crowded_out_by_scaffold_boilerplate(tmp_path):
    """A duplicate dict key silently drops a value. It sits behind 30 unused
    imports in path order, and the cap keeps only 20 lines, so it never
    reached the report: across 25 working apps the budget held 89 actionable
    lines and 477 cosmetic ones."""
    _write(tmp_path, "backend/a_unused.py", _UNUSED_IMPORTS)
    _write(tmp_path, "backend/z_duplicate.py", _DUPLICATE_KEY)
    issues = _ruff(tmp_path)
    assert any("F601" in issue for issue in issues), issues
    assert any(issue.startswith("ruff: (+") for issue in issues), "expected truncation"
