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

from besser.generators.llm.endpoint_coherence import collect_endpoint_coherence_issues
from besser.generators.llm.fix_target import finding_matches_target, parse_reported_target
from besser.generators.llm.frontend_bindings import literal_component_props
from besser.generators.llm.validation.frontend_contract import _method_button_source_issues
from besser.generators.llm.validation.frontend_schema import collect_frontend_schema_issues
from besser.generators.llm.validation.issues import _classify_issue
from besser.generators.llm.validation.python_source import _create_schema_router_mismatches
from besser.generators.llm.write_diagnostics import diagnose_written_content


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
