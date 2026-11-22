"""End-to-end tests: deleting a composite whole deletes its parts.

A part cannot outlive its whole, so the generated backend must delete the parts
with the whole instead of refusing the delete because each part "requires at
least 1" owner. Before the fix every composition behaved like a plain mandatory
association: ``DELETE /order/1/`` answered 409 while the order had lines, so a
whole could never be deleted through the API.

    Customer 1..1 -- 0..*  Order       plain association: delete stays guarded
    Order    1..1 <>- 0..* OrderLine   composition, FK on the part
    OrderLine 1..1 <>- 0..* LineNote   nested composition
    Order    1..1 <>- 0..1 Receipt     1:1 composition, FK on the part
    Report   0..1 <>- 1..1 Summary     1:1 composition, FK on the whole
    OrderLine 1..1 -- 0..* Delivery    plain association: a Delivery needs its line
    LineNote  1..1 -- 0..* Flag        plain association: a Flag needs its note
    Order    1..1 <>- 0..* Label       composition; a Label also needs a line (OrderLine 1..1 -- 0..* Label)

``<>`` marks the whole: its end carries ``is_composite=True``.
"""

import asyncio
import importlib
import os
import sys

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.backend.backend_generator import BackendGenerator


def _cls(name):
    return Class(name=name, attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="label", type=StringType),
    })


def _assoc(name, whole_end, part_end):
    return BinaryAssociation(name=name, ends={whole_end, part_end})


def _shop_model() -> DomainModel:
    customer, order, line, note = _cls("Customer"), _cls("Order"), _cls("OrderLine"), _cls("LineNote")
    receipt, report, summary = _cls("Receipt"), _cls("Report"), _cls("Summary")
    delivery, flag, label = _cls("Delivery"), _cls("Flag"), _cls("Label")
    associations = {
        _assoc("Customer_Order",
               Property(name="customer", type=customer, multiplicity=Multiplicity(1, 1)),
               Property(name="orders", type=order, multiplicity=Multiplicity(0, "*"))),
        _assoc("Order_Lines",
               Property(name="order", type=order, multiplicity=Multiplicity(1, 1), is_composite=True),
               Property(name="lines", type=line, multiplicity=Multiplicity(0, "*"))),
        _assoc("Line_Notes",
               Property(name="line", type=line, multiplicity=Multiplicity(1, 1), is_composite=True),
               Property(name="notes", type=note, multiplicity=Multiplicity(0, "*"))),
        _assoc("Order_Receipt",
               Property(name="paid_order", type=order, multiplicity=Multiplicity(1, 1), is_composite=True),
               Property(name="receipt", type=receipt, multiplicity=Multiplicity(0, 1))),
        _assoc("Report_Summary",
               Property(name="report", type=report, multiplicity=Multiplicity(0, 1), is_composite=True),
               Property(name="summary", type=summary, multiplicity=Multiplicity(1, 1))),
        _assoc("Line_Deliveries",
               Property(name="line", type=line, multiplicity=Multiplicity(1, 1)),
               Property(name="deliveries", type=delivery, multiplicity=Multiplicity(0, "*"))),
        _assoc("Order_Labels",
               Property(name="labelled_order", type=order, multiplicity=Multiplicity(1, 1), is_composite=True),
               Property(name="labels", type=label, multiplicity=Multiplicity(0, "*"))),
        _assoc("Line_Labels",
               Property(name="labelled_line", type=line, multiplicity=Multiplicity(1, 1)),
               Property(name="line_labels", type=label, multiplicity=Multiplicity(0, "*"))),
        _assoc("Note_Flags",
               Property(name="note", type=note, multiplicity=Multiplicity(1, 1)),
               Property(name="flags", type=flag, multiplicity=Multiplicity(0, "*"))),
    }
    return DomainModel(
        name="ShopModel",
        types={customer, order, line, note, receipt, report, summary, delivery, flag, label},
        associations=associations,
    )


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
    "routers.customer", "routers.order", "routers.orderline", "routers.linenote",
    "routers.receipt", "routers.report", "routers.summary", "routers.delivery", "routers.flag", "routers.label",
)


@pytest.fixture(scope="module")
def generated_backend(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("backend_composition_delete")
    BackendGenerator(model=_shop_model(), output_dir=str(output_dir)).generate()
    return output_dir


@pytest.fixture(scope="module")
def app(generated_backend):
    saved_modules = {name: sys.modules.pop(name) for name in GENERATED_MODULES if name in sys.modules}
    saved_cwd = os.getcwd()
    saved_database_url = os.environ.get("DATABASE_URL")

    database_path = (generated_backend / "test_api.db").as_posix()
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"
    sys.path.insert(0, str(generated_backend))
    os.chdir(generated_backend)
    try:
        importlib.invalidate_caches()
        main_api = importlib.import_module("main_api")
        yield main_api.app
    finally:
        os.chdir(saved_cwd)
        sys.path.remove(str(generated_backend))
        for name in list(sys.modules):
            if name in GENERATED_MODULES or name.startswith("routers"):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def request(app, method, url, **kwargs):
    async def _send():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_send())


def ok(response):
    assert response.status_code == 200, response.text
    return response


def ids(app, resource):
    return sorted(row["id"] for row in ok(request(app, "GET", f"/{resource}/")).json())


def test_deleting_a_whole_deletes_its_parts_and_their_parts(app):
    ok(request(app, "POST", "/customer/", json={"id": 1, "label": "Ada", "orders": []}))
    ok(request(app, "POST", "/order/", json={"id": 10, "label": "first", "customer": 1}))
    ok(request(app, "POST", "/order/", json={"id": 11, "label": "second", "customer": 1}))
    for line_id, order_id in ((100, 10), (101, 10), (110, 11)):
        ok(request(app, "POST", "/orderline/", json={"id": line_id, "label": "line", "order": order_id}))
    ok(request(app, "POST", "/linenote/", json={"id": 1000, "label": "fragile", "line": 100}))
    ok(request(app, "POST", "/receipt/", json={"id": 7, "label": "paid", "paid_order": 10}))

    ok(request(app, "DELETE", "/order/10/"))

    assert ids(app, "order") == [11]
    assert ids(app, "orderline") == [110]
    assert ids(app, "linenote") == []
    assert ids(app, "receipt") == []


def test_a_plain_mandatory_association_still_refuses_the_delete(app):
    ok(request(app, "POST", "/customer/", json={"id": 2, "label": "Bob", "orders": []}))
    ok(request(app, "POST", "/order/", json={"id": 20, "label": "kept", "customer": 2}))

    response = request(app, "DELETE", "/customer/2/")

    assert response.status_code == 409, response.text
    assert ids(app, "order") == [11, 20]


def test_a_part_can_still_be_deleted_on_its_own(app):
    ok(request(app, "POST", "/orderline/", json={"id": 200, "label": "extra", "order": 20}))

    ok(request(app, "DELETE", "/orderline/200/"))

    assert ids(app, "order") == [11, 20]
    assert 200 not in ids(app, "orderline")


def test_one_to_one_whole_holding_the_key_deletes_its_part(app):
    ok(request(app, "POST", "/summary/", json={"id": 5, "label": "q3"}))
    ok(request(app, "POST", "/report/", json={"id": 50, "label": "Q3 report", "summary": 5}))

    ok(request(app, "DELETE", "/report/50/"))

    assert ids(app, "report") == []
    assert ids(app, "summary") == []


def test_a_whole_is_not_deleted_while_a_part_is_required_elsewhere(app):
    """The cascade reached a part another record requires and the database error went to the client."""
    ok(request(app, "POST", "/order/", json={"id": 30, "label": "shipped", "customer": 1}))
    ok(request(app, "POST", "/orderline/", json={"id": 300, "label": "line", "order": 30}))
    ok(request(app, "POST", "/delivery/", json={"id": 3000, "label": "van", "line": 300}))

    response = request(app, "DELETE", "/order/30/")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "Cannot delete Order 30: OrderLine 300 would be deleted with it, "
        "and Delivery 3000 requires at least 1 line"
    )
    assert 30 in ids(app, "order") and 300 in ids(app, "orderline")

    bulk = request(app, "DELETE", "/order/bulk/", json=[30])
    assert bulk.status_code == 409, bulk.text
    assert bulk.json()["detail"] == response.json()["detail"]

    ok(request(app, "DELETE", "/delivery/3000/"))
    ok(request(app, "DELETE", "/order/30/"))
    assert 300 not in ids(app, "orderline")


def test_the_check_follows_nested_parts(app):
    ok(request(app, "POST", "/order/", json={"id": 31, "label": "annotated", "customer": 1}))
    ok(request(app, "POST", "/orderline/", json={"id": 310, "label": "line", "order": 31}))
    ok(request(app, "POST", "/linenote/", json={"id": 3100, "label": "fragile", "line": 310}))
    ok(request(app, "POST", "/flag/", json={"id": 9, "label": "urgent", "note": 3100}))

    response = request(app, "DELETE", "/order/31/")

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == (
        "Cannot delete Order 31: OrderLine 310 would be deleted with it, "
        "and LineNote 3100 would be deleted with it, and Flag 9 requires at least 1 note"
    )
    assert 3100 in ids(app, "linenote")


def test_parts_of_one_whole_that_require_each_other_are_deleted_together(app):
    ok(request(app, "POST", "/order/", json={"id": 32, "label": "labelled", "customer": 1}))
    ok(request(app, "POST", "/orderline/", json={"id": 320, "label": "line", "order": 32}))
    ok(request(app, "POST", "/label/", json={"id": 5, "label": "gift", "labelled_order": 32, "labelled_line": 320}))

    # On its own the line is required by the label; with the order both go.
    assert request(app, "DELETE", "/orderline/320/").status_code == 409
    ok(request(app, "DELETE", "/order/32/"))

    assert 320 not in ids(app, "orderline")
    assert ids(app, "label") == []
