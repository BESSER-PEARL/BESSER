"""The acceptance test for the derived-enum fix: a raw app can POST the entity.

No LLM anywhere. The model goes straight through ``BackendGenerator``, the
app is booted by the Phase 3 constructibility probe (the same one that runs
in a real run), and the entity carrying the derived enum has to come back
``created``.

Before the fix both shapes below returned 409 on the create route of every
raw generated app, for the two case models below::

    POST /booking/  NOT NULL constraint failed: booking.commercialStatus
    POST /order/    NOT NULL constraint failed: order.status

The derived enum column was ``NOT NULL`` with no default, the client could
not send it (it is absent from the Create schema, correctly), and nothing
assigned it server-side. The entity was unreachable, and with it everything
that needs its id. Every run had to notice and repair this before the app
did anything at all.

Two shapes because the two case models differ in what else is derived:
Booking has two derived enums alongside a derived float, Order has one.
"""
from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import (
    Class, DateType, DomainModel, Enumeration, EnumerationLiteral, FloatType,
    Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.spec_driven_agent.validation.constructibility import (
    PREFIX, collect_constructibility_issues,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _enum(name: str, *literals: str) -> Enumeration:
    return Enumeration(name=name,
                       literals={EnumerationLiteral(name=lit) for lit in literals})


def _hotel_model() -> DomainModel:
    """Hotel case: Booking, two derived enums + a derived float."""
    commercial = _enum("BookingCommercialStatus",
                       "AWAITING_PAYMENT", "CONFIRMED", "CANCELLED")
    physical = _enum("BookingPhysicalStatus",
                     "NOT_YET_ARRIVED", "CHECKED_IN", "CHECKED_OUT")
    booking = Class(name="Booking")
    booking.attributes = {
        Property(name="bookingNumber", type=StringType),
        Property(name="arrivalDate", type=DateType),
        Property(name="totalPrice", type=FloatType, is_derived=True),
        Property(name="commercialStatus", type=commercial, is_derived=True),
        Property(name="physicalStatus", type=physical, is_derived=True),
    }
    return DomainModel(name="Hotel", types={booking, commercial, physical})


def _inventory_model() -> DomainModel:
    """Inventory case: Order, one derived enum (``status``)."""
    order_status = _enum("OrderStatus", "PENDING", "SHIPPED", "CANCELLED")
    order = Class(name="Order")
    order.attributes = {
        Property(name="orderNumber", type=StringType),
        Property(name="orderDate", type=DateType),
        Property(name="totalAmount", type=FloatType, is_derived=True),
        Property(name="status", type=order_status, is_derived=True),
    }
    return DomainModel(name="Inventory", types={order, order_status})


def _generate(model: DomainModel, tmp_path) -> str:
    BackendGenerator(model=model, output_dir=str(tmp_path),
                     http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    return str(tmp_path)


@pytest.mark.parametrize("build_model,entity,route", [
    (_hotel_model, "Booking", "POST /booking/"),
    (_inventory_model, "Order", "POST /order/"),
])
def test_the_entity_with_a_derived_enum_is_created(tmp_path, build_model, entity, route):
    """FAILS before the fix: the probe reports ``create contract: ... POST
    /booking/ - observed a server/persistence failure ... NOT NULL constraint
    failed``, because the derived enum column had no value to insert."""
    issues = collect_constructibility_issues(_generate(build_model(), tmp_path))
    blocking = [i for i in issues if i.startswith(PREFIX) and route in i]
    assert not blocking, f"{entity} still cannot be created: {blocking}"


@pytest.mark.parametrize("build_model,entity", [
    (_hotel_model, "Booking"),
    (_inventory_model, "Order"),
])
def test_the_probe_reports_the_entity_created(tmp_path, build_model, entity):
    """The absence of a blocker is not the same as a successful POST — an
    unverified route produces no ``create contract:`` issue either. This
    asserts the request the probe actually made came back created."""
    from besser.spec_driven_agent.validation.constructibility import _run_probe
    from besser.spec_driven_agent.execution.process import _safe_subprocess_env

    report = _run_probe(_generate(build_model(), tmp_path), _safe_subprocess_env())
    assert report.get("boot") == "ok", report
    entry = (report.get("entities") or {}).get(entity, {})
    assert entry.get("verdict") == "created", entry
