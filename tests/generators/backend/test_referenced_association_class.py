"""An association class that another association points at needs its own key.

A shape the modeling agent commonly produces for a hotel: Booking *-* Room
through the association class ``BookedRoom``, and ``ExtraCharge`` recorded
against one ``BookedRoom``. The routers already address that link by
``BookedRoom.id`` (``POST /extracharge/``, ``/bookedroom/{id}/charges/``), but
the table was keyed by the endpoint pair only, so the foreign key named
``bookedroom.id``, a column that did not exist: SQLite refused the writes
("foreign key mismatch") and the create handler raised AttributeError.

The generated backend is loaded with ``importlib`` and driven through an ASGI
transport, like ``test_backend_assoc_class.py``.
"""

import asyncio
import importlib
import os
import sys

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, FloatType,
    IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.backend.backend_generator import BackendGenerator


def _hotel_model() -> DomainModel:
    booking = Class(name="Booking", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="code", type=StringType),
    })
    room = Class(name="Room", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="number", type=IntegerType),
    })
    extra = Class(name="ExtraCharge", attributes={
        Property(name="amount", type=FloatType),
    })
    books = BinaryAssociation(name="books", ends={
        Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
        Property(name="rooms", type=room, multiplicity=Multiplicity(1, "*")),
    })
    booked = AssociationClass(
        name="BookedRoom",
        attributes={Property(name="agreed", type=FloatType)},
        association=books,
    )
    charged = BinaryAssociation(name="charged", ends={
        Property(name="recordedAgainst", type=booked, multiplicity=Multiplicity(1, 1)),
        Property(name="charges", type=extra, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        name="Hotel", types={booking, room, extra, booked}, associations={books, charged},
    )


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    backend = tmp_path_factory.mktemp("referenced_association_class")
    BackendGenerator(model=_hotel_model(), output_dir=str(backend)).generate()

    saved_modules = {
        name: sys.modules.pop(name) for name in list(sys.modules)
        if name in GENERATED_MODULES or name.startswith("routers.")
    }
    saved_cwd = os.getcwd()
    saved_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{(backend / 'test_api.db').as_posix()}"
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        importlib.invalidate_caches()
        yield importlib.import_module("main_api").app
    finally:
        os.chdir(saved_cwd)
        sys.path.remove(str(backend))
        for name in list(sys.modules):
            if name in GENERATED_MODULES or name.startswith("routers."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def request(app, method, url, **kwargs):
    async def _send():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_send())


def test_an_extra_charge_can_be_recorded_against_a_booked_room(app):
    assert request(app, "POST", "/room/", json={"id": 7, "number": 101}).status_code == 200
    booking = request(app, "POST", "/booking/", json={
        "id": 1, "code": "B1", "rooms": [{"target": 7, "agreed": 90.0}],
    })
    assert booking.status_code == 200, booking.text

    [link] = request(app, "GET", "/bookedroom/").json()
    assert isinstance(link.get("id"), int), f"the link exposes no key to point at: {link}"

    charge = request(app, "POST", "/extracharge/", json={
        "amount": 15.0, "recordedAgainst": link["id"],
    })
    assert charge.status_code == 200, charge.text

    charges = request(app, "GET", f"/bookedroom/{link['id']}/charges/")
    assert charges.status_code == 200, charges.text
    assert [c["amount"] for c in charges.json()["charges"]] == [15.0]


def test_the_pair_routes_still_address_the_link(app):
    """The new key is additive: the link is still unique per booking/room pair
    and still reachable through the existing pair routes."""
    link = request(app, "GET", "/bookedroom/1/7/")
    assert link.status_code == 200, link.text
    assert link.json()["agreed"] == 90.0

    duplicate = request(app, "POST", "/bookedroom/", json={
        "bookings": 1, "rooms": 7, "agreed": 50.0,
    })
    assert duplicate.status_code >= 400, "the booking/room pair must stay unique"
