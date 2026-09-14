"""End-to-end tests: the generated backend enforces lower-bound multiplicities.

The ORM cannot express "a Booking needs at least one Guest" (no FK, NOT NULL or
cascade says that), so the generated routers enforce it wherever a relationship
can shrink: deleting an entity, removing a link, editing the relationship list.
The create endpoint already refused such payloads; these tests cover the other
paths on a hotel-like model:

    Booking  0..* -- 1..*  Guest       (secondary table)      a Booking needs a guest
    Employee 1..1 -- 0..*  Booking     (FK on Booking)        a Booking needs an employee
    Booking  0..* -- 1..*  Room        (ReservedRoom links)   a Booking needs a room
    Booking  1..1 -- 0..1  Invoice     (FK on Invoice)        an Invoice needs a booking
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
        Property(name="reference", type=StringType),
    })
    guest = Class(name="Guest", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    })
    employee = Class(name="Employee", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
        Property(name="last_name", type=StringType),
    })
    room = Class(name="Room", attributes={
        Property(name="number", type=IntegerType, is_id=True),
        Property(name="label", type=StringType),
    })
    invoice = Class(name="Invoice", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="amount", type=FloatType),
    })
    booking_guest = BinaryAssociation(name="Booking_Guest", ends={
        Property(name="staying", type=booking, multiplicity=Multiplicity(0, "*")),
        Property(name="guests", type=guest, multiplicity=Multiplicity(1, "*")),
    })
    employee_booking = BinaryAssociation(name="Employee_Booking", ends={
        Property(name="managed_by", type=employee, multiplicity=Multiplicity(1, 1)),
        Property(name="manages", type=booking, multiplicity=Multiplicity(0, "*")),
    })
    room_booking = BinaryAssociation(name="Room_Booking", ends={
        Property(name="booking", type=booking, multiplicity=Multiplicity(0, "*")),
        Property(name="rooms", type=room, multiplicity=Multiplicity(1, "*")),
    })
    reserved_room = AssociationClass(
        name="ReservedRoom", attributes={Property(name="price", type=FloatType)}, association=room_booking,
    )
    booking_invoice = BinaryAssociation(name="Booking_Invoice", ends={
        Property(name="billed_booking", type=booking, multiplicity=Multiplicity(1, 1)),
        Property(name="invoice", type=invoice, multiplicity=Multiplicity(0, 1)),
    })
    return DomainModel(
        name="HotelModel",
        types={booking, guest, employee, room, invoice, reserved_room},
        associations={booking_guest, employee_booking, room_booking, booking_invoice},
    )


GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib", "routers",
    "routers.booking", "routers.guest", "routers.employee", "routers.room", "routers.invoice",
    "routers.reservedroom",
)


@pytest.fixture(scope="module")
def generated_backend(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("backend_lower_bounds")
    BackendGenerator(model=_hotel_model(), output_dir=str(output_dir)).generate()
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


def setup_booking(app, booking_id, guest_ids, room_numbers, employee_id):
    """Guests, rooms and an employee exist; the booking links all of them."""
    for guest_id in guest_ids:
        ok(request(app, "POST", "/guest/", json={"id": guest_id, "name": f"guest {guest_id}"}))
    for number in room_numbers:
        ok(request(app, "POST", "/room/", json={"number": number, "label": f"room {number}"}))
    ok(request(app, "POST", "/employee/", json={"id": employee_id, "name": "Emp", "last_name": "Loyee"}))
    ok(request(app, "POST", "/booking/", json={
        "id": booking_id, "reference": f"B{booking_id}", "managed_by": employee_id,
        "guests": guest_ids, "rooms": [{"target": n, "price": 10.0} for n in room_numbers],
    }))


def test_edit_employee_keeps_its_bookings(app):
    """The screenshot bug: saving an Employee whose 'manages' list is unchanged
    used to null every booking's managed_by first and die on NOT NULL."""
    setup_booking(app, 10, guest_ids=[1], room_numbers=[101], employee_id=2)

    ok(request(app, "PUT", "/employee/2/", json={"id": 2, "name": "Emp", "last_name": "Renamed", "manages": [10]}))
    assert request(app, "GET", "/employee/2/").json()["employee"]["last_name"] == "Renamed"
    assert request(app, "GET", "/booking/10/").json()["booking"]["managed_by_id"] == 2


def test_edit_employee_cannot_orphan_a_booking(app):
    setup_booking(app, 11, guest_ids=[2], room_numbers=[102], employee_id=3)

    response = request(app, "PUT", "/employee/3/", json={"id": 3, "name": "Emp", "last_name": "L", "manages": []})
    assert response.status_code == 409, response.text
    assert "Booking 11 requires a managed_by" in response.json()["detail"]
    assert request(app, "GET", "/booking/11/").json()["booking"]["managed_by_id"] == 3


def test_delete_employee_managing_a_booking_is_refused(app):
    setup_booking(app, 12, guest_ids=[3], room_numbers=[103], employee_id=4)

    response = request(app, "DELETE", "/employee/4/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Cannot delete Employee 4: Booking 12 requires at least 1 managed_by"
    assert request(app, "GET", "/employee/4/").status_code == 200


def test_edit_booking_cannot_drop_all_guests_or_rooms(app):
    setup_booking(app, 13, guest_ids=[4, 5], room_numbers=[104], employee_id=5)
    base = {"id": 13, "reference": "B13", "managed_by": 5}

    response = request(app, "PUT", "/booking/13/", json={**base, "guests": [], "rooms": [{"target": 104, "price": 1.0}]})
    assert response.status_code == 400 and response.json()["detail"] == "At least 1 Guest(s) required"
    response = request(app, "PUT", "/booking/13/", json={**base, "guests": [4], "rooms": []})
    assert response.status_code == 400 and response.json()["detail"] == "At least 1 Room(s) required"
    # Shrinking to the minimum is fine
    ok(request(app, "PUT", "/booking/13/", json={**base, "guests": [4], "rooms": [{"target": 104, "price": 1.0}]}))
    assert request(app, "GET", "/booking/13/").json()["guest_ids"] == [4]


def test_removing_the_last_guest_link_is_refused(app):
    setup_booking(app, 14, guest_ids=[6, 7], room_numbers=[105], employee_id=6)

    ok(request(app, "DELETE", "/booking/14/guests/7/"))
    response = request(app, "DELETE", "/booking/14/guests/6/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Booking 14 requires at least 1 guests"
    assert request(app, "GET", "/booking/14/").json()["guest_ids"] == [6]


def test_deleting_the_last_guest_of_a_booking_is_refused(app):
    setup_booking(app, 15, guest_ids=[8, 9], room_numbers=[106], employee_id=7)

    ok(request(app, "DELETE", "/guest/9/"))  # not the last one
    response = request(app, "DELETE", "/guest/8/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Cannot delete Guest 8: Booking 15 requires at least 1 guests"
    assert request(app, "GET", "/booking/15/").json()["guest_ids"] == [8]


def test_deleting_the_last_room_or_its_link_is_refused(app):
    setup_booking(app, 16, guest_ids=[10], room_numbers=[107, 108], employee_id=8)

    ok(request(app, "DELETE", "/reservedroom/16/108/"))  # one room left
    response = request(app, "DELETE", "/reservedroom/16/107/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Cannot delete this ReservedRoom: Booking 16 requires at least 1 rooms"
    response = request(app, "DELETE", "/room/107/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Cannot delete Room 107: Booking 16 requires at least 1 rooms"
    # The freed room can go
    ok(request(app, "DELETE", "/room/108/"))


def test_deleting_a_booking_with_an_invoice_is_refused(app):
    setup_booking(app, 17, guest_ids=[11], room_numbers=[109], employee_id=9)
    ok(request(app, "POST", "/invoice/", json={"id": 1, "amount": 99.0, "billed_booking": 17}))

    response = request(app, "DELETE", "/booking/17/")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Cannot delete Booking 17: Invoice 1 requires at least 1 billed_booking"
    # Without the invoice the booking (and its links) can go
    ok(request(app, "DELETE", "/invoice/1/"))
    ok(request(app, "DELETE", "/booking/17/"))
    assert request(app, "GET", "/reservedroom/17/109/").status_code == 404
