"""End-to-end tests for the REST API generator on association classes and custom primary keys.

The generated backend is loaded with ``importlib`` and driven through an ASGI transport
(the installed starlette/httpx versions do not support the legacy ``TestClient(app=...)``
pattern, see ``tests/utilities/web_modeling_editor/backend/test_api_integration.py``).

The Create models that carry association class links are produced by the Pydantic
generator. This module appends its own definitions of the affected models to the
generated ``pydantic_classes.py`` so that the REST layer is exercised against the frozen
contract (``<AssociationClass>LinkCreate`` with ``target`` plus the association class
attributes) without depending on the Pydantic generator implementing it yet.
"""

import asyncio
import importlib
import os
import py_compile
import sys

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, FloatType,
    IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator


GENERATED_MODULES = ("main_api", "pydantic_classes", "sql_alchemy")

# Create models rewritten on top of the generated ones, following the association class
# link contract shared with the Pydantic generator.
LINK_CONTRACT_STUB = '''

class ReservationLinkCreate(BaseModel):
    target: int  # Seat code
    price: float


class TripCreate(BaseModel):
    id: int
    reference: str
    seats: Optional[List[ReservationLinkCreate]] = None
    tags: Optional[List[int]] = None


class ReservationCreate(BaseModel):
    seats: int
    trips: int
    price: float
'''


def _trip_seat_model() -> DomainModel:
    """Trip -- Seat N:M carrying a Reservation association class, plus a plain N:M.

    ``Seat`` is keyed by ``code`` instead of ``id`` so that the generated queries have to
    use the real primary key of the class.
    """
    trip = Class(
        name="Trip",
        attributes={
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="reference", type=StringType),
        },
    )
    seat = Class(
        name="Seat",
        attributes={
            Property(name="code", type=IntegerType, is_id=True),
            Property(name="label", type=StringType),
        },
    )
    tag = Class(
        name="Tag",
        attributes={
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="label", type=StringType),
        },
    )

    trip_seat = BinaryAssociation(
        name="trip_seat",
        ends={
            Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
            Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*")),
        },
    )
    reservation = AssociationClass(
        name="Reservation",
        attributes={Property(name="price", type=FloatType)},
        association=trip_seat,
    )

    trip_tag = BinaryAssociation(
        name="trip_tag",
        ends={
            Property(name="tagged_trips", type=trip, multiplicity=Multiplicity(0, "*")),
            Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
        },
    )

    return DomainModel(
        name="TripModel",
        types={trip, seat, tag, reservation},
        associations={trip_seat, trip_tag},
    )


@pytest.fixture(scope="module")
def generated_backend(tmp_path_factory):
    """Generate the backend of the Trip/Seat model and return its output directory."""
    output_dir = tmp_path_factory.mktemp("rest_api_assoc_class")
    model = _trip_seat_model()

    RESTAPIGenerator(model=model, output_dir=str(output_dir), backend=True).generate()
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate()
    PydanticGenerator(
        model=model, output_dir=str(output_dir), backend=True, nested_creations=False
    ).generate()

    pydantic_file = output_dir / "pydantic_classes.py"
    pydantic_file.write_text(
        pydantic_file.read_text(encoding="utf-8") + LINK_CONTRACT_STUB, encoding="utf-8"
    )
    return output_dir


@pytest.fixture(scope="module")
def app(generated_backend):
    """Import the generated FastAPI application and return it."""
    saved_modules = {
        name: sys.modules.pop(name) for name in GENERATED_MODULES if name in sys.modules
    }
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
        for name in GENERATED_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if saved_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = saved_database_url


def request(app, method, url, **kwargs):
    """Send a request to the ASGI application and return the response."""

    async def _send():
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(_send())


def create_trip(app, trip_id, reference="TR", seats=None, tags=None):
    payload = {"id": trip_id, "reference": reference}
    if seats is not None:
        payload["seats"] = seats
    if tags is not None:
        payload["tags"] = tags
    return request(app, "POST", "/trip/", json=payload)


def create_seat(app, code, label="window"):
    return request(app, "POST", "/seat/", json={"code": code, "label": label})


def create_tag(app, tag_id, label="promo"):
    return request(app, "POST", "/tag/", json={"id": tag_id, "label": label})


def test_generated_main_api_is_valid_python(generated_backend):
    """The generated application must at least compile."""
    py_compile.compile(str(generated_backend / "main_api.py"), doraise=True)


def test_association_class_replaces_secondary_table(generated_backend):
    """The N:M code paths must go through the association class, not a secondary table."""
    main_api = (generated_backend / "main_api.py").read_text(encoding="utf-8")

    assert "trip_seat.insert()" not in main_api
    assert "trip_seat.c." not in main_api
    assert "Reservation.trips_id" in main_api
    assert "Reservation.seats_id" in main_api
    # The plain N:M association keeps using its secondary table
    assert "trip_tag.insert()" in main_api
    # Seat is keyed by `code`, never by a surrogate `id`
    assert "Seat.id" not in main_api
    assert "Seat.code" in main_api


def test_create_with_association_class_links(app):
    """Creating an entity with links stores the association class rows and their attributes."""
    assert create_seat(app, 11).status_code == 200
    assert create_seat(app, 12).status_code == 200

    response = create_trip(
        app, 1, seats=[{"target": 11, "price": 30.5}, {"target": 12, "price": 40.0}]
    )
    assert response.status_code == 200, response.text
    assert sorted(response.json()["seats_ids"]) == [11, 12]

    link = request(app, "GET", "/reservation/11/1/")
    assert link.status_code == 200, link.text
    assert link.json()["price"] == 30.5


def test_create_rejects_unknown_link_target(app):
    """A link pointing at a missing entity is reported instead of being silently dropped."""
    response = create_trip(app, 2, seats=[{"target": 999, "price": 1.0}])
    assert response.status_code == 404
    assert "Seat with ID 999" in response.text


def test_entity_with_non_id_primary_key(app):
    """A class whose primary key is not named `id` is created and read back by that key."""
    assert create_seat(app, 21, label="aisle").status_code == 200

    response = request(app, "GET", "/seat/21/")
    assert response.status_code == 200, response.text
    assert response.json()["seat"]["label"] == "aisle"

    assert request(app, "GET", "/seat/2100/").status_code == 404


def test_get_returns_link_ids_and_detailed_links(app):
    """GET endpoints expose the link target ids and, in detailed mode, the link rows."""
    assert create_seat(app, 31).status_code == 200
    assert create_trip(app, 3, seats=[{"target": 31, "price": 12.0}]).status_code == 200

    response = request(app, "GET", "/trip/3/")
    assert response.status_code == 200, response.text
    assert response.json()["seats_ids"] == [31]

    detailed = request(app, "GET", "/trip/?detailed=true")
    assert detailed.status_code == 200, detailed.text
    trip = next(item for item in detailed.json() if item["id"] == 3)
    assert [seat["code"] for seat in trip["seats"]] == [31]
    assert [link["price"] for link in trip["seats_links"]] == [12.0]


def test_update_reconciles_association_class_links(app):
    """PUT removes dropped links, adds new ones and updates the attributes of kept ones."""
    for code in (41, 42, 43):
        assert create_seat(app, code).status_code == 200
    assert create_trip(app, 4, seats=[{"target": 41, "price": 5.0},
                                      {"target": 42, "price": 6.0}]).status_code == 200

    response = request(
        app,
        "PUT",
        "/trip/4/",
        json={
            "id": 4,
            "reference": "TR-updated",
            "seats": [{"target": 42, "price": 66.0}, {"target": 43, "price": 7.0}],
            # The plain N:M update path predates this fix and dereferences the field
            # unconditionally, so it has to be sent even when it stays empty.
            "tags": [],
        },
    )
    assert response.status_code == 200, response.text
    assert sorted(response.json()["seats_ids"]) == [42, 43]

    assert request(app, "GET", "/reservation/41/4/").status_code == 404
    assert request(app, "GET", "/reservation/42/4/").json()["price"] == 66.0
    assert request(app, "GET", "/reservation/43/4/").json()["price"] == 7.0


def test_relationship_endpoints_of_association_class(app):
    """The add/get/remove relationship endpoints write and read the association class rows."""
    assert create_seat(app, 51).status_code == 200
    assert create_trip(app, 5).status_code == 200

    added = request(app, "POST", "/trip/5/seats/51/", json={"price": 99.5})
    assert added.status_code == 200, added.text

    duplicate = request(app, "POST", "/trip/5/seats/51/", json={"price": 1.0})
    assert duplicate.status_code == 400

    listed = request(app, "GET", "/trip/5/seats/")
    assert listed.status_code == 200, listed.text
    assert listed.json()["seats_count"] == 1
    assert [link["price"] for link in listed.json()["seats_links"]] == [99.5]

    removed = request(app, "DELETE", "/trip/5/seats/51/")
    assert removed.status_code == 200, removed.text
    assert request(app, "GET", "/trip/5/seats/").json()["seats_count"] == 0


def test_association_class_crud_uses_both_foreign_keys(app):
    """The association class is created, updated and deleted through its two foreign keys."""
    assert create_seat(app, 61).status_code == 200
    assert create_trip(app, 6).status_code == 200

    created = request(app, "POST", "/reservation/", json={"seats": 61, "trips": 6, "price": 8.0})
    assert created.status_code == 200, created.text

    updated = request(app, "PUT", "/reservation/61/6/", json={"seats": 61, "trips": 6, "price": 9.0})
    assert updated.status_code == 200, updated.text
    assert request(app, "GET", "/reservation/61/6/").json()["price"] == 9.0

    assert request(app, "DELETE", "/reservation/61/6/").status_code == 200
    assert request(app, "GET", "/reservation/61/6/").status_code == 404


def test_association_class_create_reports_missing_target(app):
    """Creating an association class row with an unknown end is rejected."""
    assert create_trip(app, 7).status_code == 200
    response = request(app, "POST", "/reservation/", json={"seats": 998, "trips": 7, "price": 1.0})
    assert response.status_code == 404
    assert "Seat with ID 998" in response.text


def test_plain_many_to_many_still_works(app):
    """An N:M association without association class keeps using its secondary table."""
    assert create_tag(app, 81).status_code == 200
    assert create_tag(app, 82).status_code == 200

    response = create_trip(app, 8, tags=[81, 82])
    assert response.status_code == 200, response.text
    assert sorted(response.json()["tag_ids"]) == [81, 82]

    listed = request(app, "GET", "/trip/8/tags/")
    assert listed.status_code == 200, listed.text
    assert listed.json()["tags_count"] == 2
