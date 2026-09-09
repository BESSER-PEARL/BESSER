"""End-to-end tests for the per-file BackendGenerator on association classes and custom PKs.

Successor of ``tests/generators/rest_api/test_rest_api_assoc_class.py``, which exercised
the retired monolithic ``RESTAPIGenerator`` backend template against the same Trip/Seat
model. The modular ``BackendGenerator`` (``main_api.py`` + ``routers/<class>.py`` +
``database.py`` + ``bal_stdlib.py``) must satisfy the identical association-class contract:
link create with ``{target, <attrs>}``, GET link ids + detailed links, PUT reconciliation,
the relationship add/get/remove endpoints, association-class CRUD through both FKs — while
the plain N:M path keeps using its secondary table. Error responses must carry the
endpoint's actual message in ``detail`` (the generated frontend reads that field), and the
primary key — whatever its name — must be immutable through PUT.

The generated backend is loaded with ``importlib`` and driven through an ASGI transport.
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
from besser.generators.backend.backend_generator import BackendGenerator


def _trip_seat_model() -> DomainModel:
    """Trip -- Seat N:M carrying a Reservation association class, plus a plain N:M.

    ``Seat`` is keyed by ``code`` instead of ``id`` so the generated queries have
    to use the real primary key of the class.
    """
    trip = Class(name="Trip", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="reference", type=StringType),
    })
    seat = Class(name="Seat", attributes={
        Property(name="code", type=IntegerType, is_id=True),
        Property(name="label", type=StringType),
    })
    tag = Class(name="Tag", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="label", type=StringType),
    })
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*")),
    })
    reservation = AssociationClass(
        name="Reservation",
        attributes={Property(name="price", type=FloatType)},
        association=trip_seat,
    )
    trip_tag = BinaryAssociation(name="trip_tag", ends={
        Property(name="tagged_trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        name="TripModel",
        types={trip, seat, tag, reservation},
        associations={trip_seat, trip_tag},
    )

# main_api.py imports: pydantic_classes, sql_alchemy, database, routers.<class>
GENERATED_MODULES = (
    "main_api", "pydantic_classes", "sql_alchemy", "database", "bal_stdlib",
    "routers", "routers.trip", "routers.seat", "routers.tag", "routers.reservation",
)


@pytest.fixture(scope="module")
def generated_backend(tmp_path_factory):
    """Generate the per-file backend of the Trip/Seat model and return its output dir."""
    output_dir = tmp_path_factory.mktemp("backend_assoc_class")
    BackendGenerator(model=_trip_seat_model(), output_dir=str(output_dir)).generate()
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


def test_generated_backend_is_valid_python(generated_backend):
    import glob
    for f in glob.glob(str(generated_backend / "**" / "*.py"), recursive=True):
        py_compile.compile(f, doraise=True)


def test_association_class_replaces_secondary_table(generated_backend):
    main_api = (generated_backend / "main_api.py").read_text(encoding="utf-8")
    reservation = (generated_backend / "routers" / "reservation.py").read_text(encoding="utf-8")
    trip = (generated_backend / "routers" / "trip.py").read_text(encoding="utf-8")
    all_code = main_api + reservation + trip
    assert "trip_seat.insert()" not in all_code
    assert "trip_seat.c." not in all_code
    assert "Reservation.trips_id" in all_code
    assert "Reservation.seats_id" in all_code
    assert "trip_tag.insert()" in all_code  # plain N:M keeps its secondary table
    assert "Seat.id" not in all_code
    assert "Seat.code" in all_code


def test_create_with_association_class_links(app):
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
    response = create_trip(app, 2, seats=[{"target": 999, "price": 1.0}])
    assert response.status_code == 404
    # `detail` must carry the endpoint's actual message — the generated
    # frontend reads response.data.detail for its error banner, so a generic
    # "HTTP 404 error occurred" here is a regression even when `message` is
    # correct (checking response.text alone would not catch it).
    assert response.json()["detail"] == "Seat with ID 999 not found"


def test_error_detail_carries_the_actual_message(app):
    """Every 4xx must expose the endpoint's real message in `detail`."""
    response = request(app, "GET", "/seat/2101/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Seat not found"
    response = request(app, "DELETE", "/trip/9999/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Trip not found"


def test_entity_with_non_id_primary_key(app):
    assert create_seat(app, 21, label="aisle").status_code == 200
    response = request(app, "GET", "/seat/21/")
    assert response.status_code == 200, response.text
    assert response.json()["seat"]["label"] == "aisle"
    assert request(app, "GET", "/seat/2100/").status_code == 404


def test_put_never_rewrites_the_primary_key(app):
    """PUT updates attributes but keeps the PK — whatever its name.

    Rewriting Seat.code through PUT would orphan every Reservation row whose
    FK points at the old value; the PK guard must use the class's real
    primary key, not a hardcoded 'id'.
    """
    assert create_seat(app, 71, label="window").status_code == 200
    assert create_trip(app, 17, seats=[{"target": 71, "price": 10.0}]).status_code == 200

    response = request(app, "PUT", "/seat/71/", json={"code": 7100, "label": "renamed"})
    assert response.status_code == 200, response.text
    # The attribute update applied, the PK did not move
    assert request(app, "GET", "/seat/71/").json()["seat"]["label"] == "renamed"
    assert request(app, "GET", "/seat/7100/").status_code == 404
    # ...and the association-class row still points at the seat
    assert request(app, "GET", "/reservation/71/17/").json()["price"] == 10.0

    # Same guarantee for a PK literally named `id` (Trip)
    response = request(app, "PUT", "/trip/17/",
                       json={"id": 888, "reference": "kept",
                             "seats": [{"target": 71, "price": 10.0}], "tags": []})
    assert response.status_code == 200, response.text
    assert request(app, "GET", "/trip/17/").json()["seats_ids"] == [71]
    assert request(app, "GET", "/trip/888/").status_code == 404


def test_get_returns_link_ids_and_detailed_links(app):
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
    for code in (41, 42, 43):
        assert create_seat(app, code).status_code == 200
    assert create_trip(app, 4, seats=[{"target": 41, "price": 5.0},
                                      {"target": 42, "price": 6.0}]).status_code == 200
    response = request(
        app, "PUT", "/trip/4/",
        json={"id": 4, "reference": "TR-updated",
              "seats": [{"target": 42, "price": 66.0}, {"target": 43, "price": 7.0}],
              "tags": []},
    )
    assert response.status_code == 200, response.text
    assert sorted(response.json()["seats_ids"]) == [42, 43]
    assert request(app, "GET", "/reservation/41/4/").status_code == 404
    assert request(app, "GET", "/reservation/42/4/").json()["price"] == 66.0
    assert request(app, "GET", "/reservation/43/4/").json()["price"] == 7.0


def test_relationship_endpoints_of_association_class(app):
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
    assert create_trip(app, 7).status_code == 200
    response = request(app, "POST", "/reservation/", json={"seats": 998, "trips": 7, "price": 1.0})
    assert response.status_code == 404
    assert "Seat with ID 998" in response.text


def test_plain_many_to_many_still_works(app):
    assert create_tag(app, 81).status_code == 200
    assert create_tag(app, 82).status_code == 200
    response = create_trip(app, 8, tags=[81, 82])
    assert response.status_code == 200, response.text
    assert sorted(response.json()["tag_ids"]) == [81, 82]
    listed = request(app, "GET", "/trip/8/tags/")
    assert listed.status_code == 200, listed.text
    assert listed.json()["tags_count"] == 2
