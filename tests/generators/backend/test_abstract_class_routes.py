"""An abstract class has no instances of its own: its router reads across the
subclasses and offers no route that creates, changes or deletes a record.

    Vehicle (abstract)  <|-- Car, Truck          no association: one table per subclass
    Party (abstract)    <|-- Person, Company     Party 1..1 -- 0..* Note: joined tables
"""
import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Generalization, IntegerType, Multiplicity, Property, StringType,
)

SCRIPT = """
    car = client.post("/car/", json={"plate": "AB-1", "seats": 4}).json()
    truck = client.post("/truck/", json={"plate": "CD-2", "payload": 9}).json()
    person = client.post("/person/", json={"name": "Ann", "age": 41}).json()["person"]
    company = client.post("/company/", json={"name": "Acme", "staff": 12}).json()["company"]
    note = client.post("/note/", json={"text": "call back", "party": person["id"]}).json()

    def status(method, url, **kwargs):
        return client.request(method, url, **kwargs).status_code

    writes = {
        "create": status("POST", "/party/", json={"name": "Nobody"}),
        "bulk_create": status("POST", "/party/bulk/", json=[{"name": "Nobody"}]),
        "update": status("PUT", f"/party/{person['id']}/", json={"name": "Changed"}),
        "delete": status("DELETE", f"/party/{person['id']}/"),
        "bulk_delete": status("DELETE", "/party/bulk/", json=[person["id"]]),
        "create_concrete_table": status("POST", "/vehicle/", json={"plate": "ZZ-9"}),
    }
    routes = sorted(
        f"{method.upper()} {path}"
        for path, operations in client.get("/openapi.json").json()["paths"].items()
        if path.startswith(("/party/", "/vehicle/"))
        for method in operations
    )
    print(json.dumps({
        "writes": writes, "routes": routes,
        "parties": sorted(item["name"] for item in client.get("/party/").json()),
        "party": client.get(f"/party/{company['id']}/").json()["party"]["name"],
        "party_notes": client.get(f"/party/{person['id']}/notes/").json()["notes_count"],
        "vehicles": sorted(item["plate"] for item in client.get("/vehicle/").json()),
        "vehicle_count": client.get("/vehicle/count/").json()["count"],
        "subclass_update": status("PUT", f"/person/{person['id']}/", json={"name": "Ann B", "age": 42}),
    }))
"""


def _model() -> DomainModel:
    vehicle = Class(name="Vehicle", is_abstract=True, attributes={Property(name="plate", type=StringType)})
    car = Class(name="Car", attributes={Property(name="seats", type=IntegerType)})
    truck = Class(name="Truck", attributes={Property(name="payload", type=IntegerType)})
    party = Class(name="Party", is_abstract=True, attributes={Property(name="name", type=StringType)})
    person = Class(name="Person", attributes={Property(name="age", type=IntegerType)})
    company = Class(name="Company", attributes={Property(name="staff", type=IntegerType)})
    note = Class(name="Note", attributes={Property(name="text", type=StringType)})
    notes = BinaryAssociation(name="Party_Notes", ends={
        Property(name="party", type=party, multiplicity=Multiplicity(1, 1)),
        Property(name="notes", type=note, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        name="Registry", types={vehicle, car, truck, party, person, company, note}, associations={notes},
        generalizations={
            Generalization(general=vehicle, specific=car), Generalization(general=vehicle, specific=truck),
            Generalization(general=party, specific=person), Generalization(general=party, specific=company),
        })


@pytest.fixture
def result(run_generated_backend):
    return run_generated_backend(_model(), SCRIPT)


def test_an_abstract_class_has_no_write_route(result):
    # 405: the path exists for reading, the method is not offered; 404/405 for a path with no route at all.
    assert result["writes"]["create"] == 405
    assert result["writes"]["create_concrete_table"] == 405
    assert all(code in (404, 405) for code in result["writes"].values()), result["writes"]
    assert [route for route in result["routes"] if not route.startswith("GET ")] == []


def test_an_abstract_class_is_read_across_its_subclasses(result):
    assert result["parties"] == ["Acme", "Ann"]
    assert result["party"] == "Acme"
    assert result["party_notes"] == 1
    assert result["vehicles"] == ["AB-1", "CD-2"]
    assert result["vehicle_count"] == 2
    assert "GET /party/{party_id}/notes/" in result["routes"]


def test_the_subclasses_keep_their_write_routes(result):
    assert result["subclass_update"] == 200
