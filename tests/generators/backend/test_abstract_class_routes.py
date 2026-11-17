"""An abstract class has no instances of its own: no route of its router creates
one. Its other routes work on the records of the subclasses.

    Vehicle (abstract)  <|-- Car, Truck          no association: one table per subclass
    Party (abstract)    <|-- Person, Company     Party 1..1 -- 0..* Note: joined tables
                                                 Party 0..* -- 0..* Group
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
    extra = client.post("/company/", json={"name": "Bolt", "staff": 3}).json()["company"]
    group = client.post("/group/", json={"label": "vip"}).json()["group"]["id"]
    note = client.post("/note/", json={"text": "call back", "party": person["id"]}).json()

    def status(method, url, **kwargs):
        return client.request(method, url, **kwargs).status_code

    creates = {
        "create": status("POST", "/party/", json={"name": "Nobody"}),
        "bulk_create": status("POST", "/party/bulk/", json=[{"name": "Nobody"}]),
        "create_concrete_table": status("POST", "/vehicle/", json={"plate": "ZZ-9"}),
    }
    reads = {
        "parties": sorted(item["name"] for item in client.get("/party/").json()),
        "party": client.get(f"/party/{company['id']}/").json()["party"]["name"],
        "party_notes": client.get(f"/party/{person['id']}/notes/").json()["notes_count"],
        "vehicles": sorted(item["plate"] for item in client.get("/vehicle/").json()),
        "vehicle_count": client.get("/vehicle/count/").json()["count"],
    }
    through_parent = {
        "update": status("PUT", f"/party/{person['id']}/", json={"name": "Ann B"}),
        "link": status("POST", f"/party/{person['id']}/groups/{group}/"),
        "linked": client.get(f"/party/{person['id']}/").json()["group_ids"],
        "unlink": status("DELETE", f"/party/{person['id']}/groups/{group}/"),
        "delete": status("DELETE", f"/party/{extra['id']}/"),
        "bulk_delete": status("DELETE", "/party/bulk/", json=[company["id"]]),
    }
    print(json.dumps({
        "creates": creates, "reads": reads, "through_parent": through_parent,
        "after": sorted(item["name"] for item in client.get("/party/").json()),
        "person": client.get(f"/person/{person['id']}/").json()["person"],
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
    group = Class(name="Group", attributes={Property(name="label", type=StringType)})
    notes = BinaryAssociation(name="Party_Notes", ends={
        Property(name="party", type=party, multiplicity=Multiplicity(1, 1)),
        Property(name="notes", type=note, multiplicity=Multiplicity(0, "*")),
    })
    groups = BinaryAssociation(name="Party_Groups", ends={
        Property(name="members", type=party, multiplicity=Multiplicity(0, "*")),
        Property(name="groups", type=group, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        name="Registry", types={vehicle, car, truck, party, person, company, note, group},
        associations={notes, groups},
        generalizations={
            Generalization(general=vehicle, specific=car), Generalization(general=vehicle, specific=truck),
            Generalization(general=party, specific=person), Generalization(general=party, specific=company),
        })


@pytest.fixture
def result(run_generated_backend):
    return run_generated_backend(_model(), SCRIPT)


def test_no_route_creates_a_record_of_the_abstract_class(result):
    # 405: the path exists for other methods; 404/405 for the bulk path, which only deletes.
    assert result["creates"]["create"] == 405
    assert result["creates"]["create_concrete_table"] == 405
    assert result["creates"]["bulk_create"] in (404, 405)
    assert result["reads"]["parties"] == ["Acme", "Ann", "Bolt"]


def test_an_abstract_class_is_read_across_its_subclasses(result):
    reads = result["reads"]
    assert reads["party"] == "Acme"
    assert reads["party_notes"] == 1
    assert reads["vehicles"] == ["AB-1", "CD-2"]
    assert reads["vehicle_count"] == 2


def test_subclass_records_are_changed_linked_and_deleted_through_the_parent(result):
    through_parent = result["through_parent"]
    assert through_parent["update"] == 200
    assert result["person"]["name"] == "Ann B" and result["person"]["age"] == 41
    assert (through_parent["link"], through_parent["unlink"]) == (200, 200)
    assert through_parent["linked"] == [1]
    assert through_parent["delete"] == 200
    # The bulk delete route is there (it refuses: a Note requires its party).
    assert through_parent["bulk_delete"] == 409
    assert result["after"] == ["Acme", "Ann B"]
