"""Creating and updating an entity of a self-associated class (Product.parent /
Product.children).

The create endpoint validated the ``children`` ids with a loop variable named
``db_product``, the same name as the entity being created, so
``POST /product/ {"children": [leaf]}`` made the leaf its own parent and
returned the leaf. A ``parent`` sent on create was silently dropped. The update
endpoint had the same clash and also ignored ``parent``.

The other shapes of an association from a class to itself are covered below:
many-to-many (both ends), one-to-many whose to-one end sorts first by name,
and one-to-one.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Generalization, Multiplicity, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.pydantic_classes import PydanticGenerator

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    def product(client, product_id):
        return client.get(f"/product/{product_id}/").json()["product"]

    with TestClient(main_api.app) as client:
        root = client.post("/product/", json={"code": "22"}).json()["product"]
        leaf = client.post("/product/", json={"code": "2204", "parent": root["id"]}).json()["product"]
        orphan = client.post("/product/", json={"code": "2205"}).json()["product"]
        group = client.post("/product/", json={"code": "220", "children": [orphan["id"]]}).json()
        leaf_after_create = product(client, leaf["id"])
        loose = client.post("/product/", json={"code": "2206"}).json()["product"]
        updated = client.put(f"/product/{root['id']}/",
                             json={"code": "22", "children": [leaf["id"], loose["id"]]}).json()
        moved = client.put(f"/product/{leaf['id']}/", json={"code": "2204", "parent": group["product"]["id"]})
        print(json.dumps({
            "root": root,
            "leaf": leaf_after_create,
            "group": group,
            "orphan": product(client, orphan["id"]),
            "loose": product(client, loose["id"]),
            "updated": updated,
            "moved": moved.json()["product"],
        }))
""")


def _hierarchy_model() -> DomainModel:
    product = Class(name="Product", attributes={Property(name="code", type=StringType)})
    parent = Property(name="parent", type=product, multiplicity=Multiplicity(0, 1))
    children = Property(name="children", type=product, multiplicity=Multiplicity(0, "*"))
    hierarchy = BinaryAssociation(name="Product_hierarchy", ends={parent, children})
    return DomainModel(name="Catalog", types={product}, associations={hierarchy})


@pytest.fixture
def created(tmp_path):
    BackendGenerator(model=_hierarchy_model(), output_dir=str(tmp_path),
                     http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    result = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(tmp_path),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_create_with_children_returns_the_new_entity(created):
    group = created["group"]["product"]
    assert group["code"] == "220"
    assert group["parent_id"] is None
    assert created["group"]["children_ids"] == [created["orphan"]["id"]]
    assert created["orphan"]["parent_id"] == group["id"]


def test_create_with_parent_sets_the_foreign_key(created):
    assert created["leaf"]["parent_id"] == created["root"]["id"]


def test_update_with_children_keeps_the_updated_entity(created):
    updated = created["updated"]["product"]
    assert updated["id"] == created["root"]["id"]
    assert updated["parent_id"] is None
    assert created["loose"]["parent_id"] == created["root"]["id"]


def test_update_with_parent_moves_the_entity(created):
    assert created["moved"]["parent_id"] == created["group"]["product"]["id"]


def _run_client(output_dir, model: DomainModel, script: str) -> dict:
    BackendGenerator(model=model, output_dir=str(output_dir),
                     http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    preamble = "import json\nimport main_api\nfrom fastapi.testclient import TestClient\n"
    result = subprocess.run([sys.executable, "-c", preamble + textwrap.dedent(script)], cwd=str(output_dir),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _prerequisites_model() -> DomainModel:
    course = Class(name="Course", attributes={Property(name="code", type=StringType)})
    prerequisites = Property(name="prerequisites", type=course, multiplicity=Multiplicity(0, "*"))
    required_by = Property(name="requiredBy", type=course, multiplicity=Multiplicity(0, "*"))
    association = BinaryAssociation(name="Course_prerequisites", ends={prerequisites, required_by})
    return DomainModel(name="Curriculum", types={course}, associations={association})


MANY_TO_MANY_SCRIPT = """
    with TestClient(main_api.app) as client:
        def course(course_id):
            return client.get(f"/course/{course_id}/").json()

        algebra = client.post("/course/", json={"code": "ALG"}).json()
        calculus = client.post("/course/", json={"code": "CAL", "prerequisites": [algebra["course"]["id"]]}).json()
        logic = client.post("/course/", json={"code": "LOG", "requiredBy": [calculus["course"]["id"]]}).json()
        alg, cal, log = (item["course"]["id"] for item in (algebra, calculus, logic))
        after_create = {"algebra": course(alg), "calculus": course(cal), "logic": course(log)}

        updated = client.put(f"/course/{alg}/", json={"code": "ALG", "prerequisites": [log], "requiredBy": []})
        after_update = {"algebra": course(alg), "calculus": course(cal), "logic": course(log)}

        linked = client.post(f"/course/{cal}/prerequisites/{alg}/").status_code
        listed = client.get(f"/course/{cal}/prerequisites/").json()
        reverse = client.get(f"/course/{log}/requiredBy/").json()
        unlinked = client.delete(f"/course/{cal}/prerequisites/{alg}/").status_code
        after_unlink = course(cal)
        detailed = client.get("/course/", params={"detailed": True}).json()
        page = client.get("/course/paginated/", params={"detailed": True}).json()
        deleted = client.delete(f"/course/{log}/").status_code
        print(json.dumps({
            "ids": [alg, cal, log], "calculus_created": calculus, "logic_created": logic,
            "after_create": after_create, "updated": updated.json(), "after_update": after_update,
            "linked": linked, "listed": listed, "reverse": reverse, "unlinked": unlinked,
            "after_unlink": after_unlink, "detailed": detailed, "page": page, "deleted": deleted,
            "after_delete": {"algebra": course(alg), "calculus": course(cal)},
        }))
"""


@pytest.fixture(scope="module")
def many_to_many(tmp_path_factory):
    return _run_client(tmp_path_factory.mktemp("self_many_to_many"), _prerequisites_model(), MANY_TO_MANY_SCRIPT)


def test_many_to_many_create_returns_the_new_entity_with_both_ends(many_to_many):
    alg, cal, log = many_to_many["ids"]
    assert many_to_many["calculus_created"] == {
        "course": {"id": cal, "code": "CAL"}, "prerequisites_ids": [alg], "requiredBy_ids": []}
    assert many_to_many["logic_created"] == {
        "course": {"id": log, "code": "LOG"}, "prerequisites_ids": [], "requiredBy_ids": [cal]}


def test_many_to_many_link_is_read_from_both_sides(many_to_many):
    alg, cal, log = many_to_many["ids"]
    after_create = many_to_many["after_create"]
    assert after_create["algebra"]["requiredBy_ids"] == [cal]
    assert sorted(after_create["calculus"]["prerequisites_ids"]) == [alg, log]
    assert after_create["logic"]["requiredBy_ids"] == [cal]


def test_many_to_many_update_reconciles_both_ends_of_the_updated_entity(many_to_many):
    alg, cal, log = many_to_many["ids"]
    assert many_to_many["updated"] == {
        "course": {"id": alg, "code": "ALG"}, "prerequisites_ids": [log], "requiredBy_ids": []}
    after_update = many_to_many["after_update"]
    assert after_update["calculus"]["prerequisites_ids"] == [log]
    assert sorted(after_update["logic"]["requiredBy_ids"]) == [alg, cal]


def test_many_to_many_link_routes_serve_each_end(many_to_many):
    alg, cal, log = many_to_many["ids"]
    assert many_to_many["linked"] == 200
    assert sorted(item["id"] for item in many_to_many["listed"]["prerequisites"]) == [alg, log]
    assert sorted(item["id"] for item in many_to_many["reverse"]["requiredBy"]) == [alg, cal]
    assert many_to_many["unlinked"] == 200
    assert many_to_many["after_unlink"]["prerequisites_ids"] == [log]


def test_many_to_many_lists_carry_both_ends(many_to_many):
    alg, cal, log = many_to_many["ids"]
    logic = next(item for item in many_to_many["detailed"] if item["id"] == log)
    assert logic["prerequisites"] == []
    assert sorted(item["id"] for item in logic["requiredBy"]) == [alg, cal]
    page_item = next(item for item in many_to_many["page"]["data"] if item["course"]["id"] == log)
    assert sorted(page_item["requiredBy_ids"]) == [alg, cal]


def test_deleting_a_many_to_many_member_removes_its_links(many_to_many):
    assert many_to_many["deleted"] == 200
    assert many_to_many["after_delete"]["algebra"]["prerequisites_ids"] == []
    assert many_to_many["after_delete"]["calculus"]["prerequisites_ids"] == []


def test_many_to_many_create_schema_names_a_field_per_end(tmp_path):
    PydanticGenerator(model=_prerequisites_model(), backend=True, output_dir=str(tmp_path)).generate()
    schema = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")
    assert "prerequisites: Optional[List[int]] = None" in schema
    assert "requiredBy: Optional[List[int]] = None" in schema


ONE_TO_MANY_SCRIPT = """
    with TestClient(main_api.app) as client:
        lead = client.post("/employee/", json={"name": "Lea"}).json()["employee"]
        report = client.post("/employee/", json={"name": "Ray", "manager": lead["id"]}).json()["employee"]
        loose = client.post("/employee/", json={"name": "Lou"}).json()["employee"]
        head = client.post("/employee/", json={"name": "Hal", "subordinates": [lead["id"], loose["id"]]}).json()
        bulk = client.post("/employee/bulk/", json=[{"name": "Bea", "manager": head["employee"]["id"]}]).json()
        print(json.dumps({
            "report": report, "loose": loose, "head": head,
            "detailed": client.get("/employee/", params={"detailed": True}).json(),
            "lead": client.get(f"/employee/{lead['id']}/").json(),
            "bulk": client.get(f"/employee/{bulk['created_ids'][0]}/").json(),
        }))
"""


def test_one_to_many_handles_both_ends_whatever_their_names(tmp_path, employee_self_assoc_model):
    """manager sorts before subordinates: the list end was dropped on create and never returned."""
    created = _run_client(tmp_path, employee_self_assoc_model, ONE_TO_MANY_SCRIPT)
    head, lead = created["head"]["employee"], created["lead"]["employee"]
    assert head["name"] == "Hal"
    assert sorted(created["head"]["subordinates_ids"]) == sorted([lead["id"], created["loose"]["id"]])
    assert lead["manager_id"] == head["id"]
    assert created["lead"]["subordinates_ids"] == [created["report"]["id"]]
    assert created["report"]["manager_id"] == lead["id"]
    assert created["bulk"]["employee"]["manager_id"] == head["id"]
    listed = next(item for item in created["detailed"] if item["id"] == lead["id"])
    assert listed["manager"]["name"] == "Hal"
    assert [item["name"] for item in listed["subordinates"]] == ["Ray"]


def _mentoring_model() -> DomainModel:
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    mentor = Property(name="mentor", type=person, multiplicity=Multiplicity(0, 1))
    mentee = Property(name="mentee", type=person, multiplicity=Multiplicity(0, 1))
    mentoring = BinaryAssociation(name="Person_mentoring", ends={mentor, mentee})
    return DomainModel(name="School", types={person}, associations={mentoring})


ONE_TO_ONE_SCRIPT = """
    with TestClient(main_api.app) as client:
        ada = client.post("/person/", json={"name": "Ada"}).json()
        alan = client.post("/person/", json={"name": "Alan", "mentor": ada["id"]}).json()
        chained = client.get("/person/", params={"detailed": True}).json()
        # The end without a column, on create and on update, read back from the other record.
        grace = client.post("/person/", json={"name": "Grace", "mentee": ada["id"]}).json()
        ada_mentored = client.get(f"/person/{ada['id']}/").json()["person"]
        edsger = client.post("/person/", json={"name": "Edsger"}).json()
        client.put(f"/person/{grace['id']}/", json={"name": "Grace", "mentee": edsger["id"]})
        moved = {key: client.get(f"/person/{person['id']}/").json()["person"]["mentor_id"]
                 for key, person in (("ada", ada), ("edsger", edsger))}
        with_mentee = client.get("/person/", params={"detailed": True}).json()
        client.put(f"/person/{ada['id']}/", json={"name": "Ada", "mentor": alan["id"]})
        circular = client.get("/person/", params={"detailed": True})
        print(json.dumps({"ada": ada, "alan": alan, "grace": grace, "edsger": edsger, "chained": chained,
                          "ada_mentored": ada_mentored, "moved": moved, "with_mentee": with_mentee,
                          "circular_status": circular.status_code, "circular": circular.json()}))
"""


@pytest.fixture(scope="module")
def one_to_one(tmp_path_factory):
    return _run_client(tmp_path_factory.mktemp("self_one_to_one"), _mentoring_model(), ONE_TO_ONE_SCRIPT)


def test_one_to_one_writes_the_foreign_key_end(one_to_one):
    """The create read the end that holds no column (``mentee_id``) and answered 500."""
    assert one_to_one["alan"]["mentor_id"] == one_to_one["ada"]["id"]
    alan = next(item for item in one_to_one["chained"] if item["id"] == one_to_one["alan"]["id"])
    assert alan["mentor"] == {"id": one_to_one["ada"]["id"], "name": "Ada", "mentor_id": None}
    ada = next(item for item in one_to_one["chained"] if item["id"] == one_to_one["ada"]["id"])
    assert ada["mentee"]["id"] == one_to_one["alan"]["id"]


def test_one_to_one_end_without_the_column_is_stored_on_the_other_record(one_to_one):
    grace = one_to_one["grace"]["id"]
    assert one_to_one["ada_mentored"]["mentor_id"] == grace
    # Moving the mentee frees the previous one: the column is unique.
    assert one_to_one["moved"] == {"ada": None, "edsger": grace}
    listed = next(item for item in one_to_one["with_mentee"] if item["id"] == grace)
    assert listed["mentee"]["id"] == one_to_one["edsger"]["id"]


def test_one_to_one_records_pointing_at_each_other_can_be_listed(one_to_one):
    assert one_to_one["circular_status"] == 200
    ada = next(item for item in one_to_one["circular"] if item["id"] == one_to_one["ada"]["id"])
    assert ada["mentor"]["id"] == one_to_one["alan"]["id"]


def _cover_model() -> DomainModel:
    """A many-to-many to itself whose end has the name of the class."""
    staff = Class(name="Staff", attributes={Property(name="name", type=StringType)})
    association = BinaryAssociation(name="Cover", ends={
        Property(name="staff", type=staff, multiplicity=Multiplicity(0, "*")),
        Property(name="covers", type=staff, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Rota", types={staff}, associations={association})


END_NAMED_LIKE_THE_CLASS_SCRIPT = """
    with TestClient(main_api.app) as client:
        ann = client.post("/staff/", json={"name": "Ann"}).json()["staff"]["id"]
        bob = client.post("/staff/", json={"name": "Bob"}).json()["staff"]["id"]
        created = client.post("/staff/", json={"name": "Cy", "staff": [ann], "covers": [bob]}).json()
        cy = created["staff"]["id"]
        updated = client.put(f"/staff/{cy}/", json={"name": "Cy", "staff": [bob], "covers": []}).json()
        linked = client.post(f"/staff/{cy}/staff/{ann}/").status_code
        print(json.dumps({
            "ids": [ann, bob, cy], "created": created, "updated": updated, "linked": linked,
            "fetched": client.get(f"/staff/{cy}/").json(),
            "names": [item["name"] for item in client.get("/staff/").json()],
        }))
"""


def test_an_end_named_like_the_class_does_not_replace_the_entity(tmp_path):
    """``db_<end>`` was ``db_<class>``: the create returned a related record and linked it to itself."""
    result = _run_client(tmp_path, _cover_model(), END_NAMED_LIKE_THE_CLASS_SCRIPT)
    ann, bob, cy = result["ids"]
    assert result["created"] == {"staff": {"id": cy, "name": "Cy"}, "covers_ids": [bob], "staff_ids": [ann]}
    assert result["updated"] == {"staff": {"id": cy, "name": "Cy"}, "covers_ids": [], "staff_ids": [bob]}
    assert result["linked"] == 200
    assert sorted(result["fetched"]["staff_ids"]) == [ann, bob]
    assert result["names"] == ["Ann", "Bob", "Cy"]


def _engineer_model() -> DomainModel:
    """A subclass of a class that has a one-to-many to itself."""
    employee = Class(name="Employee", attributes={Property(name="name", type=StringType)})
    engineer = Class(name="Engineer", attributes={Property(name="skill", type=StringType)})
    manages = BinaryAssociation(name="Manages", ends={
        Property(name="manager", type=employee, multiplicity=Multiplicity(0, 1)),
        Property(name="subordinates", type=employee, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Org", types={employee, engineer}, associations={manages},
                       generalizations={Generalization(general=employee, specific=engineer)})


INHERITED_SCRIPT = """
    with TestClient(main_api.app) as client:
        lea = client.post("/employee/", json={"name": "Lea"}).json()["employee"]["id"]
        ray = client.post("/employee/", json={"name": "Ray"}).json()["employee"]["id"]
        lead = client.post("/engineer/", json={"name": "Eve", "skill": "ops", "subordinates": [ray]}).json()
        junior = client.post("/engineer/", json={"name": "Jo", "skill": "qa", "manager": lea}).json()
        print(json.dumps({
            "lea": lea, "ray": ray, "lead": lead, "junior": junior,
            "ray_after": client.get(f"/employee/{ray}/").json()["employee"],
        }))
"""


def test_a_subclass_handles_both_ends_of_an_inherited_one_to_many(tmp_path):
    """The list end was accepted by the subclass's create and ignored."""
    result = _run_client(tmp_path, _engineer_model(), INHERITED_SCRIPT)
    lead = result["lead"]
    assert lead["subordinates_ids"] == [result["ray"]]
    assert result["ray_after"]["manager_id"] == lead["engineer"]["id"]
    assert result["junior"]["engineer"]["manager_id"] == result["lea"]


def _tutoring_model() -> DomainModel:
    """A subclass of a class with a one-to-many and a one-to-one to itself."""
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    student = Class(name="Student", attributes={Property(name="level", type=StringType)})
    tutoring = BinaryAssociation(name="Tutoring", ends={
        Property(name="tutor", type=person, multiplicity=Multiplicity(0, 1)),
        Property(name="tutees", type=person, multiplicity=Multiplicity(0, "*")),
    })
    pairing = BinaryAssociation(name="Pairing", ends={
        Property(name="partner", type=person, multiplicity=Multiplicity(0, 1)),
        Property(name="partner_of", type=person, multiplicity=Multiplicity(0, 1)),
    })
    return DomainModel(name="Tutoring", types={person, student}, associations={tutoring, pairing},
                       generalizations={Generalization(general=person, specific=student)})


INHERITED_DETAILED_SCRIPT = """
    with TestClient(main_api.app, raise_server_exceptions=False) as client:
        ann = client.post("/person/", json={"name": "Ann"}).json()["person"]["id"]
        bo = client.post("/student/", json={"name": "Bo", "level": "2", "tutor": ann}).json()["student"]["id"]
        cy = client.post("/student/", json={"name": "Cy", "level": "1", "tutor": bo}).json()["student"]["id"]
        client.put(f"/person/{ann}/", json={"name": "Ann", "partner": bo})
        people = client.get("/person/", params={"detailed": True}).json()
        students = client.get("/student/", params={"detailed": True}).json()
        # Two students paired with each other
        client.put(f"/person/{bo}/", json={"name": "Bo", "partner_of": cy})
        client.put(f"/person/{cy}/", json={"name": "Cy", "partner_of": bo})
        paired = client.get("/student/", params={"detailed": True})
        print(json.dumps({"bo": bo, "people": people, "students": students,
                          "paired": [paired.status_code, paired.json()]}))
"""


def test_a_subclass_lists_both_ends_of_inherited_associations_to_itself(tmp_path):
    """The detailed list of the subclass lost the to-one end of each inherited association to the class."""
    result = _run_client(tmp_path, _tutoring_model(), INHERITED_DETAILED_SCRIPT)
    as_person = next(item for item in result["people"] if item["id"] == result["bo"])
    as_student = next(item for item in result["students"] if item["id"] == result["bo"])
    assert set(as_person) <= set(as_student)
    for end in ("tutor", "partner", "partner_of"):
        assert as_student[end] == as_person[end], end
    assert as_student["tutor"]["name"] == "Ann"
    assert [item["name"] for item in as_student["tutees"]] == ["Cy"]
    assert as_student["partner_of"]["name"] == "Ann"
    assert as_student["partner"] is None
    # A related record of the parent class is serialized by its columns, so a cycle ends there.
    status, paired = result["paired"]
    assert status == 200
    assert {item["name"]: item["partner_of"]["name"] for item in paired} == {"Bo": "Cy", "Cy": "Bo"}


def _folder_model() -> DomainModel:
    """A composition from a class to itself: a folder is deleted with the folders in it."""
    folder = Class(name="Folder", attributes={Property(name="name", type=StringType)})
    nesting = BinaryAssociation(name="Nesting", ends={
        Property(name="parent", type=folder, multiplicity=Multiplicity(0, 1), is_composite=True),
        Property(name="children", type=folder, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Tree", types={folder}, associations={nesting})


CYCLE_SCRIPT = """
    with TestClient(main_api.app, raise_server_exceptions=False) as client:
        def folder(name, parent=None):
            return client.post("/folder/", json={"name": name, "parent": parent}).json()["folder"]["id"]

        root = folder("root")
        inner = folder("inner", folder("middle", root))
        plain = folder("plain")
        kept = folder("kept", plain)
        client.put(f"/folder/{root}/", json={"name": "root", "parent": inner})
        cyclic = client.delete(f"/folder/{root}/")
        tree = client.delete(f"/folder/{plain}/")
        print(json.dumps({
            "cyclic": [cyclic.status_code, cyclic.json()["detail"]], "tree": tree.status_code,
            "left": sorted(item["name"] for item in client.get("/folder/").json()),
        }))
"""


def test_deleting_a_record_that_contains_itself_is_refused_readably(tmp_path):
    """The delete followed the parts round the cycle and answered 500."""
    result = _run_client(tmp_path, _folder_model(), CYCLE_SCRIPT)
    status, detail = result["cyclic"]
    assert status == 409
    assert detail.startswith("Cannot delete Folder 1: ") and detail.endswith("Folder 1 is one of its own parts")
    # A tree without a cycle is deleted with the folders in it.
    assert result["tree"] == 200
    assert result["left"] == ["inner", "middle", "root"]
