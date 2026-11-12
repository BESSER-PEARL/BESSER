"""The ``uniform_responses`` option: ``POST /x/`` and ``PUT /x/{x_id}/`` answer what ``GET /x/{x_id}/`` does.

Without it (the default, unchanged) ``GET /x/{x_id}/`` answers ``{"x": object}`` plus the ids per to-many
end, but ``POST`` and ``PUT`` answer that only when ``X`` has a to-many end, else the bare object: the
create of one class disagreed with its read, and adding a to-many end to a class changed what its create
answered. ``GET /x/paginated/?detailed=true`` had the same split per item. With the option, every class but
an association class answers the one shape on all four routes.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, Generalization, Multiplicity, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.registry import get_generator

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# Creates, updates and reads one instance of each class; prints {class: {"post", "put", "get"}}.
SHAPES_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    def obj(body, name):
        return body.get(name, body)

    answers = {}
    with TestClient(main_api.app) as client:
        def record(name, payload, changed):
            created = client.post(f"/{name}/", json=payload)
            assert created.status_code == 200, created.text
            item_id = obj(created.json(), name)["id"]
            updated = client.put(f"/{name}/{item_id}/", json={**payload, **changed})
            assert updated.status_code == 200, updated.text
            read = client.get(f"/{name}/{item_id}/")
            assert read.status_code == 200, read.text
            page = client.get(f"/{name}/paginated/", params={"detailed": True})
            assert page.status_code == 200, page.text
            answers[name] = {"post": created.json(), "put": updated.json(), "get": read.json(),
                             "paginated": page.json()["data"][-1]}
            return item_id

        record("plain", {"label": "a"}, {"label": "b"})
        owner = record("owner", {"name": "Ann"}, {"name": "Anna"})
        record("pet", {"name": "Rex", "owner": owner}, {"name": "Max"})
        course = record("course", {"title": "Maths"}, {"title": "Physics"})
        record("student", {"name": "Bo", "courses": [course]}, {"name": "Bob"})
        root = client.post("/node/", json={"code": "1"}).json()
        record("node", {"code": "11", "parent": obj(root, "node")["id"]}, {"code": "12"})
        record("animal", {"name": "Tom", "keeper": owner}, {"name": "Tim"})
        record("dog", {"name": "Fido", "breed": "lab", "keeper": owner}, {"breed": "pug"})
    print(json.dumps(answers))
""")


def _shapes_model() -> DomainModel:
    """A class of each kind: no association (Plain), only to-one ends (Pet; Animal; Dog, inherited),
    a one-to-many end (Owner), a many-to-many end (Student, Course), a self-association (Node)."""
    plain = Class(name="Plain", attributes={Property(name="label", type=StringType)})
    owner = Class(name="Owner", attributes={Property(name="name", type=StringType)})
    pet = Class(name="Pet", attributes={Property(name="name", type=StringType)})
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})
    node = Class(name="Node", attributes={Property(name="code", type=StringType)})
    animal = Class(name="Animal", attributes={Property(name="name", type=StringType)})
    dog = Class(name="Dog", attributes={Property(name="breed", type=StringType)})
    associations = {
        BinaryAssociation(name="owner_pets", ends={
            Property(name="owner", type=owner, multiplicity=Multiplicity(0, 1)),
            Property(name="pets", type=pet, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="enrolment", ends={
            Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
            Property(name="courses", type=course, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="node_tree", ends={
            Property(name="parent", type=node, multiplicity=Multiplicity(0, 1)),
            Property(name="children", type=node, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="keeper_animals", ends={
            Property(name="keeper", type=owner, multiplicity=Multiplicity(0, 1)),
            Property(name="animals", type=animal, multiplicity=Multiplicity(0, "*"))}),
    }
    return DomainModel(name="Shapes", types={plain, owner, pet, student, course, node, animal, dog},
                       associations=associations, generalizations={Generalization(general=animal, specific=dog)})


def _plain_model(with_tags: bool) -> DomainModel:
    """Plain alone, or Plain with a to-many end ``tags``."""
    plain = Class(name="Plain", attributes={Property(name="label", type=StringType)})
    if not with_tags:
        return DomainModel(name="Plain", types={plain})
    tag = Class(name="Tag", attributes={Property(name="text", type=StringType)})
    tags = BinaryAssociation(name="plain_tags", ends={
        Property(name="plain", type=plain, multiplicity=Multiplicity(0, 1)),
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Plain", types={plain, tag}, associations={tags})


PLAIN_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    with TestClient(main_api.app) as client:
        created = client.post("/plain/", json={"label": "a"}).json()
        item_id = created.get("plain", created)["id"]
        updated = client.put(f"/plain/{item_id}/", json={"label": "b"}).json()
    print(json.dumps({"post": created, "put": updated}))
""")


def _drive(model, folder, script, **options):
    BackendGenerator(model, output_dir=str(folder), **options).generate()
    result = subprocess.run([sys.executable, "-c", script], cwd=str(folder),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    return _drive(_shapes_model(), tmp_path_factory.mktemp("uniform"), SHAPES_SCRIPT, uniform_responses=True)


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    return _drive(_shapes_model(), tmp_path_factory.mktemp("historical"), SHAPES_SCRIPT)


CLASSES = ["plain", "owner", "pet", "course", "student", "node", "animal", "dog"]


@pytest.mark.parametrize("name", CLASSES)
def test_post_put_and_get_answer_one_shape(uniform, name):
    answers = uniform[name]
    get = answers["get"]
    assert set(get) - {name} == {key for key in get if key.endswith("_ids")}
    for route in ("post", "put", "paginated"):
        body = answers[route]
        assert set(body) == set(get), f"{route} /{name}/ answers {sorted(body)}, GET {sorted(get)}"
        assert set(body[name]) == set(get[name]), f"{route} /{name}/: object keys differ from GET"


@pytest.mark.parametrize("name", CLASSES)
def test_put_and_a_detailed_page_answer_the_object_get_reads_back(uniform, name):
    assert uniform[name]["put"] == uniform[name]["get"]
    assert uniform[name]["paginated"] == uniform[name]["get"]


def test_the_ids_per_to_many_end_keep_their_names(uniform):
    assert set(uniform["owner"]["post"]) == {"owner", "pets_ids", "animals_ids"}
    assert uniform["student"]["post"]["course_ids"] == [uniform["course"]["get"]["course"]["id"]]
    assert set(uniform["course"]["post"]) == {"course", "student_ids"}
    assert set(uniform["node"]["post"]) == {"node", "children_ids"}
    assert uniform["node"]["post"]["node"]["parent_id"] is not None
    assert uniform["dog"]["get"]["dog"]["breed"] == "pug"


def test_without_the_option_a_class_without_a_to_many_end_answers_the_bare_object(historical):
    """The default is the historical contract, which deployed apps and their clients read."""
    for name in ("plain", "pet", "animal", "dog"):
        for route in ("post", "put", "paginated"):
            assert "id" in historical[name][route] and name not in historical[name][route]
        assert name in historical[name]["get"]
    for name in ("owner", "course", "student", "node"):
        assert set(historical[name]["post"]) == set(historical[name]["get"])


@pytest.mark.parametrize("with_tags", [False, True])
def test_adding_a_to_many_end_keeps_where_post_and_put_put_the_object(tmp_path, with_tags):
    answers = _drive(_plain_model(with_tags), tmp_path, PLAIN_SCRIPT, uniform_responses=True)
    assert answers["post"]["plain"]["label"] == "a"
    assert answers["put"]["plain"]["label"] == "b"
    assert ("tags_ids" in answers["post"]) is with_tags


def test_without_the_option_adding_a_to_many_end_moves_the_object(tmp_path):
    """The defect the option fixes, pinned so the default is known not to have changed."""
    bare = _drive(_plain_model(False), tmp_path / "bare", PLAIN_SCRIPT)
    wrapped = _drive(_plain_model(True), tmp_path / "wrapped", PLAIN_SCRIPT)
    assert bare["post"]["label"] == "a"
    assert wrapped["post"]["plain"]["label"] == "a"


def test_off_is_the_output_of_a_generator_built_without_the_option(tmp_path):
    """``uniform_responses=False`` writes the same bytes as not passing it."""
    BackendGenerator(_shapes_model(), output_dir=str(tmp_path / "default")).generate()
    BackendGenerator(_shapes_model(), output_dir=str(tmp_path / "off"), uniform_responses=False).generate()
    for path in sorted((tmp_path / "default").rglob("*.py")):
        other = tmp_path / "off" / path.relative_to(tmp_path / "default")
        assert other.read_bytes() == path.read_bytes(), path.name


def test_association_class_routes_are_unchanged(tmp_path):
    """An association class answers the bare link on GET, POST and PUT alike, so the option leaves it,
    and its two ends' classes, which have a to-many end, answered the wrapped shape already."""
    trip = Class(name="Trip", attributes={Property(name="reference", type=StringType)})
    seat = Class(name="Seat", attributes={Property(name="label", type=StringType)})
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*"))})
    reservation = AssociationClass(name="Reservation", attributes={Property(name="note", type=StringType)},
                                   association=trip_seat)
    model = DomainModel(name="Trips", types={trip, seat, reservation}, associations={trip_seat})
    BackendGenerator(model, output_dir=str(tmp_path / "off")).generate()
    BackendGenerator(model, output_dir=str(tmp_path / "on"), uniform_responses=True).generate()
    for path in sorted((tmp_path / "off" / "routers").glob("*.py")):
        assert (tmp_path / "on" / "routers" / path.name).read_bytes() == path.read_bytes(), path.name


@pytest.mark.parametrize("uniform_responses", [False, True])
def test_web_app_passes_the_option_to_its_backend(monkeypatch, tmp_path, uniform_responses):
    from besser.generators.web_app import web_app_generator

    built = {}

    class Recorder:
        def __init__(self, model, **kwargs):
            built.update(kwargs)

        def generate(self):
            pass

    monkeypatch.setattr(web_app_generator, "BackendGenerator", Recorder)
    generator = web_app_generator.WebAppGenerator(_plain_model(False), None, output_dir=str(tmp_path),
                                                  uniform_responses=uniform_responses)
    generator._generate_backend(generator.env)
    assert built["uniform_responses"] is uniform_responses


@pytest.mark.parametrize("generator_id", ["backend", "web_app"])
def test_the_option_is_declared_off_by_default(generator_id):
    option = get_generator(generator_id).options["uniform_responses"]
    assert option["type"] == "boolean" and option["default"] is False
