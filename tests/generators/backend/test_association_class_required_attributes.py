"""Links of an association class whose attributes are required.

The relation field accepts a bare id next to the ``{target, <attributes>}``
payload. With a required attribute the id alone cannot create a link: the
request failed in the database and the client got the driver's message (409
"NOT NULL constraint failed: ..."). A bare id that names a link which already
exists changes nothing and stays valid.
"""
from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator

SCRIPT = """
    course = client.post("/course/", json={"title": "Algebra"}).json()["course"]["id"]
    club = client.post("/club/", json={"title": "Chess"}).json()["club"]["id"]
    bare = client.post("/student/", json={"name": "Ann", "courses": [course], "clubs": []})
    full = client.post("/student/", json={"name": "Bob", "courses": [{"target": course, "grade": 5}], "clubs": [club]})
    student = full.json()["student"]["id"]
    other = client.post("/course/", json={"title": "Logic"}).json()["course"]["id"]
    no_body = client.post(f"/student/{student}/courses/{other}/")
    with_body = client.post(f"/student/{student}/courses/{other}/", json={"grade": 4})
    third = client.post("/course/", json={"title": "Sets"}).json()["course"]["id"]
    kept = client.put(f"/student/{student}/", json={"name": "Bob", "courses": [course, other], "clubs": [club]})
    added = client.put(f"/student/{student}/", json={"name": "Bob", "courses": [course, other, third], "clubs": [club]})
    grades = sorted(link["grade"] for link in client.get(f"/student/{student}/courses/").json()["courses_links"])
    print(json.dumps({
        "bare": [bare.status_code, bare.json()], "full": [full.status_code, full.json()],
        "no_body": [no_body.status_code, no_body.json()], "with_body": with_body.status_code,
        "kept": [kept.status_code, kept.json()], "added": [added.status_code, added.json()], "grades": grades,
        "students": [item["name"] for item in client.get("/student/").json()],
    }))
"""


def _model() -> DomainModel:
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})
    club = Class(name="Club", attributes={Property(name="title", type=StringType)})
    enrolment = BinaryAssociation(name="Enrolment", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*")),
    })
    membership = BinaryAssociation(name="Membership", ends={
        Property(name="members", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="clubs", type=club, multiplicity=Multiplicity(0, "*")),
    })
    enrolled = AssociationClass(name="Enrolled", association=enrolment, attributes={
        Property(name="grade", type=IntegerType),
        Property(name="remark", type=StringType, is_optional=True),
    })
    club_role = AssociationClass(name="ClubRole", association=membership, attributes={
        Property(name="role", type=StringType, is_optional=True),
        Property(name="level", type=IntegerType, default_value=1),
    })
    return DomainModel(name="School", types={student, course, club, enrolled, club_role},
                       associations={enrolment, membership})


def test_a_bare_id_is_refused_when_the_link_needs_an_attribute(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)

    status, body = result["bare"]
    assert status == 422, body
    assert body["detail"] == "Enrolled requires grade"
    assert result["students"] == ["Bob"]

    # A link that needs no attribute still travels as its id.
    status, body = result["full"]
    assert status == 200, body
    assert body["courses_ids"] == [1] and body["clubs_ids"] == [1]


def test_the_link_route_names_the_missing_attribute(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)
    status, body = result["no_body"]
    assert status == 422
    assert body["detail"] == "Enrolled requires grade"
    assert result["with_body"] == 200


def test_a_bare_id_keeps_a_link_that_exists(run_generated_backend):
    """Sending back the ids a GET returned is a valid update; only a new link needs its attributes."""
    result = run_generated_backend(_model(), SCRIPT)
    status, body = result["kept"]
    assert status == 200, body
    assert sorted(body["courses_ids"]) == [1, 2]
    status, body = result["added"]
    assert status == 422 and body["detail"] == "Enrolled requires grade"
    assert result["grades"] == [4, 5]


def test_the_create_schema_accepts_a_bare_id_for_every_link(tmp_path):
    PydanticGenerator(model=_model(), backend=True, output_dir=str(tmp_path)).generate()
    schema = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")
    assert 'courses: Optional[List[Union["EnrolledLinkCreate", int]]] = None' in schema
    assert 'clubs: Optional[List[Union["ClubRoleLinkCreate", int]]] = None' in schema
