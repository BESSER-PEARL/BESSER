"""Two many-to-many associations from one class to the same class (Book.authors
and Book.reviewers, both to Person).

The router named the id list of a many-to-many after its target class, so both
associations wrote ``person_ids``: the response carried one of the two lists
and no key told which.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)
from besser.generators.backend import BackendGenerator

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

CLIENT_SCRIPT = textwrap.dedent("""
    import json
    import main_api
    from fastapi.testclient import TestClient

    with TestClient(main_api.app) as client:
        ann = client.post("/person/", json={"name": "Ann"}).json()["person"]["id"]
        bob = client.post("/person/", json={"name": "Bob"}).json()["person"]["id"]
        created = client.post("/book/", json={"title": "Maps", "authors": [ann], "reviewers": [bob]}).json()
        book = created["book"]["id"]
        updated = client.put(f"/book/{book}/", json={"title": "Maps", "authors": [bob], "reviewers": [ann, bob]}).json()
        print(json.dumps({
            "ann": ann, "bob": bob, "created": created, "updated": updated,
            "fetched": client.get(f"/book/{book}/").json(),
            "page": client.get("/book/paginated/", params={"detailed": True}).json()["data"][0],
            "person": client.get(f"/person/{ann}/").json(),
        }))
""")


def _model() -> DomainModel:
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    authorship = BinaryAssociation(name="Authorship", ends={
        Property(name="authors", type=person, multiplicity=Multiplicity(0, "*")),
        Property(name="written", type=book, multiplicity=Multiplicity(0, "*")),
    })
    reviewing = BinaryAssociation(name="Reviewing", ends={
        Property(name="reviewers", type=person, multiplicity=Multiplicity(0, "*")),
        Property(name="reviewed", type=book, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Library", types={book, person}, associations={authorship, reviewing})


@pytest.fixture(scope="module")
def responses(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("two_many_to_many")
    BackendGenerator(model=_model(), output_dir=str(output_dir),
                     http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    result = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(output_dir),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_create_returns_one_id_list_per_association(responses):
    assert responses["created"]["authors_ids"] == [responses["ann"]]
    assert responses["created"]["reviewers_ids"] == [responses["bob"]]


def test_get_and_page_return_one_id_list_per_association(responses):
    ann, bob = responses["ann"], responses["bob"]
    for body in (responses["updated"], responses["fetched"], responses["page"]):
        assert body["authors_ids"] == [bob]
        assert sorted(body["reviewers_ids"]) == [ann, bob]


def test_the_other_class_tells_its_two_ends_apart_too(responses):
    assert responses["person"]["written_ids"] == []
    assert len(responses["person"]["reviewed_ids"]) == 1
