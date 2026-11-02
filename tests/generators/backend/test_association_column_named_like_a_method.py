"""Many-to-many ends named like a ``ColumnCollection`` method (``items``, ``keys``).

The router read association-table columns as attributes, ``item_tags.c.items``,
which returns the bound ``ColumnCollection.items`` method instead of the column.
Every endpoint that touches the association table then failed with a 500: a
todo app with ``TodoItem.items`` / ``Tag.tags`` could not create a tag.
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
        status = {}
        tag = client.post("/tag/", json={"name": "urgent"})
        status["create tag"] = tag.status_code
        tag_id = tag.json()["tag"]["id"]
        item = client.post("/item/", json={"title": "a", "keys": [tag_id]})
        status["create item with tags"] = item.status_code
        item_id = item.json()["item"]["id"]
        status["get item"] = client.get(f"/item/{item_id}/").status_code
        status["list items"] = client.get("/item/", params={"detailed": True}).status_code
        status["get tag"] = client.get(f"/tag/{tag_id}/").status_code
        status["update item"] = client.put(f"/item/{item_id}/", json={"title": "b", "keys": []}).status_code
        status["link"] = client.post(f"/item/{item_id}/keys/{tag_id}/").status_code
        status["linked tags"] = client.get(f"/item/{item_id}/keys/").status_code
        linked = client.get(f"/item/{item_id}/keys/").json()
        status["unlink"] = client.delete(f"/item/{item_id}/keys/{tag_id}/").status_code
        status["delete tag"] = client.delete(f"/tag/{tag_id}/").status_code
        print(json.dumps({"status": status, "linked": linked}))
""")


@pytest.fixture
def result(tmp_path):
    item = Class(name="Item", attributes={Property(name="title", type=StringType)})
    tag = Class(name="Tag", attributes={Property(name="name", type=StringType)})
    tagged = BinaryAssociation(name="item_tags", ends={
        Property(name="items", type=item, multiplicity=Multiplicity(0, "*")),
        Property(name="keys", type=tag, multiplicity=Multiplicity(0, "*"))})
    BackendGenerator(model=DomainModel(name="Shop", types={item, tag}, associations={tagged}),
                     output_dir=str(tmp_path), http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    run = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(tmp_path),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_every_association_endpoint_succeeds(result):
    failed = {name: code for name, code in result["status"].items() if code >= 300}
    assert not failed, failed


def test_linking_reaches_the_association_table(result):
    assert "urgent" in json.dumps(result["linked"])

