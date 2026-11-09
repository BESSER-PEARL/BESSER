"""Many-to-many ends named like a SQLAlchemy built-in (``items``, ``t``).

The router read association-table columns as attributes. ``item_tags.c.items``
returns the bound ``ColumnCollection.items`` method instead of the column, and
on a fetched row ``assoc.t`` returns ``Row.t`` (the whole tuple). Every endpoint
that touches the association table then failed with a 500: a todo app with
``TodoItem.items`` / ``Tag.tags`` could not create a tag.
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
        item = client.post("/item/", json={"title": "a", "t": [tag_id]})
        status["create item with tags"] = item.status_code
        item_id = item.json()["item"]["id"]
        status["get item"] = client.get(f"/item/{item_id}/").status_code
        status["list items"] = client.get("/item/", params={"detailed": True}).status_code
        status["page items"] = client.get("/item/paginated/", params={"detailed": True}).status_code
        status["page tags"] = client.get("/tag/paginated/", params={"detailed": True}).status_code
        status["get tag"] = client.get(f"/tag/{tag_id}/").status_code
        status["update item"] = client.put(f"/item/{item_id}/", json={"title": "b", "t": []}).status_code
        status["update tag"] = client.put(f"/tag/{tag_id}/", json={"name": "soon", "items": [item_id]}).status_code
        linked = client.get(f"/item/{item_id}/t/")
        status["linked tags"] = linked.status_code
        status["unlink from item"] = client.delete(f"/item/{item_id}/t/{tag_id}/").status_code
        status["link from tag"] = client.post(f"/tag/{tag_id}/items/{item_id}/").status_code
        status["linked items"] = client.get(f"/tag/{tag_id}/items/").status_code
        status["unlink from tag"] = client.delete(f"/tag/{tag_id}/items/{item_id}/").status_code
        status["delete tag"] = client.delete(f"/tag/{tag_id}/").status_code
        print(json.dumps({"status": status, "linked": linked.json()}))
""")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("backend_method_named_ends")
    item = Class(name="Item", attributes={Property(name="title", type=StringType)})
    tag = Class(name="Tag", attributes={Property(name="name", type=StringType)})
    tagged = BinaryAssociation(name="item_tags", ends={
        Property(name="items", type=item, multiplicity=Multiplicity(0, "*")),
        Property(name="t", type=tag, multiplicity=Multiplicity(0, "*"))})
    BackendGenerator(model=DomainModel(name="Shop", types={item, tag}, associations={tagged}),
                     output_dir=str(out), http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    run = subprocess.run([sys.executable, "-c", CLIENT_SCRIPT], cwd=str(out),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_every_association_endpoint_succeeds(result):
    failed = {name: code for name, code in result["status"].items() if code >= 300}
    assert not failed, failed


def test_the_tag_set_by_the_tag_side_update_is_linked(result):
    assert "soon" in json.dumps(result["linked"])
