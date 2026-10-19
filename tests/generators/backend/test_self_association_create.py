"""Creating an entity of a self-associated class (Product.parent / Product.children).

The create endpoint validated the ``children`` ids with a loop variable named
``db_product``, the same name as the entity being created, so
``POST /product/ {"children": [leaf]}`` made the leaf its own parent and
returned the leaf. A ``parent`` sent on create was silently dropped.
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

    def product(client, product_id):
        return client.get(f"/product/{product_id}/").json()["product"]

    with TestClient(main_api.app) as client:
        root = client.post("/product/", json={"code": "22"}).json()["product"]
        leaf = client.post("/product/", json={"code": "2204", "parent": root["id"]}).json()["product"]
        orphan = client.post("/product/", json={"code": "2205"}).json()["product"]
        group = client.post("/product/", json={"code": "220", "children": [orphan["id"]]}).json()
        print(json.dumps({
            "root": root,
            "leaf": product(client, leaf["id"]),
            "group": group,
            "orphan": product(client, orphan["id"]),
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
