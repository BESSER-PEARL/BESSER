"""The router must only read fields the Create schema actually carries.

Observed on a generated hotel app: POST returned 500 for 7 of
9 entities with ``AttributeError: 'PersonCreate' object has no attribute
'createdAt'``. The app booted, every file parsed, every import resolved and the
run was reported as a clean success -- nothing exercised a write.

pydantic_classes_template deliberately drops the surrogate ``id`` and the audit
timestamps from Create (the ORM stamps them); router.py.j2 dropped only ``id``
and read the timestamps off the payload anyway. Two templates, two different
beliefs about the same schema.
"""
import ast
import re

import pytest

from besser.BUML.metamodel.structural import (
    BooleanType, Class, DateTimeType, DomainModel, Property, StringType,
)
from besser.generators.backend import BackendGenerator

SERVER_OWNED = ("createdAt", "updatedAt")


@pytest.fixture
def generated(tmp_path):
    item = Class(name="Item", attributes={
        Property(name="name", type=StringType),
        Property(name="createdAt", type=DateTimeType),
        Property(name="updatedAt", type=DateTimeType),
        Property(name="isActive", type=BooleanType),
    })
    model = DomainModel(name="Shop", types={item}, associations=set())
    BackendGenerator(model=model, output_dir=str(tmp_path),
                     http_methods=["GET", "POST", "PUT"]).generate()
    return tmp_path


def _create_schema_fields(pydantic_src: str) -> set:
    """Field names declared on ItemCreate."""
    tree = ast.parse(pydantic_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ItemCreate":
            return {
                s.target.id for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
    return set()


def test_router_never_reads_a_field_the_create_schema_lacks(generated):
    router = (generated / "routers" / "item.py").read_text(encoding="utf-8")
    declared = _create_schema_fields(
        (generated / "pydantic_classes.py").read_text(encoding="utf-8"))
    read = set(re.findall(r"item_data\.(\w+)", router)) - {"model_dump", "dict"}
    missing = sorted(read - declared)
    assert not missing, (
        f"router reads {missing} off ItemCreate, which declares {sorted(declared)} "
        "-- every POST raises AttributeError and returns 500"
    )


@pytest.mark.parametrize("field", SERVER_OWNED)
def test_audit_timestamps_are_server_owned_on_both_sides(generated, field):
    pyd = (generated / "pydantic_classes.py").read_text(encoding="utf-8")
    router = (generated / "routers" / "item.py").read_text(encoding="utf-8")
    assert field not in _create_schema_fields(pyd), f"{field} must not be in Create"
    assert f"item_data.{field}" not in router, f"router must not read {field}"


def test_ordinary_fields_are_still_read(generated):
    """Guard against over-correcting into dropping real fields."""
    router = (generated / "routers" / "item.py").read_text(encoding="utf-8")
    assert "item_data.name" in router
    assert "item_data.isActive" in router
