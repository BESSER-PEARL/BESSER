"""A class associated with itself (Product.parent / Product.children) generates
mappers SQLAlchemy can configure: the many-to-one end carries ``remote_side``."""

import importlib.util
import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, configure_mappers

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _hierarchy_model() -> DomainModel:
    product = Class(name="Product", attributes={Property(name="code", type=StringType)})
    parent = Property(name="parent", type=product, multiplicity=Multiplicity(0, 1))
    children = Property(name="children", type=product, multiplicity=Multiplicity(0, "*"))
    hierarchy = BinaryAssociation(name="Product_hierarchy", ends={parent, children})
    return DomainModel(name="Catalog", types={product}, associations={hierarchy})


def _generate_and_import(tmp_path, module_name: str):
    SQLAlchemyGenerator(model=_hierarchy_model(), output_dir=str(tmp_path)).generate(dbms="sqlite")
    path = os.path.join(str(tmp_path), "sql_alchemy.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return path, module


def test_self_association_sets_remote_side(tmp_path):
    path, _ = _generate_and_import(tmp_path, "self_assoc_remote_side")
    with open(path, encoding="utf-8") as f:
        code = f.read()
    assert "remote_side=[Product.id]" in code


def test_self_association_mappers_configure_and_round_trip(tmp_path):
    _, module = _generate_and_import(tmp_path, "self_assoc_round_trip")
    configure_mappers()  # raised ArgumentError before: both ends read as one-to-many

    engine = create_engine("sqlite://")
    module.Base.metadata.create_all(engine)
    with Session(engine) as session:
        root = module.Product(code="22")
        leaf = module.Product(code="2204")
        leaf.parent = root
        session.add_all([root, leaf])
        session.commit()
        assert session.get(module.Product, leaf.id).parent.code == "22"
        assert [c.code for c in session.get(module.Product, root.id).children] == ["2204"]
