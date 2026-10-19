"""A class associated with itself (Product.parent / Product.children, or a 1:1
Person.mentor / Person.mentee) generates mappers SQLAlchemy can configure: only
the end that owns the FK column carries ``remote_side``."""

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


def _mentoring_model() -> DomainModel:
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    mentor = Property(name="mentor", type=person, multiplicity=Multiplicity(0, 1))
    mentee = Property(name="mentee", type=person, multiplicity=Multiplicity(0, 1))
    mentoring = BinaryAssociation(name="Person_mentoring", ends={mentor, mentee})
    return DomainModel(name="School", types={person}, associations={mentoring})


def _generate_and_import(tmp_path, module_name: str, model: DomainModel = None):
    model = model or _hierarchy_model()
    SQLAlchemyGenerator(model=model, output_dir=str(tmp_path)).generate(dbms="sqlite")
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


def test_one_to_one_self_association_sets_remote_side_on_one_end(tmp_path):
    path, _ = _generate_and_import(tmp_path, "self_assoc_one_to_one_code", _mentoring_model())
    with open(path, encoding="utf-8") as f:
        code = f.read()
    # Both ends set remote_side before, which SQLAlchemy rejects as
    # "both of the same direction MANYTOONE".
    assert code.count("remote_side=[Person.id]") == 1


def test_one_to_one_self_association_mappers_configure_and_round_trip(tmp_path):
    _, module = _generate_and_import(tmp_path, "self_assoc_one_to_one", _mentoring_model())
    configure_mappers()

    engine = create_engine("sqlite://")
    module.Base.metadata.create_all(engine)
    with Session(engine) as session:
        teacher = module.Person(name="Ada")
        student = module.Person(name="Alan")
        # Whichever end holds the FK, setting one side must be readable from the other.
        teacher.mentee = student
        session.add_all([teacher, student])
        session.commit()
        assert session.get(module.Person, teacher.id).mentee.name == "Alan"
        assert session.get(module.Person, student.id).mentor.name == "Ada"
