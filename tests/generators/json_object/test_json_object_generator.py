import os
import json
import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType, IntegerType,
    BinaryAssociation, Multiplicity
)
from besser.BUML.metamodel.object import (
    ObjectModel, Object, DataValue, AttributeLink, LinkEnd, Link
)
from besser.generators.json import JSONObjectGenerator


@pytest.fixture
def object_model():
    """Create a domain model and a matching object model."""
    # Domain model
    name_prop = Property(name="name", type=StringType)
    age_prop = Property(name="age", type=IntegerType)
    person_cls = Class(name="Person", attributes={name_prop, age_prop})

    title_prop = Property(name="title", type=StringType)
    book_cls = Class(name="Book", attributes={title_prop})

    person_end = Property(name="owner", type=person_cls, multiplicity=Multiplicity(1, 1))
    book_end = Property(name="books", type=book_cls, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="Owns", ends={person_end, book_end})

    domain_model = DomainModel(
        name="LibraryDomain",
        types={person_cls, book_cls},
        associations={assoc},
    )

    # Object model (instances)
    alice_name_val = DataValue(classifier=StringType, value="Alice", name="alice_name")
    alice_age_val = DataValue(classifier=IntegerType, value=30, name="alice_age")
    alice = Object(
        name="alice",
        classifier=person_cls,
        slots=[
            AttributeLink(value=alice_name_val, attribute=name_prop),
            AttributeLink(value=alice_age_val, attribute=age_prop),
        ],
    )

    book_title_val = DataValue(classifier=StringType, value="Python 101", name="book1_title")
    book1 = Object(
        name="book1",
        classifier=book_cls,
        slots=[
            AttributeLink(value=book_title_val, attribute=title_prop),
        ],
    )

    link_end_person = LinkEnd(name="owner", association_end=person_end, object=alice)
    link_end_book = LinkEnd(name="books", association_end=book_end, object=book1)
    link = Link(name="owns_link", association=assoc, connections=[link_end_person, link_end_book])

    obj_model = ObjectModel(
        name="LibraryObjects",
        objects={alice, book1},
    )

    return obj_model


def test_json_object_generator_instantiation(object_model):
    """Test that the JSONObjectGenerator can be instantiated."""
    generator = JSONObjectGenerator(model=object_model)
    assert generator is not None


def test_json_object_generator_rejects_wrong_model():
    """Test that non-ObjectModel input raises TypeError."""
    with pytest.raises(TypeError, match="ObjectModel"):
        JSONObjectGenerator(model="not_an_object_model")


def test_json_object_generator_generate(object_model, tmpdir):
    """Test that generate() runs without errors and produces a JSON file."""
    output_dir = tmpdir.mkdir("output")
    generator = JSONObjectGenerator(
        model=object_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    # Look for .json files in the output directory
    json_files = [f for f in os.listdir(str(output_dir)) if f.endswith(".json")]
    assert len(json_files) > 0, "JSONObjectGenerator should produce a .json file"


def test_json_object_generator_valid_json(object_model, tmpdir):
    """Test that the generated output is valid JSON."""
    output_dir = tmpdir.mkdir("output")
    generator = JSONObjectGenerator(
        model=object_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    json_files = [f for f in os.listdir(str(output_dir)) if f.endswith(".json")]
    assert len(json_files) > 0

    json_path = os.path.join(str(output_dir), json_files[0])
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, (dict, list)), "Generated JSON should be a dict or list"


def test_json_object_generator_output_not_empty(object_model, tmpdir):
    """Test that the generated file is not empty."""
    output_dir = tmpdir.mkdir("output")
    generator = JSONObjectGenerator(
        model=object_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    json_files = [f for f in os.listdir(str(output_dir)) if f.endswith(".json")]
    assert len(json_files) > 0

    json_path = os.path.join(str(output_dir), json_files[0])
    assert os.path.getsize(json_path) > 0, "Generated JSON file should not be empty"
