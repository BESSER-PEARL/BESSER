import os
import pytest
from besser.generators.rdf import RDFGenerator
from besser.BUML.metamodel.structural import (
    Property, Enumeration, EnumerationLiteral,
)


@pytest.fixture
def domain_model(library_book_author_model):
    """Extend the shared Library-Book-Author model with an enumeration.

    The RDF generator tests need a MemberType enum on Author, so we add
    it on top of the shared fixture.
    """
    model = library_book_author_model

    # Add MemberType enumeration
    member_type = Enumeration(
        name="MemberType",
        literals={
            EnumerationLiteral(name="ADULT"),
            EnumerationLiteral(name="SENIOR"),
            EnumerationLiteral(name="STUDENT"),
            EnumerationLiteral(name="CHILD"),
        },
    )

    # Find the Author class and add the 'member' attribute
    author = next(t for t in model.types if t.name == "Author")
    member_prop = Property(name="member", type=member_type)
    author.attributes = author.attributes | {member_prop}

    # Add the enum to the model types
    model.types = model.types | {member_type}

    return model

def test_rdf_classes_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = RDFGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "vocabulary.ttl")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        rdf_code = f.read()

    # Check for class definitions
    assert "ex:Library rdf:type rdfs:Class" in rdf_code
    assert "ex:Book rdf:type rdfs:Class" in rdf_code
    assert "ex:Author rdf:type rdfs:Class" in rdf_code

def test_rdf_properties_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = RDFGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "vocabulary.ttl")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        rdf_code = f.read()

    # Check for property definitions
    assert "ex:name rdf:type rdf:Property" in rdf_code
    assert "ex:address rdf:type rdf:Property" in rdf_code
    assert "ex:title rdf:type rdf:Property" in rdf_code
    assert "ex:pages rdf:type rdf:Property" in rdf_code
    assert "ex:release rdf:type rdf:Property" in rdf_code
    assert "ex:email rdf:type rdf:Property" in rdf_code

def test_rdf_associations_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = RDFGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "vocabulary.ttl")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        rdf_code = f.read()

    # Check for association properties
    assert "ex:locatedIn rdf:type rdf:Property" in rdf_code or "ex:has rdf:type rdf:Property" in rdf_code
    assert "ex:writtenBy rdf:type rdf:Property" in rdf_code or "ex:publishes rdf:type rdf:Property" in rdf_code

def test_rdf_enumerations_exist(domain_model, tmpdir):
    output_dir = tmpdir.mkdir("output")
    generator = RDFGenerator(model=domain_model, output_dir=str(output_dir))
    generator.generate()

    output_file = os.path.join(str(output_dir), "vocabulary.ttl")
    assert os.path.isfile(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        rdf_code = f.read()

    # Check for enumeration class and literals
    assert "ex:MemberType rdf:type rdfs:Class" in rdf_code
    assert "ex:ADULT rdf:type ex:MemberType" in rdf_code
    assert "ex:SENIOR rdf:type ex:MemberType" in rdf_code
    assert "ex:STUDENT rdf:type ex:MemberType" in rdf_code
    assert "ex:CHILD rdf:type ex:MemberType" in rdf_code