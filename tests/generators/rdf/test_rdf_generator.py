import os
import pytest
from besser.generators.rdf import RDFGenerator
from besser.BUML.metamodel.structural import (
    Class, DomainModel, DateType, StringType, IntegerType,
    Property, BinaryAssociation, Multiplicity, Enumeration, EnumerationLiteral
)

@pytest.fixture
def domain_model():
    # Enumeration example
    member_type = Enumeration(
        name="MemberType",
        literals={
            EnumerationLiteral(name="ADULT"),
            EnumerationLiteral(name="SENIOR"),
            EnumerationLiteral(name="STUDENT"),
            EnumerationLiteral(name="CHILD")
        }
    )

    # Classes
    library = Class(name="Library", attributes={
        Property(name="name", type=StringType),
        Property(name="address", type=StringType)
    })
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType),
        Property(name="pages", type=IntegerType),
        Property(name="release", type=DateType)
    })
    author = Class(name="Author", attributes={
        Property(name="email", type=StringType),
        Property(name="member", type=member_type)
    })

    # Associations
    located_in = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    lib_book_association = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

    written_by = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    publishes = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    book_author_association = BinaryAssociation(name="book_author", ends={written_by, publishes})

    model = DomainModel(
        name="Library_model",
        types={library, book, author, member_type},
        associations={lib_book_association, book_author_association}
    )
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