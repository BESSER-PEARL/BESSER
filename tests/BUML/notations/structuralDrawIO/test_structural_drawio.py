import os
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.structuralDrawIO import structural_drawio_to_buml

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "test.drawio")
modeltest: DomainModel = structural_drawio_to_buml(drawio_file_path=model_path)

# Test the classes of the BUML output model
def test_classes():
    assert len(modeltest.get_classes()) == 6
    assert modeltest.get_class_by_name("Library")
    assert modeltest.get_class_by_name("Book")
    assert modeltest.get_class_by_name("Author")
    assert modeltest.get_class_by_name("Science")
    assert modeltest.get_class_by_name("Fantasy")
    assert modeltest.get_class_by_name("Literature")

# Test the association of the BUML output model
def test_associations():
    assert len(modeltest.associations) == 2
    for association in modeltest.associations:
        assert association.name in ["WrittenBy", "Has"]

# Test the generalizations of the BUML output model
def test_generalizations():
    assert len(modeltest.generalizations) == 3
    for generalization in modeltest.generalizations:
        assert generalization.general.name == "Book"
        assert generalization.specific.name in ["Literature", "Fantasy", "Science"]

# Test the attributes of a class
def test_attributes():
    library = modeltest.get_class_by_name("Library")
    assert len(library.attributes) == 2
    for attr in library.attributes:
        assert attr.name in ["name", "address"]

# Test the attributes of a class
def test_methods():
    author = modeltest.get_class_by_name("Author")
    assert len(author.methods) == 1
    for method in author.methods:
        assert method.name in ["notify"]

# Test the attributes of a class
def test_enumeration():
    assert len(modeltest.get_enumerations()) == 1
    for enum in modeltest.get_enumerations():
        assert enum.name == "ContactM"
        assert len(enum.literals) == 3

# Test association ends of a class
def test_association_ends():
    library = modeltest.get_class_by_name("Library")
    assert len(library.association_ends()) == 1
    association_end = library.association_ends().pop()
    assert association_end.multiplicity.min == 1
    assert association_end.multiplicity.max == 1

# Test abstract class
def test_abstract_class():
    cls2 = modeltest.get_class_by_name("Author")
    assert cls2.is_abstract is False

# Test the inherited attributes of a specific class
def test_inherited_attributes():
    literature = modeltest.get_class_by_name("Literature")
    assert len(literature.inherited_attributes()) == 3
    for attr in literature.inherited_attributes():
        assert attr.name in ["title", "pages", "edition"]
