import os
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "test.plantuml")
modeltest: DomainModel = plantuml_to_buml(plantUML_model_path=model_path)

# Test the classes of the BUML output model
def test_classes():
    assert len(modeltest.get_classes()) == 7
    assert modeltest.get_class_by_name("Library")
    assert modeltest.get_class_by_name("Book")
    assert modeltest.get_class_by_name("Author")
    assert modeltest.get_class_by_name("Science")
    assert modeltest.get_class_by_name("Fantasy")
    assert modeltest.get_class_by_name("Literature")
    assert modeltest.get_class_by_name("Platform")

# Test the association of the BUML output model
def test_associations():
    assert len(modeltest.associations) == 2
    for association in modeltest.associations:
        assert association.name in ["writtenBy", "has"]

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
    assert len(author.methods) == 2
    for method in author.methods:
        assert method.name in ["notify", "func"]

# Test the attributes of a class
def test_enumeration():
    assert len(modeltest.get_enumerations()) == 1
    for enum in modeltest.get_enumerations():
        assert enum.name == "ContactM"
        assert len(enum.literals) == 3

# Test the inherited attributes of a specific class
def test_inherited_attributes():
    literature = modeltest.get_class_by_name("Literature")
    assert len(literature.inherited_attributes()) == 3
    for attr in literature.inherited_attributes():
        assert attr.name in ["tittle", "pages", "edition"]

# Test association ends of a class
def test_abstract_class():
    library = modeltest.get_class_by_name("Library")
    assert len(library.association_ends()) == 1
    assert library.association_ends().pop().multiplicity.min == 1
    assert library.association_ends().pop().multiplicity.max == 1

# Test abstract class
def test_association_ends():
    cls1 = modeltest.get_class_by_name("Platform")
    assert cls1.is_abstract is True
    cls2 = modeltest.get_class_by_name("Author")
    assert cls2.is_abstract is False
