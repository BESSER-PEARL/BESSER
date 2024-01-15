from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.plantUML import plantuml_to_buml

modeltest: DomainModel = plantuml_to_buml(plantUML_model_path="test.plantuml")

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
        assert association.name in ["writedBy", "has"]

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

# Test the inherited attributes of a specific class
def test_inherited_attributes():
    literature = modeltest.get_class_by_name("Literature")
    assert len(literature.inherited_attributes()) == 3
    for attr in literature.inherited_attributes():
        assert attr.name in ["tittle", "pages", "edition"]

# Test association ends of a class
def test_association_ends():
    library = modeltest.get_class_by_name("Library")
    assert len(library.association_ends()) == 1
    assert library.association_ends().pop().multiplicity.min == 1
    assert library.association_ends().pop().multiplicity.max == 1