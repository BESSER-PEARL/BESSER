import pytest

from besser.BUML.metamodel.structural import NamedElement, DomainModel, Type, Class, \
    Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, \
    GeneralizationSet, AssociationClass, Enumeration, EnumerationLiteral


def test_named_element():
    named_element: NamedElement = NamedElement(name="element1")
    assert named_element.name == "element1"


def test_model_initialization():
    class1: Type = Type(name="element1")
    class2: Type = Type(name="element2")
    model: DomainModel = DomainModel(name="mymodel", types={class1, class2}, associations = None, packages = None, constraints = None)
    assert len(model.types) == 2
    model_empty: DomainModel = DomainModel(name="mymodel", types = None, associations = None, packages = None, constraints = None)
    assert len(model_empty.types) == 0


# Testing the WFR for duplicate names in a model
def test_model_duplicated_names():
    with pytest.raises(ValueError) as excinfo:
        class1: Type = Type(name="name1")
        class2: Type = Type(name="name1")
        model: DomainModel = DomainModel(name="mymodel", types={class1, class2}, associations = None, packages = None, constraints = None)
    assert "same name" in str(excinfo.value)


# Testing attributes initialization
def test_attribute_initialization():
    class1: Type = Type(name="name1")
    attribute1: Property = Property(name="attribute1", owner = class1, type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(0, 1))
    # assert attributes has proper type and multiplicity
    assert attribute1.type.name == "int"
    assert attribute1.multiplicity.min == 0
    assert attribute1.multiplicity.max == 1


# Testing attribute multiplicity and data type violations
def test_attribute_type_and_multiplicity_violation():
    with pytest.raises(ValueError) as excinfo:
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"),
                                        multiplicity=Multiplicity(0, -1))
    assert "Invalid max multiplicity" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"),
                                        multiplicity=Multiplicity(-1, 1))
    assert "Invalid min multiplicity" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"),
                                        multiplicity=Multiplicity(2, 1))
    assert "Invalid max multiplicity" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("invented_type"),
                                        multiplicity=Multiplicity(2, 1))
    assert "Invalid primitive data type" in str(excinfo.value)


# Testing class initialization
def test_class_initialization():
    class1: Class = Class(name="classA", attributes=set())
    class2: Class
    attribute1: Property = Property(name="attribute1", owner=None, type=PrimitiveDataType("int"), multiplicity=Multiplicity(0, 1))
    attribute2: Property = Property(name="attribute2", owner=None, type=PrimitiveDataType("int"), multiplicity=Multiplicity(0, 1))
    reference1: Property = Property(name="reference1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    class2: Class = Class(name="classB", attributes={attribute1, attribute2})
    assert len(class2.attributes) == 2
    print(class2)


# Testing no duplicated names in class attributes
def test_duplicated_name_class():
    with pytest.raises(ValueError) as excinfo:
        class1 : Class
        attribute1: Property = Property(name="attribute1", owner=None, type=PrimitiveDataType("int"), multiplicity=Multiplicity(0, 1))
        attribute2: Property = Property(name="attribute1", owner=None, type=PrimitiveDataType("int"), multiplicity=Multiplicity(0, 1))
        class1 = Class(name="name1", attributes={attribute1, attribute2})
    assert "A class cannot have two attributes with the same name" in str(excinfo.value)

# Testing for no more than one id attribute in class
def test_more_than_one_id_class():
    with pytest.raises(ValueError) as excinfo:
        class1 : Class
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"), is_id=True)
        attribute2: Property = Property(name="attribute2", type=PrimitiveDataType("int"), is_id=True)
        class1 = Class(name="name1", attributes={attribute1, attribute2})
    assert "A class cannot have two id attributes" in str(excinfo.value)

def test_association_initialization():
    class1: Class = Class(name="name1", attributes=set())
    class2: Class = Class(name="name2", attributes=set())
    aend1: Property = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    aend2: Property = Property(name="end2", owner=None, type=class2, multiplicity=Multiplicity(0, 1))
    association: BinaryAssociation = BinaryAssociation(name="association1", ends={aend1, aend2})
    assert len(association.ends) == 2
    assert aend1 in association.ends
    assert aend1.owner == association
    assert class1.associations == {association}
    assert class1.association_ends() == {aend2}

# Testing the creation of a binary association cannot have more than two ends
def test_binary_association():
    with pytest.raises(ValueError) as excinfo:
        class1: Type = Type(name="name1")
        aend: Property = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
        association: BinaryAssociation = BinaryAssociation(name="association1", ends={aend})
    assert "A binary association should have two ends" in str(excinfo.value)


# Testing the creation of an association class with an attribute
def test_association_class():
    class1: Class = Class(name="name1", attributes=None)
    class2: Class = Class(name="name2", attributes=None)
    aend1: Property = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    aend2: Property = Property(name="end2", owner=None, type=class2, multiplicity=Multiplicity(0, 1))
    association: BinaryAssociation = BinaryAssociation(name="association1", ends={aend1, aend2})
    attribute1: Property = Property(name="attribute1", owner=None, type=PrimitiveDataType("int"), multiplicity=Multiplicity(0, 1))
    association_class: AssociationClass = AssociationClass(name="association_class1", attributes={attribute1}, association=association)
    assert len(association_class.attributes) == 1
    assert attribute1 in association_class.attributes
    assert association_class.association.name == "association1"


def test_generalization_initialization():
    attribute1: Property = Property(name="attribute1", owner = None, type=PrimitiveDataType("int"),
                                    multiplicity=Multiplicity(0, 1))
    class1: Class = Class(name="name1", attributes={attribute1})
    class2: Class = Class(name="name2", attributes=None)
    generalization: Generalization = Generalization(general=class1, specific=class2)
    assert generalization.general == class1
    assert generalization.specific == class2
    assert class2.generalizations == {generalization}
    assert class1.specializations() == {class2}
    assert class2.parents() == {class1}
    assert class2.specializations() == set()
    assert class2.all_attributes() == {attribute1}


def test_no_generalization_loop():
    with pytest.raises(ValueError) as excinfo:
        class1: Class = Class(name="name1", attributes=None)
        generalization: Generalization = Generalization(general=class1, specific=class1)
    assert "A class cannot be a generalization of itself" in str(excinfo.value)

def test_generalization_set_initialization():
    class1: Class = Class(name="name1", attributes=None)
    class2: Class = Class(name="name2", attributes=None)
    class3: Class = Class(name="name3", attributes=None)
    generalization1: Generalization = Generalization(general=class1, specific=class2)
    generalization2: Generalization = Generalization(general=class1, specific=class3)
    generalization_set: GeneralizationSet = GeneralizationSet(name="Generalization Set", generalizations={
        generalization1,generalization2}, is_disjoint=True, is_complete=True)
    assert generalization_set.is_disjoint == True
    assert generalization_set.is_complete == True
    assert class1.generalizations == {generalization1, generalization2}
    assert class1.specializations() == {class3, class2}
    assert class2.parents() == {class1}
    assert class3.parents() == {class1}
    assert class2.specializations() == set()
    
# Testing enumeration and literals initialization
def test_enumeration_initialization():
    literal1: EnumerationLiteral = EnumerationLiteral(name="literal_1", owner=None)
    literal2: EnumerationLiteral = EnumerationLiteral(name="literal_2", owner=None)
    literal3: EnumerationLiteral = EnumerationLiteral(name="literal_3", owner=None)
    enum: Enumeration = Enumeration(name="Enumeration", literals={literal1,literal2,literal3})
    assert len(enum.literals) == 3
    assert literal1.owner == enum
    
# Testing no duplicated literal names in an enumeration
def test_duplicated_name_literal():
    with pytest.raises(ValueError) as excinfo:
        literal1: EnumerationLiteral = EnumerationLiteral(name="duplicated_name", owner=None)
        literal2: EnumerationLiteral = EnumerationLiteral(name="duplicated_name", owner=None)
        enum1: Enumeration = Enumeration(name="Enumeration", literals={literal1,literal2})
    assert "An enumeration cannot have two literals with the same name" in str(excinfo.value)
