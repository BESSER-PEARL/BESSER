import pytest
from besser.BUML.metamodel.structural import *
from besser.utilities import sort_by_timestamp

def test_named_element():
    named_element: NamedElement = NamedElement(name="element1")
    assert named_element.name == "element1"
    assert named_element.is_derived == False  # Default value should be False
    
    # Test setting is_derived
    named_element.is_derived = True
    assert named_element.is_derived == True
    
    # Test initialization with is_derived
    derived_element: NamedElement = NamedElement(name="element2", is_derived=True)
    assert derived_element.is_derived == True


def test_model_initialization():
    class1: Type = Type(name="element1")
    class2: Type = Type(name="element2")
    model: DomainModel = DomainModel(name="mymodel", types={class1, class2}, associations = None, packages = None, constraints = None)
    assert class1 in model.types
    assert class2 in model.types
    model_empty: DomainModel = DomainModel(name="mymodel", types = None, associations = None, packages = None, constraints = None)
    assert class1 not in model_empty.types
    assert class2 not in model_empty.types

# Testing the WFR for duplicate names in a model
def test_model_duplicated_names():
    with pytest.raises(ValueError) as excinfo:
        class1: Type = Type(name="name1")
        class2: Type = Type(name="name1")
        model: DomainModel = DomainModel(name="mymodel", types={class1, class2}, associations = None, packages = None, constraints = None)
    assert "The model cannot have types with duplicate names: name1" in str(excinfo.value)


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
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"),
                                        multiplicity=Multiplicity(0, 0))
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
    assert "A class cannot have attributes with duplicate names: attribute1" in str(excinfo.value)

# Testing for no more than one id attribute in class
def test_more_than_one_id_class():
    with pytest.raises(ValueError) as excinfo:
        class1 : Class
        attribute1: Property = Property(name="attribute1", type=PrimitiveDataType("int"), is_id=True)
        attribute2: Property = Property(name="attribute2", type=PrimitiveDataType("int"), is_id=True)
        class1 = Class(name="name1", attributes={attribute1, attribute2})
    assert "A class cannot have more than one attribute marked as 'id'" in str(excinfo.value)

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
    assert "A binary association must have exactly two ends" in str(excinfo.value)


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
    generalization_set: GeneralizationSet = GeneralizationSet(name="GeneralizationSet", generalizations={
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

# Testing method and parameters initialization
def test_method_initialization():
    parameter1: Parameter = Parameter(name="parameter_1", type=PrimitiveDataType(name="str"))
    parameter2: Parameter = Parameter(name="parameter_2", type=PrimitiveDataType(name="int"))
    method: Method = Method(name='method_1', is_abstract=True, parameters={parameter1, parameter2})
    cls: Class = Class(name="class_1", attributes={}, methods={method})
    method.type = cls
    assert len(method.parameters) == 2
    assert method.owner == cls
    assert method.name == "method_1"
    assert method.type.name == "class_1"

# Testing parameters with repeated name
def test_parameters_same_name():
    with pytest.raises(ValueError) as excinfo:
        parameter1: Parameter = Parameter(name="parameter_1", type=PrimitiveDataType(name="str"))
        parameter2: Parameter = Parameter(name="parameter_1", type=PrimitiveDataType(name="int"))
        method: Method = Method(name='method_1', is_abstract=True, parameters={parameter1, parameter2})
    assert "A method cannot have parameters with duplicate names: parameter_1" in str(excinfo.value)

# Testing default empty pre and post lists
def test_method_default_empty_pre_post():
    method: Method = Method(name="m1")
    assert method.pre == []
    assert method.post == []

# Testing pre and post initialization
def test_method_pre_post_initialization():
    cls: Class = Class(name="C")
    pre1: Constraint = Constraint(name="pre1", context=cls, expression="self.x > 0", language="OCL")
    post1: Constraint = Constraint(name="post1", context=cls, expression="self.x = self.x + 1", language="OCL")
    method: Method = Method(name="op", owner=cls, pre=[pre1], post=[post1])
    assert method.pre == [pre1]
    assert method.post == [post1]

# Testing pre with duplicate constraint names
def test_pre_same_name():
    cls: Class = Class(name="C")
    c1: Constraint = Constraint(name="dup", context=cls, expression="self.x > 0", language="OCL")
    c2: Constraint = Constraint(name="dup", context=cls, expression="self.x < 5", language="OCL")
    with pytest.raises(ValueError) as excinfo:
        Method(name="op", pre=[c1, c2])
    assert "A method cannot have preconditions with duplicate names: dup" in str(excinfo.value)

# Testing post with duplicate constraint names
def test_post_same_name():
    cls: Class = Class(name="C")
    c1: Constraint = Constraint(name="dup", context=cls, expression="self.x > 0", language="OCL")
    c2: Constraint = Constraint(name="dup", context=cls, expression="self.x < 5", language="OCL")
    with pytest.raises(ValueError) as excinfo:
        Method(name="op", post=[c1, c2])
    assert "A method cannot have postconditions with duplicate names: dup" in str(excinfo.value)

# Testing Constraint.expression strict typing (Pre-work B)
def test_constraint_expression_must_be_string():
    cls: Class = Class(name="C")
    with pytest.raises(TypeError, match="must be a string"):
        Constraint(name="bad", context=cls, expression=12345, language="OCL")


def test_constraint_expression_setter_rejects_non_string():
    cls: Class = Class(name="C")
    constraint: Constraint = Constraint(name="ok", context=cls, expression="self.x > 0", language="OCL")
    with pytest.raises(TypeError, match="must be a string"):
        constraint.expression = ["not", "a", "string"]


# Testing add_pre and add_post helpers
def test_method_add_pre_add_post():
    cls: Class = Class(name="C")
    method: Method = Method(name="op")
    pre1: Constraint = Constraint(name="pre1", context=cls, expression="self.x > 0", language="OCL")
    pre2: Constraint = Constraint(name="pre2", context=cls, expression="self.y > 0", language="OCL")
    post1: Constraint = Constraint(name="post1", context=cls, expression="self.x = 1", language="OCL")
    method.add_pre(pre1)
    method.add_pre(pre2)
    method.add_post(post1)
    assert method.pre == [pre1, pre2]
    assert method.post == [post1]
    # Adding duplicate-named precondition raises
    pre_dup: Constraint = Constraint(name="pre1", context=cls, expression="self.z > 0", language="OCL")
    with pytest.raises(ValueError) as excinfo:
        method.add_pre(pre_dup)
    assert "A method cannot have two preconditions with the same name: 'pre1'" in str(excinfo.value)

# Testing sort attributes by timestamp
def test_sort_attributes():
    attribute1: Property = Property(name="attribute_1", type=PrimitiveDataType(name="str"))
    attribute2: Property = Property(name="attribute_2", type=PrimitiveDataType(name="int"))
    attribute3: Property = Property(name="attribute_3", type=PrimitiveDataType(name="int"))
    cls: Class = Class(name="class", attributes={attribute1, attribute2, attribute3})
    attributes = sort_by_timestamp(cls.attributes)
    assert len(attributes) == 3
    assert isinstance(attributes, list)
    assert set(attributes) == {attribute1, attribute2, attribute3}

# Testing the classes_sorted_by_inheritance method
def test_classes_sorted_by_inheritance():
    cl1 = Class(name="c1")
    cl2 = Class(name="c2")
    cl3 = Class(name="c3")
    cl4 = Class(name="c4")
    cl5 = Class(name="c5")
    cl6 = Class(name="c6")
    cl7 = Class(name="c7")
    h1 = Generalization(general=cl7, specific=cl6)
    h2 = Generalization(general=cl6, specific=cl5)
    h3 = Generalization(general=cl5, specific=cl4)
    h4 = Generalization(general=cl4, specific=cl2)
    h5 = Generalization(general=cl4, specific=cl3)
    h6 = Generalization(general=cl3, specific=cl2)
    h7 = Generalization(general=cl2, specific=cl1)
    model = DomainModel(name="model", types={cl1,cl2,cl3,cl4,cl5,cl6,cl7}, generalizations={h1,h2,h3,h4,h5,h6,h7})
    classes = model.classes_sorted_by_inheritance()
    assert len(classes) == 7
    assert classes[0] == cl7
    assert classes[2] == cl5
    assert classes[4] == cl3
    assert classes[6] == cl1

# Testing synonyms of a Named Element
def test_synonyms():
    metadata: Metadata = Metadata()
    metadata.synonyms = ["synonym1", "synonym2", "synonym3"]
    class_a: Class = Class(name="Library", metadata=metadata)
    assert len(class_a.metadata.synonyms) == 3
    assert class_a.metadata.synonyms[0] == "synonym1"
    assert class_a.metadata.synonyms[1] == "synonym2"
    assert class_a.metadata.synonyms[2] == "synonym3"

# Testing all_parents and inherited_attributes methods
# GrandParent (attr1)
#           /           \
#    Parent1 (attr2)    Parent2 (attr3)
#          |
#    Child (attr4)
#~The test verifies:
# The all_parents() method by checking that each class correctly identifies its parent classes up the hierarchy
# The inherited_attributes() method by verifying that each class correctly inherits attributes from its parent classes
# The all_attributes() method by checking that each class has both its own attributes and inherited attributes
# The test checks multiple inheritance paths and ensures that attributes are correctly inherited through the class hierarchy.

def test_all_parents_and_inherited_attributes():
    # Create a hierarchy of classes with attributes
    attribute1: Property = Property(name="attr1", type=PrimitiveDataType("int"), multiplicity=Multiplicity(1, 1))
    attribute2: Property = Property(name="attr2", type=PrimitiveDataType("str"), multiplicity=Multiplicity(1, 1))
    attribute3: Property = Property(name="attr3", type=PrimitiveDataType("bool"), multiplicity=Multiplicity(1, 1))
    attribute4: Property = Property(name="attr4", type=PrimitiveDataType("float"), multiplicity=Multiplicity(1, 1))
    
    # Create classes with their attributes
    class_grandparent: Class = Class(name="GrandParent", attributes={attribute1})
    class_parent1: Class = Class(name="Parent1", attributes={attribute2})
    class_parent2: Class = Class(name="Parent2", attributes={attribute3})
    class_child: Class = Class(name="Child", attributes={attribute4})
    
    # Create generalizations
    generalization1: Generalization = Generalization(general=class_grandparent, specific=class_parent1)
    generalization2: Generalization = Generalization(general=class_grandparent, specific=class_parent2)
    generalization3: Generalization = Generalization(general=class_parent1, specific=class_child)
    
    # Test all_parents()
    assert class_child.parents() == {class_parent1}
    assert class_child.all_parents() == {class_parent1, class_grandparent}
    assert class_parent1.all_parents() == {class_grandparent}
    assert class_parent2.all_parents() == {class_grandparent}
    assert class_grandparent.all_parents() == set()
    
    # Test inherited_attributes()
    assert class_child.inherited_attributes() == {attribute1, attribute2}
    assert class_parent1.inherited_attributes() == {attribute1}
    assert class_parent2.inherited_attributes() == {attribute1}
    assert class_grandparent.inherited_attributes() == set()
    
    # Test all_attributes() which includes own and inherited attributes
    assert class_child.all_attributes() == {attribute1, attribute2, attribute4}
    assert class_parent1.all_attributes() == {attribute1, attribute2}
    assert class_parent2.all_attributes() == {attribute1, attribute3}
    assert class_grandparent.all_attributes() == {attribute1}

def test_named_element_blank_spaces():
    # Test that names with spaces raise ValueError
    with pytest.raises(ValueError) as excinfo:
        named_element = NamedElement(name="element with spaces")
    assert "'element with spaces' is invalid. Name cannot contain spaces." in str(excinfo.value)

    # Test that names without spaces work fine
    named_element = NamedElement(name="element_without_spaces")
    assert named_element.name == "element_without_spaces"

def test_attribute_reassignment():
    attribute1: Property = Property(name="attr1", type=StringType)
    class1: Class = Class(name="Cls1", attributes={attribute1})
    class2: Class = Class(name="Cls2", attributes=set())

    # Reassign attribute1 to class2
    class2.attributes = {attribute1}

    assert attribute1 in class2.attributes
    assert attribute1 not in class1.attributes
    assert attribute1.owner == class2


def test_package_initialization():
    # Create classes
    class1: Class = Class(name="Class1", attributes=set())
    class2: Class = Class(name="Class2", attributes=set())

    # Create associations
    aend1: Property = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    aend2: Property = Property(name="end2", owner=None, type=class2, multiplicity=Multiplicity(0, 1))
    association1: BinaryAssociation = BinaryAssociation(name="Association1", ends={aend1, aend2})

    aend3: Property = Property(name="end3", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    aend4: Property = Property(name="end4", owner=None, type=class2, multiplicity=Multiplicity(0, 1))
    association2: BinaryAssociation = BinaryAssociation(name="Association2", ends={aend3, aend4})

    # Create enumeration
    literal1: EnumerationLiteral = EnumerationLiteral(name="Literal1", owner=None)
    enumeration: Enumeration = Enumeration(name="Enumeration", literals={literal1})

    # Create package
    package: Package = Package(name="Package1", elements={class1, class2, association1, association2, enumeration})

    # Test get_classes method
    classes = package.get_classes()
    assert len(classes) == 2
    assert class1 in classes
    assert class2 in classes

    # Test get_associations method
    associations = package.get_associations()
    assert len(associations) == 2
    assert association1 in associations
    assert association2 in associations

    # Test get_enumerations method
    enumerations = package.get_enumerations()
    assert len(enumerations) == 1
    assert enumeration in enumerations

def test_domain_model_elements_recalculation():
    # Create types
    class1 = Class(name="Class1")
    class2 = Class(name="Class2")
    
    # Create associations
    end1 = Property(name="end1", type=class2)
    end2 = Property(name="end2", type=class1)
    association = BinaryAssociation(name="Assoc1", ends={end1, end2})
    
    # Create generalization
    generalization = Generalization(general=class1, specific=class2)
    
    # Create package
    package = Package(name="Package1", elements={class1})
    
    # Create constraint
    constraint = Constraint(name="Constraint1", context=class1, expression="self.name <> ''", language="OCL")
    
    # Create domain model
    domain_model = DomainModel(
        name="TestModel",
        types={class1, class2},
        associations={association},
        generalizations={generalization},
        packages={package},
        constraints={constraint}
    )
    
    # Check that elements are properly calculated
    expected_elements = {class1, class2, association, generalization, package, constraint}
    # Add primitive data types that are automatically included
    expected_elements.update(data_types)
    
    assert domain_model.elements == expected_elements


# Tests for uncertainty property
def test_element_uncertainty_default():
    """Test that Element uncertainty defaults to 0.0"""
    class1 = Class(name="TestClass")
    assert class1.uncertainty == 0.0
    
    property1 = Property(name="testProp", type=StringType)
    assert property1.uncertainty == 0.0
    
    association_end = Property(name="end1", type=class1)
    assert association_end.uncertainty == 0.0


def test_element_uncertainty_initialization():
    """Test that Element uncertainty can be set during initialization"""
    class1 = Class(name="TestClass", uncertainty=0.5)
    assert class1.uncertainty == 0.5
    
    property1 = Property(name="testProp", type=StringType, uncertainty=0.8)
    assert property1.uncertainty == 0.8
    
    # Test with association end
    class2 = Class(name="TestClass2")
    association_end = Property(name="end1", type=class2, uncertainty=0.3)
    assert association_end.uncertainty == 0.3


def test_element_uncertainty_setter_valid_values():
    """Test that uncertainty setter accepts valid values between 0 and 1"""
    class1 = Class(name="TestClass")
    
    # Test boundary values
    class1.uncertainty = 0.0
    assert class1.uncertainty == 0.0
    
    class1.uncertainty = 1.0
    assert class1.uncertainty == 1.0
    
    # Test intermediate values
    class1.uncertainty = 0.5
    assert class1.uncertainty == 0.5
    
    class1.uncertainty = 0.25
    assert class1.uncertainty == 0.25
    
    class1.uncertainty = 0.999
    assert class1.uncertainty == 0.999


def test_element_uncertainty_setter_invalid_values():
    """Test that uncertainty setter rejects invalid values outside [0,1]"""
    class1 = Class(name="TestClass")
    
    # Test values greater than 1
    with pytest.raises(ValueError) as excinfo:
        class1.uncertainty = 1.1
    assert "Uncertainty must be a probability between 0 and 1 inclusive" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        class1.uncertainty = 2.0
    assert "Uncertainty must be a probability between 0 and 1 inclusive" in str(excinfo.value)
    
    # Test negative values
    with pytest.raises(ValueError) as excinfo:
        class1.uncertainty = -0.1
    assert "Uncertainty must be a probability between 0 and 1 inclusive" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        class1.uncertainty = -1.0
    assert "Uncertainty must be a probability between 0 and 1 inclusive" in str(excinfo.value)


def test_element_uncertainty_inheritance():
    """Test that uncertainty property is inherited by all Element subclasses"""
    # Test with different Element subclasses
    class1 = Class(name="TestClass", uncertainty=0.2)
    assert class1.uncertainty == 0.2
    
    property1 = Property(name="testProp", type=StringType, uncertainty=0.3)
    assert property1.uncertainty == 0.3
    
    method1 = Method(name="testMethod", uncertainty=0.4)
    assert method1.uncertainty == 0.4
    
    parameter1 = Parameter(name="testParam", type=IntegerType, uncertainty=0.5)
    assert parameter1.uncertainty == 0.5
    
    enum1 = Enumeration(name="TestEnum", uncertainty=0.6)
    assert enum1.uncertainty == 0.6
    
    literal1 = EnumerationLiteral(name="TestLiteral", uncertainty=0.7)
    assert literal1.uncertainty == 0.7


def test_element_uncertainty_with_associations():
    """Test uncertainty property with associations"""
    class1 = Class(name="Class1", uncertainty=0.2)
    class2 = Class(name="Class2", uncertainty=0.3)
    
    end1 = Property(name="end1", type=class2, uncertainty=0.4)
    end2 = Property(name="end2", type=class1, uncertainty=0.5)
    
    association = BinaryAssociation(name="TestAssoc", ends={end1, end2}, uncertainty=0.6)
    
    assert class1.uncertainty == 0.2
    assert class2.uncertainty == 0.3
    assert end1.uncertainty == 0.4
    assert end2.uncertainty == 0.5
    assert association.uncertainty == 0.6


def test_element_uncertainty_with_generalization():
    """Test uncertainty property with generalization"""
    parent_class = Class(name="Parent", uncertainty=0.1)
    child_class = Class(name="Child", uncertainty=0.2)
    
    generalization = Generalization(general=parent_class, specific=child_class, uncertainty=0.3)
    
    assert parent_class.uncertainty == 0.1
    assert child_class.uncertainty == 0.2
    assert generalization.uncertainty == 0.3


def test_element_uncertainty_with_multiplicity():
    """Test uncertainty property with multiplicity"""
    multiplicity = Multiplicity(min_multiplicity=1, max_multiplicity=5, uncertainty=0.4)
    assert multiplicity.uncertainty == 0.4
    
    # Test with property that has multiplicity
    class1 = Class(name="TestClass")
    property1 = Property(
        name="testProp", 
        type=StringType, 
        multiplicity=multiplicity,
        uncertainty=0.7
    )
    
    assert property1.uncertainty == 0.7
    assert property1.multiplicity.uncertainty == 0.4


def test_element_uncertainty_domain_model():
    """Test uncertainty property in domain model context"""
    class1 = Class(name="Class1", uncertainty=0.1)
    class2 = Class(name="Class2", uncertainty=0.2)
    
    end1 = Property(name="end1", type=class2, uncertainty=0.3)
    end2 = Property(name="end2", type=class1, uncertainty=0.4)
    association = BinaryAssociation(name="Assoc1", ends={end1, end2}, uncertainty=0.5)
    
    generalization = Generalization(general=class1, specific=class2, uncertainty=0.6)
    
    domain_model = DomainModel(
        name="TestModel",
        types={class1, class2},
        associations={association},
        generalizations={generalization},
        uncertainty=0.8
    )
    
    assert domain_model.uncertainty == 0.8
    assert class1.uncertainty == 0.1
    assert class2.uncertainty == 0.2
    assert association.uncertainty == 0.5
    assert generalization.uncertainty == 0.6


def test_element_uncertainty_type_validation():
    """Test that uncertainty property accepts float values"""
    class1 = Class(name="TestClass")
    
    # Test with integer that should be converted to float
    class1.uncertainty = 1
    assert class1.uncertainty == 1.0
    assert isinstance(class1.uncertainty, float)
    
    class1.uncertainty = 0
    assert class1.uncertainty == 0.0
    assert isinstance(class1.uncertainty, float)
    
    # Test with explicit float
    class1.uncertainty = 0.5
    assert class1.uncertainty == 0.5
    assert isinstance(class1.uncertainty, float)
    

def test_property_is_id_roundtrip():
    """Test that is_id attribute survives a simulated JSON round-trip."""
    json_data = {
        "name": "id_attr",
        "isId": True,
        "isOptional": False 
    }
    
    prop = Property(name=json_data["name"], type=StringType, is_id=json_data["isId"], is_optional=json_data["isOptional"])
    
    assert prop.is_id is True
    
    output_json = {
        "name": prop.name,
        "isId": prop.is_id
    }
    
    assert output_json["isId"] is True


def test_property_is_id_validation():
    """Test all scenarios for Property identifiers."""
    # Check default value (is_id should be False by default)
    prop_default = Property(name="normal_attr", type=StringType)
    assert prop_default.is_id is False
    
    # Check correct initialization as an identifier
    prop = Property(name="id_attr", type=StringType, is_id=True)
    assert prop.is_id is True
    assert prop.is_optional is False
    
    # Check validation rule: conflict between is_id and is_optional during initialization
    with pytest.raises(ValueError, match="cannot be both an identifier"):
        Property(name="invalid_prop", type=StringType, is_id=True, is_optional=True)
        
    # Check validation rule: setting is_id=True on an already optional property
    prop_opt = Property(name="opt_to_id", type=StringType, is_optional=True)
    with pytest.raises(ValueError, match="cannot be both an identifier"):
        prop_opt.is_id = True
    
    # Check validation rule: setting is_optional=True on an already identifier property
    prop_id = Property(name="id_to_opt", type=StringType, is_id=True)
    with pytest.raises(ValueError, match="cannot be both an identifier"):
        prop_id.is_optional = True


def test_property_is_external_id_validation():
    """is_external_id defaults to False, round-trips, and conflicts with is_optional.

    External identifiers (issue #230) are distinct from the internal PK
    (``is_id``): a class may have several of them (composite key) and a
    property may carry both flags at once when a natural key also serves as
    the PK (e.g. ``isbn``). External ids must still be non-optional so the
    object is always identifiable by users.
    """
    # Default is False
    prop_default = Property(name="plain", type=StringType)
    assert prop_default.is_external_id is False

    # Correct initialization
    prop = Property(name="email", type=StringType, is_external_id=True)
    assert prop.is_external_id is True
    assert prop.is_optional is False

    # is_id and is_external_id may coexist (natural PK case)
    prop_both = Property(name="isbn", type=StringType, is_id=True, is_external_id=True)
    assert prop_both.is_id is True
    assert prop_both.is_external_id is True

    # is_external_id + is_optional rejected at construction
    with pytest.raises(ValueError, match="external identifier and optional"):
        Property(name="bad", type=StringType, is_external_id=True, is_optional=True)

    # Setting is_external_id=True on an already-optional property is rejected
    prop_opt = Property(name="opt", type=StringType, is_optional=True)
    with pytest.raises(ValueError, match="external identifier and optional"):
        prop_opt.is_external_id = True

    # Setting is_optional=True on an external-id property is rejected
    prop_ext = Property(name="ext", type=StringType, is_external_id=True)
    with pytest.raises(ValueError, match="external identifier and optional"):
        prop_ext.is_optional = True


def test_attribute_shadowing_validation():
    """Test that validation catches attribute shadowing between parent and child classes."""
    parent_attr = Property(name="name", type=StringType)
    parent = Class(name="Parent", attributes={parent_attr})

    child_attr = Property(name="name", type=StringType)
    child = Class(name="Child", attributes={child_attr})

    generalization = Generalization(general=parent, specific=child)

    domain_model = DomainModel(
        name="TestModel",
        types={parent, child},
        associations=set(),
        generalizations={generalization}
    )

    result = domain_model.validate(raise_exception=False)
    assert result["success"] is False
    assert any("attribute 'name'" in e and "Child" in e for e in result["errors"])


def test_no_attribute_shadowing_validation():
    """Test that validation passes when there is no attribute shadowing."""
    parent_attr = Property(name="name", type=StringType)
    parent = Class(name="Parent", attributes={parent_attr})

    child_attr = Property(name="age", type=IntegerType)
    child = Class(name="Child", attributes={child_attr})

    generalization = Generalization(general=parent, specific=child)

    domain_model = DomainModel(
        name="TestModel",
        types={parent, child},
        associations=set(),
        generalizations={generalization}
    )

    result = domain_model.validate(raise_exception=False)
    assert result["success"] is True
