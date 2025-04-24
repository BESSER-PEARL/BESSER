import pytest
import os
from besser.generators.pydantic_classes import PydanticGenerator
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    IntegerType, StringType, Multiplicity, BinaryAssociation

# Test that a file is created with the correct content
def test_file_generation():
    # Define the model to be generated
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=IntegerType),
        Property(name="attr2", type=StringType)
    })

    domain_model: DomainModel = DomainModel(name="mymodel", types={class1})
    # Generate the file
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()

    # Check if the file exists
    assert os.path.exists(output_file), "The file was not created."

    # Read the content of the file
    with open(output_file, 'r') as file:
        content = file.read()

    # Check for expected lines in the file
    assert 'from pydantic import BaseModel' in content, "Missing Pydantic import."
    assert 'class name1(BaseModel):' in content, "Class definition is incorrect."
    assert 'attr1: int' in content, "Missing or incorrect 'id' field definition."
    assert 'attr2: str' in content, "Missing or incorrect 'name' field definition."

    # Clean up (optional)
    os.remove(output_file)

def test_multiple_class_generation():
    # Define multiple classes in the model
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=IntegerType),
        Property(name="attr2", type=StringType)
    })
    class2 = Class(name="name2", attributes={
        Property(name="attr1", type=StringType),
        Property(name="attr2", type=StringType)
    })

    domain_model = DomainModel(name="CompanyModel", types={class1, class2}, associations=None, packages=None, constraints=None)
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()

    assert os.path.exists(output_file), "The file was not created."

    with open(output_file, 'r') as file:
        content = file.read()

    # Check for the correct class definitions
    assert 'class name1(BaseModel):' in content, "Employee class definition is incorrect."
    assert 'class name2(BaseModel):' in content, "Department class definition is incorrect."

    os.remove(output_file)

def test_association_handling():
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=IntegerType),
    })
    class2 = Class(name="name2", attributes={
        Property(name="attr2", type=IntegerType)
    })
    association = BinaryAssociation(name="name_assoc", ends={
        Property(name="assocs1", type=class1, multiplicity=Multiplicity(1, "*")),
        Property(name="assocs2", type=class2, multiplicity=Multiplicity(1, "*"))
    })

    domain_model = DomainModel(name="AssociationModel", types={class1, class2}, associations={association})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=True, nested_creations=True)
    pydantic_model.generate()

    assert os.path.exists(output_file), "The file was not created."

    with open(output_file, 'r') as file:
        content = file.read()

    assert 'assocs2: Optional[List[Union["name2Create", int]]] = None' in content or 'assocs1: Optional[List[Union["name1Create", int]]] = None' in content, "Associations are not handled correctly."

    os.remove(output_file)

