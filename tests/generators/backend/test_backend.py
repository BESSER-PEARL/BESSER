import os
import pytest
from besser.generators.backend import BackendGenerator
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, IntegerType, 
    BinaryAssociation, Multiplicity
)

# Define the expected output markers based on the actual debug output
pydantic_markers = [
    "class name1Create(BaseModel):",
    "attr1: int",
    "assocs2: List[int]",
    "class name2Create(BaseModel):",
    "attr2: int",
    "assocs1: List[int]"
]

# Updated SQLAlchemy markers based on the debug output (SQLAlchemy 2.0 style)
sqlalchemy_markers = [
    "class name1(Base):",
    "__tablename__ = \"name1\"",
    "id: Mapped[int] = mapped_column(primary_key=True)",
    "attr1: Mapped[int] = mapped_column(Integer)"
]

api_markers = [
    "@app.get(\"/name1/\"",
    "def get_all_name1(database: Session = Depends(get_db))",
    "name1_list = database.query(name1).all()"
]

@pytest.fixture
def domain_model():
    # Create classes
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=IntegerType),
    })
    class2 = Class(name="name2", attributes={
        Property(name="attr2", type=IntegerType)
    })
    
    # Create association between classes
    association = BinaryAssociation(
        name="name_assoc", 
        ends={
            Property(name="assocs1", type=class1, multiplicity=Multiplicity(1, "*")),
            Property(name="assocs2", type=class2, multiplicity=Multiplicity(1, "*"))
        }
    )

    # Create domain model
    model = DomainModel(
        name="Name", 
        types={class1, class2}, 
        associations={association}
    )
    
    return model

# Define the test function
def test_generator(domain_model, tmpdir):
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = BackendGenerator(model=domain_model, output_dir=str(output_dir))

    # Generate backend
    generator.generate()

    # Check if the files were created
    api_file = os.path.join(str(output_dir), "main_api.py")
    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    sqlalchemy_file = os.path.join(str(output_dir), "sql_alchemy.py")
    
    assert os.path.isfile(api_file)
    assert os.path.isfile(pydantic_file)
    assert os.path.isfile(sqlalchemy_file)

    # Read the generated files
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()
    
    with open(sqlalchemy_file, "r", encoding="utf-8") as f:
        sqlalchemy_code = f.read()
    
    with open(api_file, "r", encoding="utf-8") as f:
        api_code = f.read()

    # For debugging - print the actual content to see what we're checking against
    print("--- Generated Pydantic Code ---")
    print(pydantic_code[:500])  # Print first 500 chars to see structure
    
    print("--- Generated SQLAlchemy Code ---")
    print(sqlalchemy_code[:1000])  # Print first 1000 chars to see structure
    
    print("--- Generated API Code ---")
    print(api_code[:1000])  # Print first 1000 chars to see structure
    
    # Check for the existence of expected code snippets using more flexible approach
    for marker in pydantic_markers:
        assert marker in pydantic_code, f"Missing expected code: {marker}"
    
    for marker in sqlalchemy_markers:
        assert marker in sqlalchemy_code, f"Missing expected code: {marker}"
    
    for marker in api_markers:
        assert marker in api_code, f"Missing expected code: {marker}"
