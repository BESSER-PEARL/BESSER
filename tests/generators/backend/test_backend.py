import os
import pytest
from besser.generators.backend import BackendGenerator
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, IntegerType, StringType, FloatType, DateTimeType,
    BinaryAssociation, Multiplicity
)


@pytest.fixture
def simple_model():
    """Simple N:M relationship test"""
    # Create classes
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=IntegerType),
    })
    class2 = Class(name="name2", attributes={
        Property(name="attr2", type=IntegerType)
    })
    
    # Create N:M association between classes
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


@pytest.fixture
def relationship_model():
    """
    Test model with 1:1, N:1, and 1:N relationships.
    This tests the FK placement fix and multiplicity constraint handling.
    
    Model structure:
    - PhysicalAsset (no FK)
    - DigitalTwin (has p_asset_id FK to PhysicalAsset) - mandatory 1:1
    - Sensor (has dt_id FK to DigitalTwin) - mandatory N:1
    """
    # Create classes
    PhysicalAsset = Class(name="PhysicalAsset", attributes={
        Property(name="attribute", type=StringType),
    })
    
    DigitalTwin = Class(name="DigitalTwin", attributes={
        Property(name="attribute", type=StringType),
    })
    
    Sensor = Class(name="Sensor", attributes={
        Property(name="type", type=StringType),
        Property(name="timestamp", type=DateTimeType),
        Property(name="value", type=FloatType),
    })
    
    # Sensor N:1 DigitalTwin (Sensor MUST have a DigitalTwin)
    Sensor_DigitalTwin = BinaryAssociation(
        name="Sensor_DigitalTwin",
        ends={
            Property(name="sensors", type=Sensor, multiplicity=Multiplicity(0, 9999)),
            Property(name="dt", type=DigitalTwin, multiplicity=Multiplicity(1, 1))
        }
    )
    
    # DigitalTwin 1:1 PhysicalAsset (DigitalTwin MUST have a PhysicalAsset)
    DigitalTwin_PhysicalAsset = BinaryAssociation(
        name="DigitalTwin_PhysicalAsset",
        ends={
            Property(name="dt", type=DigitalTwin, multiplicity=Multiplicity(0, 1)),
            Property(name="p_asset", type=PhysicalAsset, multiplicity=Multiplicity(1, 1))
        }
    )

    # Create domain model
    model = DomainModel(
        name="RelationshipTest",
        types={DigitalTwin, Sensor, PhysicalAsset},
        associations={Sensor_DigitalTwin, DigitalTwin_PhysicalAsset},
    )
    
    return model


def test_simple_generator(simple_model, tmpdir):
    """Test basic N:M relationship generation"""
    # Create an instance of the generator
    output_dir = tmpdir.mkdir("output")
    generator = BackendGenerator(model=simple_model, output_dir=str(output_dir))

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

    # Pydantic checks
    pydantic_markers = [
        "class name1Create(BaseModel):",
        "attr1: int",
        "assocs2: List[int]",
        "class name2Create(BaseModel):",
        "attr2: int",
        "assocs1: List[int]"
    ]
    
    for marker in pydantic_markers:
        assert marker in pydantic_code, f"Missing expected Pydantic code: {marker}"

    # SQLAlchemy checks
    sqlalchemy_markers = [
        "class name1(Base):",
        "__tablename__ = \"name1\"",
        "id: Mapped[int] = mapped_column(primary_key=True)",
        "attr1: Mapped[int] = mapped_column(Integer)"
    ]
    
    for marker in sqlalchemy_markers:
        assert marker in sqlalchemy_code, f"Missing expected SQLAlchemy code: {marker}"

    # API checks
    api_markers = [
        "@app.get(\"/name1/\"",
        "def get_all_name1(detailed: bool = False, database: Session = Depends(get_db))",
        "name1_list = database.query(name1).all()"
    ]
    
    for marker in api_markers:
        assert marker in api_code, f"Missing expected API code: {marker}"


def test_relationship_fk_placement(relationship_model, tmpdir):
    """
    Test that FKs are placed correctly based on multiplicity constraints.
    This validates the fix for 1:1 and N:1 relationship FK placement.
    """
    output_dir = tmpdir.mkdir("output_rel")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    sqlalchemy_file = os.path.join(str(output_dir), "sql_alchemy.py")
    with open(sqlalchemy_file, "r", encoding="utf-8") as f:
        sqlalchemy_code = f.read()

    # Test 1: PhysicalAsset should NOT have any FK
    assert "class PhysicalAsset(Base):" in sqlalchemy_code
    assert "PhysicalAsset" in sqlalchemy_code
    # Extract only the class definition (before relationship definitions)
    physicalasset_section = sqlalchemy_code.split("class PhysicalAsset(Base):")[1].split("#---")[0]
    assert "ForeignKey" not in physicalasset_section, "PhysicalAsset should not have any ForeignKey"
    # Make sure it only has id (primary key) and attribute, not dt_id FK column
    assert "dt_id" not in physicalasset_section, "PhysicalAsset should not have dt_id FK"

    # Test 2: Sensor should have dt_id FK (N:1 mandatory)
    assert "class Sensor(Base):" in sqlalchemy_code
    assert "dt_id: Mapped[int] = mapped_column(ForeignKey(\"digitaltwin.id\"), nullable=False)" in sqlalchemy_code

    # Test 3: DigitalTwin should have p_asset_id FK (1:1 mandatory)
    assert "class DigitalTwin(Base):" in sqlalchemy_code
    assert "p_asset_id: Mapped[int] = mapped_column(ForeignKey(\"physicalasset.id\"), nullable=False, unique=True)" in sqlalchemy_code


def test_pydantic_multiplicity_constraints(relationship_model, tmpdir):
    """
    Test that Pydantic models respect multiplicity constraints.
    This validates the fix for required vs optional relationship fields.
    """
    output_dir = tmpdir.mkdir("output_pyd")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()

    # Test 1: DigitalTwinCreate should have REQUIRED p_asset (1:1 mandatory)
    assert "class DigitalTwinCreate(BaseModel):" in pydantic_code
    assert "p_asset: int" in pydantic_code  # No Optional!
    assert "# 1:1 Relationship (mandatory)" in pydantic_code

    # Test 2: PhysicalAssetCreate should have OPTIONAL dt (1:1 optional)
    assert "class PhysicalAssetCreate(BaseModel):" in pydantic_code
    assert "dt: Optional[int] = None" in pydantic_code
    assert "# 1:1 Relationship (optional)" in pydantic_code

    # Test 3: SensorCreate should have REQUIRED dt (N:1 mandatory)
    assert "class SensorCreate(BaseModel):" in pydantic_code
    assert "dt: int" in pydantic_code  # No Optional!
    assert "# N:1 Relationship (mandatory)" in pydantic_code

    # Test 4: DigitalTwinCreate should have OPTIONAL sensors list (1:N)
    assert "sensors: Optional[List[int]] = None" in pydantic_code
    assert "# 1:N Relationship" in pydantic_code


def test_api_fk_validation_and_assignment(relationship_model, tmpdir):
    """
    Test that REST API endpoints correctly validate and assign FKs.
    This validates the fix for FK validation and assignment in POST/PUT endpoints.
    """
    output_dir = tmpdir.mkdir("output_api")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    api_file = os.path.join(str(output_dir), "main_api.py")
    with open(api_file, "r", encoding="utf-8") as f:
        api_code = f.read()

    # Test 1: PhysicalAsset POST should NOT validate any FK (it has none)
    assert "@app.post(\"/physicalasset/\"" in api_code
    physicalasset_post_idx = api_code.find("@app.post(\"/physicalasset/\"")
    physicalasset_put_idx = api_code.find("@app.put(\"/physicalasset/", physicalasset_post_idx)
    physicalasset_section = api_code[physicalasset_post_idx:physicalasset_put_idx]
    
    # PhysicalAsset should create entity directly without FK validation
    assert "PhysicalAsset(" in physicalasset_section
    # Should not have validation checks before PhysicalAsset creation
    creation_idx = physicalasset_section.find("PhysicalAsset(")
    pre_creation = physicalasset_section[:creation_idx]
    assert "database.query" not in pre_creation or "DigitalTwin" not in pre_creation, "PhysicalAsset should not validate FKs"

    # Test 2: Sensor POST should validate dt and assign dt_id
    assert "@app.post(\"/sensor/\"" in api_code
    sensor_post_idx = api_code.find("@app.post(\"/sensor/\"")
    sensor_put_idx = api_code.find("@app.put(\"/sensor/", sensor_post_idx)
    sensor_section = api_code[sensor_post_idx:sensor_put_idx]
    
    assert "if sensor_data.dt is not None:" in sensor_section
    assert "db_dt = database.query(DigitalTwin)" in sensor_section
    assert "DigitalTwin ID is required" in sensor_section
    assert "dt_id=sensor_data.dt" in sensor_section  # Should assign FK field, not object

    # Test 3: DigitalTwin POST should validate p_asset and assign p_asset_id
    assert "@app.post(\"/digitaltwin/\"" in api_code
    digitaltwin_post_idx = api_code.find("@app.post(\"/digitaltwin/\"")
    digitaltwin_put_idx = api_code.find("@app.put(\"/digitaltwin/", digitaltwin_post_idx)
    digitaltwin_section = api_code[digitaltwin_post_idx:digitaltwin_put_idx]
    
    assert "if digitaltwin_data.p_asset is not None:" in digitaltwin_section
    assert "db_p_asset = database.query(PhysicalAsset)" in digitaltwin_section
    assert "PhysicalAsset ID is required" in digitaltwin_section
    assert "p_asset_id=digitaltwin_data.p_asset" in digitaltwin_section  # Should assign FK field

    # Test 4: Sensor PUT should update dt_id
    sensor_put_idx = api_code.find("@app.put(\"/sensor/")
    sensor_delete_idx = api_code.find("@app.delete(\"/sensor/", sensor_put_idx)
    sensor_update_section = api_code[sensor_put_idx:sensor_delete_idx]
    
    assert "if sensor_data.dt is not None:" in sensor_update_section
    assert "setattr(db_sensor, 'dt_id', sensor_data.dt)" in sensor_update_section

    # Test 5: DigitalTwin PUT should update p_asset_id
    digitaltwin_put_idx = api_code.find("@app.put(\"/digitaltwin/")
    digitaltwin_delete_idx = api_code.find("@app.delete(\"/digitaltwin/", digitaltwin_put_idx)
    digitaltwin_update_section = api_code[digitaltwin_put_idx:digitaltwin_delete_idx]
    
    assert "if digitaltwin_data.p_asset is not None:" in digitaltwin_update_section
    assert "setattr(db_digitaltwin, 'p_asset_id', digitaltwin_data.p_asset)" in digitaltwin_update_section


def test_no_circular_dependency(relationship_model, tmpdir):
    """
    Test that the generated code allows proper entity creation order without circular dependencies.
    This is a logical test - verifying that PhysicalAsset can be created first, then DigitalTwin, then Sensor.
    """
    output_dir = tmpdir.mkdir("output_circ")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    sqlalchemy_file = os.path.join(str(output_dir), "sql_alchemy.py")
    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    
    with open(sqlalchemy_file, "r", encoding="utf-8") as f:
        sqlalchemy_code = f.read()
    
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()

    # PhysicalAsset should have no FKs (can be created first)
    physicalasset_class = sqlalchemy_code.split("class PhysicalAsset(Base):")[1].split("class ")[0]
    assert "ForeignKey" not in physicalasset_class, "PhysicalAsset should not have FKs"

    # DigitalTwin should only reference PhysicalAsset (can be created second)
    digitaltwin_class = sqlalchemy_code.split("class DigitalTwin(Base):")[1].split("class ")[0] if "class DigitalTwin(Base):" in sqlalchemy_code else sqlalchemy_code.split("class DigitalTwin(Base):")[1].split("#---")[0]
    assert "ForeignKey(\"physicalasset.id\")" in digitaltwin_class
    assert "ForeignKey(\"sensor.id\")" not in digitaltwin_class, "DigitalTwin should not reference Sensor"

    # Sensor should only reference DigitalTwin (can be created third)
    sensor_class = sqlalchemy_code.split("class Sensor(Base):")[1].split("class ")[0] if "class Sensor(Base):" in sqlalchemy_code else sqlalchemy_code.split("class Sensor(Base):")[1].split("#---")[0]
    assert "ForeignKey(\"digitaltwin.id\")" in sensor_class
    assert "ForeignKey(\"physicalasset.id\")" not in sensor_class, "Sensor should not reference PhysicalAsset"


def test_pydantic_formatting(relationship_model, tmpdir):
    """
    Test that Pydantic classes have proper formatting (each field on separate line).
    """
    output_dir = tmpdir.mkdir("output_fmt")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()

    # Each class should have fields on separate lines
    lines = pydantic_code.split("\n")
    
    # Check DigitalTwinCreate has proper line breaks
    dt_start = None
    for i, line in enumerate(lines):
        if "class DigitalTwinCreate(BaseModel):" in line:
            dt_start = i
            break
    
    assert dt_start is not None, "DigitalTwinCreate class not found"
    
    # Check next few lines are individual field definitions
    field_lines = [lines[dt_start + i].strip() for i in range(1, 5) if dt_start + i < len(lines)]
    
    # At least one line should contain "attribute"
    assert any("attribute" in line for line in field_lines), "attribute field not found"
    # At least one line should contain "sensors"
    assert any("sensors" in line for line in field_lines), "sensors field not found"
    # At least one line should contain "p_asset"
    assert any("p_asset" in line for line in field_lines), "p_asset field not found"
