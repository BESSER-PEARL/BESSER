import os
import pytest
from besser.generators.backend import BackendGenerator
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, IntegerType, StringType, FloatType, DateTimeType,
    BinaryAssociation, Multiplicity, Generalization
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


def _read(output_dir, *parts):
    """Helper to read a generated file from the layered output structure."""
    path = os.path.join(str(output_dir), *parts)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_simple_generator(simple_model, tmpdir):
    """Test basic N:M relationship generation with layered structure"""
    output_dir = tmpdir.mkdir("output")
    generator = BackendGenerator(model=simple_model, output_dir=str(output_dir))
    generator.generate()

    # Check layered structure exists
    app_dir = os.path.join(str(output_dir), "app")
    assert os.path.isdir(app_dir)
    assert os.path.isdir(os.path.join(app_dir, "models"))
    assert os.path.isdir(os.path.join(app_dir, "schemas"))
    assert os.path.isdir(os.path.join(app_dir, "routers"))

    # Check per-entity files exist
    assert os.path.isfile(os.path.join(app_dir, "models", "name1.py"))
    assert os.path.isfile(os.path.join(app_dir, "schemas", "name1.py"))
    assert os.path.isfile(os.path.join(app_dir, "routers", "name1.py"))

    # Pydantic checks
    schema1 = _read(output_dir, "app", "schemas", "name1.py")
    schema2 = _read(output_dir, "app", "schemas", "name2.py")
    assert "class name1Create(BaseModel):" in schema1
    assert "attr1: int" in schema1
    assert "assocs2: List[int]" in schema1
    assert "class name2Create(BaseModel):" in schema2
    assert "attr2: int" in schema2
    assert "assocs1: List[int]" in schema2

    # SQLAlchemy checks
    model1 = _read(output_dir, "app", "models", "name1.py")
    assert "class name1(" in model1
    assert "id: Mapped[int] = mapped_column(primary_key=True)" in model1
    assert "attr1: Mapped[int] = mapped_column(Integer)" in model1

    # API checks (routers use @router. prefix, not @app.)
    router1 = _read(output_dir, "app", "routers", "name1.py")
    assert '@router.get("/", response_model=None)' in router1
    assert "def get_all_name1(detailed: bool = False, database: Session = Depends(get_db))" in router1
    assert "return database.query(name1).all()" in router1


def test_relationship_fk_placement(relationship_model, tmpdir):
    """
    Test that FKs are placed correctly based on multiplicity constraints.
    Each entity is now in its own file, making assertions cleaner.
    """
    output_dir = tmpdir.mkdir("output_rel")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    pa_model = _read(output_dir, "app", "models", "physicalasset.py")
    dt_model = _read(output_dir, "app", "models", "digitaltwin.py")
    sensor_model = _read(output_dir, "app", "models", "sensor.py")

    # Test 1: PhysicalAsset should NOT have any FK column
    assert "class PhysicalAsset(" in pa_model
    assert "mapped_column(ForeignKey(" not in pa_model
    assert "dt_id" not in pa_model

    # Test 2: Sensor should have dt_id FK (N:1 mandatory)
    assert "class Sensor(" in sensor_model
    assert "dt_id: Mapped[int] = mapped_column(ForeignKey(\"digitaltwin.id\")" in sensor_model
    dt_id_line = [line for line in sensor_model.split('\n') if 'dt_id' in line and 'mapped_column' in line][0]
    assert "nullable=True" not in dt_id_line, "Mandatory FK should not have nullable=True"

    # Test 3: DigitalTwin should have p_asset_id FK (1:1 mandatory)
    assert "class DigitalTwin(" in dt_model
    assert "p_asset_id: Mapped[int] = mapped_column(ForeignKey(\"physicalasset.id\")" in dt_model
    p_asset_id_line = [line for line in dt_model.split('\n') if 'p_asset_id' in line and 'mapped_column' in line][0]
    assert "unique=True" in p_asset_id_line, "1:1 relationship should have unique=True"
    assert "nullable=True" not in p_asset_id_line, "Mandatory FK should not have nullable=True"


def test_pydantic_multiplicity_constraints(relationship_model, tmpdir):
    """
    Test that Pydantic models respect multiplicity constraints.
    """
    output_dir = tmpdir.mkdir("output_pyd")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    dt_schema = _read(output_dir, "app", "schemas", "digitaltwin.py")
    pa_schema = _read(output_dir, "app", "schemas", "physicalasset.py")
    sensor_schema = _read(output_dir, "app", "schemas", "sensor.py")

    # Test 1: DigitalTwinCreate should have REQUIRED p_asset (1:1 mandatory)
    assert "class DigitalTwinCreate(" in dt_schema
    assert "p_asset: int" in dt_schema
    assert "# 1:1 Relationship (mandatory)" in dt_schema

    # Test 2: PhysicalAssetCreate should have OPTIONAL dt (1:1 optional)
    assert "class PhysicalAssetCreate(" in pa_schema
    assert "dt: Optional[int] = None" in pa_schema
    assert "# 1:1 Relationship (optional)" in pa_schema

    # Test 3: SensorCreate should have REQUIRED dt (N:1 mandatory)
    assert "class SensorCreate(" in sensor_schema
    assert "dt: int" in sensor_schema
    assert "# N:1 Relationship (mandatory)" in sensor_schema

    # Test 4: DigitalTwinCreate should have OPTIONAL sensors list (1:N)
    assert "sensors: Optional[List[int]] = None" in dt_schema
    assert "# 1:N Relationship" in dt_schema


def test_api_fk_validation_and_assignment(relationship_model, tmpdir):
    """
    Test that REST API endpoints correctly validate and assign FKs.
    Each entity router is now in its own file.
    """
    output_dir = tmpdir.mkdir("output_api")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    pa_router = _read(output_dir, "app", "routers", "physicalasset.py")
    sensor_router = _read(output_dir, "app", "routers", "sensor.py")
    dt_router = _read(output_dir, "app", "routers", "digitaltwin.py")

    # Test 1: PhysicalAsset POST should NOT validate any FK
    assert "@router.post(" in pa_router
    post_idx = pa_router.find("@router.post(")
    put_idx = pa_router.find("@router.put(", post_idx)
    pa_post_section = pa_router[post_idx:put_idx]
    assert "PhysicalAsset(" in pa_post_section
    creation_idx = pa_post_section.find("PhysicalAsset(")
    pre_creation = pa_post_section[:creation_idx]
    assert "database.query" not in pre_creation or "DigitalTwin" not in pre_creation

    # Test 2: Sensor POST should validate dt and assign dt_id
    post_idx = sensor_router.find("@router.post(")
    put_idx = sensor_router.find("@router.put(", post_idx)
    sensor_post_section = sensor_router[post_idx:put_idx]
    assert "if sensor_data.dt is not None:" in sensor_post_section
    assert "db_dt = database.query(DigitalTwin)" in sensor_post_section
    assert "DigitalTwin ID is required" in sensor_post_section
    assert "dt_id=sensor_data.dt" in sensor_post_section

    # Test 3: DigitalTwin POST should validate p_asset and assign p_asset_id
    post_idx = dt_router.find("@router.post(")
    put_idx = dt_router.find("@router.put(", post_idx)
    dt_post_section = dt_router[post_idx:put_idx]
    assert "if digitaltwin_data.p_asset is not None:" in dt_post_section
    assert "db_p_asset = database.query(PhysicalAsset)" in dt_post_section
    assert "PhysicalAsset ID is required" in dt_post_section
    assert "p_asset_id=digitaltwin_data.p_asset" in dt_post_section

    # Test 4: Sensor PUT should update dt_id
    put_idx = sensor_router.find("@router.put(")
    delete_idx = sensor_router.find("@router.delete(", put_idx)
    sensor_put_section = sensor_router[put_idx:delete_idx]
    assert "if sensor_data.dt is not None:" in sensor_put_section
    assert "setattr(db_sensor, 'dt_id', sensor_data.dt)" in sensor_put_section

    # Test 5: DigitalTwin PUT should update p_asset_id
    put_idx = dt_router.find("@router.put(")
    delete_idx = dt_router.find("@router.delete(", put_idx)
    dt_put_section = dt_router[put_idx:delete_idx]
    assert "if digitaltwin_data.p_asset is not None:" in dt_put_section
    assert "setattr(db_digitaltwin, 'p_asset_id', digitaltwin_data.p_asset)" in dt_put_section


def test_no_circular_dependency(relationship_model, tmpdir):
    """
    Test that the generated code allows proper entity creation order.
    With per-entity files, each model file is self-contained.
    """
    output_dir = tmpdir.mkdir("output_circ")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    pa_model = _read(output_dir, "app", "models", "physicalasset.py")
    dt_model = _read(output_dir, "app", "models", "digitaltwin.py")
    sensor_model = _read(output_dir, "app", "models", "sensor.py")

    # PhysicalAsset should have no FK columns
    assert "mapped_column(ForeignKey(" not in pa_model

    # DigitalTwin should only reference PhysicalAsset
    assert "ForeignKey(\"physicalasset.id\")" in dt_model
    assert "ForeignKey(\"sensor.id\")" not in dt_model

    # Sensor should only reference DigitalTwin
    assert "ForeignKey(\"digitaltwin.id\")" in sensor_model
    assert "ForeignKey(\"physicalasset.id\")" not in sensor_model


def test_pydantic_formatting(relationship_model, tmpdir):
    """
    Test that Pydantic classes have proper formatting (each field on separate line).
    """
    output_dir = tmpdir.mkdir("output_fmt")
    generator = BackendGenerator(model=relationship_model, output_dir=str(output_dir))
    generator.generate()

    dt_schema = _read(output_dir, "app", "schemas", "digitaltwin.py")
    lines = dt_schema.split("\n")

    # Check DigitalTwinCreate has proper line breaks
    dt_start = None
    for i, line in enumerate(lines):
        if "class DigitalTwinCreate(" in line:
            dt_start = i
            break

    assert dt_start is not None, "DigitalTwinCreate class not found"

    field_lines = [lines[dt_start + i].strip() for i in range(1, 5) if dt_start + i < len(lines)]

    assert any("attribute" in line for line in field_lines), "attribute field not found"
    assert any("sensors" in line for line in field_lines), "sensors field not found"
    assert any("p_asset" in line for line in field_lines), "p_asset field not found"


def test_rest_api_inherited_constructor_args(tmpdir):
    """
    Ensure REST API creation endpoints include constructor arguments
    from all ancestor classes (A <- B <- C), including inherited FKs.
    """
    class_a = Class(name="A", attributes={
        Property(name="a_attr", type=StringType),
    })
    class_b = Class(name="B", attributes={
        Property(name="b_attr", type=IntegerType),
    })
    class_c = Class(name="C", attributes={
        Property(name="c_attr", type=StringType),
    })
    ref = Class(name="Ref", attributes={
        Property(name="label", type=StringType),
    })

    # A <- B <- C inheritance chain
    Generalization(general=class_a, specific=class_b)
    Generalization(general=class_b, specific=class_c)

    # A has a mandatory many-to-one association to Ref (FK on A)
    a_ref = BinaryAssociation(
        name="a_ref",
        ends={
            Property(name="as", type=class_a, multiplicity=Multiplicity(0, "*")),
            Property(name="ref", type=ref, multiplicity=Multiplicity(1, 1)),
        },
    )

    model = DomainModel(
        name="InheritanceRestApiTest",
        types={class_a, class_b, class_c, ref},
        associations={a_ref},
    )

    output_dir = tmpdir.mkdir("output_inheritance")
    generator = BackendGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    c_router = _read(output_dir, "app", "routers", "c.py")
    a_schema = _read(output_dir, "app", "schemas", "a.py")
    b_schema = _read(output_dir, "app", "schemas", "b.py")
    c_schema = _read(output_dir, "app", "schemas", "c.py")
    a_model = _read(output_dir, "app", "models", "a.py")
    c_model = _read(output_dir, "app", "models", "c.py")

    # Direct create should include all ancestor attributes
    assert "a_attr=c_data.a_attr" in c_router
    assert "b_attr=c_data.b_attr" in c_router
    assert "c_attr=c_data.c_attr" in c_router
    # Inherited FK from A should be included in constructor args
    assert "ref_id=c_data.ref" in c_router

    # Bulk create should include all ancestor attributes too
    assert "a_attr=item_data.a_attr" in c_router
    assert "b_attr=item_data.b_attr" in c_router
    assert "c_attr=item_data.c_attr" in c_router
    assert "ref_id=item_data.ref" in c_router

    # Pydantic backend classes should preserve deep inheritance
    assert "class ACreate(" in a_schema
    assert "class BCreate(ACreate):" in b_schema
    assert "class CCreate(BCreate):" in c_schema
    assert "a_attr: str" in a_schema
    assert "b_attr: int" in b_schema
    assert "c_attr: str" in c_schema

    # SQLAlchemy classes should preserve deep inheritance
    assert "class A(" in a_model
    assert "class C(B):" in c_model
