import json
import os
import pytest
from besser.BUML.metamodel.structural import (
    AssociationClass, Class, DomainModel, Property, StringType, IntegerType, FloatType,
    BinaryAssociation, Multiplicity, Enumeration, EnumerationLiteral, BooleanType
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text, DataBinding
from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType, Table
from besser.generators.react import ReactGenerator


@pytest.fixture
def domain_model():
    """Create a minimal domain model for testing."""
    name_prop = Property(name="name", type=StringType)
    age_prop = Property(name="age", type=IntegerType)
    person = Class(name="Person", attributes={name_prop, age_prop})

    title_prop = Property(name="title", type=StringType)
    task = Class(name="Task", attributes={title_prop})

    person_end = Property(name="person_end", type=person, multiplicity=Multiplicity(1, 1))
    task_end = Property(name="task_end", type=task, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="PersonTask", ends={person_end, task_end})

    model = DomainModel(
        name="TestModel",
        types={person, task},
        associations={assoc},
    )
    return model


@pytest.fixture
def gui_model():
    """Create a minimal GUI model for testing."""
    text1 = Text(name="welcome_text", content="Hello World")
    screen1 = Screen(
        name="Home",
        description="Main page",
        view_elements={text1},
        is_main_page=True,
    )
    module1 = Module(name="MainModule", screens={screen1})
    gui = GUIModel(
        name="TestGUI",
        package="com.test",
        versionCode="1",
        versionName="1.0",
        modules={module1},
        description="Test GUI model",
    )
    return gui


def test_react_generator_instantiation(domain_model, gui_model):
    """Test that the ReactGenerator can be instantiated."""
    generator = ReactGenerator(model=domain_model, gui_model=gui_model)
    assert generator is not None
    assert generator.gui_model is gui_model


def test_react_generator_generate(domain_model, gui_model, tmpdir):
    """Test that generate() runs without errors and produces output files."""
    output_dir = tmpdir.mkdir("output")
    generator = ReactGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    # Verify the output directory has content
    generated_files = []
    for root, dirs, files in os.walk(str(output_dir)):
        for f in files:
            generated_files.append(os.path.join(root, f))

    assert len(generated_files) > 0, "ReactGenerator should produce output files"


def test_react_generator_creates_src_directory(domain_model, gui_model, tmpdir):
    """Test that the generator creates src directory with pages."""
    output_dir = tmpdir.mkdir("output")
    generator = ReactGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    src_dir = os.path.join(str(output_dir), "src")
    assert os.path.isdir(src_dir), "ReactGenerator should create a src directory"

    pages_dir = os.path.join(src_dir, "pages")
    assert os.path.isdir(pages_dir), "ReactGenerator should create a src/pages directory"

    # Check that at least one page TSX file was generated
    page_files = [f for f in os.listdir(pages_dir) if f.endswith(".tsx")]
    assert len(page_files) > 0, "At least one page component should be generated"


def test_react_generator_creates_app_tsx(domain_model, gui_model, tmpdir):
    """Test that App.tsx is generated for routing."""
    output_dir = tmpdir.mkdir("output")
    generator = ReactGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    app_tsx = os.path.join(str(output_dir), "src", "App.tsx")
    assert os.path.isfile(app_tsx), "App.tsx should be generated"

    with open(app_tsx, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Route" in content, "App.tsx should contain routing configuration"


# ---------------------------------------------------------------------------
# Map component React generator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def map_domain_model():
    """Domain model with a Location class for map binding tests."""
    lat_prop = Property(name="latitude", type=FloatType)
    lng_prop = Property(name="longitude", type=FloatType)
    name_prop = Property(name="store_name", type=StringType)
    location = Class(name="Location", attributes={lat_prop, lng_prop, name_prop})
    return DomainModel(name="MapModel", types={location}), location, lat_prop, lng_prop, name_prop


@pytest.fixture
def map_gui_model(map_domain_model):
    """GUI model containing a Map component with a points layer."""
    _, location_cls, lat_p, lng_p, name_p = map_domain_model
    layer = MapLayer(
        name="stores",
        layer_type=MapLayerType.points,
        data_binding=DataBinding(name="loc_binding", domain_concept=location_cls),
        latitude_field=lat_p,
        longitude_field=lng_p,
        label_field=name_p,
    )
    map_comp = Map(
        name="StoreMap",
        title="Store Locations",
        center_latitude=51.5,
        center_longitude=-0.09,
        zoom=12,
        layers=[layer],
    )
    screen = Screen(
        name="MapPage",
        description="Map screen",
        view_elements={map_comp},
        is_main_page=True,
    )
    module = Module(name="MapModule", screens={screen})
    return GUIModel(
        name="MapApp",
        package="com.test.map",
        versionCode="1",
        versionName="1.0",
        modules={module},
        description="Map GUI",
    )


def test_react_map_generates_without_error(map_domain_model, map_gui_model, tmpdir):
    """ReactGenerator runs without errors for a model containing a Map."""
    domain, *_ = map_domain_model
    output_dir = tmpdir.mkdir("map_output")
    generator = ReactGenerator(model=domain, gui_model=map_gui_model,
                                output_dir=str(output_dir))
    generator.generate()
    all_files = []
    for root, _, files in os.walk(str(output_dir)):
        all_files.extend(os.path.join(root, f) for f in files)
    assert len(all_files) > 0


def test_react_map_block_file_exists(map_domain_model, map_gui_model, tmpdir):
    """MapBlock.tsx runtime component is present in the generated output."""
    domain, *_ = map_domain_model
    output_dir = tmpdir.mkdir("map_output2")
    generator = ReactGenerator(model=domain, gui_model=map_gui_model,
                                output_dir=str(output_dir))
    generator.generate()

    map_block = None
    for root, _, files in os.walk(str(output_dir)):
        for f in files:
            if f == "MapBlock.tsx":
                map_block = os.path.join(root, f)
    assert map_block is not None, "MapBlock.tsx should be in the generated output"


def test_react_map_page_contains_mapblock(map_domain_model, map_gui_model, tmpdir):
    """The generated page TSX imports and uses <MapBlock>."""
    domain, *_ = map_domain_model
    output_dir = tmpdir.mkdir("map_output3")
    generator = ReactGenerator(model=domain, gui_model=map_gui_model,
                                output_dir=str(output_dir))
    generator.generate()

    pages_dir = os.path.join(str(output_dir), "src", "pages")
    page_files = [f for f in os.listdir(pages_dir) if f.endswith(".tsx")]
    assert page_files, "At least one page TSX should be generated"

    page_content = ""
    for pf in page_files:
        with open(os.path.join(pages_dir, pf), "r", encoding="utf-8") as f:
            page_content += f.read()

    assert "MapBlock" in page_content, "MapBlock should be referenced in the generated page"


def test_react_map_leaflet_in_package_json(map_domain_model, map_gui_model, tmpdir):
    """leaflet deps are present in package.json when the model contains a Map."""
    domain, *_ = map_domain_model
    output_dir = tmpdir.mkdir("map_output4")
    generator = ReactGenerator(model=domain, gui_model=map_gui_model,
                                output_dir=str(output_dir))
    generator.generate()

    pkg_json_path = os.path.join(str(output_dir), "package.json")
    assert os.path.isfile(pkg_json_path), "package.json must be generated"
    with open(pkg_json_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)

    deps = pkg.get("dependencies", {})
    assert "leaflet" in deps, "leaflet must be in package.json dependencies"
    assert "react-leaflet" not in deps, "react-leaflet (Hippocratic-2.1) must NOT be shipped"
    assert "leaflet.heat" in deps, "leaflet.heat must be in package.json dependencies"
    assert "@types/leaflet" in pkg.get("devDependencies", {}), "@types/leaflet must be in devDependencies"


def test_react_no_map_no_leaflet_deps(domain_model, gui_model, tmpdir):
    """leaflet deps and MapBlock.tsx are absent when the model has no Map component."""
    output_dir = tmpdir.mkdir("map_output5")
    generator = ReactGenerator(model=domain_model,
                                gui_model=gui_model,
                                output_dir=str(output_dir))
    generator.generate()

    with open(os.path.join(str(output_dir), "package.json"), "r", encoding="utf-8") as f:
        pkg = json.load(f)
    assert "leaflet" not in pkg.get("dependencies", {})
    assert "@types/leaflet" not in pkg.get("devDependencies", {})
    assert not os.path.isfile(
        os.path.join(str(output_dir), "src", "components", "runtime", "MapBlock.tsx")
    ), "MapBlock.tsx must not be generated without a Map component"


# ---------------------------------------------------------------------------
# Association class attributes in the create/edit form
# ---------------------------------------------------------------------------

def _build_booking_models(with_association_class: bool):
    """Booking -- Room N:M model with a table page, optionally with an association class."""
    reference = Property(name="reference", type=StringType)
    booking = Class(name="Booking", attributes={reference})

    number = Property(name="number", type=StringType, is_id=True)
    room = Class(name="Room", attributes={number})

    booking_end = Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*"))
    room_end = Property(name="rooms", type=room, multiplicity=Multiplicity(0, "*"))
    booking_room = BinaryAssociation(name="booking_room", ends={booking_end, room_end})

    types = {booking, room}
    if with_association_class:
        agreed_price = Property(name="agreed_price", type=FloatType)
        additional_charges = Property(name="additional_charges", type=FloatType)
        types.add(
            AssociationClass(
                name="ReservedRoom",
                attributes={agreed_price, additional_charges},
                association=booking_room,
            )
        )

    domain_model = DomainModel(
        name="BookingModel",
        types=types,
        associations={booking_room},
    )

    table = Table(
        name="BookingTable",
        title="Bookings",
        action_buttons=True,
        data_binding=DataBinding(name="booking_binding", domain_concept=booking),
    )
    screen = Screen(
        name="Bookings",
        description="Booking screen",
        view_elements={table},
        is_main_page=True,
    )
    module = Module(name="BookingModule", screens={screen})
    gui_model = GUIModel(
        name="BookingApp",
        package="com.test.booking",
        versionCode="1",
        versionName="1.0",
        modules={module},
        description="Booking GUI",
    )
    return domain_model, gui_model


@pytest.fixture
def assoc_class_models():
    """Booking -- Room N:M THROUGH the ReservedRoom association class."""
    return _build_booking_models(with_association_class=True)


@pytest.fixture
def plain_nm_models():
    """Booking -- Room plain N:M (no association class)."""
    return _build_booking_models(with_association_class=False)


def _form_columns(generator):
    """Extract the formColumns metadata of the first table of the serialized GUI model."""
    payload = json.loads(generator._build_generation_context()["components_json"])

    def walk(nodes):
        for node in nodes:
            chart = node.get("chart") or {}
            if "formColumns" in chart:
                return chart["formColumns"]
            found = walk(node.get("children") or [])
            if found is not None:
                return found
        return None

    for page in payload.get("pages", []):
        found = walk(page.get("components", []))
        if found is not None:
            return found
    return []


def test_form_column_carries_association_class(assoc_class_models):
    """A list lookup backed by an association class exposes its attributes."""
    domain_model, gui_model = assoc_class_models
    generator = ReactGenerator(model=domain_model, gui_model=gui_model)

    form_columns = _form_columns(generator)
    rooms = next(col for col in form_columns if col["field"] == "rooms")

    assert rooms["column_type"] == "lookup"
    assert rooms["type"] == "list"
    assert rooms["association_class"] == {
        "entity": "ReservedRoom",
        "fields": [
            {"name": "agreed_price", "type": "float", "required": True},
            {"name": "additional_charges", "type": "float", "required": True},
        ],
    }

    # Regular attribute columns are untouched
    assert all(
        "association_class" not in col for col in form_columns if col["field"] != "rooms"
    )


def test_form_column_without_association_class_unchanged(plain_nm_models):
    """A plain N:M lookup keeps exactly the metadata it had before."""
    domain_model, gui_model = plain_nm_models
    generator = ReactGenerator(model=domain_model, gui_model=gui_model)

    form_columns = _form_columns(generator)
    rooms = next(col for col in form_columns if col["field"] == "rooms")

    assert rooms == {
        "column_type": "lookup",
        "path": "rooms",
        "field": "rooms",
        "lookup_field": "number",
        "target_field": "number",
        "entity": "Room",
        "type": "list",
        "required": False,
    }
    assert all("association_class" not in col for col in form_columns)


def test_generated_page_embeds_association_class_metadata(assoc_class_models, tmp_path):
    """The generated page passes the association class metadata to the table."""
    domain_model, gui_model = assoc_class_models
    generator = ReactGenerator(
        model=domain_model, gui_model=gui_model, output_dir=str(tmp_path)
    )
    generator.generate()

    pages_dir = os.path.join(str(tmp_path), "src", "pages")
    page_content = ""
    for page_file in os.listdir(pages_dir):
        if page_file.endswith(".tsx"):
            with open(os.path.join(pages_dir, page_file), "r", encoding="utf-8") as f:
                page_content += f.read()

    assert "association_class" in page_content
    assert "ReservedRoom" in page_content
    assert "agreed_price" in page_content


def test_generated_table_component_renders_association_class_inputs(
    assoc_class_models, tmp_path
):
    """The generated table component reads, renders and submits association class attributes."""
    domain_model, gui_model = assoc_class_models
    generator = ReactGenerator(
        model=domain_model, gui_model=gui_model, output_dir=str(tmp_path)
    )
    generator.generate()

    component_path = os.path.join(
        str(tmp_path), "src", "components", "table", "TableComponent.tsx"
    )
    assert os.path.isfile(component_path), "TableComponent.tsx should be generated"
    with open(component_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Metadata key coming from the serializer and its normalized column property
    assert "association_class" in content
    assert "associationClass" in content
    # Per selected target inputs
    assert "setLinkAttrValue(col.field, String(targetId), linkField.name" in content
    # Edit prefill from the `<end>_links` payload
    assert "_links`]" in content
    # Submitted payload shape: [{ target: <id>, <association class attributes> }]
    assert "{ target: parseInt(v, 10) }" in content


def test_table_component_is_model_independent(assoc_class_models, plain_nm_models, tmp_path):
    """The table component is data driven: models with and without an association
    class generate byte-identical component code, so plain N:M rendering is unchanged."""
    assoc_domain, assoc_gui = assoc_class_models
    plain_domain, plain_gui = plain_nm_models

    assoc_dir = os.path.join(str(tmp_path), "with_assoc")
    plain_dir = os.path.join(str(tmp_path), "without_assoc")
    ReactGenerator(model=assoc_domain, gui_model=assoc_gui, output_dir=assoc_dir).generate()
    ReactGenerator(model=plain_domain, gui_model=plain_gui, output_dir=plain_dir).generate()

    rel_path = os.path.join("src", "components", "table", "TableComponent.tsx")
    with open(os.path.join(assoc_dir, rel_path), "rb") as f:
        assoc_bytes = f.read()
    with open(os.path.join(plain_dir, rel_path), "rb") as f:
        plain_bytes = f.read()

    assert assoc_bytes == plain_bytes

    # ... and the plain model never emits the association class metadata
    plain_pages_dir = os.path.join(plain_dir, "src", "pages")
    for page_file in os.listdir(plain_pages_dir):
        if page_file.endswith(".tsx"):
            with open(os.path.join(plain_pages_dir, page_file), "r", encoding="utf-8") as f:
                assert "association_class" not in f.read()


# ---------------------------------------------------------------------------
# Row keys: how the generated table addresses one row in the REST API
# ---------------------------------------------------------------------------

def _row_key_fields_by_entity(generator):
    """Map each table binding's entity to its serialized row_key_fields."""
    payload = json.loads(generator._build_generation_context()["components_json"])
    found = {}

    def walk(node):
        if isinstance(node, dict):
            if "row_key_fields" in node and "entity" in node:
                found[node["entity"]] = node["row_key_fields"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def test_data_binding_row_key_fields(assoc_class_models):
    """A table addresses rows by the declared primary key, and an association
    class by both foreign keys in the backend's route order - never by
    guessing the first column of the row."""
    domain_model, _ = assoc_class_models
    classes = {cls.name: cls for cls in domain_model.get_classes()}
    tables = {
        Table(name="BookingTable", title="Bookings", action_buttons=True,
              data_binding=DataBinding(name="booking_binding", domain_concept=classes["Booking"])),
        Table(name="RoomTable", title="Rooms", action_buttons=True,
              data_binding=DataBinding(name="room_binding", domain_concept=classes["Room"])),
        Table(name="ReservedRoomTable", title="Links", action_buttons=True,
              data_binding=DataBinding(name="link_binding", domain_concept=classes["ReservedRoom"])),
    }
    screen = Screen(name="Admin", description="Admin screen", view_elements=tables, is_main_page=True)
    gui_model = GUIModel(
        name="BookingApp", package="com.test.booking", versionCode="1", versionName="1.0",
        modules={Module(name="AdminModule", screens={screen})}, description="Booking GUI",
    )
    generator = ReactGenerator(model=domain_model, gui_model=gui_model)

    assert _row_key_fields_by_entity(generator) == {
        "Booking": ["id"],                          # surrogate key
        "Room": ["number"],                         # declared is_id attribute
        "ReservedRoom": ["bookings_id", "rooms_id"],  # /reservedroom/{bookings_id}/{rooms_id}/
    }


# ---------------------------------------------------------------------------
# Attribute defaults: preselected in the create form
# ---------------------------------------------------------------------------

def test_form_columns_carry_attribute_defaults_and_the_form_preselects_them(tmp_path):
    """An enumeration attribute with a model default (booking_status =
    pending_payment) must reach the form column as defaultValue, and the
    generated table must initialize a new record with it - an empty "" was
    posted before, which the backend rejects for an enumeration."""
    status = Enumeration(name="BookingStatus", literals={
        EnumerationLiteral(name="pending_payment"), EnumerationLiteral(name="confirmed"),
    })
    booking = Class(name="Booking", attributes={
        Property(name="reference", type=StringType),
        Property(name="booking_status", type=status, default_value="pending_payment"),
        Property(name="paid", type=BooleanType, default_value=False),
    })
    domain_model = DomainModel(name="DefaultsModel", types={booking, status})
    table = Table(name="BookingTable", title="Bookings", action_buttons=True,
                  data_binding=DataBinding(name="booking_binding", domain_concept=booking))
    screen = Screen(name="Bookings", description="Bookings", view_elements={table}, is_main_page=True)
    gui_model = GUIModel(name="DefaultsApp", package="com.test.defaults", versionCode="1", versionName="1.0",
                         modules={Module(name="M", screens={screen})}, description="Defaults GUI")
    generator = ReactGenerator(model=domain_model, gui_model=gui_model, output_dir=str(tmp_path))

    columns = {col["field"]: col for col in _form_columns(generator)}
    assert columns["booking_status"]["type"] == "enum"
    assert columns["booking_status"]["defaultValue"] == "pending_payment"
    assert columns["paid"]["defaultValue"] is False

    generator.generate()
    with open(os.path.join(str(tmp_path), "src", "components", "table", "TableComponent.tsx"), encoding="utf-8") as f:
        component = f.read()
    assert "defaultValue: (col as any).defaultValue ?? (col as any).default_value" in component
    assert "Preselect the model's default" in component
    # An enum left unselected is omitted from the payload instead of sent as ""
    assert "col.type === 'enum' && (value === undefined || value === null || value === '')" in component
