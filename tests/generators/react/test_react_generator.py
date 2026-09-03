import json
import os
import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType, IntegerType, FloatType,
    BinaryAssociation, Multiplicity
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text, DataBinding
from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType
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
    """leaflet and react-leaflet are present in the generated package.json."""
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
    assert "react-leaflet" in deps, "react-leaflet must be in package.json dependencies"
    assert "leaflet.heat" in deps, "leaflet.heat must be in package.json dependencies"
