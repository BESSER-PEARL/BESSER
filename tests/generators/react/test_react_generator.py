import os
import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType, IntegerType,
    BinaryAssociation, Multiplicity
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
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
