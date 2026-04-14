import os
import pytest
from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, StringType, IntegerType,
    BinaryAssociation, Multiplicity
)
from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.generators.web_app import WebAppGenerator


@pytest.fixture
def domain_model():
    """Create a minimal domain model for testing."""
    name_prop = Property(name="name", type=StringType)
    email_prop = Property(name="email", type=StringType)
    user = Class(name="User", attributes={name_prop, email_prop})

    title_prop = Property(name="title", type=StringType)
    item = Class(name="Item", attributes={title_prop})

    user_end = Property(name="user_end", type=user, multiplicity=Multiplicity(1, 1))
    item_end = Property(name="item_end", type=item, multiplicity=Multiplicity(0, "*"))
    assoc = BinaryAssociation(name="UserItem", ends={user_end, item_end})

    model = DomainModel(
        name="TestModel",
        types={user, item},
        associations={assoc},
    )
    return model


@pytest.fixture
def gui_model():
    """Create a minimal GUI model for testing."""
    text1 = Text(name="title_text", content="Welcome")
    screen1 = Screen(
        name="Dashboard",
        description="Main dashboard",
        view_elements={text1},
        is_main_page=True,
    )
    module1 = Module(name="AppModule", screens={screen1})
    gui = GUIModel(
        name="TestApp",
        package="com.test.app",
        versionCode="1",
        versionName="1.0",
        modules={module1},
        description="Test web application",
    )
    return gui


def test_web_app_generator_instantiation(domain_model, gui_model):
    """Test that the WebAppGenerator can be instantiated."""
    generator = WebAppGenerator(model=domain_model, gui_model=gui_model)
    assert generator is not None
    assert generator.gui_model is gui_model
    assert generator.agent_model is None


def test_web_app_generator_generate(domain_model, gui_model, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
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

    assert len(generated_files) > 0, "WebAppGenerator should produce output files"


def test_web_app_generator_creates_frontend_and_backend(domain_model, gui_model, tmpdir):
    """Test that the generator creates both frontend and backend directories."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    frontend_dir = os.path.join(str(output_dir), "frontend")
    backend_dir = os.path.join(str(output_dir), "backend")

    assert os.path.isdir(frontend_dir), "WebAppGenerator should create a frontend directory"
    assert os.path.isdir(backend_dir), "WebAppGenerator should create a backend directory"


def test_web_app_generator_creates_docker_compose(domain_model, gui_model, tmpdir):
    """Test that generate() creates a docker-compose.yml."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    docker_compose = os.path.join(str(output_dir), "docker-compose.yml")
    assert os.path.isfile(docker_compose), "docker-compose.yml should be generated"

    with open(docker_compose, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 0, "docker-compose.yml should not be empty"


def test_web_app_generator_creates_dockerfiles(domain_model, gui_model, tmpdir):
    """Test that Dockerfiles are generated for frontend and backend."""
    output_dir = tmpdir.mkdir("output")
    generator = WebAppGenerator(
        model=domain_model,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    frontend_dockerfile = os.path.join(str(output_dir), "frontend", "Dockerfile")
    backend_dockerfile = os.path.join(str(output_dir), "backend", "Dockerfile")

    assert os.path.isfile(frontend_dockerfile), "Frontend Dockerfile should be generated"
    assert os.path.isfile(backend_dockerfile), "Backend Dockerfile should be generated"
