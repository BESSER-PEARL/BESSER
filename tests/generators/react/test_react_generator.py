import os
import pytest
import json
import io
import wave
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
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


def test_generated_read_aloud_synthesizes_final_text_without_preview_cache(domain_model, gui_model, tmp_path):
    from besser.generators.gui.tts.base import SpeechAudio
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b'\x00\x00' * 100)
    text = next(iter(next(iter(next(iter(gui_model.modules)).screens)).view_elements))
    text.content = 'Hello World everyone'
    text.custom_attributes = {'data-besser-read-aloud': 'true', 'data-besser-audio': json.dumps({'text': 'Legacy cached text', 'assetId': 'a' * 64})}
    provider = SimpleNamespace(configuration={'provider': 'fake', 'model': 'fake'}, text2speech=MagicMock(return_value=SpeechAudio(audio.getvalue())))
    generator = ReactGenerator(model=domain_model, gui_model=gui_model, output_dir=str(tmp_path / 'app'))
    with patch('besser.generators.gui.tts.create_text2speech', return_value=provider): generator.generate()
    page = next((tmp_path / 'app' / 'src' / 'pages').glob('*.tsx')).read_text(encoding='utf-8')
    assert '<ReadAloud' in page
    assert 'import.meta.env.BASE_URL' in page
    assert next((tmp_path / 'app' / 'public' / 'audio').glob('*.wav')).read_bytes() == audio.getvalue()
    assert (tmp_path / 'app' / 'src' / 'components' / 'runtime' / 'ReadAloud.tsx').exists()
    text.content = 'Read the new instructions carefully.'
    generator = ReactGenerator(model=domain_model, gui_model=gui_model, output_dir=str(tmp_path / 'stale'))
    with patch('besser.generators.gui.tts.create_text2speech', return_value=provider): generator.generate()
    page = next((tmp_path / 'stale' / 'src' / 'pages').glob('*.tsx')).read_text(encoding='utf-8')
    assert '<ReadAloud' in page
    assert provider.text2speech.call_args.args[0] == text.content


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
