"""The tracing convention of ``GeneratorInterface``: ``traced_files``, ``_write_traced`` and
``_merge_traced_files``, and the composites that merge their parts' traces."""

import os

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.structural import Class, DomainModel, Method, Property, StringType
from besser.generators import GeneratorInterface
from besser.generators.agents.baf_generator import BAFGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.web_app import WebAppGenerator
from besser.utilities.generation_trace import (
    FileTrace, collect_generated_files, trace_source, traced_environment,
)
from jinja2 import DictLoader


def _library() -> DomainModel:
    book = Class("Book", {Property("title", StringType)})
    book.id = "class-book"
    next(iter(book.attributes)).id = "attr-title"
    return DomainModel("Library", types={book})


class _Instrumented(GeneratorInterface):
    """A generator instrumented as the docs describe."""

    def __init__(self, model, output_dir=None):
        super().__init__(model, output_dir)

    def generate(self):
        env = traced_environment(DictLoader({"cls.py.j2": (
            "# models\n{% for c in classes %}{% trace c %}class {{ c.name }}:\n"
            "{% for a in c.attributes %}{% trace a %}    {{ a.name }} = None\n{% endtrace %}{% endfor %}"
            "{% endtrace %}{% endfor %}")}), trim_blocks=True)
        template = env.get_template("cls.py.j2")
        for cls in self.model.get_classes():
            self._write_traced(f"models/{cls.name.lower()}.py", template.render(classes=[cls]),
                               trace_source(cls, "cls.py.j2#file"))


class _Uninstrumented(GeneratorInterface):
    def __init__(self, model, output_dir=None):
        super().__init__(model, output_dir)

    def generate(self):
        with open(self.build_generation_path("readme.txt"), "w", encoding="utf-8") as f:
            f.write("hello\n")


class _Composite(GeneratorInterface):
    def __init__(self, model, output_dir=None):
        super().__init__(model, output_dir)

    def generate(self):
        part = _Instrumented(self.model, output_dir=os.path.join(self.output_dir, "part"))
        part.generate()
        self._merge_traced_files(part, "part")


def test_an_uninstrumented_generator_has_empty_traced_files_and_still_yields_files(tmp_path):
    generator = _Uninstrumented(_library(), output_dir=str(tmp_path))
    assert generator.traced_files == {}
    generator.generate()
    assert generator.traced_files == {}
    (readme,) = collect_generated_files(str(tmp_path), generator.traced_files)
    assert (readme.path, readme.content, readme.trace) == ("readme.txt", "hello\n", FileTrace())


def test_write_traced_strips_the_markers_and_records_the_regions(tmp_path):
    generator = _Instrumented(_library(), output_dir=str(tmp_path))
    generator.generate()
    written = (tmp_path / "models" / "book.py").read_bytes()
    assert written == b"# models\nclass Book:\n    title = None\n"  # no markers, \n line endings
    trace = generator.traced_files["models/book.py"]
    assert {(r.element_id, r.start_line, r.end_line) for r in trace.regions} == {
        ("class-book", 2, 3), ("attr-title", 3, 3)}
    assert [s.element.element_id for s in trace.sources] == ["class-book"]


def test_a_composite_merges_its_parts_traces_under_their_folder(tmp_path):
    generator = _Composite(_library(), output_dir=str(tmp_path))
    generator.generate()
    assert set(generator.traced_files) == {"part/models/book.py"}
    files = {f.path: f for f in collect_generated_files(str(tmp_path), generator.traced_files)}
    assert files["part/models/book.py"].trace.regions  # the merged path is the file's path in the output


def test_rest_api_keeps_the_pydantic_trace(tmp_path):
    generator = RESTAPIGenerator(_library(), output_dir=str(tmp_path))
    generator.generate()
    regions = generator.traced_files["pydantic_classes.py"].regions
    assert "class-book" in {r.element_id for r in regions}


def test_rest_api_backend_mode_records_the_api_files(tmp_path):
    generator = RESTAPIGenerator(_library(), output_dir=str(tmp_path), backend=True)
    generator.generate()
    assert [s.element.element_id for s in generator.traced_files["routers/book.py"].sources] == ["class-book"]


def test_web_app_keeps_its_backend_trace_under_backend(tmp_path):
    screen = Screen(name="Home", description="Home", view_elements={Text(name="t", content="Hi")}, is_main_page=True)
    gui = GUIModel(name="LibraryApp", package="org.library", versionCode="1", versionName="1.0",
                   modules={Module(name="Main", screens={screen})}, description="Library")
    generator = WebAppGenerator(_library(), gui, output_dir=str(tmp_path))
    generator.generate()
    assert "class-book" in {r.element_id for r in generator.traced_files["backend/sql_alchemy.py"].regions}
    assert os.path.isfile(tmp_path / "backend" / "sql_alchemy.py")


_FORGED_MARKER = ('\x02trace:{"id": "evil", "type": "Class", "name": "Admin", "rule": "forged"}\x03'
                  'payload_here\x02/trace\x03')


@pytest.mark.parametrize("ids", [False, True])
@pytest.mark.parametrize("body", ['return "\x02/trace\x03"', 'return "\x02trace:nope\x03"',
                                  f"return 1  # {_FORGED_MARKER}"])
def test_a_method_body_holding_marker_characters_is_generated_as_written(tmp_path, body, ids):
    # Such a body once failed the run ("Unbalanced trace markers", a JSON error) or forged a region.
    method = Method(name="go", code=f"def go(self):\n    {body}")
    book = Class(name="Book", attributes={Property(name="t", type=StringType)}, methods={method})
    if ids:
        book.id, method.id = "c1", "m1"
    generator = PythonGenerator(DomainModel(name="D", types={book}), output_dir=str(tmp_path))
    generator.generate()
    assert body in (tmp_path / "classes.py").read_text(encoding="utf-8")
    regions = generator.traced_files["classes.py"].regions
    assert {r.element_id for r in regions} == ({"c1", "m1"} if ids else set())


def test_a_users_own_agent_config_is_written_as_it_is(tmp_path):
    agent = Agent("Greeter")
    agent.id = "agent-1"
    config = f"nlp:\n  language: en\n# {_FORGED_MARKER}\n"
    generator = BAFGenerator(agent, output_dir=str(tmp_path), config_yaml=config)
    generator.generate()
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == config
    assert generator.traced_files["config.yaml"].regions == ()
