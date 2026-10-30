"""Tests for the generator-side tracing helper (besser.utilities.generation_trace)."""

from datetime import datetime

import pytest
from jinja2 import DictLoader, Environment

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.metamodel.traceability import CodeArtifact, CodeGeneration, ModelArtifact
from besser.utilities.generation_trace import (
    ElementRef, FileTrace, GeneratedFile, TraceExtension, TraceRegion, TraceSource, build_generation_trace,
    extract_trace_regions, shift_regions,
)

TEMPLATE = """header
{% for c in classes %}{% trace c, "cls" %}
class {{ c.name }}:
    {%- for a in c.attrs %}
    {% trace a %}{{ a.name }} = 1{% endtrace %}
    {%- endfor %}
{% endtrace %}{% endfor %}
footer"""


class _Attr:
    def __init__(self, name, id):
        self.name, self.id = name, id


class _Cls:
    def __init__(self, name, id, attrs=()):
        self.name, self.id, self.attrs = name, id, list(attrs)


def _render(template: str, **context) -> str:
    env = Environment(loader=DictLoader({"t.j2": template}), extensions=[TraceExtension])
    return env.get_template("t.j2").render(**context)


def _plain(template: str, **context) -> str:
    """The template rendered with the trace tags removed, i.e. what an untraced template gives."""
    plain = template.replace('{% trace c, "cls" %}', "").replace("{% trace a %}", "").replace("{% endtrace %}", "")
    return Environment(loader=DictLoader({"t.j2": plain})).get_template("t.j2").render(**context)


def test_markers_leave_the_output_unchanged():
    classes = [_Cls("A", "a", [_Attr("x", "ax")]), _Cls("B", "b")]
    clean, _ = extract_trace_regions(_render(TEMPLATE, classes=classes))
    assert clean == _plain(TEMPLATE, classes=classes)
    assert "\x02" not in clean and "\x03" not in clean


def test_regions_cover_the_block_lines_and_nest():
    classes = [_Cls("A", "a", [_Attr("x", "ax"), _Attr("y", "ay")]), _Cls("B", "b")]
    clean, regions = extract_trace_regions(_render(TEMPLATE, classes=classes))
    lines = clean.split("\n")
    by_id = {r.element_id: r for r in regions}
    assert lines[by_id["a"].start_line - 1] == "class A:"
    assert lines[by_id["a"].end_line - 1].strip() == "y = 1"
    assert lines[by_id["ax"].start_line - 1].strip() == "x = 1"
    assert by_id["ax"].start_line == by_id["ax"].end_line
    assert by_id["a"].start_line < by_id["ax"].start_line <= by_id["a"].end_line
    assert by_id["a"].rule == "t.j2#cls"
    assert by_id["ax"].rule == "t.j2#_Attr"  # no label: the element's type


def test_element_without_id_is_rendered_untraced():
    clean, regions = extract_trace_regions(_render(TEMPLATE, classes=[_Cls("A", None)]))
    assert regions == []
    assert "class A:" in clean


def test_block_that_renders_only_whitespace_gives_no_region():
    _, regions = extract_trace_regions(_render("{% trace c %}   \n{% endtrace %}x", c=_Cls("A", "a")))
    assert regions == []


def test_metamodel_element_names_and_types_are_recorded():
    enum = Enumeration("Color", id="e1")
    _, regions = extract_trace_regions(_render("{% trace e %}enum {{ e.name }}{% endtrace %}", e=enum))
    assert regions == [TraceRegion("e1", "Enumeration", "Color", "t.j2#Enumeration", 1, 1)]


def test_unbalanced_markers_are_rejected():
    rendered = _render("{% trace c %}x{% endtrace %}", c=_Cls("A", "a"))
    with pytest.raises(ValueError):
        extract_trace_regions(rendered[:-3])
    with pytest.raises(ValueError):
        extract_trace_regions("\x02/trace\x03")


def test_shift_regions_moves_every_line():
    region = TraceRegion("a", "Class", "A", "r", 3, 5)
    assert shift_regions([region], 1) == [TraceRegion("a", "Class", "A", "r", 4, 6)]


def _build(files, cap: int = 1000):
    now = datetime(2026, 10, 1)
    return build_generation_trace(
        files, model_id="diagram-1", model_name="Library", generator="sqlalchemy", generator_version="8.5.0",
        actor_name="BESSER User", max_stored_content_bytes=cap, started_at=now, ended_at=now)


def _trace(content: str, cap: int):
    regions = (TraceRegion("c1", "Class", "Book", "t#class", 2, 3), TraceRegion("c1", "Class", "Book", "t#rel", 5, 5))
    return _build([GeneratedFile("out.py", content, FileTrace(regions))], cap)


def test_build_generation_trace_links_each_region():
    trace = _trace("h\nclass Book:\n  x\n\nrel\n", cap=1000)
    assert trace.validate()["success"]
    (operation,) = trace.operations
    assert isinstance(operation, CodeGeneration)
    assert [a.name for a in operation.performed_by] == ["BESSER User"]
    model, file = trace.artifacts[:2]
    assert isinstance(model, ModelArtifact) and model.element_id == "diagram-1"
    assert isinstance(file, CodeArtifact) and file.content is not None
    # One model artifact per element, one fragment per region, one link per region plus model -> file.
    book = trace.model_artifact_for("c1")
    assert book.parent is model
    fragments = trace.fragments_of(file)
    assert [(f.start_line, f.end_line) for f in fragments] == [(2, 3), (5, 5)]
    assert len(trace.trace_links) == 3
    assert {link.target for link in trace.trace_links if link.source is book} == set(fragments)
    assert all(link.confidence.value == 1.0 for link in trace.trace_links)


def test_build_generation_trace_keeps_only_the_hash_above_the_cap():
    content = "h\nclass Book:\n  x\n\nrel\n"
    _, file = _trace(content, cap=len(content) - 1).artifacts[:2]
    assert file.content is None and file.content_hash


def test_build_generation_trace_over_several_files():
    """A backend run: a file with blocks, a file generated as a whole from a class, and boilerplate."""
    book, close = ElementRef("c1", "Class", "Book"), ElementRef("m1", "Method", "close")
    files = [
        GeneratedFile("main_api.py", "app\ninclude(book)\n",
                      FileTrace((TraceRegion("c1", "Class", "Book", "main_api.py.j2#router_include", 2, 2),))),
        GeneratedFile("routers/book.py", "router for book\n", FileTrace(sources=(TraceSource(book, "router.py.j2#router"),))),
        GeneratedFile("routers/book_methods.py", "def close():\n  pass\n",
                      FileTrace((TraceRegion("m1", "Method", "close", "router_methods.py.j2#method", 1, 2, owner=book),))),
        GeneratedFile("database.py", "engine\n"),
    ]
    trace = _build(files)
    assert trace.validate()["success"]
    model = trace.model_artifact_for("diagram-1")
    by_path = {a.path: a for a in trace.artifacts if isinstance(a, CodeArtifact) and a.start_line is None}
    assert set(by_path) == {"main_api.py", "routers/book.py", "routers/book_methods.py", "database.py"}
    # The model is linked to every file; the class to its router as a whole; the method nests under its class.
    assert {link.target.path for link in trace.trace_links if link.source is model} == set(by_path)
    book_artifact, close_artifact = trace.model_artifact_for("c1"), trace.model_artifact_for(close.element_id)
    assert by_path["routers/book.py"] in {link.target for link in trace.trace_links if link.source is book_artifact}
    assert close_artifact.parent is book_artifact
    # Content is kept only for files with fragments (the views show their lines).
    assert by_path["main_api.py"].content is not None and by_path["routers/book_methods.py"].content is not None
    assert by_path["routers/book.py"].content is None and by_path["database.py"].content is None
    assert all(f.content_hash for f in by_path.values())


def test_a_traced_block_records_its_element_owner():
    from besser.BUML.metamodel.structural import Class, Method
    method = Method("close")
    owner = Class("Book", methods={method})
    owner.id, method.id = "c1", "m1"
    _, (region,) = extract_trace_regions(_render("{% trace m %}def close(): pass{% endtrace %}", m=method))
    assert region.owner == ElementRef("c1", "Class", "Book")
