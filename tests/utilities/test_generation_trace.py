"""Tests for the generator-side tracing helper (besser.utilities.generation_trace)."""

from datetime import datetime

import pytest
from jinja2 import DictLoader, Environment

from besser.BUML.metamodel.structural import Class, Enumeration
from besser.BUML.metamodel.traceability import CodeArtifact, ModelArtifact, OperationType
from besser.utilities.generation_trace import (
    TraceExtension, TraceRegion, build_generation_trace, extract_trace_regions, shift_regions,
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


def _trace(content: str, cap: int):
    regions = [TraceRegion("c1", "Class", "Book", "t#class", 2, 3), TraceRegion("c1", "Class", "Book", "t#rel", 5, 5)]
    now = datetime(2026, 10, 1)
    return build_generation_trace(
        regions, file_path="out.py", content=content, model_id="diagram-1", model_name="Library",
        generator="sqlalchemy", generator_version="8.5.0", actor_name="BESSER User",
        max_stored_content_bytes=cap, started_at=now, ended_at=now)


def test_build_generation_trace_links_each_region():
    trace = _trace("h\nclass Book:\n  x\n\nrel\n", cap=1000)
    assert trace.validate()["success"]
    (operation,) = trace.operations
    assert operation.operation_type is OperationType.CODE_GENERATION
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
