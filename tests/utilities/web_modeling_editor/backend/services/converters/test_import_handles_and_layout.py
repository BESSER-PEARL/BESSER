"""Importing a ``.py`` project must produce a diagram the v4 editor can render.

Live evidence (React Flow migration): importing a class project dropped every
association because the edges carried ``sourceHandle: "Right"`` /
``targetHandle: "Left"`` while the editor's handles are lowercase
(``right``, ``left``, ``top-right``...), so React Flow could not resolve them.
The same import gave every class a fixed size on a fixed grid, so classes with
many attributes overlapped; a BPMN ``.py`` exported by ``development`` lost its
edge routing (it stores ``bounds`` + ``path`` but no ``points``); and object
attribute rows lacked ``visibility``, which makes the frontend's v3 round-trip
fall back to name parsing and drop ``attributeType``.
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel, EndEvent, Process, SequenceFlow, StartEvent, Task, TaskType,
)
from besser.utilities.buml_code_builder.bpmn_model_builder import bpmn_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
    bpmn_buml_to_json, bpmn_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)

# ``HandleId`` in packages/library/lib/nodes/wrappers/DefaultNodeWrapper.tsx.
V4_HANDLE_IDS = {
    "top-left", "top-mid-left", "top", "top-mid-right", "top-right",
    "right-top", "right-mid-top", "right", "right-mid-bottom", "right-bottom",
    "bottom-right", "bottom-mid-right", "bottom", "bottom-mid-left", "bottom-left",
    "left-bottom", "left-mid-bottom", "left", "left-mid-top", "left-top",
}

_ATTRS_AUTHOR = ", ".join(f'Property(name="a{i}", type=StringType)' for i in range(9))
_ATTRS_BOOK = ", ".join(f'Property(name="b{i}", type=IntegerType)' for i in range(12))

CLASS_PROJECT_PY = f'''
from besser.BUML.metamodel.structural import *
Author = Class(name="Author")
Book = Class(name="Book")
Library = Class(name="Library")
Publisher = Class(name="Publisher")
Author.attributes = {{{_ATTRS_AUTHOR}}}
Book.attributes = {{{_ATTRS_BOOK}}}
Library.attributes = {{Property(name="name", type=StringType)}}
Publisher.attributes = {{Property(name="name", type=StringType)}}
writes = BinaryAssociation(name="writes", ends={{
    Property(name="authors", type=Author, multiplicity=Multiplicity(1, "*")),
    Property(name="books", type=Book, multiplicity=Multiplicity(0, "*"))}})
holds = BinaryAssociation(name="holds", ends={{
    Property(name="library", type=Library, multiplicity=Multiplicity(1, 1)),
    Property(name="stock", type=Book, multiplicity=Multiplicity(0, "*"))}})
domain_model = DomainModel(name="Lib", types={{Author, Book, Library, Publisher}},
                           associations={{writes, holds}})
'''


def _class_diagram(project: dict) -> dict:
    diagrams = project["diagrams"]["ClassDiagram"]
    diagram = diagrams[0] if isinstance(diagrams, list) else diagrams
    return diagram["model"]


def _overlaps(nodes):
    hits = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            ax, ay, bx, by = a["position"]["x"], a["position"]["y"], b["position"]["x"], b["position"]["y"]
            if ax < bx + b["width"] and bx < ax + a["width"] and ay < by + b["height"] and by < ay + a["height"]:
                hits.append((a["data"].get("name"), b["data"].get("name")))
    return hits


def _assert_valid_handles(edges):
    for edge in edges:
        if edge["type"] == "ClassLinkRel":
            continue  # edge-anchored; canonical "Center"/"Up", never handed to React Flow
        assert edge["sourceHandle"] in V4_HANDLE_IDS, edge
        assert edge["targetHandle"] in V4_HANDLE_IDS, edge


def test_imported_class_project_keeps_associations_renderable():
    model = _class_diagram(project_to_json(CLASS_PROJECT_PY))
    associations = [e for e in model["edges"] if e["type"] == "ClassBidirectional"]
    assert len(associations) == 2
    _assert_valid_handles(model["edges"])


def test_imported_classes_are_sized_from_content_and_do_not_overlap():
    model = _class_diagram(project_to_json(CLASS_PROJECT_PY))
    classes = {n["data"]["name"]: n for n in model["nodes"] if n["type"] == "class"}
    # Frontend calculateMinHeight: header 40 + 30 per row, snapped up to 10.
    assert classes["Book"]["height"] == 40 + 30 * 12
    assert classes["Author"]["height"] == 40 + 30 * 9
    assert classes["Library"]["height"] == 40 + 30
    assert _overlaps(list(model["nodes"])) == []


def test_long_member_names_widen_the_class():
    code = CLASS_PROJECT_PY.replace(
        'Library.attributes = {Property(name="name"',
        'Library.attributes = {Property(name="a_really_long_attribute_name_for_width"', 1,
    )
    assert code != CLASS_PROJECT_PY
    model = _class_diagram(project_to_json(code))
    library = next(n for n in model["nodes"] if n["data"].get("name") == "Library")
    assert library["width"] > 300
    assert _overlaps(list(model["nodes"])) == []


def _class_json_with_handles(source_handle, target_handle):
    return {
        "title": "Handles",
        "model": {
            "version": "4.0.0", "type": "ClassDiagram",
            "nodes": [
                {"id": "c1", "type": "class", "position": {"x": 10, "y": 20}, "width": 160, "height": 70,
                 "data": {"name": "A", "attributes": [], "methods": []}},
                {"id": "c2", "type": "class", "position": {"x": 500, "y": 40}, "width": 160, "height": 70,
                 "data": {"name": "B", "attributes": [], "methods": []}},
            ],
            "edges": [
                {"id": "e1", "type": "ClassBidirectional", "source": "c1", "target": "c2",
                 "sourceHandle": source_handle, "targetHandle": target_handle,
                 "data": {"name": "ab", "sourceRole": "a", "sourceMultiplicity": "1",
                          "targetRole": "b", "targetMultiplicity": "*", "points": []}},
            ],
        },
    }


@pytest.mark.parametrize(
    "stored, expected",
    [(("Up", "Down"), ("top", "bottom")),  # older (development) export
     (("Right", "Topleft"), ("right", "top-left")),
     (("right-top", "bottom"), ("right-top", "bottom"))],  # already v4
)
def test_saved_handles_are_accepted_and_emitted_as_v4_ids(stored, expected):
    domain = process_class_diagram(_class_json_with_handles(*stored))
    out = class_buml_to_json(domain)
    edge = next(e for e in out["edges"] if e["type"] == "ClassBidirectional")
    assert (edge["sourceHandle"], edge["targetHandle"]) == expected
    # Saved positions from the diagram are kept, not re-laid out.
    positions = {n["data"]["name"]: n["position"] for n in out["nodes"]}
    assert positions == {"A": {"x": 10, "y": 20}, "B": {"x": 500, "y": 40}}


@pytest.mark.parametrize(
    "v3, v4",
    [("Up", "top"), ("Right", "right"), ("Down", "bottom"), ("Left", "left"),
     ("Upright", "right-top"), ("Downleft", "left-bottom"),
     ("RightTop", "top-right"), ("LeftBottom", "bottom-left"),
     ("Bottomleft", "bottom-left"), ("Topright", "top-right"),
     ("left-mid-top", "left-mid-top")],
)
def test_normalize_handle_mirrors_frontend_convertV3HandleToV4(v3, v4):
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._node_builders import (
        normalize_handle,
    )
    assert normalize_handle(v3) == v4


def test_object_rows_carry_visibility_and_type_and_objects_do_not_overlap():
    domain_json = class_buml_to_json(process_class_diagram({
        "title": "D",
        "model": {"version": "4.0.0", "type": "ClassDiagram", "edges": [], "nodes": [
            {"id": "c1", "type": "class", "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
             "data": {"name": "Book", "attributes": [
                 {"id": f"a{i}", "name": f"f{i}", "attributeType": "int", "visibility": "public"}
                 for i in range(8)], "methods": []}},
        ]},
    }))
    kwargs = ", ".join(f"f{i}={i}" for i in range(8))
    content = "\n".join(
        f'b{n} = Book("b{n}").attributes({kwargs}).build()' for n in range(6)
    )
    out = object_buml_to_json(content, domain_json)
    objects = [n for n in out["nodes"] if n["type"] == "objectName"]
    assert len(objects) == 6
    for row in objects[0]["data"]["attributes"]:
        assert row["visibility"] == "public"
        assert row["attributeType"] == "int"
    assert _overlaps(objects) == []


def _development_bpmn_model():
    """A BPMN model as a development-era ``.py`` stores it: flow layout has
    ``bounds`` + relative ``path`` and no ``points``."""
    s = StartEvent(name="start")
    t = Task(name="work", task_type=TaskType.USER)
    e = EndEvent(name="end")
    s.layout = {"id": "s1", "owner": None, "bounds": {"x": 0, "y": 0, "width": 40, "height": 40}}
    t.layout = {"id": "t1", "owner": None, "bounds": {"x": 200, "y": 100, "width": 160, "height": 60}}
    e.layout = {"id": "e1", "owner": None, "bounds": {"x": 500, "y": 0, "width": 40, "height": 40}}
    f1 = SequenceFlow(s, t)
    f1.layout = {
        "id": "f1", "owner": None,
        "bounds": {"x": 40, "y": 20, "width": 160, "height": 110},
        "path": [{"x": 0, "y": 0}, {"x": 80, "y": 0}, {"x": 80, "y": 110}, {"x": 160, "y": 110}],
        "source_direction": "Right", "target_direction": "Left", "isManuallyLayouted": True,
    }
    f2 = SequenceFlow(t, e)
    p = Process(name="P", flow_nodes={s, t, e}, sequence_flows={f1, f2})
    return BPMNModel(name="Dev", processes={p})


def test_development_bpmn_flow_path_becomes_v4_points():
    expected = [{"x": 40, "y": 20}, {"x": 120, "y": 20}, {"x": 120, "y": 130}, {"x": 200, "y": 130}]
    for out in (bpmn_object_to_json(_development_bpmn_model()),
                bpmn_buml_to_json(bpmn_model_to_code(_development_bpmn_model()))):
        edges = {e["id"]: e for e in out["edges"]}
        assert edges["f1"]["data"]["points"] == expected
        _assert_valid_handles(out["edges"])
