"""
Roundtrip tests for JSON -> BUML -> JSON converters.

All tests are written against the v4 wire shape (``{nodes, edges}``).
See ``docs/source/migrations/uml-v4-shape.md``.
"""

import json
from collections import Counter

import pytest

from besser.BUML.metamodel.structural import MethodImplementationType
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.state_machine_processor import (
    process_state_machine,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.state_machine_converter import (
    state_machine_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
    process_nn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
    nn_model_to_json,
)


# ---------------------------------------------------------------------------
# v4 helpers
# ---------------------------------------------------------------------------

def _model(payload):
    """Return the inner v4 model dict — accepts both the {title, model} fixture
    envelope and the bare ``{nodes, edges, ...}`` converter output."""
    if isinstance(payload, dict) and "nodes" in payload and "edges" in payload:
        return payload
    return (payload or {}).get("model") or {}


def _nodes(payload):
    return _model(payload).get("nodes") or []


def _edges(payload):
    return _model(payload).get("edges") or []


def _nodes_by_type(payload, node_type):
    return [n for n in _nodes(payload) if n.get("type") == node_type]


def _edges_by_type(payload, edge_type):
    return [e for e in _edges(payload) if e.get("type") == edge_type]


def _data(node):
    return node.get("data") or {}


def _node_by_id(payload, node_id):
    for n in _nodes(payload):
        if n.get("id") == node_id:
            return n
    return None


def _class_nodes(payload):
    """Return all v4 class nodes (concrete, abstract, enumeration)."""
    return _nodes_by_type(payload, "class")


def _class_node_by_name(payload, class_name):
    for n in _class_nodes(payload):
        if (_data(n).get("name") or "") == class_name:
            return n
    return None


def _class_names_by_kind(payload, *, abstract=False, enumeration=False):
    """Return the set of names for class nodes matching the given stereotype."""
    out = set()
    for n in _class_nodes(payload):
        data = _data(n)
        st = (data.get("stereotype") or "").strip().lower()
        if abstract and st == "abstract":
            out.add(data.get("name"))
        elif enumeration and st == "enumeration":
            out.add(data.get("name"))
        elif not abstract and not enumeration and not st:
            out.add(data.get("name"))
    return out


def _all_class_names(payload):
    return {(_data(n).get("name") or "") for n in _class_nodes(payload)}


def _attribute_names_for_class(payload, class_name):
    node = _class_node_by_name(payload, class_name)
    if not node:
        return set()
    return {(a.get("name") or "") for a in _data(node).get("attributes") or []}


def _attributes_for_class(payload, class_name):
    """Return a name -> attribute-row dict for the given class."""
    node = _class_node_by_name(payload, class_name)
    if not node:
        return {}
    return {a.get("name"): a for a in _data(node).get("attributes") or [] if a.get("name")}


def _method_names_for_class(payload, class_name):
    """Return the *bare* method names (stripped of visibility / signature)."""
    node = _class_node_by_name(payload, class_name)
    if not node:
        return set()
    out = set()
    for m in _data(node).get("methods") or []:
        sig = m.get("name", "") or ""
        if sig and sig[0] in "+-#~":
            sig = sig[1:].lstrip()
        if "(" in sig:
            out.add(sig[:sig.index("(")].strip())
        else:
            out.add(sig.strip())
    return out


# ---------------------------------------------------------------------------
# Fixtures: hand-authored v4 payloads.
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_class_diagram_json():
    """A minimal v4 class diagram with two classes, attributes, a method,
    and a bidirectional association."""
    return {
        "title": "LibraryModel",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-book",
                    "type": "class",
                    "position": {"x": 0, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Book",
                        "stereotype": None,
                        "attributes": [
                            {"id": "n-book-a-title", "name": "title", "attributeType": "str", "visibility": "public", "isOptional": False, "isId": False, "isExternalId": False},
                            {"id": "n-book-a-pages", "name": "pages", "attributeType": "int", "visibility": "private", "isOptional": True, "isId": False, "isExternalId": False},
                        ],
                        "methods": [
                            {"id": "n-book-m-summary", "name": "+ summary(verbose: bool): str"},
                        ],
                    },
                },
                {
                    "id": "n-author",
                    "type": "class",
                    "position": {"x": 300, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Author",
                        "stereotype": None,
                        "attributes": [
                            {"id": "n-author-a-name", "name": "name", "attributeType": "str", "visibility": "public", "isOptional": False},
                        ],
                        "methods": [],
                    },
                },
            ],
            "edges": [
                {
                    "id": "e-writes",
                    "source": "n-author",
                    "target": "n-book",
                    "type": "ClassBidirectional",
                    "sourceHandle": "Right",
                    "targetHandle": "Left",
                    "data": {
                        "name": "writes",
                        "sourceRole": "author",
                        "sourceMultiplicity": "1..*",
                        "targetRole": "book",
                        "targetMultiplicity": "0..*",
                        "points": [],
                    },
                },
            ],
        },
    }


@pytest.fixture
def class_diagram_with_inheritance_json():
    return {
        "title": "VehicleModel",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-vehicle",
                    "type": "class",
                    "position": {"x": 0, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Vehicle",
                        "stereotype": "abstract",
                        "attributes": [
                            {"id": "a-speed", "name": "speed", "attributeType": "float", "visibility": "public"},
                        ],
                        "methods": [],
                    },
                },
                {
                    "id": "n-car",
                    "type": "class",
                    "position": {"x": 0, "y": 200},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Car",
                        "stereotype": None,
                        "attributes": [
                            {"id": "a-doors", "name": "doors", "attributeType": "int", "visibility": "public"},
                        ],
                        "methods": [],
                    },
                },
            ],
            "edges": [
                {
                    "id": "e-inherit",
                    "source": "n-car",
                    "target": "n-vehicle",
                    "type": "ClassInheritance",
                    "sourceHandle": "Top",
                    "targetHandle": "Bottom",
                    "data": {"points": []},
                },
            ],
        },
    }


@pytest.fixture
def class_diagram_with_enum_json():
    return {
        "title": "OrderModel",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-status",
                    "type": "class",
                    "position": {"x": 300, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "OrderStatus",
                        "stereotype": "enumeration",
                        "attributes": [
                            {"id": "lit-pending", "name": "PENDING", "attributeType": "str", "visibility": "public"},
                            {"id": "lit-shipped", "name": "SHIPPED", "attributeType": "str", "visibility": "public"},
                            {"id": "lit-delivered", "name": "DELIVERED", "attributeType": "str", "visibility": "public"},
                        ],
                        "methods": [],
                    },
                },
                {
                    "id": "n-order",
                    "type": "class",
                    "position": {"x": 0, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Order",
                        "stereotype": None,
                        "attributes": [
                            {"id": "a-status", "name": "status", "attributeType": "OrderStatus", "visibility": "public", "isOptional": False},
                            {"id": "a-total", "name": "total", "attributeType": "float", "visibility": "public", "isOptional": False},
                        ],
                        "methods": [],
                    },
                },
            ],
            "edges": [],
        },
    }


@pytest.fixture
def class_diagram_with_composition_json():
    return {
        "title": "CompanyModel",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [
                {
                    "id": "n-company",
                    "type": "class",
                    "position": {"x": 0, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Company",
                        "stereotype": None,
                        "attributes": [
                            {"id": "a-cname", "name": "companyName", "attributeType": "str", "visibility": "public"},
                        ],
                        "methods": [],
                    },
                },
                {
                    "id": "n-dept",
                    "type": "class",
                    "position": {"x": 300, "y": 0},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "Department",
                        "stereotype": None,
                        "attributes": [
                            {"id": "a-dname", "name": "deptName", "attributeType": "str", "visibility": "public"},
                        ],
                        "methods": [],
                    },
                },
            ],
            "edges": [
                {
                    "id": "e-has",
                    "source": "n-company",
                    "target": "n-dept",
                    "type": "ClassComposition",
                    "sourceHandle": "Right",
                    "targetHandle": "Left",
                    "data": {
                        "name": "has_departments",
                        "sourceRole": "company",
                        "sourceMultiplicity": "1",
                        "targetRole": "department",
                        "targetMultiplicity": "1..*",
                        "points": [],
                    },
                },
            ],
        },
    }


@pytest.fixture
def state_machine_json():
    return {
        "title": "OrderStateMachine",
        "model": {
            "version": "4.0.0",
            "type": "StateMachineDiagram",
            "nodes": [
                {
                    "id": "n-init",
                    "type": "StateInitialNode",
                    "position": {"x": -850, "y": -280},
                    "width": 45, "height": 45,
                    "measured": {"width": 45, "height": 45},
                    "data": {"name": ""},
                },
                {
                    "id": "n-idle",
                    "type": "State",
                    "position": {"x": -550, "y": -300},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "idle",
                        "bodies": [{"id": "body-idle", "name": "idle_body"}],
                        "fallbackBodies": [],
                    },
                },
                {
                    "id": "n-proc",
                    "type": "State",
                    "position": {"x": -60, "y": -300},
                    "width": 160, "height": 100,
                    "measured": {"width": 160, "height": 100},
                    "data": {
                        "name": "processing",
                        "bodies": [{"id": "body-proc", "name": "process_body"}],
                        "fallbackBodies": [],
                    },
                },
                {
                    "id": "code-idle",
                    "type": "StateCodeBlock",
                    "position": {"x": -970, "y": 80},
                    "width": 580, "height": 200,
                    "measured": {"width": 580, "height": 200},
                    "data": {
                        "name": "idle_body",
                        "code": "def idle_body(session):\n    session.reply('Waiting for input.')",
                        "language": "python",
                    },
                },
                {
                    "id": "code-proc",
                    "type": "StateCodeBlock",
                    "position": {"x": -360, "y": 80},
                    "width": 580, "height": 200,
                    "measured": {"width": 580, "height": 200},
                    "data": {
                        "name": "process_body",
                        "code": "def process_body(session):\n    session.reply('Processing...')",
                        "language": "python",
                    },
                },
                {
                    "id": "code-start",
                    "type": "StateCodeBlock",
                    "position": {"x": 250, "y": 80},
                    "width": 580, "height": 200,
                    "measured": {"width": 580, "height": 200},
                    "data": {
                        "name": "start_event",
                        "code": "def start_event(session):\n    return session.message",
                        "language": "python",
                    },
                },
            ],
            "edges": [
                {
                    "id": "t-init",
                    "source": "n-init",
                    "target": "n-idle",
                    "type": "StateTransition",
                    "sourceHandle": "Right",
                    "targetHandle": "Left",
                    "data": {"name": "", "points": []},
                },
                {
                    "id": "t-start",
                    "source": "n-idle",
                    "target": "n-proc",
                    "type": "StateTransition",
                    "sourceHandle": "Right",
                    "targetHandle": "Left",
                    "data": {"name": "start_event", "points": []},
                },
            ],
        },
    }


def _build_object_diagram_v4_fixture(
    class_diagram_v4, *, include_attributes=True, attribute_style="split",
):
    """Build a v4 object diagram referencing the given v4 class diagram.

    Two objects (book1 of Book, author1 of Author) and a link between them.

    ``attribute_style`` selects the attribute-row shape: ``"split"`` is the
    canonical v4 form (separate ``name`` / ``value`` fields, per
    ``ObjectNodeAttribute``); ``"legacy"`` bakes the combined
    ``"+ name: type = value"`` string into ``name`` (older fixtures).
    """
    book_id = "n-book"
    author_id = "n-author"
    # Re-emit the reference class diagram via the converter so the link's
    # ``associationId`` matches a real edge id in the reference.
    cd_buml = process_class_diagram(class_diagram_v4)
    ref = class_buml_to_json(cd_buml)
    # Keep the v4 reference data — both class diagrams are v4.
    nodes = [
        {
            "id": "obj-book1",
            "type": "objectName",
            "position": {"x": 0, "y": 0},
            "width": 200, "height": 100,
            "measured": {"width": 200, "height": 100},
            "data": {
                "name": "book1 : Book",
                "classId": book_id,
                "className": "Book",
                "attributes": [],
                "methods": [],
            },
        },
        {
            "id": "obj-author1",
            "type": "objectName",
            "position": {"x": 300, "y": 0},
            "width": 200, "height": 100,
            "measured": {"width": 200, "height": 100},
            "data": {
                "name": "author1 : Author",
                "classId": author_id,
                "className": "Author",
                "attributes": [],
                "methods": [],
            },
        },
    ]
    if include_attributes:
        if attribute_style == "split":
            nodes[0]["data"]["attributes"] = [
                {"id": "oa-title", "name": "title", "attributeType": "str",
                 "value": "TheGreatGatsby"},
            ]
            nodes[1]["data"]["attributes"] = [
                {"id": "oa-aname", "name": "name", "attributeType": "str",
                 "value": "Fitzgerald"},
            ]
        else:
            nodes[0]["data"]["attributes"] = [
                {"id": "oa-title", "name": "+ title: str = TheGreatGatsby"},
            ]
            nodes[1]["data"]["attributes"] = [
                {"id": "oa-aname", "name": "+ name: str = Fitzgerald"},
            ]

    # Pick the first ClassBidirectional edge from the reference for the link's
    # associationId.
    association_id = None
    for e in ref.get("edges") or []:
        if e.get("type") in ("ClassBidirectional", "ClassUnidirectional", "ClassComposition"):
            association_id = e.get("id")
            break

    edges = [
        {
            "id": "link-writes",
            "source": "obj-author1",
            "target": "obj-book1",
            "type": "ObjectLink",
            "sourceHandle": "Right",
            "targetHandle": "Left",
            "data": {"name": "writes_link", "associationId": association_id, "points": []},
        },
    ]

    return {
        "title": "LibraryObjects",
        "model": {
            "version": "4.0.0",
            "type": "ObjectDiagram",
            "nodes": nodes,
            "edges": edges,
            "referenceDiagramData": ref,
        },
    }


@pytest.fixture
def object_diagram_json(minimal_class_diagram_json):
    return _build_object_diagram_v4_fixture(minimal_class_diagram_json, include_attributes=False)


@pytest.fixture
def object_diagram_with_attrs_json(minimal_class_diagram_json):
    return _build_object_diagram_v4_fixture(minimal_class_diagram_json, include_attributes=True)


@pytest.fixture
def object_diagram_with_legacy_attrs_json(minimal_class_diagram_json):
    return _build_object_diagram_v4_fixture(
        minimal_class_diagram_json, include_attributes=True, attribute_style="legacy",
    )


# ===========================================================================
# Class Diagram Roundtrip Tests
# ===========================================================================

class TestClassDiagramRoundtrip:
    """JSON -> BUML -> JSON roundtrip for class diagrams."""

    def test_classes_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        assert _all_class_names(minimal_class_diagram_json) == _all_class_names(result)

    def test_attributes_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        for class_name in ("Book", "Author"):
            assert _attribute_names_for_class(minimal_class_diagram_json, class_name) == \
                _attribute_names_for_class(result, class_name), \
                f"Attribute mismatch for class '{class_name}'"

    def test_attribute_types_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        attrs = _attributes_for_class(result, "Book")
        assert attrs["title"]["attributeType"] == "str"
        assert attrs["pages"]["attributeType"] == "int"

    def test_attribute_visibility_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        attrs = _attributes_for_class(result, "Book")
        assert attrs["title"]["visibility"] == "public"
        assert attrs["pages"]["visibility"] == "private"

    def test_attribute_optional_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        attrs = _attributes_for_class(result, "Book")
        assert attrs["title"]["isOptional"] is False
        assert attrs["pages"]["isOptional"] is True

    def test_attribute_is_id_and_external_id_preserved(self, minimal_class_diagram_json):
        """``isId`` / ``isExternalId`` flags survive the roundtrip."""
        # Mutate the fixture in place: title is id, pages is external id.
        for node in minimal_class_diagram_json["model"]["nodes"]:
            if (node.get("data") or {}).get("name") != "Book":
                continue
            for attr in node["data"]["attributes"]:
                if attr["name"] == "title":
                    attr["isId"] = True
                    attr["isExternalId"] = False
                    attr["isOptional"] = False
                elif attr["name"] == "pages":
                    attr["isId"] = False
                    attr["isExternalId"] = True
                    attr["isOptional"] = False

        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        attrs = _attributes_for_class(result, "Book")
        assert attrs["title"].get("isId") is True
        assert attrs["title"].get("isExternalId") is False
        assert attrs["pages"].get("isId") is False
        assert attrs["pages"].get("isExternalId") is True

    def test_methods_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        assert _method_names_for_class(minimal_class_diagram_json, "Book") == \
            _method_names_for_class(result, "Book")

    def test_method_parameters_preserved(self, minimal_class_diagram_json):
        """The converter emits the canonical inspector shape: a bare method
        ``name`` plus structured ``parameters`` rows and ``returnType``
        mirrored onto ``attributeType`` (never a fused signature string)."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        book_node = _class_node_by_name(result, "Book")
        methods = _data(book_node).get("methods") or []

        assert len(methods) == 1
        method = methods[0]
        assert method.get("name") == "summary"
        assert method.get("returnType") == "str"
        assert method.get("attributeType") == "str"
        params = method.get("parameters") or []
        assert len(params) == 1
        assert params[0].get("name") == "verbose"
        assert params[0].get("parameterType") == "bool"

    def test_association_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        bidir_edges = _edges_by_type(result, "ClassBidirectional")
        assert len(bidir_edges) == 1
        edge_data = _data(bidir_edges[0])

        roles = {edge_data.get("sourceRole"), edge_data.get("targetRole")}
        assert roles == {"author", "book"}

        mults = {edge_data.get("sourceMultiplicity"), edge_data.get("targetMultiplicity")}
        assert "1..*" in mults
        assert "0..*" in mults

    def test_association_name_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        bidir_edges = _edges_by_type(result, "ClassBidirectional")
        assert len(bidir_edges) == 1
        assert _data(bidir_edges[0]).get("name") == "writes"

    def test_inheritance_preserved(self, class_diagram_with_inheritance_json):
        domain_model = process_class_diagram(class_diagram_with_inheritance_json)
        result = class_buml_to_json(domain_model)

        assert _all_class_names(result) == {"Vehicle", "Car"}
        assert _class_names_by_kind(result, abstract=True) == {"Vehicle"}
        # Concrete (no stereotype) classes.
        assert _class_names_by_kind(result) == {"Car"}

        inherit_edges = _edges_by_type(result, "ClassInheritance")
        assert len(inherit_edges) == 1
        edge = inherit_edges[0]
        source = _node_by_id(result, edge["source"])
        target = _node_by_id(result, edge["target"])
        assert _data(source).get("name") == "Car"
        assert _data(target).get("name") == "Vehicle"

    def test_enumeration_preserved(self, class_diagram_with_enum_json):
        domain_model = process_class_diagram(class_diagram_with_enum_json)
        result = class_buml_to_json(domain_model)

        assert _class_names_by_kind(result, enumeration=True) == {"OrderStatus"}

        enum_node = _class_node_by_name(result, "OrderStatus")
        assert enum_node is not None
        literal_names = {a.get("name") for a in _data(enum_node).get("attributes") or []}
        assert literal_names == {"PENDING", "SHIPPED", "DELIVERED"}

    def test_enum_attribute_type_preserved(self, class_diagram_with_enum_json):
        domain_model = process_class_diagram(class_diagram_with_enum_json)
        result = class_buml_to_json(domain_model)
        attrs = _attributes_for_class(result, "Order")
        assert attrs["status"]["attributeType"] == "OrderStatus"

    def test_composition_preserved(self, class_diagram_with_composition_json):
        domain_model = process_class_diagram(class_diagram_with_composition_json)
        result = class_buml_to_json(domain_model)

        comp_edges = _edges_by_type(result, "ClassComposition")
        assert len(comp_edges) == 1
        edge_data = _data(comp_edges[0])
        roles = {edge_data.get("sourceRole"), edge_data.get("targetRole")}
        assert roles == {"company", "department"}

    def test_unidirectional_association(self):
        """A legacy unidirectional association survives the roundtrip as a one-way association."""
        json_data = {
            "title": "UniModel",
            "model": {
                "version": "4.0.0",
                "type": "ClassDiagram",
                "nodes": [
                    {
                        "id": "n-a", "type": "class",
                        "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {"name": "ClassA", "stereotype": None,
                                 "attributes": [], "methods": []},
                    },
                    {
                        "id": "n-b", "type": "class",
                        "position": {"x": 300, "y": 0}, "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {"name": "ClassB", "stereotype": None,
                                 "attributes": [], "methods": []},
                    },
                ],
                "edges": [
                    {
                        "id": "e-uni", "source": "n-a", "target": "n-b",
                        "type": "ClassUnidirectional",
                        "sourceHandle": "Right", "targetHandle": "Left",
                        "data": {
                            "name": "uses",
                            "sourceRole": "classA", "sourceMultiplicity": "1",
                            "targetRole": "classB", "targetMultiplicity": "0..1",
                            "points": [],
                        },
                    },
                ],
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)

        # A legacy ClassUnidirectional comes back as ClassBidirectional with explicit
        # per-end navigability (source non-navigable), orientation unchanged.
        assert _edges_by_type(result, "ClassUnidirectional") == []
        bi_edges = _edges_by_type(result, "ClassBidirectional")
        assert len(bi_edges) == 1
        edge_data = _data(bi_edges[0])
        assert edge_data.get("name") == "uses"
        assert edge_data["sourceRole"] == "classA" and edge_data["sourceNavigable"] is False
        assert edge_data["targetRole"] == "classB" and edge_data["targetNavigable"] is True

    def test_diagram_type_preserved(self, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        assert result["type"] == "ClassDiagram"

    def test_double_roundtrip_stable(self, minimal_class_diagram_json):
        dm1 = process_class_diagram(minimal_class_diagram_json)
        json1 = class_buml_to_json(dm1)

        json1_input = {"title": "LibraryModel", "model": json1}
        dm2 = process_class_diagram(json1_input)
        json2 = class_buml_to_json(dm2)

        assert _all_class_names(json1) == _all_class_names(json2)
        assert _class_names_by_kind(json1, enumeration=True) == \
            _class_names_by_kind(json2, enumeration=True)
        for cls_name in _all_class_names(json1):
            assert _attribute_names_for_class(json1, cls_name) == \
                _attribute_names_for_class(json2, cls_name)
            assert _method_names_for_class(json1, cls_name) == \
                _method_names_for_class(json2, cls_name)

    def test_method_with_code_roundtrip(self):
        json_data = {
            "title": "CodeModel",
            "model": {
                "version": "4.0.0",
                "type": "ClassDiagram",
                "nodes": [
                    {
                        "id": "n-calc", "type": "class",
                        "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {
                            "name": "Calculator", "stereotype": None,
                            "attributes": [],
                            "methods": [
                                {
                                    "id": "m-add",
                                    "name": "+ add(a: int, b: int): int",
                                    "code": "def add(self, a, b):\n    return a + b",
                                    "implementationType": "code",
                                },
                            ],
                        },
                    },
                ],
                "edges": [],
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)

        calc_node = _class_node_by_name(result, "Calculator")
        methods = _data(calc_node).get("methods") or []
        assert len(methods) == 1
        assert methods[0].get("code") == "def add(self, a, b):\n    return a + b"
        assert methods[0].get("implementationType") == "code"

    def test_abstract_stereotype_emitted_capitalized(self, class_diagram_with_inheritance_json):
        """The frontend compares stereotypes case-sensitively against the
        capitalized canonical forms — emit exactly 'Abstract' (the
        processor keeps reading case-insensitively, so lowercase input
        still ingests)."""
        domain_model = process_class_diagram(class_diagram_with_inheritance_json)
        result = class_buml_to_json(domain_model)
        vehicle = _class_node_by_name(result, "Vehicle")
        assert _data(vehicle).get("stereotype") == "Abstract"

    def test_enumeration_stereotype_emitted_capitalized(self, class_diagram_with_enum_json):
        domain_model = process_class_diagram(class_diagram_with_enum_json)
        result = class_buml_to_json(domain_model)
        status = _class_node_by_name(result, "OrderStatus")
        assert _data(status).get("stereotype") == "Enumeration"

    def test_comment_round_trip(self, minimal_class_diagram_json):
        """``comment`` nodes tethered by ``CommentLink`` edges land on the
        class metadata and re-emit with the same node / edge types."""
        model = minimal_class_diagram_json["model"]
        model["nodes"].append({
            "id": "n-note",
            "type": "comment",
            "position": {"x": -300, "y": 0},
            "width": 160, "height": 75,
            "measured": {"width": 160, "height": 75},
            "data": {"name": "Books are the unit of lending."},
        })
        model["edges"].append({
            "id": "e-note",
            "source": "n-note",
            "target": "n-book",
            "type": "CommentLink",
            "sourceHandle": "Right",
            "targetHandle": "Left",
            "data": {"points": []},
        })

        domain_model = process_class_diagram(minimal_class_diagram_json)
        book = next(t for t in domain_model.types if t.name == "Book")
        assert book.metadata is not None
        assert book.metadata.description == "Books are the unit of lending."

        result = class_buml_to_json(domain_model)
        comments = _nodes_by_type(result, "comment")
        assert [_data(c).get("name") for c in comments] == [
            "Books are the unit of lending.",
        ]
        links = _edges_by_type(result, "CommentLink")
        assert len(links) == 1
        linked = {links[0]["source"], links[0]["target"]}
        assert comments[0]["id"] in linked
        book_node = _class_node_by_name(result, "Book")
        assert book_node["id"] in linked

    def test_legacy_comments_link_spelling_still_ingests(self, minimal_class_diagram_json):
        """Older fixtures use ``Comments`` / ``Link`` — keep accepting them."""
        model = minimal_class_diagram_json["model"]
        model["nodes"].append({
            "id": "n-note",
            "type": "Comments",
            "position": {"x": -300, "y": 0},
            "width": 160, "height": 75,
            "measured": {"width": 160, "height": 75},
            "data": {"name": "Legacy note."},
        })
        model["edges"].append({
            "id": "e-note",
            "source": "n-note",
            "target": "n-book",
            "type": "Link",
            "sourceHandle": "Right",
            "targetHandle": "Left",
            "data": {"points": []},
        })
        domain_model = process_class_diagram(minimal_class_diagram_json)
        book = next(t for t in domain_model.types if t.name == "Book")
        assert book.metadata is not None
        assert book.metadata.description == "Legacy note."

    def test_method_with_neural_network_roundtrip(self):
        """A method implemented by a neural network keeps its type and NN reference."""
        json_data = {
            "title": "NNModel",
            "model": {
                "version": "4.0.0",
                "type": "ClassDiagram",
                "nodes": [
                    {
                        "id": "cls-classifier", "type": "class",
                        "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {
                            "name": "ImageClassifier", "stereotype": None,
                            "attributes": [],
                            "methods": [
                                {
                                    "id": "meth-predict", "name": "predict",
                                    "visibility": "public",
                                    "parameters": [{"id": "p-image", "name": "image", "parameterType": "str"}],
                                    "returnType": "str", "attributeType": "str",
                                    "implementationType": "neural_network",
                                    "neuralNetworkId": "nn-diagram-1",
                                },
                            ],
                        },
                    },
                ],
                "edges": [],
            },
        }
        domain_model = process_class_diagram(json_data)
        method = next(iter(domain_model.get_class_by_name("ImageClassifier").methods))
        assert method.implementation_type == MethodImplementationType.NEURAL_NETWORK
        assert domain_model.method_diagram_refs[("ImageClassifier", "predict")]["neuralNetworkId"] == "nn-diagram-1"

        result = class_buml_to_json(domain_model)
        methods = _data(_class_node_by_name(result, "ImageClassifier"))["methods"]
        assert len(methods) == 1
        assert methods[0].get("implementationType") == "neural_network"
        assert methods[0].get("neuralNetworkId") == "nn-diagram-1"
        assert "quantumCircuitId" not in methods[0]

    def test_attribute_default_value_preserved(self):
        json_data = {
            "title": "DefaultsModel",
            "model": {
                "version": "4.0.0",
                "type": "ClassDiagram",
                "nodes": [
                    {
                        "id": "n-config", "type": "class",
                        "position": {"x": 0, "y": 0}, "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {
                            "name": "Config", "stereotype": None,
                            "attributes": [
                                {"id": "a-retries", "name": "retries", "attributeType": "int",
                                 "visibility": "public", "isOptional": False, "defaultValue": "3"},
                            ],
                            "methods": [],
                        },
                    },
                ],
                "edges": [],
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)
        attrs = _attributes_for_class(result, "Config")
        assert attrs["retries"].get("defaultValue") == "3"


# ===========================================================================
# Association navigability roundtrip tests
# ===========================================================================

_OMIT = object()


def _with_writes_relationship(diagram_json, rel_type, source_navigable=_OMIT, target_navigable=_OMIT):
    """Return a copy of *diagram_json* whose Author -> Book 'writes' edge has the given
    type and per-end ``sourceNavigable`` / ``targetNavigable`` values (``_OMIT`` leaves
    the key out, as legacy payloads do)."""
    data = json.loads(json.dumps(diagram_json))
    edge = next(e for e in data["model"]["edges"] if e["id"] == "e-writes")
    edge["type"] = rel_type
    if source_navigable is not _OMIT:
        edge["data"]["sourceNavigable"] = source_navigable
    if target_navigable is not _OMIT:
        edge["data"]["targetNavigable"] = target_navigable
    return data


def _roundtrip_writes(diagram_json):
    """Run JSON -> BUML -> JSON and return (domain_model, the single output association edge)."""
    domain_model = process_class_diagram(diagram_json)
    result = class_buml_to_json(domain_model)
    nodes_by_id = {n["id"]: n for n in _nodes(result)}
    edges = [
        e for e in _edges(result)
        if e.get("type") in ("ClassBidirectional", "ClassUnidirectional", "ClassComposition", "ClassAggregation")
    ]
    assert len(edges) == 1
    edge = edges[0]
    edge["_source_class"] = _data(nodes_by_id[edge["source"]])["name"]
    edge["_target_class"] = _data(nodes_by_id[edge["target"]])["name"]
    return domain_model, edge


class TestAssociationNavigabilityRoundtrip:
    """Per-end navigability survives JSON -> BUML -> JSON with the drawn orientation kept."""

    @pytest.mark.parametrize("in_type, src_nav, tgt_nav, out_type, out_src, out_tgt", [
        ("ClassBidirectional", True, True, "ClassBidirectional", True, True),
        ("ClassBidirectional", False, True, "ClassBidirectional", False, True),
        ("ClassBidirectional", True, False, "ClassBidirectional", True, False),
        # Legacy payload without the flags: ClassUnidirectional => source non-navigable
        ("ClassUnidirectional", _OMIT, _OMIT, "ClassBidirectional", False, True),
        # Legacy payload without the flags: plain association => both navigable
        ("ClassBidirectional", _OMIT, _OMIT, "ClassBidirectional", True, True),
        ("ClassComposition", True, True, "ClassComposition", True, True),
        # Composition whose composite (whole, target) end is non-navigable
        ("ClassComposition", True, False, "ClassComposition", True, False),
        # BUML has no aggregation flag: aggregation comes back as ClassBidirectional, navigability kept
        ("ClassAggregation", True, False, "ClassBidirectional", True, False),
        ("ClassAggregation", False, True, "ClassBidirectional", False, True),
    ])
    def test_navigability_and_orientation_preserved(
        self, minimal_class_diagram_json, in_type, src_nav, tgt_nav, out_type, out_src, out_tgt
    ):
        payload = _with_writes_relationship(minimal_class_diagram_json, in_type, src_nav, tgt_nav)
        domain_model, edge = _roundtrip_writes(payload)
        edge_data = _data(edge)

        assert edge["type"] == out_type
        assert edge_data["sourceNavigable"] is out_src
        assert edge_data["targetNavigable"] is out_tgt
        # Orientation as drawn: Author (role "author") is the source, Book (role "book") the target
        assert (edge["_source_class"], edge_data["sourceRole"]) == ("Author", "author")
        assert (edge["_target_class"], edge_data["targetRole"]) == ("Book", "book")
        assert domain_model.ocl_warnings == []

    def test_navigability_maps_to_metamodel_ends(self, minimal_class_diagram_json):
        payload = _with_writes_relationship(minimal_class_diagram_json, "ClassBidirectional", False, True)
        domain_model = process_class_diagram(payload)
        assoc = next(a for a in domain_model.associations if a.name == "writes")
        by_name = {end.name: end for end in assoc.ends}
        assert by_name["author"].is_navigable is False
        assert by_name["book"].is_navigable is True

    def test_saved_endpoint_layout_stays_on_its_end(self, minimal_class_diagram_json):
        """A one-way association keeps each saved endpoint handle on the end it was drawn on,
        even when the drawn source sorts after the target."""
        payload = _with_writes_relationship(minimal_class_diagram_json, "ClassBidirectional", True, False)
        edge_in = next(e for e in payload["model"]["edges"] if e["id"] == "e-writes")
        # Draw Book -> Author (the roles sort the other way round).
        edge_in["source"], edge_in["target"] = "n-book", "n-author"
        edge_in["data"]["sourceRole"], edge_in["data"]["targetRole"] = "book", "author"
        edge_in["sourceHandle"], edge_in["targetHandle"] = "Left", "Right"
        edge_in["data"]["points"] = [{"x": 0, "y": 0}, {"x": 140, "y": 0}]

        _, edge = _roundtrip_writes(payload)
        edge_data = _data(edge)
        assert edge["_source_class"] == "Book"
        assert edge_data["sourceRole"] == "book"
        assert edge_data["sourceNavigable"] is True
        assert edge["sourceHandle"] == "Left"
        assert edge["_target_class"] == "Author"
        assert edge_data["targetNavigable"] is False
        assert edge["targetHandle"] == "Right"
        assert edge_data["points"] == [{"x": 0, "y": 0}, {"x": 140, "y": 0}]

    def test_no_navigable_end_is_corrected_with_warning(self, minimal_class_diagram_json):
        payload = _with_writes_relationship(minimal_class_diagram_json, "ClassBidirectional", False, False)
        domain_model, edge = _roundtrip_writes(payload)

        assert _data(edge)["sourceNavigable"] is False
        assert _data(edge)["targetNavigable"] is True
        assert any("e-writes" in w and "at least one end must be navigable" in w
                   for w in domain_model.ocl_warnings)

    def test_composition_part_end_non_navigable_is_corrected_with_warning(self, minimal_class_diagram_json):
        payload = _with_writes_relationship(minimal_class_diagram_json, "ClassComposition", False, True)
        domain_model, edge = _roundtrip_writes(payload)

        assert edge["type"] == "ClassComposition"
        assert _data(edge)["sourceNavigable"] is True
        assert _data(edge)["targetNavigable"] is True
        assert any("e-writes" in w and "part (source) end of a composition must be navigable" in w
                   for w in domain_model.ocl_warnings)

    @pytest.mark.parametrize("bad_value", ["false", None, 0, "true"])
    def test_non_bool_navigable_falls_back_with_warning(self, minimal_class_diagram_json, bad_value):
        """A non-boolean ``sourceNavigable`` is not coerced (``"false"`` must not become True):
        the legacy default for the relationship type is used and a warning is recorded."""
        payload = _with_writes_relationship(minimal_class_diagram_json, "ClassUnidirectional", bad_value, True)
        domain_model, edge = _roundtrip_writes(payload)

        # Legacy default for a ClassUnidirectional source end is non-navigable
        assert _data(edge)["sourceNavigable"] is False
        assert _data(edge)["targetNavigable"] is True
        assert any("e-writes" in w and "sourceNavigable must be true or false" in w
                   for w in domain_model.ocl_warnings)


# ===========================================================================
# State Machine Roundtrip Tests
# ===========================================================================

class TestStateMachineRoundtrip:
    """JSON -> BUML -> JSON roundtrip for state machine diagrams."""

    def test_states_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)
        states = _nodes_by_type(result, "State")
        names = {(_data(s).get("name") or "") for s in states}
        assert names == {"idle", "processing"}

    def test_initial_state_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        initial_nodes = _nodes_by_type(result, "StateInitialNode")
        assert len(initial_nodes) == 1
        initial_id = initial_nodes[0]["id"]

        transitions = _edges_by_type(result, "StateTransition")
        from_initial = [t for t in transitions if t["source"] == initial_id]
        assert len(from_initial) == 1

        target_node = _node_by_id(result, from_initial[0]["target"])
        assert _data(target_node).get("name") == "idle"

    def test_transitions_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        initial_ids = {n["id"] for n in _nodes_by_type(result, "StateInitialNode")}
        transitions = [
            t for t in _edges_by_type(result, "StateTransition")
            if t["source"] not in initial_ids
        ]
        assert len(transitions) == 1
        t = transitions[0]
        assert _data(t).get("name") == "start_event"

        source_node = _node_by_id(result, t["source"])
        target_node = _node_by_id(result, t["target"])
        assert _data(source_node).get("name") == "idle"
        assert _data(target_node).get("name") == "processing"

    def test_bodies_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        for state in _nodes_by_type(result, "State"):
            data = _data(state)
            bodies = data.get("bodies") or []
            if data.get("name") == "idle":
                assert len(bodies) == 1
                assert bodies[0].get("name") == "idle_body"
            elif data.get("name") == "processing":
                assert len(bodies) == 1
                assert bodies[0].get("name") == "process_body"

    def test_code_blocks_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        code_blocks = _nodes_by_type(result, "StateCodeBlock")
        names = {_data(cb).get("name") for cb in code_blocks}
        assert "idle_body" in names
        assert "process_body" in names
        assert "start_event" in names

    def test_code_block_content_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        for cb in _nodes_by_type(result, "StateCodeBlock"):
            data = _data(cb)
            name = data.get("name")
            code = data.get("code") or ""
            if name == "idle_body":
                assert "idle_body" in code
                assert "session.reply" in code
            elif name == "process_body":
                assert "process_body" in code
                assert "Processing" in code
            elif name == "start_event":
                assert "start_event" in code
                assert "session.message" in code

    def test_diagram_type_preserved(self, state_machine_json):
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)
        assert result["type"] == "StateMachineDiagram"

    def test_state_machine_with_guard(self):
        json_data = {
            "title": "GuardTest",
            "model": {
                "version": "4.0.0",
                "type": "StateMachineDiagram",
                "nodes": [
                    {
                        "id": "n-init", "type": "StateInitialNode",
                        "position": {"x": -850, "y": -280},
                        "width": 45, "height": 45,
                        "measured": {"width": 45, "height": 45},
                        "data": {"name": ""},
                    },
                    {
                        "id": "n-a", "type": "State",
                        "position": {"x": -550, "y": -300},
                        "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {"name": "stateA", "bodies": [], "fallbackBodies": []},
                    },
                    {
                        "id": "n-b", "type": "State",
                        "position": {"x": -60, "y": -300},
                        "width": 160, "height": 100,
                        "measured": {"width": 160, "height": 100},
                        "data": {"name": "stateB", "bodies": [], "fallbackBodies": []},
                    },
                ],
                "edges": [
                    {
                        "id": "t-init", "source": "n-init", "target": "n-a",
                        "type": "StateTransition",
                        "sourceHandle": "Right", "targetHandle": "Left",
                        "data": {"name": "", "points": []},
                    },
                    {
                        "id": "t-guard", "source": "n-a", "target": "n-b",
                        "type": "StateTransition",
                        "sourceHandle": "Right", "targetHandle": "Left",
                        "data": {"name": "trigger_event", "guard": "x > 10", "points": []},
                    },
                ],
            },
        }
        sm = process_state_machine(json_data)
        result = state_machine_object_to_json(sm)

        guarded = [t for t in _edges_by_type(result, "StateTransition") if _data(t).get("guard")]
        assert len(guarded) == 1
        assert _data(guarded[0]).get("guard") == "x > 10"
        assert _data(guarded[0]).get("name") == "trigger_event"

    def test_comment_round_trip(self, state_machine_json):
        """``comment`` nodes tethered by ``CommentLink`` edges land on the
        state metadata and re-emit with the same node / edge types."""
        model = state_machine_json["model"]
        model["nodes"].append({
            "id": "n-note",
            "type": "comment",
            "position": {"x": -900, "y": -500},
            "width": 160, "height": 75,
            "measured": {"width": 160, "height": 75},
            "data": {"name": "Waits for the next order."},
        })
        model["edges"].append({
            "id": "e-note",
            "source": "n-note",
            "target": "n-idle",
            "type": "CommentLink",
            "sourceHandle": "Right",
            "targetHandle": "Left",
            "data": {"points": []},
        })

        sm = process_state_machine(state_machine_json)
        idle = next(s for s in sm.states if s.name == "idle")
        assert idle.metadata is not None
        assert idle.metadata.description == "Waits for the next order."

        result = state_machine_object_to_json(sm)
        comments = _nodes_by_type(result, "comment")
        assert [_data(c).get("name") for c in comments] == [
            "Waits for the next order.",
        ]
        links = _edges_by_type(result, "CommentLink")
        assert len(links) == 1
        linked = {links[0]["source"], links[0]["target"]}
        assert comments[0]["id"] in linked
        idle_node = next(
            n for n in _nodes_by_type(result, "State")
            if _data(n).get("name") == "idle"
        )
        assert idle_node["id"] in linked

    def test_double_roundtrip_stable(self, state_machine_json):
        sm1 = process_state_machine(state_machine_json)
        json1 = state_machine_object_to_json(sm1)

        json1_input = {"title": "OrderStateMachine", "model": json1}
        sm2 = process_state_machine(json1_input)
        json2 = state_machine_object_to_json(sm2)

        states1 = {_data(s).get("name") for s in _nodes_by_type(json1, "State")}
        states2 = {_data(s).get("name") for s in _nodes_by_type(json2, "State")}
        assert states1 == states2

        code1 = {_data(cb).get("name") for cb in _nodes_by_type(json1, "StateCodeBlock")}
        code2 = {_data(cb).get("name") for cb in _nodes_by_type(json2, "StateCodeBlock")}
        assert code1 == code2


# ===========================================================================
# Object Diagram Roundtrip Tests
# ===========================================================================

class TestObjectDiagramRoundtrip:
    """JSON -> BUML roundtrip for object diagrams.

    No metamodel-to-JSON converter exists for ObjectModel that runs without
    parsing generated code, so these tests verify that the BUML ObjectModel
    produced from JSON contains the correct data.
    """

    def test_objects_created(self, object_diagram_json, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        # The processor preserves the full ``"name : Class"`` label.
        names = {obj.name or "" for obj in obj_model.objects}
        assert any("book1" in n for n in names)
        assert any("author1" in n for n in names)

    def test_object_classifiers(self, object_diagram_json, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        for obj in obj_model.objects:
            label = obj.name or ""
            classifier_name = obj.classifier.name
            if "book1" in label:
                assert classifier_name == "Book"
            elif "author1" in label:
                assert classifier_name == "Author"

    def test_object_count(self, object_diagram_json, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)
        assert len(obj_model.objects) == 2

    def test_object_links_created(self, object_diagram_json, minimal_class_diagram_json):
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        all_links = set()
        for obj in obj_model.objects:
            for link in obj.links:
                all_links.add(link)

        assert len(all_links) >= 1
        link = next(iter(all_links))
        assert link.association is not None
        end_types = {end.type.name for end in link.association.ends}
        assert "Book" in end_types
        assert "Author" in end_types

    def test_object_attribute_values(self, object_diagram_with_attrs_json, minimal_class_diagram_json):
        """Canonical v4 rows keep name and runtime value in separate
        fields (``ObjectNodeAttribute.value``)."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_with_attrs_json, domain_model)

        for obj in obj_model.objects:
            label = obj.name or ""
            slot_values = {slot.attribute.name: slot.value.value for slot in obj.slots}
            if "book1" in label:
                assert slot_values.get("title") == "TheGreatGatsby"
            elif "author1" in label:
                assert slot_values.get("name") == "Fitzgerald"

    def test_object_attribute_values_legacy_combined_strings(
        self, object_diagram_with_legacy_attrs_json, minimal_class_diagram_json,
    ):
        """Legacy rows bake ``"+ name: type = value"`` into ``name`` —
        the fallback parser must keep ingesting them."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(
            object_diagram_with_legacy_attrs_json, domain_model,
        )

        slot_map: dict = {}
        for obj in obj_model.objects:
            for slot in obj.slots:
                slot_map[slot.attribute.name] = slot.value.value
        assert slot_map == {"title": "TheGreatGatsby", "name": "Fitzgerald"}

    def test_user_model_nodes_and_user_model_link(
        self, object_diagram_json, minimal_class_diagram_json,
    ):
        """UserDiagram payloads use ``UserModelName`` nodes whose rows
        carry ``value`` / ``defaultValue`` plus an ``attributeOperator``
        comparator, linked by ``UserModelLink`` edges — all must ingest
        exactly like their Object-diagram counterparts."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        model = object_diagram_json["model"]
        for node in model["nodes"]:
            node["type"] = "UserModelName"
        model["nodes"][0]["data"]["attributes"] = [
            {"id": "ua-title", "name": "title", "attributeType": "str",
             "attributeOperator": "==", "value": "TheGreatGatsby"},
        ]
        model["nodes"][1]["data"]["attributes"] = [
            # ``defaultValue`` is the secondary carrier when ``value`` is
            # absent (``UserModelAttributeRow``).
            {"id": "ua-name", "name": "name", "attributeType": "str",
             "attributeOperator": "==", "defaultValue": "Fitzgerald"},
        ]
        for edge in model["edges"]:
            edge["type"] = "UserModelLink"

        obj_model = process_object_diagram(object_diagram_json, domain_model)
        assert len(obj_model.objects) == 2

        slot_map: dict = {}
        for obj in obj_model.objects:
            for slot in obj.slots:
                slot_map[slot.attribute.name] = slot.value.value
        assert slot_map == {"title": "TheGreatGatsby", "name": "Fitzgerald"}

        all_links = {link for obj in obj_model.objects for link in obj.links}
        assert len(all_links) >= 1
        link = next(iter(all_links))
        end_types = {end.type.name for end in link.association.ends}
        assert end_types == {"Book", "Author"}

    def test_user_model_legacy_embedded_operator_string(
        self, object_diagram_json, minimal_class_diagram_json,
    ):
        """Legacy user rows embed ``"name <op> value"`` in ``name`` —
        the comparator split must keep working as a fallback."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        model = object_diagram_json["model"]
        for node in model["nodes"]:
            node["type"] = "UserModelName"
        model["nodes"][0]["data"]["attributes"] = [
            {"id": "ua-title", "name": "title == TheGreatGatsby",
             "attributeOperator": "=="},
        ]
        obj_model = process_object_diagram(object_diagram_json, domain_model)
        book1 = next(o for o in obj_model.objects if "book1" in (o.name or ""))
        slot_values = {s.attribute.name: s.value.value for s in book1.slots}
        assert slot_values == {"title": "TheGreatGatsby"}

    @staticmethod
    def _user_diagram_json(attribute_row):
        """A v4 UserDiagram with one ``UserModelName`` node (``book1 : Book``)."""
        return {
            "title": "UserProfile",
            "model": {
                "version": "4.0.0",
                "type": "UserDiagram",
                "nodes": [
                    {
                        "id": "u-book1", "type": "UserModelName",
                        "position": {"x": 0, "y": 0}, "width": 200, "height": 100,
                        "measured": {"width": 200, "height": 100},
                        "data": {
                            "name": "book1", "className": "Book",
                            "attributes": [attribute_row], "methods": [],
                        },
                    },
                ],
                "edges": [],
            },
        }

    def test_user_model_equality_attribute_value(self, minimal_class_diagram_json):
        """UserModelAttribute equality criteria ('==') are parsed, not dropped.

        Regression test: the editor renders equality as a single '=' (e.g.
        "title = TheGreatGatsby") while storing attributeOperator as '=='. The
        parser must split on the displayed symbol; otherwise every '=='
        criterion falls through with an empty value ("attributes are empty").
        """
        domain_model = process_class_diagram(minimal_class_diagram_json)
        user_diagram_json = self._user_diagram_json({
            "id": "u-title", "name": "title = TheGreatGatsby",
            "attributeOperator": "==",
        })
        obj_model = process_object_diagram(user_diagram_json, domain_model)

        book = next(obj for obj in obj_model.objects if "book1" in (obj.name or ""))
        slot_values = {slot.attribute.name: slot.value.value for slot in book.slots}
        assert slot_values.get("title") == "TheGreatGatsby"

    def test_user_model_stored_operator_equality(self, minimal_class_diagram_json):
        """A manually-typed '==' criterion splits on '==', not the first '='.

        Regression: splitting every '==' on the displayed '=' captured
        "= TheGreatGatsby" as the value and broke type conversion. The stored
        operator must be tried first.
        """
        domain_model = process_class_diagram(minimal_class_diagram_json)
        user_diagram_json = self._user_diagram_json({
            "id": "u-title", "name": "title == TheGreatGatsby",
            "attributeOperator": "==",
        })
        obj_model = process_object_diagram(user_diagram_json, domain_model)
        book = next(obj for obj in obj_model.objects if "book1" in (obj.name or ""))
        slot_values = {slot.attribute.name: slot.value.value for slot in book.slots}
        assert slot_values.get("title") == "TheGreatGatsby"

    def test_user_model_blank_criterion_skipped(self, minimal_class_diagram_json):
        """A blank criterion ("title = ") is skipped, not materialised as an
        empty-valued slot (which would break type conversion / OCL evaluation).
        """
        domain_model = process_class_diagram(minimal_class_diagram_json)
        user_diagram_json = self._user_diagram_json({
            "id": "u-title", "name": "title = ", "attributeOperator": "==",
        })
        obj_model = process_object_diagram(user_diagram_json, domain_model)
        book = next(obj for obj in obj_model.objects if "book1" in (obj.name or ""))
        slot_names = {slot.attribute.name for slot in book.slots}
        assert "title" not in slot_names

    def test_user_model_blank_value_row_skipped(self, minimal_class_diagram_json):
        """A v4 row whose structured ``value`` is empty is skipped as well."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        user_diagram_json = self._user_diagram_json({
            "id": "u-pages", "name": "pages", "attributeType": "int",
            "attributeOperator": "==", "value": "",
        })
        obj_model = process_object_diagram(user_diagram_json, domain_model)
        book = next(obj for obj in obj_model.objects if "book1" in (obj.name or ""))
        assert "pages" not in {slot.attribute.name for slot in book.slots}


# ===========================================================================
# NN Diagram Roundtrip
# ===========================================================================

def _nn_layer_attrs(node):
    """Return the ``data.attributes`` dict on an NN layer node."""
    attrs = (node.get("data") or {}).get("attributes") or {}
    return attrs if isinstance(attrs, dict) else {}


class TestNNDiagramRoundtrip:
    """Roundtrip tests for NNDiagram."""

    @staticmethod
    def _minimal_nn_json():
        """A minimal v4 NN diagram: container + Conv2D + Linear + configuration + datasets."""
        return {
            "title": "TestNet",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "size": {"width": 1400, "height": 740},
                "nodes": [
                    {"id": "c1", "type": "NNContainer",
                     "position": {"x": 0, "y": 0}, "width": 600, "height": 200,
                     "measured": {"width": 600, "height": 200},
                     "data": {"name": "TestNet"}},
                    {"id": "l1", "type": "Conv2DLayer",
                     "parentId": "c1",
                     "position": {"x": 20, "y": 60}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "c1",
                              "attributes": {
                                  "name": "c1",
                                  "kernel_dim": "[3, 3]",
                                  "out_channels": "16",
                              }}},
                    {"id": "l2", "type": "LinearLayer",
                     "parentId": "c1",
                     "position": {"x": 150, "y": 60}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "lin",
                              "attributes": {
                                  "name": "lin",
                                  "out_features": "10",
                              }}},
                    {"id": "cfg", "type": "Configuration",
                     "position": {"x": 700, "y": 0}, "width": 160, "height": 200,
                     "measured": {"width": 160, "height": 200},
                     "data": {"name": "Configuration",
                              "attributes": {
                                  "batch_size": "32",
                                  "epochs": "10",
                                  "learning_rate": "0.001",
                                  "optimizer": "adam",
                                  "loss_function": "crossentropy",
                                  "metrics": "[accuracy]",
                              }}},
                    {"id": "ds1", "type": "TrainingDataset",
                     "position": {"x": 0, "y": 400}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "TrainingDataset",
                              "attributes": {
                                  "name": "train",
                                  "path_data": "/data/train",
                                  "task_type": "multi_class",
                                  "input_format": "images",
                                  "shape": "[32, 32, 3]",
                                  "normalize": "false",
                              }}},
                    {"id": "ds2", "type": "TestDataset",
                     "position": {"x": 300, "y": 400}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "TestDataset",
                              "attributes": {
                                  "name": "test",
                                  "path_data": "/data/test",
                              }}},
                ],
                "edges": [
                    {"id": "r1", "source": "l1", "target": "l2",
                     "type": "NNNext",
                     "sourceHandle": "Right", "targetHandle": "Left",
                     "data": {"name": "next", "points": []}},
                ],
            },
        }

    def _roundtrip(self, json_in):
        nn = process_nn_diagram(json_in)
        return nn_model_to_json(nn)

    def test_diagram_type_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        assert out["type"] == "NNDiagram"

    def test_container_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        containers = _nodes_by_type(out, "NNContainer")
        assert len(containers) == 1
        assert _data(containers[0]).get("name") == "TestNet"

    def test_layers_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        assert len(_nodes_by_type(out, "Conv2DLayer")) == 1
        assert len(_nodes_by_type(out, "LinearLayer")) == 1

    def test_nnnext_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        assert len(_edges_by_type(out, "NNNext")) == 1

    def test_configuration_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        configs = _nodes_by_type(out, "Configuration")
        assert len(configs) == 1
        attrs = _nn_layer_attrs(configs[0])
        assert attrs.get("batch_size") == "32"
        assert attrs.get("optimizer") == "adam"

    def test_training_dataset_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        training = _nodes_by_type(out, "TrainingDataset")
        assert len(training) == 1
        attrs = _nn_layer_attrs(training[0])
        assert attrs.get("name") == "train"
        assert attrs.get("path_data") == "/data/train"
        assert attrs.get("task_type") == "multi_class"
        assert attrs.get("input_format") == "images"
        assert attrs.get("shape") == "[32, 32, 3]"
        assert attrs.get("normalize") == "false"

    def test_test_dataset_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        testing = _nodes_by_type(out, "TestDataset")
        assert len(testing) == 1
        attrs = _nn_layer_attrs(testing[0])
        assert attrs.get("name") == "test"
        assert attrs.get("path_data") == "/data/test"
        assert "input_format" not in attrs

    def test_double_roundtrip_stable(self):
        first = self._roundtrip(self._minimal_nn_json())
        second = self._roundtrip(first)

        def counts(j):
            return Counter(n.get("type") for n in _nodes(j))
        assert counts(first) == counts(second)

    def test_nn_model_to_json_is_deterministic(self):
        """Identical NN input must produce byte-identical JSON across calls."""
        import json
        nn = process_nn_diagram(self._minimal_nn_json())
        first = json.dumps(nn_model_to_json(nn), sort_keys=True)
        second = json.dumps(nn_model_to_json(nn), sort_keys=True)
        assert first == second

    def test_nnreference_cycle_is_rejected(self):
        cyclic = {
            "title": "CyclicRefs",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "nodes": [
                    {"id": "cA", "type": "NNContainer",
                     "position": {"x": 0, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "A"}},
                    {"id": "cB", "type": "NNContainer",
                     "position": {"x": 500, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "B"}},
                    {"id": "rA", "type": "NNReference",
                     "parentId": "cA",
                     "position": {"x": 20, "y": 60}, "width": 140, "height": 40,
                     "measured": {"width": 140, "height": 40},
                     "data": {"name": "ref-to-B", "referenceTarget": "B"}},
                    {"id": "rB", "type": "NNReference",
                     "parentId": "cB",
                     "position": {"x": 520, "y": 60}, "width": 140, "height": 40,
                     "measured": {"width": 140, "height": 40},
                     "data": {"name": "ref-to-A", "referenceTarget": "A"}},
                ],
                "edges": [],
            },
        }
        with pytest.raises(ValueError, match="NNReference cycle"):
            process_nn_diagram(cyclic)

    def test_padding_type_valid_roundtrips_when_explicit(self):
        """When the user explicitly picks padding_type='valid' (the metamodel
        default), the sidecar ``mark_explicit`` path must make sure the JSON
        output still carries the attribute."""
        payload = self._minimal_nn_json()
        # Add padding_type='valid' to the Conv2D layer's attributes dict.
        for n in payload["model"]["nodes"]:
            if n["id"] == "l1":
                n["data"]["attributes"]["padding_type"] = "valid"
        out = self._roundtrip(payload)
        convs = _nodes_by_type(out, "Conv2DLayer")
        assert convs
        attrs = _nn_layer_attrs(convs[0])
        assert attrs.get("padding_type") == "valid"

    def test_flatten_start_dim_and_end_dim_defaults_roundtrip_when_explicit(self):
        """Flatten layer with start_dim=1 and end_dim=-1 (both metamodel
        defaults) must round-trip when set explicitly in the source JSON."""
        base = self._minimal_nn_json()
        base["model"]["nodes"].append({
            "id": "fl1", "type": "FlattenLayer",
            "parentId": "c1",
            "position": {"x": 300, "y": 60}, "width": 110, "height": 110,
            "measured": {"width": 110, "height": 110},
            "data": {"name": "flat",
                     "attributes": {"name": "flat", "start_dim": "1", "end_dim": "-1"}},
        })
        # Connect fl1 into the existing l1 -> l2 chain.
        base["model"]["edges"].append({
            "id": "rfl1", "source": "l2", "target": "fl1",
            "type": "NNNext",
            "sourceHandle": "Right", "targetHandle": "Left",
            "data": {"name": "next", "points": []},
        })
        out = self._roundtrip(base)
        flat = _nodes_by_type(out, "FlattenLayer")
        assert flat
        attrs = _nn_layer_attrs(flat[0])
        assert "start_dim" in attrs
        assert "end_dim" in attrs

    def test_tensorop_concatenate_preserves_float_in_layers_of_tensors(self):
        """TensorOp layers_of_tensors with a leading float must preserve it as a
        float (not its string repr) after parsing."""
        payload = {
            "title": "op",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "nodes": [
                    {"id": "c", "type": "NNContainer",
                     "position": {"x": 0, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "op"}},
                    {"id": "sup", "type": "LinearLayer",
                     "parentId": "c",
                     "position": {"x": 20, "y": 60}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "other",
                              "attributes": {"name": "other", "out_features": "4"}}},
                    {"id": "t", "type": "TensorOp",
                     "parentId": "c",
                     "position": {"x": 160, "y": 60}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "op1",
                              "attributes": {
                                  "name": "op1",
                                  "tns_type": "concatenate",
                                  "concatenate_dim": "1",
                                  "layers_of_tensors": "[0.5, 'other']",
                              }}},
                ],
                "edges": [
                    {"id": "r1", "source": "sup", "target": "t",
                     "type": "NNNext",
                     "sourceHandle": "Right", "targetHandle": "Left",
                     "data": {"name": "next", "points": []}},
                ],
            },
        }
        nn = process_nn_diagram(payload)
        top = [m for m in nn.modules if type(m).__name__ == "TensorOp"][0]
        assert top.layers_of_tensors[0] == 0.5
        assert isinstance(top.layers_of_tensors[0], float)
        assert top.layers_of_tensors[1] == "other"

    def test_transitive_nnreference_cycle_detected(self):
        cyclic = {
            "title": "transitive",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "nodes": [
                    {"id": "A", "type": "NNContainer",
                     "position": {"x": 0, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "A"}},
                    {"id": "B", "type": "NNContainer",
                     "position": {"x": 500, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "B"}},
                    {"id": "C", "type": "NNContainer",
                     "position": {"x": 1000, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "C"}},
                    {"id": "rA", "type": "NNReference", "parentId": "A",
                     "position": {"x": 20, "y": 60}, "width": 140, "height": 40,
                     "measured": {"width": 140, "height": 40},
                     "data": {"name": "A->B", "referenceTarget": "B"}},
                    {"id": "rB", "type": "NNReference", "parentId": "B",
                     "position": {"x": 520, "y": 60}, "width": 140, "height": 40,
                     "measured": {"width": 140, "height": 40},
                     "data": {"name": "B->C", "referenceTarget": "C"}},
                    {"id": "rC", "type": "NNReference", "parentId": "C",
                     "position": {"x": 1020, "y": 60}, "width": 140, "height": 40,
                     "measured": {"width": 140, "height": 40},
                     "data": {"name": "C->A", "referenceTarget": "A"}},
                ],
                "edges": [],
            },
        }
        with pytest.raises(ValueError, match="NNReference cycle"):
            process_nn_diagram(cyclic)

    def test_conv3d_kernel_dim_wrong_length_raises_with_layer_name(self):
        payload = {
            "title": "conv",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "nodes": [
                    {"id": "c", "type": "NNContainer",
                     "position": {"x": 0, "y": 0}, "width": 400, "height": 200,
                     "measured": {"width": 400, "height": 200},
                     "data": {"name": "net"}},
                    {"id": "l", "type": "Conv3DLayer", "parentId": "c",
                     "position": {"x": 20, "y": 60}, "width": 110, "height": 110,
                     "measured": {"width": 110, "height": 110},
                     "data": {"name": "volume",
                              "attributes": {"name": "volume", "kernel_dim": "[3, 3]",
                                             "out_channels": "8"}}},
                ],
                "edges": [],
            },
        }
        with pytest.raises(ValueError, match=r"Conv3D.*volume.*kernel_dim.*expected 3"):
            process_nn_diagram(payload)

    @staticmethod
    def _single_layer_json(layer_type, attrs, edges=None):
        """Build a v4 NN diagram with a single layer of the given ``layer_type``.

        ``attrs`` is a dict of attributeName -> string-value entries that
        becomes the layer's ``data.attributes`` dict.
        """
        nodes = [
            {"id": "c", "type": "NNContainer",
             "position": {"x": 0, "y": 0}, "width": 400, "height": 200,
             "measured": {"width": 400, "height": 200},
             "data": {"name": "net"}},
            {"id": "L", "type": layer_type, "parentId": "c",
             "position": {"x": 20, "y": 60}, "width": 110, "height": 110,
             "measured": {"width": 110, "height": 110},
             "data": {"name": attrs.get("name", layer_type),
                      "attributes": dict(attrs)}},
        ]
        return {
            "title": "single",
            "model": {
                "version": "4.0.0",
                "type": "NNDiagram",
                "nodes": nodes,
                "edges": edges or [],
            },
        }

    def test_lstm_roundtrip(self):
        payload = self._single_layer_json("LSTMLayer", {
            "name": 'lstm1',
            "hidden_size": '128',
            "input_size": '64',
            "return_type": 'last',
            "bias": 'false',
            "hx_source": 'encoder_h',
            "hidden_state_var": 'h_0',
            "hidden_unused": 'true',
            "hidden_subscript_source": 'h',
            "hidden_subscript_target": 'h_t',
            "cell_state_var": 'c_0',
            "cell_unused": 'false',
            "is_layer_call": 'true',
            "input_var": 'seq_in',
            "output_var": 'seq_out',
        })
        out = self._roundtrip(payload)
        lstm = _nodes_by_type(out, "LSTMLayer")
        assert lstm, 'LSTMLayer missing from output'
        attrs = _nn_layer_attrs(lstm[0])
        assert attrs.get('name') == 'lstm1'
        assert attrs.get('hidden_size') == '128'
        assert attrs.get('input_size') == '64'
        assert attrs.get('return_type') == 'last'
        assert attrs.get('bias') == 'false'
        assert attrs.get('hx_source') == 'encoder_h'
        assert attrs.get('hidden_state_var') == 'h_0'
        assert attrs.get('hidden_unused') == 'true'
        assert attrs.get('hidden_subscript_source') == 'h'
        assert attrs.get('hidden_subscript_target') == 'h_t'
        assert attrs.get('cell_state_var') == 'c_0'
        assert attrs.get('cell_unused') == 'false'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'seq_in'
        assert attrs.get('output_var') == 'seq_out'

    def test_gru_roundtrip(self):
        payload = self._single_layer_json("GRULayer", {
            "name": 'gru1',
            "hidden_size": '32',
            "bias": 'false',
            "hx_source": 'context',
            "hidden_state_var": 'h',
            "hidden_unused": 'true',
            "hidden_subscript_source": 'hs',
            "hidden_subscript_target": 'ht',
            "is_layer_call": 'true',
            "input_var": 'in',
            "output_var": 'out',
        })
        out = self._roundtrip(payload)
        gru = _nodes_by_type(out, "GRULayer")
        assert gru, 'GRULayer missing from output'
        attrs = _nn_layer_attrs(gru[0])
        assert attrs.get('name') == 'gru1'
        assert attrs.get('hidden_size') == '32'
        assert attrs.get('bias') == 'false'
        assert attrs.get('hx_source') == 'context'
        assert attrs.get('hidden_state_var') == 'h'
        assert attrs.get('hidden_unused') == 'true'
        assert attrs.get('hidden_subscript_source') == 'hs'
        assert attrs.get('hidden_subscript_target') == 'ht'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'in'
        assert attrs.get('output_var') == 'out'

    def test_embedding_roundtrip(self):
        payload = self._single_layer_json("EmbeddingLayer", {
            "name": 'emb',
            "num_embeddings": '1000',
            "embedding_dim": '64',
            "padding_idx": '0',
            "permute_in": 'true',
            "permute_out": 'false',
            "is_layer_call": 'true',
            "input_var": 'tokens',
            "output_var": 'embeddings',
        })
        out = self._roundtrip(payload)
        emb = _nodes_by_type(out, "EmbeddingLayer")
        assert emb, 'EmbeddingLayer missing from output'
        attrs = _nn_layer_attrs(emb[0])
        assert attrs.get('num_embeddings') == '1000'
        assert attrs.get('embedding_dim') == '64'
        assert attrs.get('padding_idx') == '0'
        assert attrs.get('permute_in') == 'true'
        assert attrs.get('permute_out') == 'false'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'tokens'
        assert attrs.get('output_var') == 'embeddings'

    def test_batchnorm_roundtrip(self):
        payload = self._single_layer_json("BatchNormalizationLayer", {
            "name": 'bn',
            "num_features": '64',
            "batch_normalization.dimension": '2D',
            "eps": '0.001',
            "momentum": '0.05',
            "affine": 'false',
            "track_running_stats": 'true',
            "is_layer_call": 'true',
            "input_var": 'x_bn',
            "output_var": 'y_bn',
        })
        out = self._roundtrip(payload)
        bn = _nodes_by_type(out, "BatchNormalizationLayer")
        assert bn, 'BatchNormalizationLayer missing from output'
        attrs = _nn_layer_attrs(bn[0])
        assert attrs.get('num_features') == '64'
        assert attrs.get('batch_normalization.dimension') == '2D'
        assert attrs.get('eps') == '0.001'
        assert attrs.get('momentum') == '0.05'
        assert attrs.get('affine') == 'false'
        assert attrs.get('track_running_stats') == 'true'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'x_bn'
        assert attrs.get('output_var') == 'y_bn'

    def test_layernorm_roundtrip(self):
        payload = self._single_layer_json("LayerNormalizationLayer", {
            "name": 'ln',
            "normalized_shape": '[32, 32]',
            "eps": '0.0001',
            "affine": 'true',
            "is_layer_call": 'false',
            "input_var": 'norm_in',
            "output_var": 'norm_out',
        })
        out = self._roundtrip(payload)
        ln = _nodes_by_type(out, "LayerNormalizationLayer")
        assert ln, 'LayerNormalizationLayer missing from output'
        attrs = _nn_layer_attrs(ln[0])
        assert attrs.get('normalized_shape') == '[32, 32]'
        assert attrs.get('eps') == '0.0001'
        assert attrs.get('affine') == 'true'
        assert attrs.get('is_layer_call') == 'false'
        assert attrs.get('input_var') == 'norm_in'
        assert attrs.get('output_var') == 'norm_out'

    def test_dropout_roundtrip(self):
        payload = self._single_layer_json("DropoutLayer", {
            "name": 'drop',
            "rate": '0.5',
            "dimension": '2D',
            "permute_in": 'false',
            "permute_out": 'true',
            "is_layer_call": 'false',
            "input_var": 'drop_in',
            "output_var": 'drop_out',
        })
        out = self._roundtrip(payload)
        drop = _nodes_by_type(out, "DropoutLayer")
        assert drop, 'DropoutLayer missing from output'
        attrs = _nn_layer_attrs(drop[0])
        assert attrs.get('rate') == '0.5'
        assert attrs.get('dimension') == '2D'
        assert attrs.get('permute_in') == 'false'
        assert attrs.get('permute_out') == 'true'
        assert attrs.get('is_layer_call') == 'false'
        assert attrs.get('input_var') == 'drop_in'
        assert attrs.get('output_var') == 'drop_out'

    def test_conv1d_roundtrip(self):
        payload = self._single_layer_json("Conv1DLayer", {
            "name": 'c1',
            "kernel_dim": '[3]',
            "out_channels": '16',
            "dilation": '[2]',
            "groups": '2',
            "bias": 'false',
            "is_layer_call": 'true',
            "input_var": 'x_in',
            "output_var": 'x_out',
        })
        out = self._roundtrip(payload)
        c1 = _nodes_by_type(out, "Conv1DLayer")
        assert c1, 'Conv1DLayer missing from output'
        attrs = _nn_layer_attrs(c1[0])
        assert attrs.get('kernel_dim') == '[3]'
        assert attrs.get('out_channels') == '16'
        assert attrs.get('dilation') == '[2]'
        assert attrs.get('groups') == '2'
        assert attrs.get('bias') == 'false'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'x_in'
        assert attrs.get('output_var') == 'x_out'

    def test_conv2d_roundtrip(self):
        payload = self._single_layer_json("Conv2DLayer", {
            "name": 'c2',
            "kernel_dim": '[3, 3]',
            "out_channels": '32',
            "dilation": '[1, 1]',
            "groups": '1',
            "bias": 'true',
            "is_layer_call": 'false',
            "input_var": 'img',
            "output_var": 'feat',
        })
        out = self._roundtrip(payload)
        c2 = _nodes_by_type(out, "Conv2DLayer")
        assert c2, 'Conv2DLayer missing from output'
        attrs = _nn_layer_attrs(c2[0])
        assert attrs.get('kernel_dim') == '[3, 3]'
        assert attrs.get('out_channels') == '32'
        assert attrs.get('dilation') == '[1, 1]'
        assert attrs.get('groups') == '1'
        assert attrs.get('bias') == 'true'
        assert attrs.get('is_layer_call') == 'false'
        assert attrs.get('input_var') == 'img'
        assert attrs.get('output_var') == 'feat'

    def test_conv3d_roundtrip(self):
        payload = self._single_layer_json("Conv3DLayer", {
            "name": 'c3',
            "kernel_dim": '[3, 3, 3]',
            "out_channels": '64',
            "dilation": '[2, 2, 2]',
            "groups": '4',
            "bias": 'false',
            "is_layer_call": 'true',
            "input_var": 'volume_in',
            "output_var": 'volume_out',
        })
        out = self._roundtrip(payload)
        c3 = _nodes_by_type(out, "Conv3DLayer")
        assert c3, 'Conv3DLayer missing from output'
        attrs = _nn_layer_attrs(c3[0])
        assert attrs.get('kernel_dim') == '[3, 3, 3]'
        assert attrs.get('out_channels') == '64'
        assert attrs.get('dilation') == '[2, 2, 2]'
        assert attrs.get('groups') == '4'
        assert attrs.get('bias') == 'false'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'volume_in'
        assert attrs.get('output_var') == 'volume_out'

    def test_linear_roundtrip(self):
        payload = self._single_layer_json("LinearLayer", {
            "name": 'fc',
            "out_features": '10',
            "bias": 'false',
            "is_layer_call": 'true',
            "input_var": 'hidden',
            "output_var": 'logits',
        })
        out = self._roundtrip(payload)
        fc = _nodes_by_type(out, "LinearLayer")
        assert fc, 'LinearLayer missing from output'
        attrs = _nn_layer_attrs(fc[0])
        assert attrs.get('out_features') == '10'
        assert attrs.get('bias') == 'false'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_var') == 'hidden'
        assert attrs.get('output_var') == 'logits'

    def test_pooling_roundtrip(self):
        # `dimension` is qualified to disambiguate from BatchNormalization.
        payload = self._single_layer_json("PoolingLayer", {
            "name": "pool", "pooling_type": "max",
            "pooling.dimension": "2D", "kernel_dim": "[2, 2]",
        })
        out = self._roundtrip(payload)
        pool = _nodes_by_type(out, "PoolingLayer")
        assert pool
        attrs = _nn_layer_attrs(pool[0])
        assert attrs.get("pooling_type") == "max"
        assert attrs.get("pooling.dimension") == "2D"
        assert attrs.get("kernel_dim") == "[2, 2]"

    def test_simple_rnn_roundtrip(self):
        payload = self._single_layer_json("RNNLayer", {
            "name": 'rnn1',
            "hidden_size": '64',
            "bias": 'true',
            "hx_source": 'prev_h',
            "hidden_state_var": 'h_init',
            "hidden_unused": 'false',
            "hidden_subscript_source": 'h_src',
            "hidden_subscript_target": 'h_tgt',
            "is_layer_call": 'false',
            "input_var": 'x',
            "output_var": 'y',
        })
        out = self._roundtrip(payload)
        rnn = _nodes_by_type(out, "RNNLayer")
        assert rnn, 'RNNLayer missing from output'
        attrs = _nn_layer_attrs(rnn[0])
        assert attrs.get('hidden_size') == '64'
        assert attrs.get('bias') == 'true'
        assert attrs.get('hx_source') == 'prev_h'
        assert attrs.get('hidden_state_var') == 'h_init'
        assert attrs.get('hidden_unused') == 'false'
        assert attrs.get('hidden_subscript_source') == 'h_src'
        assert attrs.get('hidden_subscript_target') == 'h_tgt'
        assert attrs.get('is_layer_call') == 'false'
        assert attrs.get('input_var') == 'x'
        assert attrs.get('output_var') == 'y'

    @staticmethod
    def _tensor_op_json(tns_type, extra_attrs):
        """Build an NN diagram with a TensorOp preceded by two LinearLayers
        named ``layer_a`` and ``layer_b``."""
        attrs = {"name": f"{tns_type}_op", "tns_type": tns_type, **extra_attrs}
        payload = TestNNDiagramRoundtrip._single_layer_json("TensorOp", attrs)
        nodes = payload["model"]["nodes"]
        edges = payload["model"]["edges"]
        for sup_id, sup_name, sup_x in (("supA", "layer_a", 200), ("supB", "layer_b", 320)):
            nodes.append({
                "id": sup_id, "type": "LinearLayer", "parentId": "c",
                "position": {"x": sup_x, "y": 60},
                "width": 110, "height": 110,
                "measured": {"width": 110, "height": 110},
                "data": {"name": sup_name,
                         "attributes": {"name": sup_name, "out_features": "4"}},
            })
        edges.append({
            "id": "rel_a_b", "source": "supA", "target": "supB",
            "type": "NNNext",
            "sourceHandle": "Right", "targetHandle": "Left",
            "data": {"name": "next", "points": []},
        })
        edges.append({
            "id": "rel_b_op", "source": "supB", "target": "L",
            "type": "NNNext",
            "sourceHandle": "Right", "targetHandle": "Left",
            "data": {"name": "next", "points": []},
        })
        return payload

    def test_tensorop_concatenate_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("concatenate", {
            "concatenate_dim": '1',
            "layers_of_tensors": "['layer_a', 'layer_b']",
            "actual_vars": "['output', 'hidden']",
            "input_var": 'concat_in',
            "output_var": 'concat_out',
            "permute_in": 'true',
            "permute_out": 'false',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        assert ops, 'TensorOp missing from output'
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'concatenate'
        assert attrs.get('concatenate_dim') == '1'
        assert attrs.get('actual_vars') == "['output', 'hidden']"
        assert attrs.get('input_var') == 'concat_in'
        assert attrs.get('output_var') == 'concat_out'
        assert attrs.get('permute_in') == 'true'
        assert attrs.get('permute_out') == 'false'

    def test_tensorop_multiply_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("multiply", {
            "layers_of_tensors": "['layer_a', 'layer_b']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get("tns_type") == "multiply"

    def test_tensorop_matmultiply_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("matmultiply", {
            "layers_of_tensors": "['layer_a', 'layer_b']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get("tns_type") == "matmultiply"

    def test_tensorop_reshape_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("reshape", {
            "layers_of_tensors": "['layer_a']",
            "reshape_dim": '[1, -1]',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'reshape'
        assert attrs.get('reshape_dim') == '[1, -1]'

    def test_tensorop_transpose_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("transpose", {
            "layers_of_tensors": "['layer_a']",
            "transpose_dim": '[0, 2]',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'transpose'
        assert attrs.get('transpose_dim') == '[0, 2]'

    def test_tensorop_permute_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("permute", {
            "permute_dim": "[0, 3, 1, 2]",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get("tns_type") == "permute"
        assert attrs.get("permute_dim") == "[0, 3, 1, 2]"

    def test_tensorop_binop_add_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("binop_add", {
            "layers_of_tensors": "['layer_a', 'layer_b']",
            "actual_vars": "['output', 'hidden']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'binop_add'
        assert attrs.get('actual_vars') == "['output', 'hidden']"

    def test_tensorop_mean_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("mean", {
            "reduce_dim": '1',
            "layers_of_tensors": "['layer_a']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'mean'
        assert attrs.get('reduce_dim') == '1'

    def test_tensorop_max_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("max", {
            "reduce_dim": '0',
            "reduce_keepdims": 'true',
            "layers_of_tensors": "['layer_a']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'max'
        assert attrs.get('reduce_dim') == '0'
        assert attrs.get('reduce_keepdims') == 'true'

    def test_tensorop_squeeze_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("squeeze", {
            "reduce_dim": '2',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'squeeze'
        assert attrs.get('reduce_dim') == '2'
        assert attrs.get('input_var') == 'x'

    def test_tensorop_subscript_roundtrip(self):
        # subscript_indices format: list of dicts with type='index' or 'slice'
        # Represents [0, :, 1:3]
        indices = [
            {"type": "index", "value": 0},
            {"type": "slice", "start": None, "stop": None, "step": None},
            {"type": "slice", "start": 1, "stop": 3, "step": None},
        ]
        out = self._roundtrip(self._tensor_op_json("subscript", {
            "subscript_indices": str(indices),
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'subscript'
        # The round-trip preserves the dict structure *and* emits it as real
        # JSON: a Python repr (single quotes / None) makes the frontend's
        # JSON.parse throw.
        emitted = attrs.get('subscript_indices')
        assert json.loads(emitted) == indices
        assert "'" not in emitted and 'None' not in emitted
        assert attrs.get('input_var') == 'x'

    def test_tensorop_subscript_accepts_json_payload(self):
        """A JSON-shaped subscript_indices value (what the converter now emits,
        and therefore what a re-imported diagram carries) must parse."""
        indices = [{"type": "index", "value": 2}]
        out = self._roundtrip(self._tensor_op_json("subscript", {
            "subscript_indices": json.dumps(indices),
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert json.loads(attrs['subscript_indices']) == indices

    def test_tensorop_subscript_malformed_raises(self):
        """Malformed subscript_indices must surface a ValueError naming the
        field instead of silently dropping the user's value."""
        with pytest.raises(ValueError, match=r"subscript_indices"):
            self._roundtrip(self._tensor_op_json("subscript", {
                "subscript_indices": '[{type: index,',
            }))

    def test_tensorop_pad_amount_malformed_raises(self):
        """Same contract for pad_amount (was also swallowed to None)."""
        with pytest.raises(ValueError, match=r"pad_amount"):
            self._roundtrip(self._tensor_op_json("pad", {
                "pad_amount": '[[1, 1',
            }))

    def test_tensorop_shape_dim_attribute_roundtrip(self):
        """The shape_dim *attribute* (distinct from the shape_dim tns_type,
        covered below) was parsed then dropped: it never reached the TensorOp
        constructor and neither exporter emitted it."""
        out = self._roundtrip(self._tensor_op_json("shape_dim", {
            "reduce_dim": '0',
            "shape_dim": '2',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'shape_dim'
        assert attrs.get('reduce_dim') == '0'
        assert attrs.get('shape_dim') == '2'

    def test_tensorop_repeat_dim_keeps_mixed_list(self):
        """repeat_dim is declared list[int | str]; parse_list_of_ints used to
        return None on the first non-int item, losing the whole value."""
        out = self._roundtrip(self._tensor_op_json("repeat", {
            "repeat_dim": "[1, 'op_3', 1]",
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('repeat_dim') == "[1, 'op_3', 1]"

    def test_batch_norm_permute_flags_roundtrip(self):
        """BatchNormLayer gained permute_in/permute_out in the metamodel;
        neither converter direction used to carry them."""
        out = self._roundtrip(self._single_layer_json("BatchNormalizationLayer", {
            "name": 'bn1',
            "num_features": '32',
            "batch_normalization.dimension": '2D',
            "permute_in": 'true',
            "permute_out": 'true',
        }))
        layers = _nodes_by_type(out, "BatchNormalizationLayer")
        attrs = _nn_layer_attrs(layers[0])
        assert attrs.get('permute_in') == 'true'
        assert attrs.get('permute_out') == 'true'

    def test_hand_written_buml_model_keeps_explicit_attributes(self):
        """A model built directly in Python never populates the explicit-attr
        sidecar. BUML -> JSON must still emit every non-default attribute."""
        from besser.BUML.metamodel.nn import NN as _NN, Conv2D as _Conv2D
        nn = _NN(name='HandWritten')
        nn.add_layer(_Conv2D(
            name='c1', kernel_dim=[3, 3], out_channels=8, bias=False,
            permute_in=True, permute_out=True, is_layer_call=True,
            dilation=[2, 2], groups=4, input_reused=True,
        ))
        out = nn_model_to_json(nn)
        convs = _nodes_by_type(out, "Conv2DLayer")
        assert convs, 'Conv2D missing from output'
        attrs = _nn_layer_attrs(convs[0])
        assert attrs.get('bias') == 'false'
        assert attrs.get('permute_in') == 'true'
        assert attrs.get('permute_out') == 'true'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_reused') == 'true'
        assert attrs.get('dilation') == '[2, 2]'
        assert attrs.get('groups') == '4'

    def test_buml_ast_import_keeps_explicit_attributes(self):
        """Same guarantee for /get-json-model: the AST parser rebuilds the
        model without touching the explicit-attr sidecar, so BUML -> JSON has
        to fall back to default comparison there too."""
        from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
            nn_buml_to_json,
        )
        buml = "\n".join([
            "from besser.BUML.metamodel.nn import NN, Conv2D",
            "main_nn = NN(name='AstNet')",
            "layer_0 = Conv2D(name='c1', kernel_dim=[3, 3], out_channels=8, "
            "bias=False, permute_in=True, permute_out=True, is_layer_call=True, "
            "dilation=[2, 2], groups=4, input_reused=True)",
            "main_nn.add_layer(layer_0)",
        ])
        out = nn_buml_to_json(buml)
        convs = _nodes_by_type(out, "Conv2DLayer")
        assert convs, 'Conv2D missing from output'
        attrs = _nn_layer_attrs(convs[0])
        assert attrs.get('bias') == 'false'
        assert attrs.get('permute_in') == 'true'
        assert attrs.get('permute_out') == 'true'
        assert attrs.get('is_layer_call') == 'true'
        assert attrs.get('input_reused') == 'true'
        assert attrs.get('dilation') == '[2, 2]'
        assert attrs.get('groups') == '4'

    def test_tensorop_interpolate_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("interpolate", {
            "interpolate_size": '[256, 256]',
            "interpolate_mode": 'bilinear',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'interpolate'
        assert attrs.get('interpolate_size') == '[256, 256]'
        assert attrs.get('interpolate_mode') == 'bilinear'
        assert attrs.get('input_var') == 'x'

    def test_tensorop_pad_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("pad", {
            "pad_amount": '[[1, 1], [2, 2]]',
            "pad_mode": 'constant',
            "pad_value": '0.0',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'pad'
        assert attrs.get('pad_amount') == '[[1, 1], [2, 2]]'
        assert attrs.get('pad_mode') == 'constant'
        assert attrs.get('pad_value') == '0.0'
        assert attrs.get('input_var') == 'x'

    def test_tensorop_dropout_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("dropout", {
            "dropout_rate": '0.3',
            "dropout_training_aware": 'true',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'dropout'
        assert attrs.get('dropout_rate') == '0.3'
        assert attrs.get('dropout_training_aware') == 'true'
        assert attrs.get('input_var') == 'x'

    def test_tensorop_split_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("split", {
            "split_dim": '1',
            "split_sizes": '3',
            "output_vars": "['chunk1', 'chunk2', 'chunk3']",
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'split'
        assert attrs.get('split_dim') == '1'
        assert attrs.get('split_sizes') == '3'
        assert attrs.get('output_vars') == "['chunk1', 'chunk2', 'chunk3']"
        assert attrs.get('input_var') == 'x'

    def test_tensorop_identity_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json("identity", {
            "layers_of_tensors": "['layer_a']",
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'identity'

    def test_tensorop_shape_dim_roundtrip(self):
        # shape_dim tns_type requires reduce_dim parameter
        out = self._roundtrip(self._tensor_op_json("shape_dim", {
            "reduce_dim": '2',
            "input_var": 'x',
        }))
        ops = _nodes_by_type(out, "TensorOp")
        attrs = _nn_layer_attrs(ops[0])
        assert attrs.get('tns_type') == 'shape_dim'
        assert attrs.get('reduce_dim') == '2'

    def test_nn_container_vars_roundtrip(self):
        """NNContainer preserves input_var and return_vars."""
        payload = self._minimal_nn_json()
        container_node = next(n for n in payload["model"]["nodes"] if n["id"] == "c1")
        container_node["data"]["input_var"] = "x_input"
        container_node["data"]["return_vars"] = ["output1", "output2"]
        out = self._roundtrip(payload)
        containers = _nodes_by_type(out, "NNContainer")
        assert len(containers) == 1
        container = _data(containers[0])
        assert container["input_var"] == "x_input"
        assert container["return_vars"] == ["output1", "output2"]

    def test_pooling_invalid_pooling_type_lists_allowed_values(self):
        payload = self._single_layer_json("PoolingLayer", {
            "name": "p", "pooling_type": "avg",  # legacy / typo
            "pooling.dimension": "2D",
        })
        with pytest.raises(ValueError, match=r"Allowed values"):
            process_nn_diagram(payload)

    def test_conv_permute_in_true_roundtrips(self):
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json("Conv2DLayer", {
            "name": "c1", "kernel_dim": "[3, 3]",
            "out_channels": "16", "permute_in": "true",
        })
        nn = process_nn_diagram(payload)
        convs = [m for m in nn.modules if type(m).__name__ == "Conv2D"]
        assert convs and convs[0].permute_in is True
        assert is_explicit(convs[0], "permute_in")

    def test_rnn_bidirectional_dropout_batch_first_roundtrip(self):
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json("LSTMLayer", {
            "name": "lstm1", "hidden_size": "128",
            "bidirectional": "true", "dropout": "0.3", "batch_first": "true",
        })
        nn = process_nn_diagram(payload)
        lstms = [m for m in nn.modules if type(m).__name__ == "LSTMLayer"]
        assert lstms
        lstm = lstms[0]
        assert lstm.bidirectional is True
        assert lstm.dropout == 0.3
        assert lstm.batch_first is True
        assert is_explicit(lstm, "bidirectional")
        assert is_explicit(lstm, "dropout")
        assert is_explicit(lstm, "batch_first")

    def test_tensorop_invalid_tns_type_raises(self):
        payload = self._tensor_op_json("concatinate", {  # typo
            "concatenate_dim": "1",
            "layers_of_tensors": "['a', 'b']",
        })
        with pytest.raises(ValueError, match=r"invalid tns_type.*Allowed values"):
            process_nn_diagram(payload)

    def test_conv_permute_in_false_explicitly_marked(self):
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json("Conv2DLayer", {
            "name": "c1", "kernel_dim": "[3, 3]",
            "out_channels": "16", "permute_in": "false",
        })
        nn = process_nn_diagram(payload)
        convs = [m for m in nn.modules if type(m).__name__ == "Conv2D"]
        assert convs and convs[0].permute_in is False
        assert is_explicit(convs[0], "permute_in")

    def test_sibling_sub_nn_same_name_emits_distinct_containers(self):
        from besser.BUML.metamodel.nn import NN, LinearLayer
        nn = NN(name="Parent")
        block_a = NN(name="Block")
        block_a.add_layer(LinearLayer(name="a_lin", out_features=4))
        block_b = NN(name="Block")
        block_b.add_layer(LinearLayer(name="b_lin", out_features=8))
        nn.add_sub_nn(block_a)
        nn.add_sub_nn(block_b)
        out = nn_model_to_json(nn)
        containers = _nodes_by_type(out, "NNContainer")
        assert len(containers) == 3, (
            f"expected 3 containers (parent + 2 same-named sub-NNs), got {len(containers)}"
        )

    def test_conv_invalid_padding_type_lists_allowed_values(self):
        payload = self._single_layer_json("Conv2DLayer", {
            "name": "c1", "kernel_dim": "[3, 3]",
            "out_channels": "16", "padding_type": "Same",
        })
        with pytest.raises(ValueError, match=r"padding_type.*Allowed values"):
            process_nn_diagram(payload)

    def test_pooling_permute_in_roundtrips_via_converter(self):
        from besser.BUML.metamodel.nn import NN, PoolingLayer
        from besser.utilities.buml_code_builder.nn_explicit_attrs import mark_explicit
        nn = NN(name="PoolPermute")
        pool = PoolingLayer(
            name="p", pooling_type="max", dimension="2D", kernel_dim=[2, 2],
        )
        pool.permute_in = True
        mark_explicit(pool, "permute_in")
        nn.add_layer(pool)
        out = nn_model_to_json(nn)
        pools = _nodes_by_type(out, "PoolingLayer")
        assert pools, "PoolingLayer missing from output"
        attrs = _nn_layer_attrs(pools[0])
        assert attrs.get("permute_in") == "true"

    def test_parse_float_rejects_european_decimal(self):
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            parse_float,
        )
        with pytest.raises(ValueError, match=r"not a valid number"):
            parse_float("0,5")

    def test_configuration_invalid_optimizer_lists_allowed_values(self):
        payload = self._single_layer_json("Configuration", {
            "batch_size": "32", "epochs": "10", "learning_rate": "0.001",
            "optimizer": "rmsprop",  # not in whitelist
            "loss_function": "crossentropy",
            "metrics": "[accuracy]",
        })
        with pytest.raises(ValueError, match=r"optimizer.*Allowed values"):
            process_nn_diagram(payload)
