"""
Roundtrip tests for JSON -> BUML -> JSON converters.

Validates that essential data survives a full conversion cycle:
  1. Class diagram:  JSON -> process_class_diagram -> DomainModel -> class_buml_to_json -> JSON
  2. State machine:  JSON -> process_state_machine -> StateMachine -> state_machine_object_to_json -> JSON
  3. Object diagram: JSON -> process_object_diagram -> ObjectModel (data verification only;
                     there is no ObjectModel-to-JSON converter that works from metamodel objects)

Layout positions (bounds, path) are intentionally NOT compared because the
BUML-to-JSON converter re-computes layout from scratch.
"""

import pytest

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
# Helpers
# ---------------------------------------------------------------------------

def _resolve_elements(json_data):
    """Resolve the elements dict from either input format (model.elements) or output format (elements)."""
    elements = json_data.get("elements", {})
    if not elements:
        elements = (json_data.get("model") or {}).get("elements", {})
    return elements


def _resolve_relationships(json_data):
    """Resolve the relationships dict from either input format (model.relationships) or output format (relationships)."""
    rels = json_data.get("relationships", {})
    if not rels:
        rels = (json_data.get("model") or {}).get("relationships", {})
    return rels


def _extract_elements_by_type(json_result, element_type):
    """Return a list of elements matching the given type from the JSON result."""
    elements = _resolve_elements(json_result)
    return [e for e in elements.values() if e.get("type") == element_type]


def _extract_relationships_by_type(json_result, rel_type):
    """Return a list of relationships matching the given type from the JSON result."""
    relationships = _resolve_relationships(json_result)
    return [r for r in relationships.values() if r.get("type") == rel_type]


def _get_class_names(json_data):
    """Extract class/abstract-class names from a JSON diagram."""
    elements = _resolve_elements(json_data)
    return {
        e["name"]
        for e in elements.values()
        if e.get("type") in ("Class", "AbstractClass")
    }


def _get_enum_names(json_data):
    """Extract enumeration names from a JSON diagram."""
    elements = _resolve_elements(json_data)
    return {
        e["name"]
        for e in elements.values()
        if e.get("type") == "Enumeration"
    }


def _get_attribute_names_for_class(json_data, class_name):
    """Return attribute names belonging to a given class in the JSON."""
    elements = _resolve_elements(json_data)
    # Find the class element
    class_elem = None
    for e in elements.values():
        if e.get("type") in ("Class", "AbstractClass") and e.get("name") == class_name:
            class_elem = e
            break
    if not class_elem:
        return set()
    # Resolve attribute IDs
    attr_names = set()
    for attr_id in class_elem.get("attributes", []):
        attr_elem = elements.get(attr_id, {})
        name = attr_elem.get("name", "")
        # New format uses separate 'name' field; legacy encodes "vis name: type" in name
        if attr_elem.get("attributeType"):
            attr_names.add(name)
        else:
            # Legacy: parse "+ name: type" or just "name"
            attr_names.add(name)
    return attr_names


def _get_method_names_for_class(json_data, class_name):
    """Return method names belonging to a given class in the JSON.

    Method names in the output JSON are full signatures like
    "+ doStuff(a: int): str".  This helper extracts just the bare name
    (the part before the opening parenthesis, stripped of visibility prefix).
    """
    elements = _resolve_elements(json_data)
    class_elem = None
    for e in elements.values():
        if e.get("type") in ("Class", "AbstractClass") and e.get("name") == class_name:
            class_elem = e
            break
    if not class_elem:
        return set()
    names = set()
    for method_id in class_elem.get("methods", []):
        method_elem = elements.get(method_id, {})
        sig = method_elem.get("name", "")
        # Strip visibility symbol if present
        if sig and sig[0] in "+-#~":
            sig = sig[1:].lstrip()
        # Extract name before '('
        if "(" in sig:
            bare = sig[:sig.index("(")].strip()
        else:
            bare = sig.strip()
        names.add(bare)
    return names


# ===========================================================================
# Fixtures: reusable JSON payloads
# ===========================================================================

@pytest.fixture
def minimal_class_diagram_json():
    """A minimal class diagram with two classes, attributes, a method, and a bidirectional association."""
    # Element IDs are arbitrary but must be consistent within the fixture
    return {
        "title": "LibraryModel",
        "model": {
            "elements": {
                "cls-book": {
                    "id": "cls-book",
                    "name": "Book",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-title", "attr-pages"],
                    "methods": ["meth-summary"],
                },
                "attr-title": {
                    "id": "attr-title",
                    "name": "title",
                    "type": "ClassAttribute",
                    "owner": "cls-book",
                    "visibility": "public",
                    "attributeType": "str",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                },
                "attr-pages": {
                    "id": "attr-pages",
                    "name": "pages",
                    "type": "ClassAttribute",
                    "owner": "cls-book",
                    "visibility": "private",
                    "attributeType": "int",
                    "isOptional": True,
                    "bounds": {"x": 0, "y": 70, "width": 159, "height": 30},
                },
                "meth-summary": {
                    "id": "meth-summary",
                    "name": "+ summary(verbose: bool): str",
                    "type": "ClassMethod",
                    "owner": "cls-book",
                    "bounds": {"x": 0, "y": 100, "width": 159, "height": 30},
                },
                "cls-author": {
                    "id": "cls-author",
                    "name": "Author",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 300, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-name"],
                    "methods": [],
                },
                "attr-name": {
                    "id": "attr-name",
                    "name": "name",
                    "type": "ClassAttribute",
                    "owner": "cls-author",
                    "visibility": "public",
                    "attributeType": "str",
                    "isOptional": False,
                    "bounds": {"x": 300, "y": 40, "width": 159, "height": 30},
                },
            },
            "relationships": {
                "rel-writes": {
                    "id": "rel-writes",
                    "name": "writes",
                    "type": "ClassBidirectional",
                    "source": {
                        "element": "cls-author",
                        "multiplicity": "1..*",
                        "role": "author",
                    },
                    "target": {
                        "element": "cls-book",
                        "multiplicity": "0..*",
                        "role": "book",
                    },
                },
            },
        },
    }


@pytest.fixture
def class_diagram_with_inheritance_json():
    """A class diagram with an abstract base class and an inheritance relationship."""
    return {
        "title": "VehicleModel",
        "model": {
            "elements": {
                "cls-vehicle": {
                    "id": "cls-vehicle",
                    "name": "Vehicle",
                    "type": "AbstractClass",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-speed"],
                    "methods": [],
                },
                "attr-speed": {
                    "id": "attr-speed",
                    "name": "speed",
                    "type": "ClassAttribute",
                    "owner": "cls-vehicle",
                    "visibility": "public",
                    "attributeType": "float",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                },
                "cls-car": {
                    "id": "cls-car",
                    "name": "Car",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 0, "y": 200, "width": 160, "height": 100},
                    "attributes": ["attr-doors"],
                    "methods": [],
                },
                "attr-doors": {
                    "id": "attr-doors",
                    "name": "doors",
                    "type": "ClassAttribute",
                    "owner": "cls-car",
                    "visibility": "public",
                    "attributeType": "int",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 240, "width": 159, "height": 30},
                },
            },
            "relationships": {
                "rel-inherit": {
                    "id": "rel-inherit",
                    "name": "",
                    "type": "ClassInheritance",
                    "source": {"element": "cls-car"},
                    "target": {"element": "cls-vehicle"},
                },
            },
        },
    }


@pytest.fixture
def class_diagram_with_enum_json():
    """A class diagram with an enumeration and a class that uses it as an attribute type."""
    return {
        "title": "OrderModel",
        "model": {
            "elements": {
                "enum-status": {
                    "id": "enum-status",
                    "name": "OrderStatus",
                    "type": "Enumeration",
                    "owner": None,
                    "bounds": {"x": 300, "y": 0, "width": 160, "height": 100},
                    "attributes": ["lit-pending", "lit-shipped", "lit-delivered"],
                    "methods": [],
                },
                "lit-pending": {
                    "id": "lit-pending",
                    "name": "PENDING",
                    "type": "ClassAttribute",
                    "owner": "enum-status",
                    "bounds": {"x": 300, "y": 40, "width": 159, "height": 30},
                },
                "lit-shipped": {
                    "id": "lit-shipped",
                    "name": "SHIPPED",
                    "type": "ClassAttribute",
                    "owner": "enum-status",
                    "bounds": {"x": 300, "y": 70, "width": 159, "height": 30},
                },
                "lit-delivered": {
                    "id": "lit-delivered",
                    "name": "DELIVERED",
                    "type": "ClassAttribute",
                    "owner": "enum-status",
                    "bounds": {"x": 300, "y": 100, "width": 159, "height": 30},
                },
                "cls-order": {
                    "id": "cls-order",
                    "name": "Order",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-status", "attr-total"],
                    "methods": [],
                },
                "attr-status": {
                    "id": "attr-status",
                    "name": "status",
                    "type": "ClassAttribute",
                    "owner": "cls-order",
                    "visibility": "public",
                    "attributeType": "OrderStatus",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                },
                "attr-total": {
                    "id": "attr-total",
                    "name": "total",
                    "type": "ClassAttribute",
                    "owner": "cls-order",
                    "visibility": "public",
                    "attributeType": "float",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 70, "width": 159, "height": 30},
                },
            },
            "relationships": {},
        },
    }


@pytest.fixture
def class_diagram_with_composition_json():
    """A class diagram with a composition relationship."""
    return {
        "title": "CompanyModel",
        "model": {
            "elements": {
                "cls-company": {
                    "id": "cls-company",
                    "name": "Company",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-cname"],
                    "methods": [],
                },
                "attr-cname": {
                    "id": "attr-cname",
                    "name": "companyName",
                    "type": "ClassAttribute",
                    "owner": "cls-company",
                    "visibility": "public",
                    "attributeType": "str",
                    "isOptional": False,
                    "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                },
                "cls-department": {
                    "id": "cls-department",
                    "name": "Department",
                    "type": "Class",
                    "owner": None,
                    "bounds": {"x": 300, "y": 0, "width": 160, "height": 100},
                    "attributes": ["attr-dname"],
                    "methods": [],
                },
                "attr-dname": {
                    "id": "attr-dname",
                    "name": "deptName",
                    "type": "ClassAttribute",
                    "owner": "cls-department",
                    "visibility": "public",
                    "attributeType": "str",
                    "isOptional": False,
                    "bounds": {"x": 300, "y": 40, "width": 159, "height": 30},
                },
            },
            "relationships": {
                "rel-has": {
                    "id": "rel-has",
                    "name": "has_departments",
                    "type": "ClassComposition",
                    "source": {
                        "element": "cls-company",
                        "multiplicity": "1",
                        "role": "company",
                    },
                    "target": {
                        "element": "cls-department",
                        "multiplicity": "1..*",
                        "role": "department",
                    },
                },
            },
        },
    }


@pytest.fixture
def state_machine_json():
    """A state machine with two states, a transition, and code blocks for body and event."""
    return {
        "title": "OrderStateMachine",
        "model": {
            "elements": {
                "initial": {
                    "id": "initial",
                    "name": "",
                    "type": "StateInitialNode",
                    "owner": None,
                    "bounds": {"x": -850, "y": -280, "width": 45, "height": 45},
                },
                "state-idle": {
                    "id": "state-idle",
                    "name": "idle",
                    "type": "State",
                    "owner": None,
                    "bounds": {"x": -550, "y": -300, "width": 160, "height": 100},
                    "bodies": ["body-idle-body"],
                    "fallbackBodies": [],
                },
                "body-idle-body": {
                    "id": "body-idle-body",
                    "name": "idle_body",
                    "type": "StateBody",
                    "owner": "state-idle",
                    "bounds": {"x": -549.5, "y": -259.5, "width": 159, "height": 30},
                },
                "state-processing": {
                    "id": "state-processing",
                    "name": "processing",
                    "type": "State",
                    "owner": None,
                    "bounds": {"x": -60, "y": -300, "width": 160, "height": 100},
                    "bodies": ["body-proc-body"],
                    "fallbackBodies": [],
                },
                "body-proc-body": {
                    "id": "body-proc-body",
                    "name": "process_body",
                    "type": "StateBody",
                    "owner": "state-processing",
                    "bounds": {"x": -59.5, "y": -259.5, "width": 159, "height": 30},
                },
                "code-idle-body": {
                    "id": "code-idle-body",
                    "name": "idle_body",
                    "type": "StateCodeBlock",
                    "owner": None,
                    "bounds": {"x": -970, "y": 80, "width": 580, "height": 200},
                    "code": "def idle_body(session):\n    session.reply('Waiting for input.')",
                    "language": "python",
                },
                "code-proc-body": {
                    "id": "code-proc-body",
                    "name": "process_body",
                    "type": "StateCodeBlock",
                    "owner": None,
                    "bounds": {"x": -360, "y": 80, "width": 580, "height": 200},
                    "code": "def process_body(session):\n    session.reply('Processing...')",
                    "language": "python",
                },
                "code-start-event": {
                    "id": "code-start-event",
                    "name": "start_event",
                    "type": "StateCodeBlock",
                    "owner": None,
                    "bounds": {"x": 250, "y": 80, "width": 580, "height": 200},
                    "code": "def start_event(session):\n    return session.message",
                    "language": "python",
                },
            },
            "relationships": {
                "trans-init": {
                    "id": "trans-init",
                    "name": "",
                    "type": "StateTransition",
                    "source": {"element": "initial", "direction": "Right"},
                    "target": {"element": "state-idle", "direction": "Left"},
                },
                "trans-start": {
                    "id": "trans-start",
                    "name": "start_event",
                    "type": "StateTransition",
                    "source": {"element": "state-idle", "direction": "Right"},
                    "target": {"element": "state-processing", "direction": "Left"},
                },
            },
        },
    }


def _build_object_diagram_fixture(minimal_class_diagram_json, include_attributes=True):
    """Build an object diagram JSON referencing the minimal class diagram.

    Creates two objects (book1 of Book, author1 of Author) and a link between them.
    When *include_attributes* is True, objects include attribute slots (which
    triggers the DataValue code path).
    """
    # Produce the reference class diagram JSON via roundtrip
    domain_model = process_class_diagram(minimal_class_diagram_json)
    ref_json = class_buml_to_json(domain_model)
    ref_elements = ref_json["elements"]
    ref_relationships = ref_json["relationships"]

    # Find class IDs and association ID from the produced JSON
    book_class_id = None
    author_class_id = None
    for eid, elem in ref_elements.items():
        if elem.get("type") in ("Class", "AbstractClass"):
            if elem["name"] == "Book":
                book_class_id = eid
            elif elem["name"] == "Author":
                author_class_id = eid

    association_id = None
    for rid, rel in ref_relationships.items():
        if rel.get("type") in ("ClassBidirectional", "ClassUnidirectional", "ClassComposition"):
            association_id = rid
            break

    elements = {
        "obj-book1": {
            "id": "obj-book1",
            "name": "book1",
            "type": "ObjectName",
            "owner": None,
            "bounds": {"x": 0, "y": 0, "width": 200, "height": 100},
            "attributes": [],
            "methods": [],
            "classId": book_class_id,
            "className": "Book",
        },
        "obj-author1": {
            "id": "obj-author1",
            "name": "author1",
            "type": "ObjectName",
            "owner": None,
            "bounds": {"x": 300, "y": 0, "width": 200, "height": 100},
            "attributes": [],
            "methods": [],
            "classId": author_class_id,
            "className": "Author",
        },
    }

    if include_attributes:
        elements["oattr-title"] = {
            "id": "oattr-title",
            "name": "+ title: str = TheGreatGatsby",
            "type": "ObjectAttribute",
            "owner": "obj-book1",
            "bounds": {"x": 0, "y": 30, "width": 199, "height": 30},
        }
        elements["obj-book1"]["attributes"] = ["oattr-title"]

        elements["oattr-aname"] = {
            "id": "oattr-aname",
            "name": "+ name: str = Fitzgerald",
            "type": "ObjectAttribute",
            "owner": "obj-author1",
            "bounds": {"x": 300, "y": 30, "width": 199, "height": 30},
        }
        elements["obj-author1"]["attributes"] = ["oattr-aname"]

    return {
        "title": "LibraryObjects",
        "model": {
            "elements": elements,
            "relationships": {
                "link-writes": {
                    "id": "link-writes",
                    "name": "writes_link",
                    "type": "ObjectLink",
                    "source": {"element": "obj-author1"},
                    "target": {"element": "obj-book1"},
                    "associationId": association_id,
                },
            },
            "referenceDiagramData": ref_json,
        },
    }


@pytest.fixture
def object_diagram_json(minimal_class_diagram_json):
    """Object diagram without attribute values (avoids DataValue name validation bug)."""
    return _build_object_diagram_fixture(minimal_class_diagram_json, include_attributes=False)


@pytest.fixture
def object_diagram_with_attrs_json(minimal_class_diagram_json):
    """Object diagram with attribute values (hits DataValue name validation)."""
    return _build_object_diagram_fixture(minimal_class_diagram_json, include_attributes=True)


# ===========================================================================
# Class Diagram Roundtrip Tests
# ===========================================================================

class TestClassDiagramRoundtrip:
    """JSON -> BUML -> JSON roundtrip for class diagrams."""

    def test_classes_preserved(self, minimal_class_diagram_json):
        """Class names survive the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        original_classes = _get_class_names(minimal_class_diagram_json)
        result_classes = _get_class_names(result)
        assert original_classes == result_classes

    def test_attributes_preserved(self, minimal_class_diagram_json):
        """Attribute names for each class survive the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        for class_name in ("Book", "Author"):
            orig_attrs = _get_attribute_names_for_class(minimal_class_diagram_json, class_name)
            result_attrs = _get_attribute_names_for_class(result, class_name)
            assert orig_attrs == result_attrs, (
                f"Attribute mismatch for class '{class_name}': {orig_attrs} vs {result_attrs}"
            )

    def test_attribute_types_preserved(self, minimal_class_diagram_json):
        """Attribute types survive the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        # Find Book's attributes in the result
        book_attrs = {}
        for e in result_elements.values():
            if e.get("type") == "ClassAttribute" and e.get("attributeType"):
                # Find owner class name
                owner_id = e.get("owner")
                owner = result_elements.get(owner_id, {})
                if owner.get("name") == "Book":
                    book_attrs[e["name"]] = e["attributeType"]

        assert book_attrs.get("title") == "str"
        assert book_attrs.get("pages") == "int"

    def test_attribute_visibility_preserved(self, minimal_class_diagram_json):
        """Attribute visibility survives the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        visibilities = {}
        for e in result_elements.values():
            if e.get("type") == "ClassAttribute" and e.get("visibility"):
                owner_id = e.get("owner")
                owner = result_elements.get(owner_id, {})
                if owner.get("name") == "Book":
                    visibilities[e["name"]] = e["visibility"]

        assert visibilities.get("title") == "public"
        assert visibilities.get("pages") == "private"

    def test_attribute_optional_preserved(self, minimal_class_diagram_json):
        """Attribute isOptional flag survives the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        optionals = {}
        for e in result_elements.values():
            if e.get("type") == "ClassAttribute" and "isOptional" in e:
                owner_id = e.get("owner")
                owner = result_elements.get(owner_id, {})
                if owner.get("name") == "Book":
                    optionals[e["name"]] = e["isOptional"]

        assert optionals.get("title") is False
        assert optionals.get("pages") is True

    def test_attribute_is_id_and_external_id_preserved(self, minimal_class_diagram_json):
        """`isId` and `isExternalId` flags survive the JSON -> BUML -> JSON roundtrip.

        Regression guard for the is_id work (PR #486) and the is_external_id
        work (issues #230 / #225). We mutate the fixture in place rather than
        building a new one so any future change to the fixture shape doesn't
        cause drift.
        """
        # Mutate the fixture: mark 'title' as id, 'pages' as external id.
        # (isOptional must be False for both — metamodel enforces it.)
        for element in minimal_class_diagram_json["model"]["elements"].values():
            if element.get("type") != "ClassAttribute":
                continue
            if element["name"] == "title":
                element["isId"] = True
                element["isExternalId"] = False
                element["isOptional"] = False
            elif element["name"] == "pages":
                element["isId"] = False
                element["isExternalId"] = True
                element["isOptional"] = False

        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        flags_by_name = {}
        for element in result["elements"].values():
            if element.get("type") != "ClassAttribute":
                continue
            owner = result["elements"].get(element.get("owner"), {})
            if owner.get("name") != "Book":
                continue
            flags_by_name[element["name"]] = {
                "isId": element.get("isId"),
                "isExternalId": element.get("isExternalId"),
            }

        assert flags_by_name["title"] == {"isId": True, "isExternalId": False}
        assert flags_by_name["pages"] == {"isId": False, "isExternalId": True}

    def test_methods_preserved(self, minimal_class_diagram_json):
        """Method names survive the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        orig_methods = _get_method_names_for_class(minimal_class_diagram_json, "Book")
        result_methods = _get_method_names_for_class(result, "Book")
        assert orig_methods == result_methods

    def test_method_parameters_preserved(self, minimal_class_diagram_json):
        """Method parameters survive the roundtrip (encoded in the signature string)."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        method_signatures = []
        for e in result_elements.values():
            if e.get("type") == "ClassMethod":
                method_signatures.append(e["name"])

        # The original method is: "+ summary(verbose: bool): str"
        # After roundtrip, the signature should contain the same info
        assert len(method_signatures) == 1
        sig = method_signatures[0]
        assert "summary" in sig
        assert "verbose" in sig
        assert "bool" in sig
        assert "str" in sig  # return type

    def test_association_preserved(self, minimal_class_diagram_json):
        """Bidirectional association survives the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        bidir_rels = _extract_relationships_by_type(result, "ClassBidirectional")
        assert len(bidir_rels) == 1
        rel = bidir_rels[0]

        # Verify roles survive
        roles = {rel["source"]["role"], rel["target"]["role"]}
        assert roles == {"author", "book"}

        # Verify multiplicities survive
        mults = {rel["source"]["multiplicity"], rel["target"]["multiplicity"]}
        assert "1..*" in mults
        assert "0..*" in mults

    def test_association_name_preserved(self, minimal_class_diagram_json):
        """Association name survives the roundtrip."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)

        bidir_rels = _extract_relationships_by_type(result, "ClassBidirectional")
        assert len(bidir_rels) == 1
        assert bidir_rels[0]["name"] == "writes"

    def test_inheritance_preserved(self, class_diagram_with_inheritance_json):
        """Inheritance relationship and abstract class flag survive the roundtrip."""
        domain_model = process_class_diagram(class_diagram_with_inheritance_json)
        result = class_buml_to_json(domain_model)

        # Verify classes
        assert _get_class_names(result) == {"Vehicle", "Car"}

        # Verify Vehicle is abstract
        abstract_classes = _extract_elements_by_type(result, "AbstractClass")
        assert len(abstract_classes) == 1
        assert abstract_classes[0]["name"] == "Vehicle"

        # Verify concrete Car class
        concrete_classes = _extract_elements_by_type(result, "Class")
        assert len(concrete_classes) == 1
        assert concrete_classes[0]["name"] == "Car"

        # Verify inheritance relationship
        inherit_rels = _extract_relationships_by_type(result, "ClassInheritance")
        assert len(inherit_rels) == 1

        # Source (specific) should be Car, target (general) should be Vehicle
        rel = inherit_rels[0]
        result_elements = result["elements"]
        source_elem = result_elements[rel["source"]["element"]]
        target_elem = result_elements[rel["target"]["element"]]
        assert source_elem["name"] == "Car"
        assert target_elem["name"] == "Vehicle"

    def test_enumeration_preserved(self, class_diagram_with_enum_json):
        """Enumerations and their literals survive the roundtrip."""
        domain_model = process_class_diagram(class_diagram_with_enum_json)
        result = class_buml_to_json(domain_model)

        # Verify enumeration exists
        enum_names = _get_enum_names(result)
        assert "OrderStatus" in enum_names

        # Verify literals
        result_elements = result["elements"]
        enum_elem = None
        for e in result_elements.values():
            if e.get("type") == "Enumeration" and e.get("name") == "OrderStatus":
                enum_elem = e
                break

        assert enum_elem is not None
        literal_names = set()
        for lit_id in enum_elem.get("attributes", []):
            lit = result_elements.get(lit_id, {})
            literal_names.add(lit.get("name"))

        assert literal_names == {"PENDING", "SHIPPED", "DELIVERED"}

    def test_enum_attribute_type_preserved(self, class_diagram_with_enum_json):
        """An attribute typed with an enumeration survives the roundtrip."""
        domain_model = process_class_diagram(class_diagram_with_enum_json)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        # Find the Order class's 'status' attribute
        for e in result_elements.values():
            if e.get("type") == "ClassAttribute" and e.get("name") == "status":
                assert e["attributeType"] == "OrderStatus"
                break
        else:
            pytest.fail("Could not find 'status' attribute in result JSON")

    def test_composition_preserved(self, class_diagram_with_composition_json):
        """Composition relationship survives the roundtrip."""
        domain_model = process_class_diagram(class_diagram_with_composition_json)
        result = class_buml_to_json(domain_model)

        comp_rels = _extract_relationships_by_type(result, "ClassComposition")
        assert len(comp_rels) == 1

        rel = comp_rels[0]
        roles = {rel["source"]["role"], rel["target"]["role"]}
        assert roles == {"company", "department"}

    def test_unidirectional_association(self):
        """Unidirectional association survives the roundtrip."""
        json_data = {
            "title": "UniModel",
            "model": {
                "elements": {
                    "cls-a": {
                        "id": "cls-a",
                        "name": "ClassA",
                        "type": "Class",
                        "owner": None,
                        "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                        "attributes": [],
                        "methods": [],
                    },
                    "cls-b": {
                        "id": "cls-b",
                        "name": "ClassB",
                        "type": "Class",
                        "owner": None,
                        "bounds": {"x": 300, "y": 0, "width": 160, "height": 100},
                        "attributes": [],
                        "methods": [],
                    },
                },
                "relationships": {
                    "rel-uni": {
                        "id": "rel-uni",
                        "name": "uses",
                        "type": "ClassUnidirectional",
                        "source": {
                            "element": "cls-a",
                            "multiplicity": "1",
                            "role": "classA",
                        },
                        "target": {
                            "element": "cls-b",
                            "multiplicity": "0..1",
                            "role": "classB",
                        },
                    },
                },
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)

        uni_rels = _extract_relationships_by_type(result, "ClassUnidirectional")
        assert len(uni_rels) == 1
        assert uni_rels[0]["name"] == "uses"

    def test_diagram_type_preserved(self, minimal_class_diagram_json):
        """The output JSON has the correct diagram type."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        result = class_buml_to_json(domain_model)
        assert result["type"] == "ClassDiagram"

    def test_double_roundtrip_stable(self, minimal_class_diagram_json):
        """Two consecutive roundtrips produce equivalent essential data."""
        # First roundtrip
        dm1 = process_class_diagram(minimal_class_diagram_json)
        json1 = class_buml_to_json(dm1)

        # Second roundtrip: wrap in expected format
        json1_input = {"title": "LibraryModel", "model": json1}
        dm2 = process_class_diagram(json1_input)
        json2 = class_buml_to_json(dm2)

        # Compare essential data
        assert _get_class_names(json1) == _get_class_names(json2)
        assert _get_enum_names(json1) == _get_enum_names(json2)
        for cls_name in _get_class_names(json1):
            assert _get_attribute_names_for_class(json1, cls_name) == \
                   _get_attribute_names_for_class(json2, cls_name)
            assert _get_method_names_for_class(json1, cls_name) == \
                   _get_method_names_for_class(json2, cls_name)

    def test_method_with_code_roundtrip(self):
        """Method code attribute survives the roundtrip."""
        json_data = {
            "title": "CodeModel",
            "model": {
                "elements": {
                    "cls-calc": {
                        "id": "cls-calc",
                        "name": "Calculator",
                        "type": "Class",
                        "owner": None,
                        "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                        "attributes": [],
                        "methods": ["meth-add"],
                    },
                    "meth-add": {
                        "id": "meth-add",
                        "name": "+ add(a: int, b: int): int",
                        "type": "ClassMethod",
                        "owner": "cls-calc",
                        "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                        "code": "def add(self, a, b):\n    return a + b",
                        "implementationType": "code",
                    },
                },
                "relationships": {},
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)

        method_elems = _extract_elements_by_type(result, "ClassMethod")
        assert len(method_elems) == 1
        assert method_elems[0].get("code") == "def add(self, a, b):\n    return a + b"
        assert method_elems[0].get("implementationType") == "code"

    def test_attribute_default_value_preserved(self):
        """Attribute default values survive the roundtrip."""
        json_data = {
            "title": "DefaultsModel",
            "model": {
                "elements": {
                    "cls-config": {
                        "id": "cls-config",
                        "name": "Config",
                        "type": "Class",
                        "owner": None,
                        "bounds": {"x": 0, "y": 0, "width": 160, "height": 100},
                        "attributes": ["attr-retries"],
                        "methods": [],
                    },
                    "attr-retries": {
                        "id": "attr-retries",
                        "name": "retries",
                        "type": "ClassAttribute",
                        "owner": "cls-config",
                        "visibility": "public",
                        "attributeType": "int",
                        "isOptional": False,
                        "defaultValue": "3",
                        "bounds": {"x": 0, "y": 40, "width": 159, "height": 30},
                    },
                },
                "relationships": {},
            },
        }
        domain_model = process_class_diagram(json_data)
        result = class_buml_to_json(domain_model)

        result_elements = result["elements"]
        for e in result_elements.values():
            if e.get("type") == "ClassAttribute" and e.get("name") == "retries":
                assert e.get("defaultValue") == "3"
                break
        else:
            pytest.fail("Could not find 'retries' attribute in result JSON")


# ===========================================================================
# State Machine Roundtrip Tests
# ===========================================================================

class TestStateMachineRoundtrip:
    """JSON -> BUML -> JSON roundtrip for state machine diagrams."""

    def test_states_preserved(self, state_machine_json):
        """State names survive the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        state_elems = _extract_elements_by_type(result, "State")
        state_names = {s["name"] for s in state_elems}
        assert state_names == {"idle", "processing"}

    def test_initial_state_preserved(self, state_machine_json):
        """The initial state marker survives the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        # There should be a StateInitialNode
        initial_nodes = _extract_elements_by_type(result, "StateInitialNode")
        assert len(initial_nodes) == 1

        # There should be a transition from the initial node to a state
        transitions = _extract_relationships_by_type(result, "StateTransition")
        initial_transitions = [
            t for t in transitions
            if result["elements"].get(t["source"]["element"], {}).get("type") == "StateInitialNode"
        ]
        assert len(initial_transitions) == 1

        # The target of the initial transition should be "idle"
        target_id = initial_transitions[0]["target"]["element"]
        target_elem = result["elements"][target_id]
        assert target_elem["name"] == "idle"

    def test_transitions_preserved(self, state_machine_json):
        """Non-initial transitions survive the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        transitions = _extract_relationships_by_type(result, "StateTransition")
        # Filter out initial node transitions
        state_transitions = [
            t for t in transitions
            if result["elements"].get(t["source"]["element"], {}).get("type") != "StateInitialNode"
        ]
        assert len(state_transitions) == 1

        t = state_transitions[0]
        assert t["name"] == "start_event"

        # Source should be "idle", target should be "processing"
        source_elem = result["elements"][t["source"]["element"]]
        target_elem = result["elements"][t["target"]["element"]]
        assert source_elem["name"] == "idle"
        assert target_elem["name"] == "processing"

    def test_bodies_preserved(self, state_machine_json):
        """State body associations survive the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        # Check that states have their body references
        state_elems = _extract_elements_by_type(result, "State")
        for state in state_elems:
            if state["name"] == "idle":
                body_ids = state.get("bodies", [])
                assert len(body_ids) == 1
                body_elem = result["elements"][body_ids[0]]
                assert body_elem["name"] == "idle_body"
                assert body_elem["type"] == "StateBody"
            elif state["name"] == "processing":
                body_ids = state.get("bodies", [])
                assert len(body_ids) == 1
                body_elem = result["elements"][body_ids[0]]
                assert body_elem["name"] == "process_body"
                assert body_elem["type"] == "StateBody"

    def test_code_blocks_preserved(self, state_machine_json):
        """Code block elements (body and event functions) survive the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        code_blocks = _extract_elements_by_type(result, "StateCodeBlock")
        code_names = {cb["name"] for cb in code_blocks}

        # Body functions
        assert "idle_body" in code_names
        assert "process_body" in code_names
        # Event function
        assert "start_event" in code_names

    def test_code_block_content_preserved(self, state_machine_json):
        """Code block source code survives the roundtrip."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)

        code_blocks = _extract_elements_by_type(result, "StateCodeBlock")
        for cb in code_blocks:
            if cb["name"] == "idle_body":
                assert "idle_body" in cb["code"]
                assert "session.reply" in cb["code"]
            elif cb["name"] == "process_body":
                assert "process_body" in cb["code"]
                assert "Processing" in cb["code"]
            elif cb["name"] == "start_event":
                assert "start_event" in cb["code"]
                assert "session.message" in cb["code"]

    def test_diagram_type_preserved(self, state_machine_json):
        """The output JSON has the correct diagram type."""
        sm = process_state_machine(state_machine_json)
        result = state_machine_object_to_json(sm)
        assert result["type"] == "StateMachineDiagram"

    def test_state_machine_with_guard(self):
        """Transition guard conditions survive the roundtrip."""
        json_data = {
            "title": "GuardTest",
            "model": {
                "elements": {
                    "initial": {
                        "id": "initial",
                        "name": "",
                        "type": "StateInitialNode",
                        "owner": None,
                        "bounds": {"x": -850, "y": -280, "width": 45, "height": 45},
                    },
                    "state-a": {
                        "id": "state-a",
                        "name": "stateA",
                        "type": "State",
                        "owner": None,
                        "bounds": {"x": -550, "y": -300, "width": 160, "height": 100},
                        "bodies": [],
                        "fallbackBodies": [],
                    },
                    "state-b": {
                        "id": "state-b",
                        "name": "stateB",
                        "type": "State",
                        "owner": None,
                        "bounds": {"x": -60, "y": -300, "width": 160, "height": 100},
                        "bodies": [],
                        "fallbackBodies": [],
                    },
                },
                "relationships": {
                    "trans-init": {
                        "id": "trans-init",
                        "name": "",
                        "type": "StateTransition",
                        "source": {"element": "initial"},
                        "target": {"element": "state-a"},
                    },
                    "trans-guarded": {
                        "id": "trans-guarded",
                        "name": "trigger_event",
                        "type": "StateTransition",
                        "source": {"element": "state-a"},
                        "target": {"element": "state-b"},
                        "guard": "x > 10",
                    },
                },
            },
        }
        sm = process_state_machine(json_data)
        result = state_machine_object_to_json(sm)

        transitions = _extract_relationships_by_type(result, "StateTransition")
        guarded = [t for t in transitions if t.get("guard")]
        assert len(guarded) == 1
        assert guarded[0]["guard"] == "x > 10"
        assert guarded[0]["name"] == "trigger_event"

    def test_double_roundtrip_stable(self, state_machine_json):
        """Two consecutive roundtrips produce equivalent essential data."""
        # First roundtrip
        sm1 = process_state_machine(state_machine_json)
        json1 = state_machine_object_to_json(sm1)

        # Second roundtrip
        json1_input = {"title": "OrderStateMachine", "model": json1}
        sm2 = process_state_machine(json1_input)
        json2 = state_machine_object_to_json(sm2)

        # Compare essential data
        states1 = {s["name"] for s in _extract_elements_by_type(json1, "State")}
        states2 = {s["name"] for s in _extract_elements_by_type(json2, "State")}
        assert states1 == states2

        code1 = {cb["name"] for cb in _extract_elements_by_type(json1, "StateCodeBlock")}
        code2 = {cb["name"] for cb in _extract_elements_by_type(json2, "StateCodeBlock")}
        assert code1 == code2


# ===========================================================================
# Object Diagram Roundtrip Tests
# ===========================================================================

class TestObjectDiagramRoundtrip:
    """JSON -> BUML roundtrip for object diagrams.

    Since there is no ObjectModel-to-JSON converter that works from the
    metamodel object, these tests verify that the BUML ObjectModel produced
    from JSON contains the correct data rather than doing a full roundtrip.

    Note: Some tests are marked xfail because DataValue.__init__ passes
    name="" to NamedElement which now rejects empty names.  This is a
    pre-existing bug (DataValue default name conflicts with NamedElement
    validation), not specific to the converter code.
    """

    def test_objects_created(self, object_diagram_json, minimal_class_diagram_json):
        """Objects (without attribute slots) are created from the JSON input."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        obj_names = {obj.name for obj in obj_model.objects}
        assert "book1" in obj_names
        assert "author1" in obj_names

    def test_object_classifiers(self, object_diagram_json, minimal_class_diagram_json):
        """Objects have the correct classifier (class) assigned."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        classifiers = {obj.name: obj.classifier.name for obj in obj_model.objects}
        assert classifiers["book1"] == "Book"
        assert classifiers["author1"] == "Author"

    def test_object_count(self, object_diagram_json, minimal_class_diagram_json):
        """The correct number of objects is created."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        assert len(obj_model.objects) == 2

    def test_object_links_created(self, object_diagram_json, minimal_class_diagram_json):
        """Links between objects are correctly created."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_json, domain_model)

        # Collect all links from all objects
        all_links = set()
        for obj in obj_model.objects:
            for link in obj.links:
                all_links.add(link)

        assert len(all_links) >= 1

        # Verify the link connects to the correct association
        link = next(iter(all_links))
        assert link.association is not None
        end_types = {end.type.name for end in link.association.ends}
        assert "Book" in end_types
        assert "Author" in end_types

    def test_object_attribute_values(self, object_diagram_with_attrs_json, minimal_class_diagram_json):
        """Object attribute (slot) values are correctly parsed."""
        domain_model = process_class_diagram(minimal_class_diagram_json)
        obj_model = process_object_diagram(object_diagram_with_attrs_json, domain_model)

        for obj in obj_model.objects:
            if obj.name == "book1":
                slot_values = {
                    slot.attribute.name: slot.value.value
                    for slot in obj.slots
                }
                assert slot_values.get("title") == "TheGreatGatsby"
            elif obj.name == "author1":
                slot_values = {
                    slot.attribute.name: slot.value.value
                    for slot in obj.slots
                }
                assert slot_values.get("name") == "Fitzgerald"


# ===========================================================================
# NN Diagram Roundtrip: JSON -> NN -> JSON
# ===========================================================================

class TestNNDiagramRoundtrip:
    """Roundtrip tests for NNDiagram: JSON -> NN -> JSON preserves essential data."""

    @staticmethod
    def _minimal_nn_json():
        """A minimal NN diagram: container + Conv2D + Linear + configuration + datasets."""
        return {
            "title": "TestNet",
            "model": {
                "type": "NNDiagram",
                "version": "3.0.0",
                "size": {"width": 1400, "height": 740},
                "elements": {
                    "c1": {"id": "c1", "type": "NNContainer", "name": "TestNet", "owner": None, "bounds": {"x": 0, "y": 0, "width": 600, "height": 200}},
                    "l1": {"id": "l1", "type": "Conv2DLayer", "name": "Conv2DLayer", "owner": "c1", "bounds": {"x": 20, "y": 60, "width": 110, "height": 110}, "attributes": ["l1a1", "l1a2", "l1a3"], "methods": []},
                    "l1a1": {"id": "l1a1", "type": "NameAttributeConv2D", "attributeName": "name", "value": "c1", "owner": "l1"},
                    "l1a2": {"id": "l1a2", "type": "KernelDimAttributeConv2D", "attributeName": "kernel_dim", "value": "[3, 3]", "owner": "l1"},
                    "l1a3": {"id": "l1a3", "type": "OutChannelsAttributeConv2D", "attributeName": "out_channels", "value": "16", "owner": "l1"},
                    "l2": {"id": "l2", "type": "LinearLayer", "name": "LinearLayer", "owner": "c1", "bounds": {"x": 150, "y": 60, "width": 110, "height": 110}, "attributes": ["l2a1", "l2a2"], "methods": []},
                    "l2a1": {"id": "l2a1", "type": "NameAttributeLinear", "attributeName": "name", "value": "lin", "owner": "l2"},
                    "l2a2": {"id": "l2a2", "type": "OutFeaturesAttributeLinear", "attributeName": "out_features", "value": "10", "owner": "l2"},
                    "cfg": {"id": "cfg", "type": "Configuration", "name": "Configuration", "owner": None, "bounds": {"x": 700, "y": 0, "width": 160, "height": 200}, "attributes": ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5", "cfg6"], "methods": []},
                    "cfg1": {"id": "cfg1", "type": "BatchSizeAttributeConfiguration", "attributeName": "batch_size", "value": "32", "owner": "cfg"},
                    "cfg2": {"id": "cfg2", "type": "EpochsAttributeConfiguration", "attributeName": "epochs", "value": "10", "owner": "cfg"},
                    "cfg3": {"id": "cfg3", "type": "LearningRateAttributeConfiguration", "attributeName": "learning_rate", "value": "0.001", "owner": "cfg"},
                    "cfg4": {"id": "cfg4", "type": "OptimizerAttributeConfiguration", "attributeName": "optimizer", "value": "adam", "owner": "cfg"},
                    "cfg5": {"id": "cfg5", "type": "LossFunctionAttributeConfiguration", "attributeName": "loss_function", "value": "crossentropy", "owner": "cfg"},
                    "cfg6": {"id": "cfg6", "type": "MetricsAttributeConfiguration", "attributeName": "metrics", "value": "[accuracy]", "owner": "cfg"},
                    "ds1": {"id": "ds1", "type": "TrainingDataset", "name": "TrainingDataset", "owner": None, "bounds": {"x": 0, "y": 400, "width": 110, "height": 110}, "attributes": ["ds1a1", "ds1a2", "ds1a3", "ds1a4", "ds1a5", "ds1a6"], "methods": []},
                    "ds1a1": {"id": "ds1a1", "attributeName": "name", "value": "train", "owner": "ds1"},
                    "ds1a2": {"id": "ds1a2", "attributeName": "path_data", "value": "/data/train", "owner": "ds1"},
                    "ds1a3": {"id": "ds1a3", "attributeName": "task_type", "value": "multi_class", "owner": "ds1"},
                    "ds1a4": {"id": "ds1a4", "attributeName": "input_format", "value": "images", "owner": "ds1"},
                    "ds1a5": {"id": "ds1a5", "attributeName": "shape", "value": "[32, 32, 3]", "owner": "ds1"},
                    "ds1a6": {"id": "ds1a6", "attributeName": "normalize", "value": "false", "owner": "ds1"},
                    "ds2": {"id": "ds2", "type": "TestDataset", "name": "TestDataset", "owner": None, "bounds": {"x": 300, "y": 400, "width": 110, "height": 110}, "attributes": ["ds2a1", "ds2a2"], "methods": []},
                    "ds2a1": {"id": "ds2a1", "attributeName": "name", "value": "test", "owner": "ds2"},
                    "ds2a2": {"id": "ds2a2", "attributeName": "path_data", "value": "/data/test", "owner": "ds2"},
                },
                "relationships": {
                    "r1": {"id": "r1", "type": "NNNext", "source": {"element": "l1"}, "target": {"element": "l2"}, "name": "next"},
                },
                "interactive": {"elements": {}, "relationships": {}},
                "assessments": {},
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
        containers = _extract_elements_by_type(out, "NNContainer")
        assert len(containers) == 1
        assert containers[0]["name"] == "TestNet"

    def test_layers_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        assert len(_extract_elements_by_type(out, "Conv2DLayer")) == 1
        assert len(_extract_elements_by_type(out, "LinearLayer")) == 1

    def test_nnnext_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        nexts = _extract_relationships_by_type(out, "NNNext")
        assert len(nexts) == 1

    def test_configuration_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        configs = _extract_elements_by_type(out, "Configuration")
        assert len(configs) == 1
        elements = _resolve_elements(out)
        attr_values = {elements[aid]["attributeName"]: elements[aid]["value"] for aid in configs[0]["attributes"]}
        assert attr_values["batch_size"] == "32"
        assert attr_values["optimizer"] == "adam"

    def test_training_dataset_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        training = _extract_elements_by_type(out, "TrainingDataset")
        assert len(training) == 1
        elements = _resolve_elements(out)
        attrs = {elements[aid]["attributeName"]: elements[aid]["value"] for aid in training[0]["attributes"]}
        assert attrs["name"] == "train"
        assert attrs["path_data"] == "/data/train"
        assert attrs["task_type"] == "multi_class"
        assert attrs["input_format"] == "images"
        assert attrs["shape"] == "[32, 32, 3]"
        assert attrs["normalize"] == "false"

    def test_test_dataset_preserved(self):
        out = self._roundtrip(self._minimal_nn_json())
        testing = _extract_elements_by_type(out, "TestDataset")
        assert len(testing) == 1
        elements = _resolve_elements(out)
        attrs = {elements[aid]["attributeName"]: elements[aid]["value"] for aid in testing[0]["attributes"]}
        assert attrs["name"] == "test"
        assert attrs["path_data"] == "/data/test"
        assert "input_format" not in attrs

    def test_double_roundtrip_stable(self):
        """Second roundtrip yields the same structural shape."""
        from collections import Counter
        first = self._roundtrip(self._minimal_nn_json())
        second = self._roundtrip(first)
        def counts(j):
            return Counter(e.get("type") for e in _resolve_elements(j).values() if "Attribute" not in (e.get("type") or ""))
        assert counts(first) == counts(second)

    def test_nn_model_to_json_is_deterministic(self):
        """Identical NN input must produce byte-identical JSON across calls.

        Locks in the uuid5 + thread-local counter scheme in nn_diagram_converter.
        Regression guard: if anyone reintroduces uuid.uuid4() for IDs, the two
        serializations will diverge and this test fails.
        """
        import json
        nn = process_nn_diagram(self._minimal_nn_json())
        first = json.dumps(nn_model_to_json(nn), sort_keys=True)
        second = json.dumps(nn_model_to_json(nn), sort_keys=True)
        assert first == second, "nn_model_to_json output must be deterministic"

    def test_nnreference_cycle_is_rejected(self):
        """Two NNContainers that reference each other must raise a clear error."""
        cyclic = {
            "title": "CyclicRefs",
            "model": {
                "type": "NNDiagram",
                "version": "3.0.0",
                "size": {"width": 1200, "height": 600},
                "elements": {
                    "cA": {"id": "cA", "type": "NNContainer", "name": "A", "owner": None, "bounds": {"x": 0, "y": 0, "width": 400, "height": 200}},
                    "cB": {"id": "cB", "type": "NNContainer", "name": "B", "owner": None, "bounds": {"x": 500, "y": 0, "width": 400, "height": 200}},
                    "rA": {"id": "rA", "type": "NNReference", "name": "ref-to-B", "owner": "cA", "referencedNN": "B", "bounds": {"x": 20, "y": 60, "width": 140, "height": 40}},
                    "rB": {"id": "rB", "type": "NNReference", "name": "ref-to-A", "owner": "cB", "referencedNN": "A", "bounds": {"x": 520, "y": 60, "width": 140, "height": 40}},
                },
                "relationships": {},
                "interactive": {"elements": {}, "relationships": {}},
                "assessments": {},
            },
        }
        with pytest.raises(ValueError, match="NNReference cycle"):
            process_nn_diagram(cyclic)

    def test_padding_type_valid_roundtrips_when_explicit(self):
        """When the user explicitly picks padding_type='valid' (the metamodel
        default), the sidecar ``mark_explicit`` path must make sure the JSON
        output still carries the attribute. Without sidecar gating, the legacy
        ``!= 'valid'`` guard silently strips it."""
        payload = self._minimal_nn_json()
        conv_id = 'l1'
        payload['model']['elements'][conv_id]['attributes'].append('l1pt')
        payload['model']['elements']['l1pt'] = {
            'id': 'l1pt', 'type': 'PaddingTypeAttributeConv2D',
            'attributeName': 'padding_type', 'value': 'valid', 'owner': conv_id,
        }
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        conv_elts = _extract_elements_by_type(out, 'Conv2DLayer')
        assert conv_elts, 'expected at least one Conv2DLayer in output'
        attr_names = {
            elements[aid]['attributeName']: elements[aid]['value']
            for aid in conv_elts[0]['attributes']
        }
        assert attr_names.get('padding_type') == 'valid', \
            f'padding_type=valid must round-trip when set explicitly, got {attr_names}'

    def test_flatten_start_dim_and_end_dim_defaults_roundtrip_when_explicit(self):
        """Flatten layer with start_dim=1 and end_dim=-1 (both metamodel
        defaults) must round-trip when set explicitly in the source JSON."""
        base = self._minimal_nn_json()
        base['model']['elements']['fl1'] = {
            'id': 'fl1', 'type': 'FlattenLayer', 'name': 'FlattenLayer',
            'owner': 'c1',
            'bounds': {'x': 300, 'y': 60, 'width': 110, 'height': 110},
            'attributes': ['fl1n', 'fl1s', 'fl1e'], 'methods': [],
        }
        base['model']['elements']['fl1n'] = {
            'id': 'fl1n', 'type': 'NameAttributeFlatten',
            'attributeName': 'name', 'value': 'flat', 'owner': 'fl1',
        }
        base['model']['elements']['fl1s'] = {
            'id': 'fl1s', 'type': 'StartDimAttributeFlatten',
            'attributeName': 'start_dim', 'value': '1', 'owner': 'fl1',
        }
        base['model']['elements']['fl1e'] = {
            'id': 'fl1e', 'type': 'EndDimAttributeFlatten',
            'attributeName': 'end_dim', 'value': '-1', 'owner': 'fl1',
        }
        out = self._roundtrip(base)
        elements = _resolve_elements(out)
        flat = _extract_elements_by_type(out, 'FlattenLayer')
        assert flat, 'FlattenLayer missing from output'
        attrs = {
            elements[aid]['attributeName']: elements[aid]['value']
            for aid in flat[0]['attributes']
        }
        assert 'start_dim' in attrs, f'start_dim should round-trip when explicit; got {attrs}'
        assert 'end_dim' in attrs, f'end_dim should round-trip when explicit; got {attrs}'

    def test_tensorop_concatenate_preserves_float_in_layers_of_tensors(self):
        """TensorOp ``layers_of_tensors=[0.5, 'name']`` must preserve the 0.5
        as a float (not the string '0.5') after round-trip, so downstream
        generators can distinguish numeric weights from layer-name references."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            process_nn_diagram,
        )
        payload = {
            'title': 'op',
            'model': {
                'type': 'NNDiagram', 'version': '3.0.0',
                'size': {'width': 1000, 'height': 400},
                'elements': {
                    'c': {'id': 'c', 'type': 'NNContainer', 'name': 'op',
                          'owner': None,
                          'bounds': {'x': 0, 'y': 0, 'width': 400, 'height': 200}},
                    't': {'id': 't', 'type': 'TensorOp', 'name': 'TensorOp',
                          'owner': 'c',
                          'bounds': {'x': 20, 'y': 60, 'width': 110, 'height': 110},
                          'attributes': ['tn', 'tt', 'tc', 'tl'], 'methods': []},
                    'tn': {'id': 'tn', 'type': 'NameAttributeTensorOp',
                           'attributeName': 'name', 'value': 'op1', 'owner': 't'},
                    'tt': {'id': 'tt', 'type': 'TnsTypeAttributeTensorOp',
                           'attributeName': 'tns_type', 'value': 'concatenate',
                           'owner': 't'},
                    'tc': {'id': 'tc', 'type': 'ConcatenateDimAttributeTensorOp',
                           'attributeName': 'concatenate_dim', 'value': '1', 'owner': 't'},
                    'tl': {'id': 'tl',
                           'type': 'LayersOfTensorsAttributeTensorOp',
                           'attributeName': 'layers_of_tensors',
                           'value': "[0.5, 'other']", 'owner': 't'},
                },
                'relationships': {},
                'interactive': {'elements': {}, 'relationships': {}},
                'assessments': {},
            },
        }
        nn = process_nn_diagram(payload)
        top = [m for m in nn.modules if type(m).__name__ == 'TensorOp'][0]
        assert top.layers_of_tensors[0] == 0.5, \
            f"first item should be float 0.5, got {top.layers_of_tensors[0]!r}"
        assert isinstance(top.layers_of_tensors[0], float)
        assert top.layers_of_tensors[1] == 'other'

    def test_transitive_nnreference_cycle_detected(self):
        """Three-container NNReference cycle A→B→C→A must raise with the
        cycle chain in the error message."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            process_nn_diagram,
        )
        cyclic = {
            'title': 'transitive',
            'model': {
                'type': 'NNDiagram', 'version': '3.0.0',
                'size': {'width': 1800, 'height': 600},
                'elements': {
                    'A': {'id': 'A', 'type': 'NNContainer', 'name': 'A', 'owner': None,
                          'bounds': {'x': 0, 'y': 0, 'width': 400, 'height': 200}},
                    'B': {'id': 'B', 'type': 'NNContainer', 'name': 'B', 'owner': None,
                          'bounds': {'x': 500, 'y': 0, 'width': 400, 'height': 200}},
                    'C': {'id': 'C', 'type': 'NNContainer', 'name': 'C', 'owner': None,
                          'bounds': {'x': 1000, 'y': 0, 'width': 400, 'height': 200}},
                    'rA': {'id': 'rA', 'type': 'NNReference', 'name': 'A->B', 'owner': 'A',
                           'referencedNN': 'B',
                           'bounds': {'x': 20, 'y': 60, 'width': 140, 'height': 40}},
                    'rB': {'id': 'rB', 'type': 'NNReference', 'name': 'B->C', 'owner': 'B',
                           'referencedNN': 'C',
                           'bounds': {'x': 520, 'y': 60, 'width': 140, 'height': 40}},
                    'rC': {'id': 'rC', 'type': 'NNReference', 'name': 'C->A', 'owner': 'C',
                           'referencedNN': 'A',
                           'bounds': {'x': 1020, 'y': 60, 'width': 140, 'height': 40}},
                },
                'relationships': {},
                'interactive': {'elements': {}, 'relationships': {}},
                'assessments': {},
            },
        }
        with pytest.raises(ValueError, match="NNReference cycle"):
            process_nn_diagram(cyclic)

    def test_conv3d_kernel_dim_wrong_length_raises_with_layer_name(self):
        """Conv3D with kernel_dim=[3,3] must raise a user-facing error that
        names the layer and expected dimensionality."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            process_nn_diagram,
        )
        payload = {
            'title': 'conv',
            'model': {
                'type': 'NNDiagram', 'version': '3.0.0',
                'size': {'width': 800, 'height': 400},
                'elements': {
                    'c': {'id': 'c', 'type': 'NNContainer', 'name': 'net',
                          'owner': None,
                          'bounds': {'x': 0, 'y': 0, 'width': 400, 'height': 200}},
                    'l': {'id': 'l', 'type': 'Conv3DLayer', 'name': 'Conv3DLayer',
                          'owner': 'c',
                          'bounds': {'x': 20, 'y': 60, 'width': 110, 'height': 110},
                          'attributes': ['ln', 'lk', 'lo'], 'methods': []},
                    'ln': {'id': 'ln', 'type': 'NameAttributeConv3D',
                           'attributeName': 'name', 'value': 'volume', 'owner': 'l'},
                    'lk': {'id': 'lk', 'type': 'KernelDimAttributeConv3D',
                           'attributeName': 'kernel_dim',
                           'value': '[3, 3]', 'owner': 'l'},
                    'lo': {'id': 'lo', 'type': 'OutChannelsAttributeConv3D',
                           'attributeName': 'out_channels', 'value': '8', 'owner': 'l'},
                },
                'relationships': {},
                'interactive': {'elements': {}, 'relationships': {}},
                'assessments': {},
            },
        }
        with pytest.raises(ValueError, match=r"Conv3D.*volume.*kernel_dim.*expected 3"):
            process_nn_diagram(payload)

    @staticmethod
    def _single_layer_json(layer_type: str, attrs: list, rels: dict = None):
        """Build a minimal NN diagram with a single layer of the given type.

        ``attrs`` is a list of (suffix, attributeName, value) tuples. Each
        becomes an attribute element owned by the layer.
        """
        layer_id = 'L'
        attr_elements = {}
        attr_ids = []
        for idx, (suffix, aname, avalue) in enumerate(attrs):
            aid = f'a{idx}'
            attr_ids.append(aid)
            attr_elements[aid] = {
                'id': aid,
                'type': f'{aname.title().replace("_", "")}Attribute{suffix}'.replace(' ', ''),
                'attributeName': aname,
                'value': avalue,
                'owner': layer_id,
            }
        return {
            'title': 'single',
            'model': {
                'type': 'NNDiagram', 'version': '3.0.0',
                'size': {'width': 800, 'height': 400},
                'elements': {
                    'c': {'id': 'c', 'type': 'NNContainer', 'name': 'net', 'owner': None,
                          'bounds': {'x': 0, 'y': 0, 'width': 400, 'height': 200}},
                    layer_id: {'id': layer_id, 'type': layer_type, 'name': layer_type, 'owner': 'c',
                               'bounds': {'x': 20, 'y': 60, 'width': 110, 'height': 110},
                               'attributes': attr_ids, 'methods': []},
                    **attr_elements,
                },
                'relationships': rels or {},
                'interactive': {'elements': {}, 'relationships': {}},
                'assessments': {},
            },
        }

    def test_lstm_roundtrip(self):
        payload = self._single_layer_json('LSTMLayer', [
            ('LSTM', 'name', 'lstm1'),
            ('LSTM', 'hidden_size', '128'),
            ('LSTM', 'input_size', '64'),
            ('LSTM', 'return_type', 'last'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        lstm = _extract_elements_by_type(out, 'LSTMLayer')
        assert lstm, 'LSTMLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in lstm[0]['attributes']}
        assert attrs.get('name') == 'lstm1'
        assert attrs.get('hidden_size') == '128'
        assert attrs.get('input_size') == '64'
        assert attrs.get('return_type') == 'last'

    def test_gru_roundtrip(self):
        payload = self._single_layer_json('GRULayer', [
            ('GRU', 'name', 'gru1'),
            ('GRU', 'hidden_size', '32'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        gru = _extract_elements_by_type(out, 'GRULayer')
        assert gru, 'GRULayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in gru[0]['attributes']}
        assert attrs.get('name') == 'gru1'
        assert attrs.get('hidden_size') == '32'

    def test_embedding_roundtrip(self):
        payload = self._single_layer_json('EmbeddingLayer', [
            ('Embedding', 'name', 'emb'),
            ('Embedding', 'num_embeddings', '1000'),
            ('Embedding', 'embedding_dim', '64'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        emb = _extract_elements_by_type(out, 'EmbeddingLayer')
        assert emb, 'EmbeddingLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in emb[0]['attributes']}
        assert attrs.get('num_embeddings') == '1000'
        assert attrs.get('embedding_dim') == '64'

    def test_batchnorm_roundtrip(self):
        payload = self._single_layer_json('BatchNormalizationLayer', [
            ('BatchNormalization', 'name', 'bn'),
            ('BatchNormalization', 'num_features', '64'),
            ('BatchNormalization', 'dimension', '2D'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        bn = _extract_elements_by_type(out, 'BatchNormalizationLayer')
        assert bn, 'BatchNormalizationLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in bn[0]['attributes']}
        assert attrs.get('num_features') == '64'
        assert attrs.get('dimension') == '2D'

    def test_layernorm_roundtrip(self):
        payload = self._single_layer_json('LayerNormalizationLayer', [
            ('LayerNormalization', 'name', 'ln'),
            ('LayerNormalization', 'normalized_shape', '[32, 32]'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        ln = _extract_elements_by_type(out, 'LayerNormalizationLayer')
        assert ln, 'LayerNormalizationLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ln[0]['attributes']}
        assert attrs.get('normalized_shape') == '[32, 32]'

    def test_dropout_roundtrip(self):
        payload = self._single_layer_json('DropoutLayer', [
            ('Dropout', 'name', 'drop'),
            ('Dropout', 'rate', '0.5'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        drop = _extract_elements_by_type(out, 'DropoutLayer')
        assert drop, 'DropoutLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in drop[0]['attributes']}
        assert attrs.get('rate') == '0.5'

    def test_conv1d_roundtrip(self):
        payload = self._single_layer_json('Conv1DLayer', [
            ('Conv1D', 'name', 'c1'),
            ('Conv1D', 'kernel_dim', '[3]'),
            ('Conv1D', 'out_channels', '16'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        c1 = _extract_elements_by_type(out, 'Conv1DLayer')
        assert c1, 'Conv1DLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in c1[0]['attributes']}
        assert attrs.get('kernel_dim') == '[3]'
        assert attrs.get('out_channels') == '16'

    def test_pooling_roundtrip(self):
        payload = self._single_layer_json('PoolingLayer', [
            ('Pooling', 'name', 'pool'),
            ('Pooling', 'pooling_type', 'max'),
            ('Pooling', 'dimension', '2D'),
            ('Pooling', 'kernel_dim', '[2, 2]'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        pool = _extract_elements_by_type(out, 'PoolingLayer')
        assert pool, 'PoolingLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in pool[0]['attributes']}
        assert attrs.get('pooling_type') == 'max'
        assert attrs.get('dimension') == '2D'
        assert attrs.get('kernel_dim') == '[2, 2]'

    def test_simple_rnn_roundtrip(self):
        payload = self._single_layer_json('RNNLayer', [
            ('RNN', 'name', 'rnn1'),
            ('RNN', 'hidden_size', '64'),
        ])
        out = self._roundtrip(payload)
        elements = _resolve_elements(out)
        rnn = _extract_elements_by_type(out, 'RNNLayer')
        assert rnn, 'RNNLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in rnn[0]['attributes']}
        assert attrs.get('hidden_size') == '64'

    @staticmethod
    def _tensor_op_json(tns_type: str, extra_attrs: list):
        """Build a minimal NN diagram containing a single TensorOp."""
        attrs = [('TensorOp', 'name', f'{tns_type}_op'),
                 ('TensorOp', 'tns_type', tns_type)] + extra_attrs
        return TestNNDiagramRoundtrip._single_layer_json('TensorOp', attrs)

    def test_tensorop_concatenate_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('concatenate', [
            ('TensorOp', 'concatenate_dim', '1'),
            ('TensorOp', 'layers_of_tensors', "['layer_a', 'layer_b']"),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        assert ops, 'TensorOp missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'concatenate'
        assert attrs.get('concatenate_dim') == '1'

    def test_tensorop_multiply_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('multiply', [
            ('TensorOp', 'layers_of_tensors', "['a', 'b']"),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'multiply'

    def test_tensorop_matmultiply_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('matmultiply', [
            ('TensorOp', 'layers_of_tensors', "['a', 'b']"),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'matmultiply'

    def test_tensorop_reshape_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('reshape', [
            ('TensorOp', 'reshape_dim', '[1, -1]'),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'reshape'
        assert attrs.get('reshape_dim') == '[1, -1]'

    def test_tensorop_transpose_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('transpose', [
            ('TensorOp', 'transpose_dim', '[0, 2, 1]'),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'transpose'
        assert attrs.get('transpose_dim') == '[0, 2, 1]'

    def test_tensorop_permute_roundtrip(self):
        out = self._roundtrip(self._tensor_op_json('permute', [
            ('TensorOp', 'permute_dim', '[0, 3, 1, 2]'),
        ]))
        elements = _resolve_elements(out)
        ops = _extract_elements_by_type(out, 'TensorOp')
        attrs = {elements[aid]['attributeName']: elements[aid]['value'] for aid in ops[0]['attributes']}
        assert attrs.get('tns_type') == 'permute'
        assert attrs.get('permute_dim') == '[0, 3, 1, 2]'

    def test_pooling_invalid_pooling_type_lists_allowed_values(self):
        """Unknown pooling_type raises with the whitelist in the message."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            process_nn_diagram,
        )
        payload = self._single_layer_json('PoolingLayer', [
            ('Pooling', 'name', 'p'),
            ('Pooling', 'pooling_type', 'avg'),  # legacy / typo
            ('Pooling', 'dimension', '2D'),
        ])
        with pytest.raises(ValueError, match=r"Allowed values"):
            process_nn_diagram(payload)

    def test_conv_permute_in_true_roundtrips(self):
        """permute_in=True must round-trip via the sidecar (iter-7/8). A
        regression dropping the ``bool`` end-to-end path would turn
        permute_in back into an int or tuple, or drop it entirely."""
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json('Conv2DLayer', [
            ('Conv2D', 'name', 'c1'),
            ('Conv2D', 'kernel_dim', '[3, 3]'),
            ('Conv2D', 'out_channels', '16'),
            ('Conv2D', 'permute_in', 'true'),
        ])
        nn = process_nn_diagram(payload)
        convs = [m for m in nn.modules if type(m).__name__ == 'Conv2D']
        assert convs and convs[0].permute_in is True
        assert is_explicit(convs[0], 'permute_in')

    def test_rnn_bidirectional_dropout_batch_first_roundtrip(self):
        """LSTM with bidirectional/dropout/batch_first set explicitly must
        round-trip all three via the sidecar."""
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json('LSTMLayer', [
            ('LSTM', 'name', 'lstm1'),
            ('LSTM', 'hidden_size', '128'),
            ('LSTM', 'bidirectional', 'true'),
            ('LSTM', 'dropout', '0.3'),
            ('LSTM', 'batch_first', 'true'),
        ])
        nn = process_nn_diagram(payload)
        lstms = [m for m in nn.modules if type(m).__name__ == 'LSTMLayer']
        assert lstms
        lstm = lstms[0]
        assert lstm.bidirectional is True
        assert lstm.dropout == 0.3
        assert lstm.batch_first is True
        assert is_explicit(lstm, 'bidirectional')
        assert is_explicit(lstm, 'dropout')
        assert is_explicit(lstm, 'batch_first')

    def test_tensorop_invalid_tns_type_raises(self):
        """Unknown tns_type raises with the whitelist in the message."""
        payload = self._tensor_op_json('concatinate', [  # typo
            ('TensorOp', 'concatenate_dim', '1'),
            ('TensorOp', 'layers_of_tensors', "['a', 'b']"),
        ])
        with pytest.raises(ValueError, match=r"invalid tns_type.*Allowed values"):
            process_nn_diagram(payload)

    def test_conv_permute_in_false_explicitly_marked(self):
        """permute_in=False must round-trip via the sidecar too — an
        explicit False is semantically different from "unset" and must
        reach the generator."""
        from besser.utilities.buml_code_builder.nn_explicit_attrs import is_explicit
        payload = self._single_layer_json('Conv2DLayer', [
            ('Conv2D', 'name', 'c1'),
            ('Conv2D', 'kernel_dim', '[3, 3]'),
            ('Conv2D', 'out_channels', '16'),
            ('Conv2D', 'permute_in', 'false'),
        ])
        nn = process_nn_diagram(payload)
        convs = [m for m in nn.modules if type(m).__name__ == 'Conv2D']
        assert convs and convs[0].permute_in is False
        assert is_explicit(convs[0], 'permute_in'), \
            "explicit False must be tracked via mark_explicit too"

    def test_sibling_sub_nn_same_name_emits_distinct_containers(self):
        """Two sibling sub-NNs with identical display names must emit
        distinct containers — pins the iter-17 id-keyed sub_nn_ids fix."""
        from besser.BUML.metamodel.nn import NN, LinearLayer
        # Build the NN directly (testing the converter side)
        nn = NN(name='Parent')
        block_a = NN(name='Block')
        block_a.add_layer(LinearLayer(name='a_lin', out_features=4))
        block_b = NN(name='Block')
        block_b.add_layer(LinearLayer(name='b_lin', out_features=8))
        nn.add_sub_nn(block_a)
        nn.add_sub_nn(block_b)
        out = nn_model_to_json(nn)
        containers = _extract_elements_by_type(out, 'NNContainer')
        # Parent + 2 sibling sub-NN containers = 3
        assert len(containers) == 3, (
            f'expected 3 containers (parent + 2 same-named sub-NNs), got {len(containers)}'
        )

    def test_conv_invalid_padding_type_lists_allowed_values(self):
        """Unknown padding_type raises with the whitelist in the message
        (mirrors pooling_type/optimizer/etc. pattern)."""
        payload = self._single_layer_json('Conv2DLayer', [
            ('Conv2D', 'name', 'c1'),
            ('Conv2D', 'kernel_dim', '[3, 3]'),
            ('Conv2D', 'out_channels', '16'),
            ('Conv2D', 'padding_type', 'Same'),  # wrong capitalization
        ])
        with pytest.raises(ValueError, match=r"padding_type.*Allowed values"):
            process_nn_diagram(payload)

    def test_pooling_permute_in_roundtrips_via_converter(self):
        """PoolingLayer permute_in/out must round-trip through the BUML→JSON
        converter — before iter-19 the Pooling branch of _module_fields
        silently dropped them even though processor+builder handled them."""
        from besser.BUML.metamodel.nn import NN, PoolingLayer
        from besser.utilities.buml_code_builder.nn_explicit_attrs import mark_explicit
        nn = NN(name='PoolPermute')
        pool = PoolingLayer(
            name='p', pooling_type='max', dimension='2D', kernel_dim=[2, 2],
        )
        pool.permute_in = True
        mark_explicit(pool, 'permute_in')
        nn.add_layer(pool)
        out = nn_model_to_json(nn)
        elements = _resolve_elements(out)
        pools = _extract_elements_by_type(out, 'PoolingLayer')
        assert pools, 'PoolingLayer missing from output'
        attrs = {elements[aid]['attributeName']: elements[aid]['value']
                 for aid in pools[0]['attributes']}
        assert attrs.get('permute_in') == 'true', \
            f'permute_in must round-trip on Pooling; got attrs={attrs}'

    def test_parse_float_rejects_european_decimal(self):
        """parse_float with '0,5' (European comma) raises rather than
        silently returning the default."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            parse_float,
        )
        with pytest.raises(ValueError, match=r"not a valid number"):
            parse_float("0,5")

    def test_configuration_invalid_optimizer_lists_allowed_values(self):
        """Unknown optimizer raises with the whitelist in the message."""
        from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
            process_nn_diagram,
        )
        payload = self._single_layer_json('Configuration', [
            ('Configuration', 'batch_size', '32'),
            ('Configuration', 'epochs', '10'),
            ('Configuration', 'learning_rate', '0.001'),
            ('Configuration', 'optimizer', 'rmsprop'),  # not in whitelist
            ('Configuration', 'loss_function', 'crossentropy'),
            ('Configuration', 'metrics', '[accuracy]'),
        ])
        with pytest.raises(ValueError, match=r"optimizer.*Allowed values"):
            process_nn_diagram(payload)
