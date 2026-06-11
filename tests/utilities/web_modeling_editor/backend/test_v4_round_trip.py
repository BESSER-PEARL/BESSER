"""Round-trip tests for the v4 wire shape.

Each test loads a hand-authored v4 fixture from ``tests/fixtures/v4/``,
runs it through the JSON->BUML processor, then back through the
BUML->JSON converter, and asserts the structural shape survives.

We deliberately compare a *structural projection* of the model rather
than byte equality: layout positions are recomputed by the converter
and uuid-keyed ids may not match when the metamodel has its own ids.
The properties verified are:

- diagram type matches
- nodes are produced
- edges are produced (when the fixture has any)
- node ``type`` distribution matches the input
- edge ``type`` distribution matches the input
- key ``data.name`` values for top-level nodes survive
"""

from __future__ import annotations

import json
import os
from collections import Counter

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
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.nn_diagram_processor import (
    process_nn_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
    nn_model_to_json,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "fixtures", "v4"
)


def _load(name: str) -> dict:
    path = os.path.normpath(os.path.join(FIXTURES_DIR, name))
    with open(path) as f:
        return json.load(f)


def _v4_data(d: dict) -> dict:
    """Return the inner v4 ``model`` dict from a top-level fixture / output."""
    if "nodes" in d or "edges" in d:
        return d
    return d.get("model") or {}


def _node_types(d: dict) -> Counter:
    inner = _v4_data(d)
    return Counter(n.get("type") for n in inner.get("nodes") or [])


def _edge_types(d: dict) -> Counter:
    inner = _v4_data(d)
    return Counter(e.get("type") for e in inner.get("edges") or [])


def _top_level_names(d: dict) -> set[str]:
    inner = _v4_data(d)
    return {
        (n.get("data") or {}).get("name")
        for n in inner.get("nodes") or []
        if not n.get("parentId")
    }


# ---------------------------------------------------------------------------
# Class diagram round-trip
# ---------------------------------------------------------------------------

def test_v4_class_diagram_round_trip():
    fixture = _load("class_diagram_basic.json")
    domain_model = process_class_diagram(fixture)
    re_emitted = class_buml_to_json(domain_model)
    assert re_emitted.get("type") == "ClassDiagram"
    # ``nodes`` and ``edges`` must be present at the top level.
    assert isinstance(re_emitted.get("nodes"), list)
    assert isinstance(re_emitted.get("edges"), list)
    # Node-type distribution preserved (Book, Author both `class`,
    # Genre `class` with stereotype 'Enumeration' also surfaces as 'class';
    # the OCL sticky note keeps its own node type).
    types_in = _node_types(fixture)
    types_out = _node_types(re_emitted)
    assert types_out["class"] == types_in["class"]
    assert types_out["ClassOCLConstraint"] == types_in["ClassOCLConstraint"]
    # Edge-type distribution preserved (incl. the visual OCL link).
    edges_in = _edge_types(fixture)
    edges_out = _edge_types(re_emitted)
    assert edges_in["ClassBidirectional"] == edges_out["ClassBidirectional"]
    assert edges_in["ClassOCLLink"] == edges_out["ClassOCLLink"]
    # Top-level class names round-trip.
    names = _top_level_names(re_emitted)
    assert "Book" in names
    assert "Author" in names
    assert "Genre" in names


def test_v4_class_diagram_stereotype_capitalized_on_emit():
    """The frontend compares stereotypes case-sensitively against the
    capitalized canonical forms — the converter must emit 'Enumeration'
    (and 'Abstract'), never the lowercase variants."""
    fixture = _load("class_diagram_basic.json")
    re_emitted = class_buml_to_json(process_class_diagram(fixture))
    genre = next(
        n for n in re_emitted["nodes"]
        if (n.get("data") or {}).get("name") == "Genre"
    )
    assert genre["data"]["stereotype"] == "Enumeration"


def test_v4_class_diagram_method_rows_structured():
    """Method rows re-emit in the canonical inspector shape: bare name,
    structured ``parameters`` and ``returnType`` mirrored onto
    ``attributeType``."""
    fixture = _load("class_diagram_basic.json")
    re_emitted = class_buml_to_json(process_class_diagram(fixture))
    book = next(
        n for n in re_emitted["nodes"]
        if (n.get("data") or {}).get("name") == "Book"
    )
    methods = book["data"]["methods"]
    assert len(methods) == 1
    method = methods[0]
    assert method["name"] == "summary"
    assert method["returnType"] == "str"
    assert method["attributeType"] == "str"
    assert [(p["name"], p["parameterType"]) for p in method["parameters"]] == [
        ("verbose", "bool"),
    ]


def test_v4_library_ocl_template_round_trip():
    """The real Library_OCL webapp template (4 classes, 11 sticky-note OCL
    constraints incl. method pre/post) survives JSON -> BUML -> JSON with
    every constraint re-emitted as a visible, linked node."""
    template = _load("library_ocl_template.json")
    domain_model = process_class_diagram(
        {"title": template.get("title") or "Library_OCL", "model": template}
    )
    assert domain_model.ocl_warnings == []
    # 7 class-level invariants in ``constraints``; the other 4 are method
    # pre/post conditions routed onto ``Method.pre`` / ``Method.post``.
    assert len(domain_model.constraints) == 7
    pre_post = sum(
        len(getattr(m, "pre", []) or []) + len(getattr(m, "post", []) or [])
        for t in domain_model.types if isinstance(getattr(t, "methods", None), (set, list))
        for m in t.methods
    )
    assert pre_post == 4

    re_emitted = class_buml_to_json(domain_model)
    types_out = _node_types(re_emitted)
    edges_out = _edge_types(re_emitted)
    assert types_out["class"] == 4
    assert types_out["ClassOCLConstraint"] == 11
    assert edges_out["ClassOCLLink"] == 11
    assert edges_out["ClassBidirectional"] == 2


# ---------------------------------------------------------------------------
# State machine round-trip
# ---------------------------------------------------------------------------

def test_v4_state_machine_round_trip():
    fixture = _load("state_machine_basic.json")
    sm = process_state_machine(fixture)
    re_emitted = state_machine_object_to_json(sm)
    assert re_emitted.get("type") == "StateMachineDiagram"
    assert isinstance(re_emitted.get("nodes"), list)
    assert isinstance(re_emitted.get("edges"), list)
    types_out = _node_types(re_emitted)
    # Two named states + an initial node.
    assert types_out["State"] == 2
    assert types_out["StateInitialNode"] >= 1
    edges_out = _edge_types(re_emitted)
    # Initial transition + Red->Green transition.
    assert edges_out["StateTransition"] >= 2


# ---------------------------------------------------------------------------
# Agent diagram round-trip
# ---------------------------------------------------------------------------

def test_v4_agent_diagram_round_trip():
    fixture = _load("agent_diagram_basic.json")
    agent = process_agent_diagram(fixture)
    assert agent is not None
    assert any(s.name == "Greeting" for s in agent.states)
    assert any(i.name == "greet" for i in agent.intents)
    # The intent's training utterances come from ``data.training_phrases``.
    greet = next(i for i in agent.intents if i.name == "greet")
    assert sorted(greet.training_sentences) == ["hello", "hi"]
    # The LLM body's system prompt rides on the body row name and lands on
    # ``LLMReply.prompt``.
    ai_help = next(s for s in agent.states if s.name == "AIHelp")
    llm_actions = [a for a in ai_help.body.actions if type(a).__name__ == "LLMReply"]
    assert [a.prompt for a in llm_actions] == ["You are a friendly gym assistant."]
    # The comment node tethered to Greeting lands on the state metadata.
    greeting = next(s for s in agent.states if s.name == "Greeting")
    assert greeting.metadata is not None
    assert greeting.metadata.description == "Welcomes the user."


def test_v4_agent_diagram_full_chain_re_emits_v4_shape():
    """JSON -> Agent metamodel -> generated Python -> JSON: the v4 node /
    edge types, the LLM prompt, the training phrases and the comment all
    survive the full export/import chain."""
    import tempfile

    from besser.utilities.buml_code_builder.agent_model_builder import (
        agent_model_to_code,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.agent_diagram_converter import (
        agent_buml_to_json,
    )

    fixture = _load("agent_diagram_basic.json")
    agent = process_agent_diagram(fixture)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "agent.py")
        agent_model_to_code(agent, path)
        with open(path, encoding="utf-8") as f:
            re_emitted = agent_buml_to_json(f.read())

    types_out = _node_types(re_emitted)
    edges_out = _edge_types(re_emitted)
    assert types_out["AgentState"] == 2
    assert types_out["AgentIntent"] == 1
    assert types_out["comment"] == 1
    assert edges_out["CommentLink"] == 1

    nodes = re_emitted["nodes"]
    ai_help = next(
        n for n in nodes
        if n["type"] == "AgentState" and n["data"]["name"] == "AIHelp"
    )
    assert [(b["name"], b["replyType"]) for b in ai_help["data"]["bodies"]] == [
        ("You are a friendly gym assistant.", "llm"),
    ]
    intent = next(n for n in nodes if n["type"] == "AgentIntent")
    assert [p["name"] for p in intent["data"]["training_phrases"]] == ["hi", "hello"]
    comment = next(n for n in nodes if n["type"] == "comment")
    assert comment["data"]["name"] == "Welcomes the user."


def test_v4_agent_intent_legacy_bodies_fallback():
    """Older fixtures store training utterances under ``data.bodies`` —
    the processor must keep accepting that spelling."""
    fixture = _load("agent_diagram_basic.json")
    for node in fixture["model"]["nodes"]:
        if node.get("type") == "AgentIntent":
            node["data"]["bodies"] = node["data"].pop("training_phrases")
    agent = process_agent_diagram(fixture)
    greet = next(i for i in agent.intents if i.name == "greet")
    assert sorted(greet.training_sentences) == ["hello", "hi"]


# ---------------------------------------------------------------------------
# Object diagram round-trip (ingest only — there is no metamodel-to-v4
# emitter for ObjectModel that runs without parsing generated code).
# ---------------------------------------------------------------------------

def test_v4_object_diagram_ingest():
    cd_fixture = _load("class_diagram_basic.json")
    domain_model = process_class_diagram(cd_fixture)

    od_fixture = _load("object_diagram_basic.json")
    obj_model = process_object_diagram(od_fixture, domain_model)
    assert obj_model is not None
    # The legacy processor keeps the full ``"name : Class"`` label.
    assert any("myBook" in (o.name or "") for o in obj_model.objects)
    # v4 attribute rows keep name and runtime value in separate fields
    # (``ObjectNodeAttribute.value``) — both slots must ingest, typed.
    my_book = next(o for o in obj_model.objects if "myBook" in (o.name or ""))
    slot_values = {s.attribute.name: s.value.value for s in my_book.slots}
    assert slot_values == {"title": "1984", "pages": 328}


# ---------------------------------------------------------------------------
# Reference class diagram extraction (generation router)
# ---------------------------------------------------------------------------

def test_extract_reference_class_diagram_accepts_v4_shape():
    from besser.utilities.web_modeling_editor.backend.routers.generation_router import (
        _extract_reference_class_diagram,
    )

    cd_fixture = _load("class_diagram_basic.json")
    payload = {
        "title": "Objects",
        "model": {
            "type": "ObjectDiagram",
            "nodes": [],
            "edges": [],
            "referenceDiagramData": {
                "title": cd_fixture["title"],
                "model": cd_fixture["model"],
            },
        },
    }
    extracted = _extract_reference_class_diagram(payload["model"])
    assert extracted is not None
    assert extracted["title"] == "Library"
    # The extracted payload feeds straight into ``process_class_diagram``.
    domain_model = process_class_diagram(extracted)
    assert any(t.name == "Book" for t in domain_model.types)


def test_extract_reference_class_diagram_rejects_v3_shape():
    from besser.utilities.web_modeling_editor.backend.routers.generation_router import (
        _extract_reference_class_diagram,
    )

    legacy = {
        "referenceDiagramData": {
            "title": "Old",
            "model": {"elements": {}, "relationships": {}},
        },
    }
    assert _extract_reference_class_diagram(legacy) is None


# ---------------------------------------------------------------------------
# GUI processor v4 class-model lookups
# ---------------------------------------------------------------------------

def test_gui_processor_resolves_v4_class_and_method_rows():
    """v4 keeps the class name on ``node.data.name`` and method rows inside
    ``data.methods`` (not as top-level nodes) — the GUI processor's lookup
    helpers must resolve both."""
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.processor import (
        _element_name,
        _find_method_row,
    )

    cd_fixture = _load("class_diagram_basic.json")
    class_model = cd_fixture["model"]
    book_node = next(
        n for n in class_model["nodes"]
        if (n.get("data") or {}).get("name") == "Book"
    )
    assert _element_name(book_node) == "Book"
    row = _find_method_row(book_node, "n-book-m-summary")
    assert row is not None
    assert row["name"] == "summary"
    assert _find_method_row(book_node, "missing-id") is None
    # Legacy dicts (name at the top level, rows at the top level) keep working.
    legacy_el = {"name": "Legacy", "methods": [{"id": "m-1", "name": "+ run()"}]}
    assert _element_name(legacy_el) == "Legacy"
    assert _find_method_row(legacy_el, "m-1")["name"] == "+ run()"


# ---------------------------------------------------------------------------
# NN diagram round-trip
# ---------------------------------------------------------------------------

def test_v4_nn_diagram_round_trip():
    fixture = _load("nn_diagram_basic.json")
    nn = process_nn_diagram(fixture)
    re_emitted = nn_model_to_json(nn)
    assert re_emitted.get("type") == "NNDiagram"
    assert isinstance(re_emitted.get("nodes"), list)
    types_out = _node_types(re_emitted)
    # Two LinearLayer + one NNContainer.
    assert types_out.get("LinearLayer", 0) == 2
    assert types_out.get("NNContainer", 0) >= 1
    edges_out = _edge_types(re_emitted)
    assert edges_out.get("NNNext", 0) >= 1
