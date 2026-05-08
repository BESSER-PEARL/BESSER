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
    # Genre `class` with stereotype 'enumeration' also surfaces as 'class').
    types_in = _node_types(fixture)
    types_out = _node_types(re_emitted)
    assert types_out["class"] == types_in["class"]
    # Edge-type distribution preserved.
    edges_in = _edge_types(fixture)
    edges_out = _edge_types(re_emitted)
    assert edges_in["ClassBidirectional"] == edges_out["ClassBidirectional"]
    # Top-level class names round-trip.
    names = _top_level_names(re_emitted)
    assert "Book" in names
    assert "Author" in names
    assert "Genre" in names


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
    # We only verify the ingest succeeds; the buml-to-json side parses
    # generated Python source which is out of scope here.
    assert agent is not None
    assert any(s.name == "Greeting" for s in agent.states)
    assert any(i.name == "greet" for i in agent.intents)


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
