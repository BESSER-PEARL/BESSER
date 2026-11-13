"""adopt_layout: an imported diagram laid out like the diagram it replaces (issue #774).

The models come from the real converters (B-UML -> v4 JSON), the way
``/get-json-model`` builds them, so the tests see the ids and grid positions an
import actually produces.
"""

import copy

from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code
from besser.utilities.buml_code_builder.state_machine_builder import state_machine_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    agent_buml_to_json,
    class_buml_to_json,
    process_state_machine,
    state_machine_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.converters.layout_adoption import (
    adopt_existing_layout,
    adopt_layout,
)


def agent_source(tmp_path, refined=False):
    """The greeting agent; ``refined`` renames ``good`` and adds a ``farewell`` state."""
    agent = Agent("Greeter")
    hello = agent.new_intent("hello_intent", ["hi", "hello"])
    bye = agent.new_intent("bye_intent", ["bye"])
    initial = agent.new_state("initial", initial=True)
    greeting = agent.new_state("greeting")
    good = agent.new_state("feeling_good" if refined else "good")
    greeting.set_body(Body("greeting_body", actions=[AgentReply("Hello!")]))
    initial.when_intent_matched(hello).go_to(greeting)
    greeting.when_intent_matched(hello).go_to(good)
    good.go_to(initial)
    if refined:
        farewell = agent.new_state("farewell")
        greeting.when_intent_matched(bye).go_to(farewell)
    path = tmp_path / ("refined.py" if refined else "agent.py")
    agent_model_to_code(agent, str(path))
    return path.read_text()


def laid_out(model):
    """``model`` as a user left it: every node moved off the converter's grid, every edge bent."""
    model = copy.deepcopy(model)
    for index, node in enumerate(model["nodes"]):
        node["position"] = {"x": 37 + 230 * index, "y": 11 - 170 * index}
    for edge in model["edges"]:
        edge["data"]["points"] = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        edge["data"]["isManuallyLayouted"] = True
        edge["sourceHandle"], edge["targetHandle"] = "bottom", "top"
    return model


def by_name(model):
    return {node["data"]["name"]: node for node in model["nodes"]}


def library_model(extra_class=None, rename_author=None):
    string = PrimitiveDataType("str")
    book = Class("Book", attributes={Property("title", string), Property("isbn", string)})
    author = Class(rename_author or "Author", attributes={Property("name", string)})
    writes = BinaryAssociation("writes", ends={
        Property("books", book, multiplicity=Multiplicity(0, "*")),
        Property("authors", author, multiplicity=Multiplicity(1, "*")),
    })
    types = {book, author}
    if extra_class:
        types.add(Class(extra_class, attributes={Property("code", string)}))
    return class_buml_to_json(DomainModel("Library", types=types, associations={writes}))


def _ids(model):
    return {node["id"] for node in model["nodes"]}, {edge["id"] for edge in model["edges"]}


def test_unchanged_agent_keeps_every_position_and_id(tmp_path):
    existing = laid_out(agent_buml_to_json(agent_source(tmp_path)))
    imported = agent_buml_to_json(agent_source(tmp_path))
    assert _ids(imported)[0].isdisjoint(_ids(existing)[0])  # the import alone gives fresh ids

    model, report = adopt_layout(existing, imported)

    assert model["nodes"] == existing["nodes"]
    assert model["edges"] == existing["edges"]
    assert set(model["components"]) == set(existing["components"])
    assert report["matched"] == 3 and report["added"] == [] and report["notInImport"] == []
    assert report["edgesKept"] == 3 and report["edgesAdded"] == 0


def test_unchanged_class_diagram_keeps_positions_ids_and_attribute_ids():
    existing = laid_out(library_model())
    imported = library_model()
    assert _ids(imported)[0].isdisjoint(_ids(existing)[0])
    row_ids = {row["id"] for node in imported["nodes"] for row in node["data"]["attributes"]}
    assert row_ids.isdisjoint({row["id"] for node in existing["nodes"] for row in node["data"]["attributes"]})

    model, report = adopt_layout(existing, imported)

    assert model["nodes"] == existing["nodes"]  # positions, sizes, ids and attribute row ids
    assert model["edges"] == existing["edges"]  # id, handles and bend points of the association
    assert report["matched"] == 2


def test_new_state_is_placed_next_to_its_neighbour_without_overlap(tmp_path):
    existing = laid_out(agent_buml_to_json(agent_source(tmp_path)))
    model, report = adopt_layout(existing, agent_buml_to_json(agent_source(tmp_path, refined=True)))

    old, new = by_name(existing), by_name(model)
    for name in ("initial", "greeting"):
        assert new[name]["id"] == old[name]["id"]
        assert new[name]["position"] == old[name]["position"]
    farewell = new["farewell"]
    assert report["added"] == [{"id": farewell["id"], "type": "AgentState", "name": "farewell",
                                "placedNextTo": "greeting"}]
    greeting = new["greeting"]["position"]
    assert abs(farewell["position"]["x"] - greeting["x"]) + abs(farewell["position"]["y"] - greeting["y"]) < 600

    def box(node):
        return (node["position"]["x"], node["position"]["y"],
                node["position"]["x"] + node["width"], node["position"]["y"] + node["height"])

    for other in model["nodes"]:
        if other is not farewell:
            a, b = box(farewell), box(other)
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], other["data"]["name"]


def test_renamed_state_keeps_its_place_when_its_connections_say_so(tmp_path):
    existing = laid_out(agent_buml_to_json(agent_source(tmp_path)))
    model, report = adopt_layout(existing, agent_buml_to_json(agent_source(tmp_path, refined=True)))

    old_good = by_name(existing)["good"]
    renamed = by_name(model)["feeling_good"]
    assert (renamed["id"], renamed["position"]) == (old_good["id"], old_good["position"])
    assert report["renamed"] == [{"id": old_good["id"], "type": "AgentState", "from": "good", "to": "feeling_good"}]
    assert report["notInImport"] == []
    # The three transitions the file still has keep their id and bend points; the new one is routed afresh.
    old_edges = {edge["id"]: edge for edge in existing["edges"]}
    kept = [edge for edge in model["edges"] if edge["id"] in old_edges]
    assert len(kept) == 3 and all(edge["data"] == old_edges[edge["id"]]["data"] for edge in kept)
    assert report["edgesKept"] == 3 and report["edgesAdded"] == 1


def test_unconnected_renamed_class_reads_as_removed_and_added():
    existing = laid_out(library_model(extra_class="Shelf"))
    model, report = adopt_layout(existing, library_model(extra_class="Rack"))

    old_shelf = by_name(existing)["Shelf"]
    assert by_name(model)["Rack"]["id"] != old_shelf["id"]
    assert report["renamed"] == []
    assert [item["name"] for item in report["added"]] == ["Rack"]
    assert report["notInImport"] == [{"id": old_shelf["id"], "type": "class", "name": "Shelf"}]


def test_removed_state_is_reported_and_its_edges_dropped(tmp_path):
    existing = laid_out(agent_buml_to_json(agent_source(tmp_path, refined=True)))
    model, report = adopt_layout(existing, agent_buml_to_json(agent_source(tmp_path)))

    old_farewell = by_name(existing)["farewell"]
    assert report["notInImport"] == [{"id": old_farewell["id"], "type": "AgentState", "name": "farewell"}]
    assert old_farewell["id"] not in {e["source"] for e in model["edges"]} | {e["target"] for e in model["edges"]}


def test_same_named_nodes_pair_in_order_and_no_id_is_used_twice():
    existing = laid_out(library_model())
    duplicate = copy.deepcopy(existing["nodes"][0])
    duplicate["id"] = "dup-old"
    duplicate["position"] = {"x": 900, "y": 900}
    existing["nodes"].append(duplicate)
    imported = library_model()
    extra = copy.deepcopy(imported["nodes"][0])
    extra["id"] = "dup-new"
    imported["nodes"].append(extra)
    second = copy.deepcopy(imported["nodes"][0])
    second["id"] = "dup-third"
    imported["nodes"].append(second)

    model, _ = adopt_layout(existing, imported)

    ids = [node["id"] for node in model["nodes"]]
    assert len(ids) == len(set(ids))
    name = existing["nodes"][0]["data"]["name"]
    same_name = [node for node in model["nodes"] if node["data"]["name"] == name]
    assert [node["id"] for node in same_name[:2]] == [existing["nodes"][0]["id"], "dup-old"]
    assert same_name[1]["position"] == {"x": 900, "y": 900}


def test_edge_drops_its_bend_points_when_an_end_grew():
    existing = laid_out(library_model())
    imported = library_model()
    book = by_name(imported)["Book"]
    book["height"] += 200
    book["measured"]["height"] += 200

    model, report = adopt_layout(existing, imported)

    old_edge = existing["edges"][0]
    edge = model["edges"][0]
    assert edge["id"] == old_edge["id"]
    assert (edge["sourceHandle"], edge["targetHandle"]) == ("bottom", "top")
    assert edge["data"]["points"] == [] and "isManuallyLayouted" not in edge["data"]
    assert report["edgesRerouted"] == 1
    assert by_name(model)["Book"]["position"] == by_name(existing)["Book"]["position"]


def test_state_machine_round_trip_keeps_positions_and_ids():
    def state(node_id, name, x, y):
        return {"id": node_id, "type": "State", "position": {"x": x, "y": y}, "width": 160, "height": 40,
                "data": {"name": name, "bodies": [], "fallbackBodies": []}}

    existing = {
        "version": "4.0.0", "type": "StateMachineDiagram", "title": "Door",
        "nodes": [
            {"id": "i", "type": "StateInitialNode", "position": {"x": -300, "y": 5}, "width": 45, "height": 45,
             "data": {"name": ""}},
            state("a", "Closed", -120, 0), state("b", "Open", 260, 140),
        ],
        "edges": [
            {"id": "t0", "source": "i", "target": "a", "type": "StateTransition",
             "sourceHandle": "right", "targetHandle": "left", "data": {"name": "", "points": []}},
            {"id": "t1", "source": "a", "target": "b", "type": "StateTransition",
             "sourceHandle": "bottom", "targetHandle": "left", "data": {"name": "push", "points": []}},
        ],
    }
    imported = state_machine_to_json(state_machine_to_code(process_state_machine({"title": "Door",
                                                                                   "model": existing})))

    model, report = adopt_layout(existing, imported)

    states = {n["data"]["name"]: n for n in model["nodes"] if n["type"] == "State"}
    assert {name: (n["id"], n["position"]) for name, n in states.items()} == {
        "Closed": ("a", {"x": -120, "y": 0}), "Open": ("b", {"x": 260, "y": 140}),
    }
    push = next(e for e in model["edges"] if e["data"].get("name") == "push")
    assert (push["id"], push["sourceHandle"]) == ("t1", "bottom")
    assert report["notInImport"] == []


def test_unusable_existing_model_leaves_the_import_as_is(tmp_path):
    imported = agent_buml_to_json(agent_source(tmp_path))
    for text, reason in [
        ("not json {", "not valid JSON"),
        ('{"elements": {}, "relationships": {}}', "not a diagram model with nodes"),
        ('{"type": "ClassDiagram", "nodes": []}', "the existing diagram is a ClassDiagram"),
    ]:
        model, warnings = adopt_existing_layout(text, imported, "AgentDiagram")
        assert model is imported
        assert [w["code"] for w in warnings] == ["layout-not-adopted"] and reason in warnings[0]["message"]

    _, warnings = adopt_existing_layout('{"nodes": []}', imported, "BPMNDiagram")
    assert [w["code"] for w in warnings] == ["layout-not-adopted"]


def test_empty_existing_diagram_keeps_the_imported_layout(tmp_path):
    imported = agent_buml_to_json(agent_source(tmp_path))
    model, warnings = adopt_existing_layout('{"type": "AgentDiagram", "nodes": [], "edges": []}',
                                            imported, "AgentDiagram")
    assert [n["position"] for n in model["nodes"]] == [n["position"] for n in imported["nodes"]]
    assert [w["code"] for w in warnings] == ["layout-adopted"] + ["element-added"] * 3
