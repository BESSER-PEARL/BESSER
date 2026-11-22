"""Tests for the KnowledgeGraph → Python code builder.

The builder's output is never imported — ``kg_buml_to_json`` strips the import
block and loads the rest through ``safe_load_buml``, an AST allowlist. These
tests therefore always assert through that reader rather than against the
emitted text, so a change that produces plausible-looking code the reader
cannot load still fails.
"""

from __future__ import annotations

import pytest

from tests.BUML.metamodel.kg.axiom_samples import AXIOM_SAMPLES
from besser.BUML.metamodel.kg import (
    EquivalentClassesAxiom,
    ImportAxiom,
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.buml_code_builder.kg_model_builder import kg_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters import (
    kg_buml_to_json,
)


def _round_trip(kg: KnowledgeGraph, tmp_path) -> dict:
    """Generate code for ``kg`` and read it back through the project reader."""
    path = tmp_path / "kg_model.py"
    kg_model_to_code(kg, str(path))
    return kg_buml_to_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(name="sample_kg")
    person = KGClass(id="c1", label="Person", iri="http://ex.org/Person")
    agent = KGClass(id="c2", label="Agent", iri="http://ex.org/Agent")
    knows = KGProperty(id="p1", label="knows", iri="http://ex.org/knows")
    alice = KGIndividual(id="i1", label="Alice", iri="http://ex.org/alice")
    age = KGLiteral(id="l1", value="30", datatype="http://www.w3.org/2001/XMLSchema#integer")
    blank = KGBlank(id="b1", label="")
    for node in (person, agent, knows, alice, age, blank):
        kg.add_node(node)
    kg.add_edge(KGEdge(
        id="e1", source=person, target=agent, label="subClassOf",
        iri="http://www.w3.org/2000/01/rdf-schema#subClassOf",
    ))
    kg.add_edge(KGEdge(id="e2", source=alice, target=age, label="age", iri="http://ex.org/age"))
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["c1", "c2"]))
    return kg


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def test_round_trip_preserves_nodes_edges_and_axioms(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)

    assert model["type"] == "KnowledgeGraphDiagram"
    assert len(model["nodes"]) == 6
    assert len(model["edges"]) == 2
    assert len(model["axioms"]) == 1

    by_id = {n["id"]: n for n in model["nodes"]}
    assert by_id["c1"]["nodeType"] == "class"
    assert by_id["p1"]["nodeType"] == "property"
    assert by_id["i1"]["nodeType"] == "individual"
    assert by_id["l1"]["nodeType"] == "literal"
    assert by_id["b1"]["nodeType"] == "blank"
    assert by_id["c1"]["iri"] == "http://ex.org/Person"


def test_round_trip_preserves_literal_value_and_datatype(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)
    literal = {n["id"]: n for n in model["nodes"]}["l1"]
    assert literal["value"] == "30"
    assert literal["datatype"] == "http://www.w3.org/2001/XMLSchema#integer"


def test_round_trip_preserves_edge_endpoints(sample_kg, tmp_path):
    model = _round_trip(sample_kg, tmp_path)
    edges = {e["id"]: e for e in model["edges"]}
    assert edges["e1"]["source"] == "c1"
    assert edges["e1"]["target"] == "c2"
    assert edges["e1"]["iri"].endswith("subClassOf")


def test_emits_the_section_banner_the_project_reader_looks_for(sample_kg, tmp_path):
    # ``project_to_code`` only prepends a numbered header when a project holds
    # more than one KG, so a lone KG relies entirely on this banner to be found.
    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path))
    assert "# KNOWLEDGE_GRAPH MODEL" in path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Constraint payloads
# ----------------------------------------------------------------------


def test_constraint_specs_survive_the_round_trip(tmp_path):
    # Losing metadata silently disarms several preflight detectors, so this is
    # the payload that matters most.
    kg = KnowledgeGraph(name="constrained")
    node_c = KGNodeConstraint(id="nc1", label="Shape", metadata={
        "constraintSpecs": [
            {"kind": "disjointWith", "vocab": ["owl"],
             "params": {"classes": ["http://ex.org/Agent"]}},
        ],
    })
    prop_c = KGPropertyConstraint(id="pc1", label="MinOne", metadata={
        "constraintSpecs": [
            {"kind": "minCardinality", "vocab": ["owl", "shacl"], "params": {"value": 1}},
        ],
    })
    kg.add_node(node_c)
    kg.add_node(prop_c)

    model = _round_trip(kg, tmp_path)
    by_id = {n["id"]: n for n in model["nodes"]}
    assert by_id["nc1"]["nodeType"] == "nodeConstraint"
    assert by_id["pc1"]["nodeType"] == "propertyConstraint"
    assert by_id["nc1"]["metadata"]["constraintSpecs"][0]["kind"] == "disjointWith"
    assert by_id["pc1"]["metadata"]["constraintSpecs"][0]["params"]["value"] == 1


def test_all_axiom_kinds_round_trip(tmp_path):
    """Every axiom class, with every field value, survives builder -> reader.

    Driven by ``KGAxiom.__subclasses__()`` (through the shared samples) so a new
    class is covered the day it is added, and compares the values, not just the
    kinds.
    """
    assert set(AXIOM_SAMPLES) == set(KGAxiom.__subclasses__())
    kg = KnowledgeGraph(name="axioms")
    kg.add_node(KGClass(id="c1", label="A"))
    for axiom_cls, kwargs in AXIOM_SAMPLES.items():
        kg.add_axiom(axiom_cls(**kwargs))

    model = _round_trip(kg, tmp_path)

    expected = sorted(
        ({"kind": cls.__name__, **kwargs} for cls, kwargs in AXIOM_SAMPLES.items()),
        key=lambda a: (a["kind"], str(a)),
    )
    assert sorted(model["axioms"], key=lambda a: (a["kind"], str(a))) == expected


def test_an_axiom_with_no_source_ontology_round_trips(tmp_path):
    kg = KnowledgeGraph(name="axioms")
    kg.add_axiom(ImportAxiom(target_iri="http://ex.org/other", source_ontology_iri=None))

    (axiom,) = _round_trip(kg, tmp_path)["axioms"]

    assert axiom == {"kind": "ImportAxiom", "target_iri": "http://ex.org/other", "source_ontology_iri": None}


# ----------------------------------------------------------------------
# Hostile input and edge cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("label", [
    'quote " inside',
    "apostrophe ' inside",
    "back\\slash",
    "new\nline",
    "carriage\rreturn",
    "unicode ünïcødé ☃",
    "'''triple''' and \"\"\"quotes\"\"\"",
    "trailing backslash \\",
])
def test_hostile_labels_round_trip_verbatim(label, tmp_path):
    # The generated source is exec()'d, so a mis-escaped label is a syntax
    # error or worse rather than a cosmetic defect.
    kg = KnowledgeGraph(name="hostile")
    kg.add_node(KGClass(id="c1", label=label, iri="http://ex.org/C"))
    model = _round_trip(kg, tmp_path)
    assert model["nodes"][0]["label"] == label


def test_labels_that_are_not_identifiers_do_not_collide(tmp_path):
    # Two labels sanitising to the same variable name must still produce two
    # distinct nodes.
    kg = KnowledgeGraph(name="collisions")
    kg.add_node(KGClass(id="c1", label="my class", iri="http://ex.org/1"))
    kg.add_node(KGClass(id="c2", label="my-class", iri="http://ex.org/2"))
    kg.add_node(KGClass(id="c3", label="my/class", iri="http://ex.org/3"))
    kg.add_node(KGClass(id="c4", label="", iri="http://ex.org/4"))
    kg.add_node(KGClass(id="c5", label="123", iri="http://ex.org/5"))
    kg.add_node(KGClass(id="c6", label="class", iri="http://ex.org/6"))

    model = _round_trip(kg, tmp_path)
    assert len(model["nodes"]) == 6
    assert {n["id"] for n in model["nodes"]} == {"c1", "c2", "c3", "c4", "c5", "c6"}


def test_empty_graph_round_trips(tmp_path):
    # ``{}`` is an empty dict, not an empty set — the builder has to emit
    # ``set()`` for the nodes and edges arguments.
    kg = KnowledgeGraph(name="empty")
    model = _round_trip(kg, tmp_path)
    assert model["nodes"] == []
    assert model["edges"] == []
    assert model["axioms"] == []


def test_output_is_deterministic(sample_kg, tmp_path):
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    kg_model_to_code(sample_kg, str(first))
    kg_model_to_code(sample_kg, str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_model_var_name_is_configurable(sample_kg, tmp_path):
    # project_to_code passes kg_model_1, kg_model_2, … for multi-KG projects.
    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path), model_var_name="kg_model_2")
    text = path.read_text(encoding="utf-8")
    assert "kg_model_2 = KnowledgeGraph(" in text
    assert kg_buml_to_json(text)["nodes"]


# ---------------------------------------------------------------------------
# Generated code must always be loadable
# ---------------------------------------------------------------------------


def test_non_finite_floats_round_trip(tmp_path):
    """``repr(float('nan'))`` is ``nan``: a name Python does not define.

    The emitted module used to fail to load with NameError.
    """
    kg = KnowledgeGraph(name="floats")
    node = KGIndividual(id="measurement", label="Measurement")
    node.set_metadata_item("readings", [float("nan"), float("inf"), float("-inf"), 1.5])
    kg.add_node(node)

    model = _round_trip(kg, tmp_path)

    readings = next(
        n["metadata"]["readings"] for n in model["nodes"] if n["id"] == "measurement"
    )
    assert readings[0] != readings[0]            # nan is the only value unequal to itself
    assert readings[1] == float("inf")
    assert readings[2] == float("-inf")
    assert readings[3] == 1.5


def test_unknown_node_type_raises_instead_of_being_relabelled():
    """Defaulting to KGIndividual silently rewrote the node's kind."""
    from besser.BUML.metamodel.kg import KGNode
    from besser.utilities.buml_code_builder.kg_model_builder import _node_class_name

    class FutureKGNode(KGNode):
        pass

    with pytest.raises(TypeError, match="unexpected KGNode type FutureKGNode"):
        _node_class_name(FutureKGNode(id="n1", label="N1"))


def test_a_label_with_a_superscript_generates_loadable_code(tmp_path):
    """``str.isalnum()`` accepts ``²``, the parser does not: ``kg_node_area_m²``
    was a SyntaxError."""
    kg = KnowledgeGraph(name="units")
    kg.add_node(KGClass(id="c1", label="area m²"))
    kg.add_node(KGClass(id="c2", label="①½"))
    path = tmp_path / "kg_model.py"
    kg_model_to_code(kg, str(path))
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert {n["label"] for n in kg_buml_to_json(source)["nodes"]} == {"area m²", "①½"}


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def test_starts_with_the_besser_version_header(sample_kg, tmp_path):
    from besser.utilities.buml_code_builder.common import buml_header

    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path))
    assert path.read_text(encoding="utf-8").startswith(buml_header())


def test_a_title_goes_into_the_one_banner(sample_kg, tmp_path):
    path = tmp_path / "kg_model.py"
    kg_model_to_code(sample_kg, str(path), title='My "KG"\nnext')
    lines = path.read_text(encoding="utf-8").splitlines()

    banners = [line for line in lines if "KNOWLEDGE_GRAPH MODEL" in line]
    assert banners == ["# KNOWLEDGE_GRAPH MODEL: \"My 'KG' next\" #"]


def test_a_project_with_two_knowledge_graphs_round_trips(tmp_path):
    """The project used to write its own numbered header right above the
    builder's banner; the importer took the banner as the section start, found
    the numbered header's section empty, and failed on the other."""
    from besser.BUML.metamodel.project import Metadata, Project
    from besser.utilities.buml_code_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import project_to_json

    def make(name):
        cls = KGClass(id=f"{name}_c", label="C")
        ind = KGIndividual(id=f"{name}_i", label="I")
        return KnowledgeGraph(
            name=name, nodes={cls, ind}, edges={KGEdge(id=f"{name}_e", source=cls, target=ind)},
            axioms=[EquivalentClassesAxiom(class_ids=[f"{name}_c"])],
        )

    project = Project(name="proj", models=[make("kga"), make("kgb")], owner="o",
                      metadata=Metadata(description="d"))
    path = tmp_path / "project.py"
    project_to_code(project, str(path))
    source = path.read_text(encoding="utf-8")

    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    assert {namespace["kg_model_1"].name, namespace["kg_model_2"].name} == {"kga", "kgb"}

    diagrams = project_to_json(source)["diagrams"]["KnowledgeGraphDiagram"]
    assert sorted(d["title"] for d in diagrams) == ["kga", "kgb"]
    by_title = {d["title"]: d["model"] for d in diagrams}
    for name in ("kga", "kgb"):
        model = by_title[name]
        assert {n["id"] for n in model["nodes"]} == {f"{name}_c", f"{name}_i"}
        assert [e["id"] for e in model["edges"]] == [f"{name}_e"]
        assert model["axioms"] == [{"kind": "EquivalentClassesAxiom", "class_ids": [f"{name}_c"]}]


def test_a_project_with_one_knowledge_graph_round_trips(sample_kg, tmp_path):
    from besser.BUML.metamodel.project import Metadata, Project
    from besser.utilities.buml_code_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import project_to_json

    project = Project(name="proj", models=[sample_kg], owner="o", metadata=Metadata(description="d"))
    path = tmp_path / "project.py"
    project_to_code(project, str(path))
    source = path.read_text(encoding="utf-8")

    assert source.count("# Generated by BESSER") == 1
    (diagram,) = project_to_json(source)["diagrams"]["KnowledgeGraphDiagram"]
    assert len(diagram["model"]["nodes"]) == 6


def _project_round_trip(models, tmp_path, retitle=None):
    """Export a project with ``models``, read it back as editor JSON and, when
    ``retitle`` is given, set those KG titles and take the JSON back to B-UML and
    out again, as the editor does on every save and reload.

    Returns:
        list: The KnowledgeGraphDiagram entries of the final project JSON.
    """
    from besser.BUML.metamodel.project import Metadata, Project
    from besser.utilities.buml_code_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        json_to_buml_project,
        project_to_json,
    )

    project = Project(name="proj", models=models, owner="o", metadata=Metadata(description="d"))
    first = tmp_path / "first.py"
    project_to_code(project, str(first))
    project_json = project_to_json(first.read_text(encoding="utf-8"))
    if retitle is None:
        return project_json["diagrams"]["KnowledgeGraphDiagram"]

    for diagram, title in zip(project_json["diagrams"]["KnowledgeGraphDiagram"], retitle):
        diagram["title"] = title
    second = tmp_path / "second.py"
    project_to_code(json_to_buml_project(ProjectInput(**project_json)), str(second))
    return project_to_json(second.read_text(encoding="utf-8"))["diagrams"]["KnowledgeGraphDiagram"]


def test_a_single_knowledge_graph_keeps_its_title_through_a_project_round_trip(sample_kg, tmp_path):
    """The banner of a lone KG carried no title, so it came back as the default
    "Knowledge Graph Diagram"; and the KG's name is sanitized, so the title the
    user typed has to travel separately."""
    (diagram,) = _project_round_trip([sample_kg], tmp_path, retitle=["My Ontology v2"])

    assert diagram["title"] == "My Ontology v2"
    assert len(diagram["model"]["nodes"]) == 6


def test_two_knowledge_graphs_with_the_same_name_both_come_back_told_apart(tmp_path):
    def make(node_id):
        return KnowledgeGraph(name="kg", nodes={KGClass(id=node_id, label="C")})

    diagrams = _project_round_trip([make("a"), make("b")], tmp_path)

    assert [(d["title"], [n["id"] for n in d["model"]["nodes"]]) for d in diagrams] == [
        ("kg (1)", ["a"]),
        ("kg (2)", ["b"]),
    ]


def test_distinct_titles_do_not_grow_a_number_on_each_round_trip(tmp_path):
    graphs = [KnowledgeGraph(name="kg", nodes={KGClass(id=i, label="C")}) for i in ("a", "b")]

    diagrams = _project_round_trip(graphs, tmp_path, retitle=["People", "Places"])

    assert [d["title"] for d in diagrams] == ["People", "Places"]


def test_a_set_in_metadata_is_written_in_a_stable_order(tmp_path):
    """A set's iteration order follows string hashing, which changes per run."""
    from besser.utilities.buml_code_builder.kg_model_builder import _literal

    assert _literal({"tags": {"b", "c", "a"}}) == "{'tags': ['a', 'b', 'c']}"
    assert _literal(frozenset({3, 1, 2})) == "[1, 2, 3]"


def test_a_scalar_subclass_is_written_as_its_base_type():
    """``repr`` of a ``str`` / ``int`` subclass is not a literal the loader accepts."""
    import enum

    from besser.utilities.buml_code_builder.kg_model_builder import _literal

    class Iri(str):
        def __repr__(self):
            return f"Iri({str(self)!r})"

    class Level(enum.IntEnum):
        HIGH = 3

    assert _literal(Iri("http://ex.org/a")) == "'http://ex.org/a'"
    assert _literal(Level.HIGH) == "3"


@pytest.mark.parametrize("value", [object(), b"bytes", 1j])
def test_a_value_that_is_not_json_compatible_is_refused(value, tmp_path):
    """It used to be written as its ``str()``, changing its type on the way back."""
    kg = KnowledgeGraph(name="bad")
    node = KGIndividual(id="i1", label="I")
    node.set_metadata_item("value", value)
    kg.add_node(node)

    with pytest.raises(TypeError, match="JSON-compatible"):
        kg_model_to_code(kg, str(tmp_path / "kg_model.py"))
