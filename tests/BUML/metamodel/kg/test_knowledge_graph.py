"""Tests for the Knowledge Graph metamodel."""

import pytest

from tests.BUML.metamodel.kg.axiom_samples import AXIOM_SAMPLES
from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    EquivalentClassesAxiom,
    ImportAxiom,
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)


def test_knowledge_graph_accepts_all_node_kinds():
    c = KGClass(id="http://ex/Person", label="Person", iri="http://ex/Person")
    i = KGIndividual(id="http://ex/alice", label="Alice", iri="http://ex/alice")
    p = KGProperty(id="http://ex/knows", label="knows", iri="http://ex/knows")
    lit = KGLiteral(id="lit:1", value="30", datatype="http://www.w3.org/2001/XMLSchema#integer")
    b = KGBlank(id="_:b0", label="b0")
    kg = KnowledgeGraph(name="graph", nodes={c, i, p, lit, b})

    assert kg.get_node("http://ex/Person") is c
    assert kg.get_node("lit:1") is lit
    assert len(kg.nodes) == 5


def test_edge_endpoints_must_be_in_graph():
    c = KGClass(id="c1", label="C1")
    i = KGIndividual(id="i1", label="I1")
    kg = KnowledgeGraph(name="graph", nodes={c, i})
    kg.add_edge(KGEdge(id="e1", source=i, target=c, label="type"))
    assert len(kg.edges) == 1

    stray = KGIndividual(id="stray", label="stray")
    with pytest.raises(ValueError):
        kg.add_edge(KGEdge(id="e2", source=i, target=stray))


def test_literal_requires_value():
    with pytest.raises(ValueError):
        KGLiteral(id="lit:2", value=None)  # type: ignore[arg-type]


def test_node_id_must_be_non_empty():
    with pytest.raises(ValueError):
        KGClass(id="", label="X")


def test_graph_name_rejects_spaces():
    # Inherited NamedElement.name validation disallows spaces.
    with pytest.raises(ValueError):
        KnowledgeGraph(name="has space")


# --- metadata field on KGNode / KGEdge -------------------------------------


def test_kgnode_metadata_defaults_to_empty_dict():
    n = KGClass(id="c1", label="C1")
    assert n.metadata == {}
    n.set_metadata_item("kind", "restriction")
    assert n.metadata == {"kind": "restriction"}


def test_kgnode_metadata_accepts_dict_via_constructor():
    payload = {"kind": "restriction", "on_property": "http://ex/age", "value": 1}
    n = KGBlank(id="_:b0", label="b0", metadata=payload)
    assert n.metadata == payload


def test_kgnode_metadata_setter_rejects_non_dict():
    n = KGClass(id="c1", label="C1")
    with pytest.raises(ValueError):
        n.metadata = "not-a-dict"  # type: ignore[assignment]


def test_kgnode_metadata_does_not_affect_identity():
    a = KGClass(id="c1", label="C1", metadata={"foo": 1})
    b = KGClass(id="c1", label="C1", metadata={"bar": 2})
    # Equality and hash are id-based, not metadata-based.
    assert a == b
    assert hash(a) == hash(b)


def test_kgliteral_metadata_propagates():
    lit = KGLiteral(id="lit:1", value="30", metadata={"source": "ABox"})
    assert lit.metadata == {"source": "ABox"}


def test_kgedge_metadata_defaults_and_setter():
    c = KGClass(id="c1", label="C1")
    i = KGIndividual(id="i1", label="I1")

    e = KGEdge(id="e1", source=i, target=c, label="type")
    assert e.metadata == {}

    e2 = KGEdge(id="e2", source=i, target=c, label="type", metadata={"derived_from": "e1"})
    assert e2.metadata == {"derived_from": "e1"}

    with pytest.raises(ValueError):
        e2.metadata = 42  # type: ignore[assignment]


# --- KnowledgeGraph.axioms -------------------------------------------------


def test_knowledge_graph_axioms_default_empty():
    kg = KnowledgeGraph(name="graph")
    assert kg.axioms == []


def test_add_axiom_appends_and_validates_type():
    kg = KnowledgeGraph(name="graph")
    kg.add_axiom(EquivalentClassesAxiom(class_ids=["c1", "c2"]))
    kg.add_axiom(DisjointClassesAxiom(class_ids=["c1", "c3"]))
    assert len(kg.axioms) == 2
    assert isinstance(kg.axioms[0], EquivalentClassesAxiom)
    assert isinstance(kg.axioms[0], KGAxiom)

    with pytest.raises(ValueError):
        kg.add_axiom("not-an-axiom")  # type: ignore[arg-type]


def test_axioms_setter_rejects_non_list_and_bad_elements():
    kg = KnowledgeGraph(name="graph")
    with pytest.raises(ValueError):
        kg.axioms = {"not": "a list"}  # type: ignore[assignment]
    with pytest.raises(ValueError):
        kg.axioms = ["not an axiom"]  # type: ignore[list-item]


def test_every_axiom_class_has_a_sample():
    """Guard for the tests below: a new axiom class must come with an example."""
    assert set(AXIOM_SAMPLES) == set(KGAxiom.__subclasses__())


@pytest.mark.parametrize("axiom_cls", AXIOM_SAMPLES, ids=lambda c: c.__name__)
def test_each_axiom_class_stores_and_reports_its_fields(axiom_cls):
    kwargs = AXIOM_SAMPLES[axiom_cls]
    assert tuple(kwargs) == axiom_cls.FIELDS, "the sample must name every field, in constructor order"

    axiom = axiom_cls(**kwargs)
    kg = KnowledgeGraph(name="graph")
    kg.add_axiom(axiom)

    assert kg.axioms == [axiom]
    for name, value in kwargs.items():
        assert getattr(axiom, name) == value
    assert axiom.to_dict() == kwargs
    assert axiom_cls(*kwargs.values()) == axiom, "positional order is FIELDS order"
    assert axiom_cls(**kwargs) is not axiom and axiom_cls(**kwargs) == axiom
    assert repr(axiom).startswith(f"{axiom_cls.__name__}(")
    assert all(repr(v) in repr(axiom) for v in kwargs.values())


@pytest.mark.parametrize("axiom_cls", AXIOM_SAMPLES, ids=lambda c: c.__name__)
def test_axiom_equality_is_by_value_and_type(axiom_cls):
    kwargs = AXIOM_SAMPLES[axiom_cls]
    other = axiom_cls(**kwargs)
    first_field = axiom_cls.FIELDS[0]
    setattr(other, first_field, [*kwargs[first_field], "x"] if isinstance(kwargs[first_field], list) else "x")

    assert axiom_cls(**kwargs) != other
    assert axiom_cls(**kwargs) != "not an axiom"
    with pytest.raises(TypeError):
        hash(axiom_cls(**kwargs))


@pytest.mark.parametrize("axiom_cls", AXIOM_SAMPLES, ids=lambda c: c.__name__)
def test_axiom_fields_reject_values_of_the_wrong_type(axiom_cls):
    axiom = axiom_cls(**AXIOM_SAMPLES[axiom_cls])

    for name, sample in AXIOM_SAMPLES[axiom_cls].items():
        before = getattr(axiom, name)
        bad_values = ["abc", ["ok", 3], 7] if isinstance(sample, list) else [7, ["x"]]
        for bad in bad_values:
            with pytest.raises(ValueError, match=name):
                setattr(axiom, name, bad)
        assert getattr(axiom, name) == before, "a rejected assignment leaves the field unchanged"
    with pytest.raises(ValueError):
        axiom_cls(**{**AXIOM_SAMPLES[axiom_cls], axiom_cls.FIELDS[0]: 7})
    with pytest.raises(TypeError):
        axiom_cls(not_a_field="x")


@pytest.mark.parametrize("axiom_cls", [c for c, kw in AXIOM_SAMPLES.items()
                                       if any(isinstance(v, list) for v in kw.values())],
                         ids=lambda c: c.__name__)
def test_axiom_lists_are_copied_in_and_out(axiom_cls):
    kwargs = {k: (list(v) if isinstance(v, list) else v) for k, v in AXIOM_SAMPLES[axiom_cls].items()}
    list_field = next(k for k, v in kwargs.items() if isinstance(v, list))
    axiom = axiom_cls(**kwargs)

    kwargs[list_field].append("mutated-by-caller")           # the caller's list is not aliased
    getattr(axiom, list_field).append("mutated-via-getter")  # nor is the one handed out

    assert getattr(axiom, list_field) == AXIOM_SAMPLES[axiom_cls][list_field]


def test_the_optional_import_field_accepts_none():
    assert ImportAxiom(target_iri="http://ex.org/o").source_ontology_iri is None
    with pytest.raises(ValueError):
        ImportAxiom(target_iri=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Node id uniqueness and validate()
# ---------------------------------------------------------------------------


def test_duplicate_node_ids_are_rejected_across_node_types():
    """``KGNode.__hash__`` keys on (type, id), so a set alone does not catch it.

    When both survived, ``kg_model_to_code`` reattached an edge to whichever
    node it found first and the other was dropped.
    """
    kg = KnowledgeGraph(name="dup")
    kg.add_node(KGClass(id="x", label="X"))

    with pytest.raises(ValueError, match="id 'x'"):
        kg.add_node(KGIndividual(id="x", label="X individual"))


def test_duplicate_node_ids_are_rejected_by_the_nodes_setter():
    kg = KnowledgeGraph(name="dup")
    with pytest.raises(ValueError, match="duplicate ids"):
        kg.nodes = {KGClass(id="x", label="X"), KGIndividual(id="x", label="X")}


def test_validate_returns_the_standard_shape_for_a_sound_graph():
    kg = KnowledgeGraph(name="ok")
    a = KGClass(id="A", label="A")
    b = KGClass(id="B", label="B")
    kg.nodes = {a, b}
    kg.add_edge(KGEdge(id="e1", source=a, target=b, iri="http://ex.org/p"))

    result = kg.validate()

    assert result == {"success": True, "errors": [], "warnings": []}


def test_validate_reports_a_dangling_edge_instead_of_letting_it_pass():
    kg = KnowledgeGraph(name="dangling")
    a = KGClass(id="A", label="A")
    ghost = KGClass(id="Ghost", label="Ghost")
    kg.nodes = {a, ghost}
    kg.add_edge(KGEdge(id="e1", source=a, target=ghost, iri="http://ex.org/p"))

    # Assigning nodes directly is what the setter allows, and what skips
    # add_edge's endpoint check.
    kg.nodes = {a}

    result = kg.validate(raise_exception=False)
    assert result["success"] is False
    assert any("dangling" in e for e in result["errors"])

    with pytest.raises(ValueError, match="dangling"):
        kg.validate()


# ---------------------------------------------------------------------------
# IRI local names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iri, expected",
    [
        ("http://example.org/ns#Person", "Person"),
        ("http://example.org/ns/Person", "Person"),
        ("urn:besser:Thing", "Thing"),                   # no # or / at all
        ("http://purl.org/ontology/bibo/", "bibo"),      # trailing separator
        ("http://example.org/ns#", "ns"),
        ("Person", "Person"),
        ("", ""),
        (None, ""),
    ],
)
def test_local_name(iri, expected):
    from besser.BUML.metamodel.kg.iri import local_name

    assert local_name(iri) == expected


# ---------------------------------------------------------------------------
# metadata is a read-only view; changes go through the validating setter
# ---------------------------------------------------------------------------

from besser.BUML.metamodel.kg import KGNodeConstraint, KGPropertyConstraint  # noqa: E402

_VALID_SPEC = {"kind": "minCardinality", "vocab": ["owl", "shacl"], "params": {"value": 1}}


@pytest.mark.parametrize("make", [
    lambda: KGClass(id="c1", label="C"),
    lambda: KGNodeConstraint(id="nc", label="N"),
    lambda: KGPropertyConstraint(id="pc", label="P"),
    lambda: KGEdge(id="e", source=KGClass(id="a"), target=KGClass(id="b")),
], ids=["class", "node_constraint", "property_constraint", "edge"])
def test_metadata_cannot_be_written_through_the_getter(make):
    thing = make()

    with pytest.raises(TypeError):
        thing.metadata["constraintSpecs"] = [{"kind": "nonsense"}]
    with pytest.raises(AttributeError):
        thing.metadata.update({"x": 1})
    assert thing.metadata == {}


def test_set_metadata_item_and_update_metadata_change_a_plain_node():
    node = KGClass(id="c1", label="C", metadata={"keep": 1})

    node.set_metadata_item("kind", "restriction")
    node.update_metadata({"a": 1, "b": 2})

    assert dict(node.metadata) == {"keep": 1, "kind": "restriction", "a": 1, "b": 2}


def test_a_constraint_node_rejects_an_invalid_spec_however_it_arrives():
    node = KGNodeConstraint(id="nc", label="N", metadata={"constraintSpecs": [_VALID_SPEC]})

    with pytest.raises(ValueError, match="invalid"):
        node.set_metadata_item("constraintSpecs", [{"kind": "nonsense"}])
    with pytest.raises(ValueError, match="invalid"):
        node.update_metadata({"constraintSpecs": [_VALID_SPEC, {"kind": "nonsense"}]})
    with pytest.raises(ValueError, match="must be a list"):
        node.set_metadata_item("constraintSpecs", "not-a-list")
    assert node.get_specs() == [_VALID_SPEC], "a rejected change leaves the node as it was"


def test_other_metadata_keys_can_change_on_a_constraint_node():
    node = KGPropertyConstraint(id="pc", label="P", metadata={"constraintSpecs": [_VALID_SPEC]})

    node.set_metadata_item("source", "owl")

    assert node.metadata["source"] == "owl" and node.get_specs() == [_VALID_SPEC]


def test_editing_the_specs_you_read_does_not_change_the_node():
    node = KGNodeConstraint(id="nc", label="N", metadata={"constraintSpecs": [_VALID_SPEC]})

    node.metadata["constraintSpecs"].append({"kind": "nonsense"})        # a copy
    node.metadata["constraintSpecs"][0]["params"]["value"] = 99          # a deep copy
    node.get_specs()[0]["kind"] = "other"

    assert node.get_specs() == [_VALID_SPEC]


def test_the_specs_you_pass_in_are_copied_too():
    mine = [dict(_VALID_SPEC, params={"value": 1})]
    node = KGNodeConstraint(id="nc", label="N", metadata={"constraintSpecs": mine})

    mine.append({"kind": "nonsense"})
    mine[0]["params"]["value"] = 99

    assert node.get_specs() == [_VALID_SPEC]


def test_add_and_remove_spec_still_work():
    node = KGNodeConstraint(id="nc", label="N")

    node.add_spec(_VALID_SPEC)
    node.add_spec(dict(_VALID_SPEC, params={"value": 2}))
    node.remove_spec(0)

    assert [s["params"]["value"] for s in node.get_specs()] == [2]
    with pytest.raises(ValueError):
        node.add_spec({"kind": "nonsense"})


def test_validate_reports_a_constraint_spec_that_got_around_the_setter():
    kg = KnowledgeGraph(name="graph")
    node = KGNodeConstraint(id="nc", label="N", metadata={"constraintSpecs": [_VALID_SPEC]})
    kg.add_node(node)
    # State the setter never saw, e.g. a graph unpickled from before it validated.
    node._KGAnnotatedElement__metadata["constraintSpecs"] = [{"kind": "nonsense"}]

    result = kg.validate(raise_exception=False)

    assert not result["success"]
    assert "invalid constraint spec at index 0" in result["errors"][0]
    assert "'nc'" in result["errors"][0]


def test_a_mapping_is_accepted_wherever_a_dict_was():
    source = KGBlank(id="_:b", label="", metadata={"kind": "restriction"})

    copy_of = KGBlank(id="_:c", label="", metadata=source.metadata)   # the view itself

    assert dict(copy_of.metadata) == {"kind": "restriction"}


# ---------------------------------------------------------------------------
# Axioms: the list is the graph's own, ids are required, references checked
# ---------------------------------------------------------------------------


def test_the_axioms_getter_hands_out_a_copy():
    """Appending to the returned list bypassed ``add_axiom``'s type check."""
    kg = KnowledgeGraph(name="graph", axioms=[EquivalentClassesAxiom(class_ids=["c1"])])

    kg.axioms.append("not an axiom")

    assert kg.axioms == [EquivalentClassesAxiom(class_ids=["c1"])]


def test_validate_reports_an_axiom_that_is_not_a_kg_axiom():
    kg = KnowledgeGraph(name="graph")
    kg._KnowledgeGraph__axioms.append("not an axiom")   # state the setter never saw

    result = kg.validate(raise_exception=False)

    assert not result["success"]
    assert result["errors"] == ["KnowledgeGraph.axioms[0] is a str, not a KGAxiom."]


_REQUIRED_ID_FIELDS = [
    (cls, name)
    for cls, kwargs in AXIOM_SAMPLES.items()
    for name, value in kwargs.items()
    if isinstance(value, str) and not (cls is ImportAxiom and name == "source_ontology_iri")
]


@pytest.mark.parametrize("axiom_cls, field_name", _REQUIRED_ID_FIELDS,
                         ids=lambda v: v if isinstance(v, str) else v.__name__)
def test_a_required_id_cannot_be_left_out_or_empty(axiom_cls, field_name):
    """``SubPropertyOfAxiom()`` used to be a valid axiom about nothing."""
    kwargs = dict(AXIOM_SAMPLES[axiom_cls])

    with pytest.raises(ValueError, match=f"{field_name} must be a non-empty str"):
        axiom_cls(**{**kwargs, field_name: ""})
    with pytest.raises(TypeError):
        axiom_cls(**{k: v for k, v in kwargs.items() if k != field_name})
    axiom = axiom_cls(**kwargs)
    with pytest.raises(ValueError):
        setattr(axiom, field_name, "")
    assert getattr(axiom, field_name) == kwargs[field_name]


@pytest.mark.parametrize("axiom_cls", [c for c, kw in AXIOM_SAMPLES.items()
                                       if any(isinstance(v, list) for v in kw.values())],
                         ids=lambda c: c.__name__)
def test_an_id_list_cannot_hold_an_empty_id(axiom_cls):
    kwargs = AXIOM_SAMPLES[axiom_cls]
    list_field = next(k for k, v in kwargs.items() if isinstance(v, list))

    with pytest.raises(ValueError, match="list of str, none of them empty"):
        axiom_cls(**{**kwargs, list_field: ["ok", ""]})


@pytest.mark.parametrize("axiom_cls", AXIOM_SAMPLES, ids=lambda c: c.__name__)
def test_referenced_ids_lists_the_node_ids_by_field(axiom_cls):
    kwargs = AXIOM_SAMPLES[axiom_cls]
    refs = axiom_cls(**kwargs).referenced_ids()

    if axiom_cls is ImportAxiom:
        assert refs == {}, "an import names ontologies, not nodes"
    else:
        assert refs == {k: (v if isinstance(v, list) else [v]) for k, v in kwargs.items()}


@pytest.mark.parametrize("axiom_cls", AXIOM_SAMPLES, ids=lambda c: c.__name__)
def test_validate_flags_every_axiom_id_that_names_no_node(axiom_cls):
    kwargs = AXIOM_SAMPLES[axiom_cls]
    kg = KnowledgeGraph(name="graph", axioms=[axiom_cls(**kwargs)])

    result = kg.validate()

    expected = sum(len(ids) for ids in axiom_cls(**kwargs).referenced_ids().values())
    assert result["success"] is True, "a dangling axiom reference is a warning, not an error"
    assert len(result["warnings"]) == expected
    assert all("is not a node in the graph" in w for w in result["warnings"])


def test_validate_is_quiet_when_every_axiom_id_is_a_node():
    kg = KnowledgeGraph(name="graph", nodes={KGClass(id="c1"), KGClass(id="c2")},
                        axioms=[EquivalentClassesAxiom(class_ids=["c1", "c2"])])

    assert kg.validate() == {"success": True, "errors": [], "warnings": []}


# ---------------------------------------------------------------------------
# label / iri types, and metadata shared by nodes and edges
# ---------------------------------------------------------------------------

_MAKERS = {
    "node": lambda: KGClass(id="c1", label="C"),
    "edge": lambda: KGEdge(id="e", source=KGClass(id="a"), target=KGClass(id="b")),
}


@pytest.mark.parametrize("kind", _MAKERS)
@pytest.mark.parametrize("field_name", ["label", "iri"])
def test_label_and_iri_must_be_strings(kind, field_name):
    thing = _MAKERS[kind]()

    with pytest.raises(ValueError, match=f"{field_name} must be a str or None"):
        setattr(thing, field_name, 5)
    setattr(thing, field_name, None)
    assert getattr(thing, field_name) == ("" if field_name == "label" else None)


def test_a_non_string_label_is_refused_by_the_constructor():
    with pytest.raises(ValueError, match="label"):
        KGIndividual(id="i1", label=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", _MAKERS)
def test_nested_metadata_cannot_be_changed_through_the_getter(kind):
    """The read-only view used to be shallow on plain nodes and edges."""
    thing = _MAKERS[kind]()
    thing.metadata = {"members": ["a"], "nested": {"k": 1}}

    thing.metadata["members"].append("b")
    thing.metadata["nested"]["k"] = 2

    assert dict(thing.metadata) == {"members": ["a"], "nested": {"k": 1}}


@pytest.mark.parametrize("kind", _MAKERS)
def test_the_metadata_passed_in_is_deep_copied(kind):
    thing = _MAKERS[kind]()
    mine = {"members": ["a"]}
    thing.metadata = mine

    mine["members"].append("b")

    assert thing.metadata["members"] == ["a"]


@pytest.mark.parametrize("kind", _MAKERS)
def test_metadata_helpers_work_on_nodes_and_edges_alike(kind):
    thing = _MAKERS[kind]()

    thing.set_metadata_item("a", 1)
    thing.update_metadata({"b": 2})

    assert dict(thing.metadata) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match=f"{type(thing).__name__}.metadata must be a dict"):
        thing.metadata = ["not", "a", "mapping"]
