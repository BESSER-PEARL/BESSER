"""End-to-end tests for the Knowledge Graph backend.

Covers:
  1. rdflib → KnowledgeGraph → JSON produces the expected shape.
  2. JSON → process_kg_diagram → kg_to_json is an identity (round-trip).
  3. POST /import-owl returns a DiagramExportResponse that /validate-diagram accepts.
  4. A project holding a KG survives project_to_code → project_to_json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph
from besser.utilities.web_modeling_editor.backend.services.converters import (
    kg_to_json,
    process_kg_diagram,
)


TTL_FIXTURE = """
@prefix : <http://ex.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:Person a owl:Class ; rdfs:label "Person" .
:knows  a owl:ObjectProperty ; rdfs:label "knows" .
:age    a owl:DatatypeProperty .
:alice  a :Person ; :knows :bob ; :age "30"^^xsd:integer ; rdfs:label "Alice" .
:bob    a :Person .
_:b0    a :Person .
""".strip()


@pytest.fixture
def ttl_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.ttl"
    p.write_text(TTL_FIXTURE, encoding="utf-8")
    return p


def test_owl_to_kg_classification(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))

    # At minimum: 1 class (Person), 2 individuals (alice, bob), 2 properties
    # (knows, age), 1 literal with value "30", and 1 blank node.
    kinds = {}
    for n in kg.nodes:
        kinds.setdefault(type(n).__name__, []).append(n)
    assert "KGClass" in kinds and any(n.id.endswith("Person") for n in kinds["KGClass"])
    assert "KGIndividual" in kinds and any(n.id.endswith("alice") for n in kinds["KGIndividual"])
    assert "KGProperty" in kinds and any(n.id.endswith("knows") for n in kinds["KGProperty"])
    assert "KGBlank" in kinds
    assert "KGLiteral" in kinds
    assert any(n.value == "30" for n in kinds["KGLiteral"])

    # One edge per triple in the fixture.
    assert len(kg.edges) == 11


def test_kg_json_round_trip(ttl_path: Path):
    kg = owl_file_to_knowledge_graph(str(ttl_path))
    j1 = kg_to_json(kg)
    kg2 = process_kg_diagram(j1)
    j2 = kg_to_json(kg2)
    assert j1 == j2


# The tiny fixture above carries no axioms, which is why a missing entry in the
# processor's axiom table went unnoticed: EquivalentPropertiesAxiom was dropped
# on the way back in and bibo.ttl returned 51 of its 61 axioms. These two
# fixtures do carry axioms -- the same pair test_kg_to_rdf.py parametrizes over.
_AXIOM_BEARING_FIXTURES = [
    Path(__file__).resolve().parents[3] / "fixtures" / "complex_kg.ttl",
    Path(__file__).resolve().parents[3]
    / "BUML" / "notations" / "kg_to_buml" / "fixtures" / "bibo" / "bibo.ttl",
]


@pytest.mark.parametrize("fixture", _AXIOM_BEARING_FIXTURES, ids=lambda p: p.name)
def test_kg_json_round_trip_preserves_axioms(fixture: Path):
    assert fixture.is_file(), f"missing fixture: {fixture}"
    kg = owl_file_to_knowledge_graph(str(fixture))
    assert kg.axioms, "fixture carries no axioms, so it cannot guard this"

    j1 = kg_to_json(kg)
    kg2 = process_kg_diagram(j1)
    j2 = kg_to_json(kg2)

    assert len(kg2.axioms) == len(kg.axioms)
    assert sorted(type(a).__name__ for a in kg2.axioms) == sorted(
        type(a).__name__ for a in kg.axioms
    )
    assert j1 == j2


def test_every_axiom_subclass_round_trips():
    """No hand-maintained list to drift: the table is derived, so assert that."""
    from besser.BUML.metamodel.kg import KGAxiom
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.kg_diagram_processor import (  # noqa: E501
        _AXIOM_KIND_TO_CLASS,
    )

    assert set(_AXIOM_KIND_TO_CLASS) == {c.__name__ for c in KGAxiom.__subclasses__()}


def test_malformed_axiom_raises_instead_of_vanishing():
    from besser.utilities.web_modeling_editor.backend.services.exceptions import (
        ConversionError,
    )

    base = {"type": "KnowledgeGraphDiagram", "version": "1.0.0", "nodes": [], "edges": []}

    with pytest.raises(ConversionError, match="Unknown KG axiom kind"):
        process_kg_diagram({"model": {**base, "axioms": [{"kind": "NoSuchAxiom"}]}})

    with pytest.raises(ConversionError, match="missing 'kind'"):
        process_kg_diagram({"model": {**base, "axioms": [{"class_ids": ["A", "B"]}]}})

    with pytest.raises(ConversionError, match="Malformed"):
        process_kg_diagram(
            {"model": {**base, "axioms": [{"kind": "HasKeyAxiom", "bogus_field": 1}]}}
        )

    # A known field of the wrong type is as malformed as an unknown one: the
    # axiom's validating setters reject it, and it must not vanish either.
    with pytest.raises(ConversionError, match="Malformed 'HasKeyAxiom'.*class_id"):
        process_kg_diagram(
            {"model": {**base, "axioms": [{"kind": "HasKeyAxiom", "class_id": 7}]}}
        )
    with pytest.raises(ConversionError, match="Malformed 'EquivalentClassesAxiom'.*list of str"):
        process_kg_diagram(
            {"model": {**base, "axioms": [{"kind": "EquivalentClassesAxiom", "class_ids": "AB"}]}}
        )


def test_import_owl_endpoint_and_validation(client, ttl_path: Path):
    with ttl_path.open("rb") as fh:
        resp = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("tiny.ttl", fh.read(), "text/turtle")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["diagramType"] == "KnowledgeGraphDiagram"
    model = body["model"]
    assert model["type"] == "KnowledgeGraphDiagram"
    assert isinstance(model["nodes"], list) and len(model["nodes"]) >= 6
    assert isinstance(model["edges"], list) and len(model["edges"]) == 11

    # The validation endpoint runs the same checks as the Refine KG modal.
    # This fixture deliberately contains `_:b0 a :Person`, so it reports a
    # BLANK_NODE_INSTANCE finding rather than passing.
    valid_resp = client.post(
        "/besser_api/validate-diagram",
        json={"title": body["title"], "model": model},
    )
    assert valid_resp.status_code == 200
    payload = valid_resp.json()
    assert payload["isValid"] is False
    assert any("BLANK_NODE_INSTANCE" in e for e in payload["errors"])


def test_import_owl_rejects_unsupported_extension(client, tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("not an ontology", encoding="utf-8")
    with p.open("rb") as fh:
        resp = client.post(
            "/besser_api/import-owl",
            files={"owl_file": ("bad.txt", fh.read(), "text/plain")},
        )
    assert resp.status_code == 415


# ----------------------------------------------------------------------
# Single-diagram round trip
# ----------------------------------------------------------------------


def test_single_diagram_buml_round_trip(client, ttl_path: Path):
    """A lone KG must survive /export-buml -> /get-json-model.

    Every other diagram type round-trips at both the project *and* the
    single-diagram level; the KG only had the project half, so /export-buml
    fell through to "Unsupported or missing diagram type" and
    /get-json-model's type sniffing had no KG branch at all.
    """
    imported = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("tiny.ttl", ttl_path.read_bytes(), "text/turtle")},
    )
    assert imported.status_code == 200, imported.text
    model = imported.json()["model"]

    exported = client.post(
        "/besser_api/export-buml",
        json={"title": "Tiny KG", "model": model, "generator": "buml"},
    )
    assert exported.status_code == 200, exported.text
    source = exported.content.decode("utf-8")
    assert "KnowledgeGraph(" in source

    reimported = client.post(
        "/besser_api/get-json-model",
        files={"buml_file": ("knowledge_graph.py", source.encode("utf-8"), "text/x-python")},
    )
    assert reimported.status_code == 200, reimported.text
    body = reimported.json()

    assert body["diagramType"] == "KnowledgeGraphDiagram"
    assert body["model"]["type"] == "KnowledgeGraphDiagram"
    assert {n["id"] for n in body["model"]["nodes"]} == {n["id"] for n in model["nodes"]}
    assert {e["id"] for e in body["model"]["edges"]} == {e["id"] for e in model["edges"]}


def test_export_buml_rejects_unknown_diagram_type(client):
    """The terminal else-branch must still reject genuinely unknown types."""
    resp = client.post(
        "/besser_api/export-buml",
        json={"title": "x", "model": {"type": "NotADiagram"}, "generator": "buml"},
    )
    assert resp.status_code == 400
    assert "NotADiagram" in resp.json()["detail"]


# ----------------------------------------------------------------------
# Project-level round trip
# ----------------------------------------------------------------------


def test_project_export_preserves_knowledge_graph(tmp_path: Path):
    """A KG in a project must survive the BUML source round trip.

    Before the KG code builder existed, ``project_to_code`` had no bucket for
    ``KnowledgeGraph`` and dropped it silently, while ``project_to_json``
    already knew how to read a ``KNOWLEDGE_GRAPH`` section that nothing ever
    wrote. This pins both halves together.
    """
    from besser.BUML.metamodel.project import Project
    from besser.BUML.metamodel.structural import Class, DomainModel, Metadata
    from besser.utilities.buml_code_builder.project_builder import project_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        project_to_json,
    )

    kg = owl_file_to_knowledge_graph(str(_write_ttl(tmp_path)))
    expected_nodes = len(kg.nodes)
    expected_edges = len(kg.edges)

    domain_model = DomainModel(name="dm")
    domain_model.add_type(Class(name="Person", attributes=set()))

    project = Project(
        name="kg_project",
        models=[domain_model, kg],
        metadata=Metadata(description="round trip"),
    )

    out = tmp_path / "project.py"
    project_to_code(project, str(out))
    content = out.read_text(encoding="utf-8")
    assert "KNOWLEDGE_GRAPH MODEL" in content

    result = project_to_json(content)
    kg_diagrams = result["diagrams"].get("KnowledgeGraphDiagram") or []
    assert len(kg_diagrams) == 1

    model = kg_diagrams[0]["model"]
    assert model["type"] == "KnowledgeGraphDiagram"
    assert len(model["nodes"]) == expected_nodes
    assert len(model["edges"]) == expected_edges

    # The sibling class diagram must be unaffected.
    assert len(result["diagrams"].get("ClassDiagram") or []) == 1


def _write_ttl(tmp_path: Path) -> Path:
    path = tmp_path / "roundtrip.ttl"
    path.write_text(TTL_FIXTURE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Security: uploaded .py is loaded through the AST allowlist, never exec()'d
# ---------------------------------------------------------------------------


def test_kg_buml_loader_blocks_subclass_walk_escape():
    """The classic restricted-builtins escape must not reach the interpreter.

    ``().__class__.__base__.__subclasses__()`` needs no builtin name at all, so
    the previous ``exec`` with a trimmed ``__builtins__`` let it through. It is
    the payload that motivated moving KG onto ``safe_load_buml``.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        kg_buml_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.exceptions import (
        ConversionError,
    )

    payload = (
        "kg_model = KnowledgeGraph(name='x')\n"
        "stolen = ().__class__.__base__.__subclasses__()\n"
    )
    with pytest.raises(ConversionError):
        kg_buml_to_json(payload)


def test_kg_buml_loader_blocks_dunder_attribute_access():
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        kg_buml_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.exceptions import (
        ConversionError,
    )

    payload = (
        "kg_model = KnowledgeGraph(name='x')\n"
        "leak = kg_model.__class__.__init__.__globals__\n"
    )
    with pytest.raises(ConversionError):
        kg_buml_to_json(payload)


def test_kg_buml_loader_still_loads_generated_code(tmp_path: Path):
    """The allowlist must not cost us the real round-trip it exists to protect."""
    from besser.utilities.buml_code_builder import kg_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        kg_buml_to_json,
    )

    kg = owl_file_to_knowledge_graph(str(_write_ttl(tmp_path)))
    out = tmp_path / "kg_model.py"
    kg_model_to_code(model=kg, file_path=str(out), model_var_name="kg_model")

    model = kg_buml_to_json(out.read_text(encoding="utf-8"))

    assert model["type"] == "KnowledgeGraphDiagram"
    assert {n["id"] for n in model["nodes"]} == {n.id for n in kg.nodes}
    assert len(model["edges"]) == len(kg.edges)


def _write_ttl(tmp_path: Path) -> Path:
    p = tmp_path / "loader.ttl"
    p.write_text(TTL_FIXTURE, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Resource limits: a DTD must be refused before rdflib expands it
# ---------------------------------------------------------------------------


# Nested internal entities: ~200 bytes that expand to roughly 3**9 copies of
# the payload. rdflib builds its RDF/XML parser with xml.sax.make_parser() and
# never caps expansion, so the cost is paid synchronously inside the request.
_BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE rdf:RDF [
  <!ENTITY a "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;"> <!ENTITY c "&b;&b;&b;"> <!ENTITY d "&c;&c;&c;">
  <!ENTITY e "&d;&d;&d;"> <!ENTITY f "&e;&e;&e;"> <!ENTITY g "&f;&f;&f;">
  <!ENTITY h "&g;&g;&g;"> <!ENTITY i "&h;&h;&h;"> <!ENTITY j "&i;&i;&i;">
]>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <rdf:Description rdf:about="http://ex.org/Boom">
    <rdfs:label>&j;</rdfs:label>
  </rdf:Description>
</rdf:RDF>
"""


def test_import_owl_rejects_a_dtd_before_parsing(client):
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("bomb.owl", _BILLION_LAUGHS.encode("utf-8"), "application/rdf+xml")},
    )

    assert resp.status_code == 400
    assert "DTD" in resp.json()["detail"]


def test_import_owl_still_accepts_a_plain_ontology(client, ttl_path: Path):
    """The DTD guard must not reject the ontologies we actually import."""
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("tiny.ttl", ttl_path.read_bytes(), "text/turtle")},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["diagramType"] == "KnowledgeGraphDiagram"


def test_import_owl_returns_the_anomalies_it_skipped(client):
    from tests.utilities.test_owl_importer_constructs import _BAD_QUALIFIED_COUNT_TTL

    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("bad.ttl", _BAD_QUALIFIED_COUNT_TTL.encode("utf-8"), "text/turtle")},
    )

    assert resp.status_code == 200, resp.text
    assert [w["code"] for w in resp.json()["warnings"]] == ["SHACL_QUALIFIED_COUNT_INVALID"]


def test_import_owl_has_an_empty_warning_list_when_nothing_was_skipped(client, ttl_path: Path):
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("tiny.ttl", ttl_path.read_bytes(), "text/turtle")},
    )

    assert resp.json()["warnings"] == []
