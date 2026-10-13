"""Feed the explicit fixture ontologies through the pipeline so the recording
plugin captures them. Only meaningful under ``-p tests.kg_golden.plugin``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("KG_GOLDEN_DIR"), reason="KG_GOLDEN_DIR not set")

_ROOT = Path(__file__).resolve().parents[1]
_KG = _ROOT / "BUML" / "notations" / "kg_to_buml"

SEEDS = [
    _ROOT / "fixtures" / "complex_kg.ttl",
    _KG / "fixtures" / "bibo" / "bibo.ttl",
    _KG / "kg_example.ttl",
]


def _run_all(kg) -> None:
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
    from besser.BUML.notations.kg_to_buml.consistency import check_kg_consistency
    from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    kg_to_class_diagram(kg)
    kg_to_rdf(kg)
    knowledge_graph_to_rdf(kg, vocab="both")
    # The KG is recorded on entry, so a pipeline stage that raises on it (as
    # pyshacl does on bibo + its shapes: a property shape without sh:path) still
    # lands in the corpus -- and its exception becomes part of the golden.
    try:
        check_kg_consistency(kg, inference="rdfs")
    except Exception:
        pass


@pytest.mark.parametrize("path", SEEDS, ids=lambda p: p.name)
def test_seed_fixture(path: Path):
    from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

    _run_all(owl_file_to_knowledge_graph(str(path)))


def test_seed_bibo_with_shapes(tmp_path: Path):
    """bibo + its SHACL shapes in one graph, as the determinism test builds it."""
    from rdflib import Graph

    from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

    graph = Graph()
    graph.parse(str(_KG / "fixtures" / "bibo" / "bibo.ttl"), format="turtle")
    graph.parse(str(_KG / "fixtures" / "bibo" / "bibo-shapes.ttl"), format="turtle")
    combined = tmp_path / "bibo-combined.ttl"
    graph.serialize(destination=str(combined), format="turtle")
    _run_all(owl_file_to_knowledge_graph(str(combined)))
