"""Preflight issue ids must be unique: the editor keys checkboxes, radio groups and
decisions by id, so two rows sharing one id corrupt each other's state."""

from __future__ import annotations

import os
import tempfile

from besser.BUML.notations.kg_to_buml.preflight import analyze_kg_for_class_diagram
from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph

_SYMMETRIC_INVERSES_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:Org a owl:Class .
ex:parentOrganization a owl:ObjectProperty ; rdfs:domain ex:Org ; rdfs:range ex:Org ;
    owl:inverseOf ex:subOrganization .
ex:subOrganization a owl:ObjectProperty ; rdfs:domain ex:Org ; rdfs:range ex:Org ;
    owl:inverseOf ex:parentOrganization .
"""


def test_inverse_declared_in_both_directions_yields_one_issue():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "inverses.ttl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_SYMMETRIC_INVERSES_TTL)
        report = analyze_kg_for_class_diagram(owl_file_to_knowledge_graph(path))

    ids = [issue.id for issue in report.issues]
    assert len(ids) == len(set(ids))
    assert sum(1 for i in report.issues if i.code == "INVERSE_PROPERTY") == 1
    assert report.issue_count == len(report.issues)


def test_keep_as_is_decision_is_not_reported_again_after_round_trip():
    """Accepting a "keep" recommendation changes nothing in the graph, so the review has to
    be recorded on it — otherwise the same issue comes back after every refinement."""
    from besser.BUML.notations.kg_to_buml.resolutions import dispatch_decision
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.kg_diagram_converter import (
        kg_to_json,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.kg_diagram_processor import (
        process_kg_diagram,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "inverses.ttl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_SYMMETRIC_INVERSES_TTL)
        kg = owl_file_to_knowledge_graph(path)

    issue = next(i for i in analyze_kg_for_class_diagram(kg).issues if i.code == "INVERSE_PROPERTY")
    assert issue.recommended_action.key == "keep_separate"

    resolved = dispatch_decision(kg, issue, "accept")
    assert not any(i.code == "INVERSE_PROPERTY" for i in analyze_kg_for_class_diagram(resolved).issues)

    # The frontend holds the KG as JSON between calls: the review must survive that.
    reloaded = process_kg_diagram(kg_to_json(resolved))
    assert not any(i.code == "INVERSE_PROPERTY" for i in analyze_kg_for_class_diagram(reloaded).issues)

    # The input graph is untouched, and the issue is still reported there.
    assert any(i.code == "INVERSE_PROPERTY" for i in analyze_kg_for_class_diagram(kg).issues)
