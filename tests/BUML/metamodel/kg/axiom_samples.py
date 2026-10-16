"""One fully-populated example of every axiom class, shared by the KG tests.

``test_every_axiom_class_has_a_sample`` fails when a new ``KGAxiom`` subclass is
added without one, so the tests that iterate :data:`AXIOM_SAMPLES` cannot
silently skip it (the hand-written lists they replace skipped two classes).
"""

from __future__ import annotations

from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    DisjointUnionAxiom,
    EquivalentClassesAxiom,
    EquivalentPropertiesAxiom,
    HasKeyAxiom,
    ImportAxiom,
    InversePropertiesAxiom,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)

#: axiom class -> constructor kwargs naming *every* field, with distinct values.
AXIOM_SAMPLES = {
    EquivalentClassesAxiom: {"class_ids": ["c1", "c2"]},
    EquivalentPropertiesAxiom: {"property_ids": ["p1", "p2"]},
    DisjointClassesAxiom: {"class_ids": ["c1", "c3"]},
    DisjointUnionAxiom: {"union_class_id": "c1", "part_class_ids": ["c2", "c3"]},
    SubPropertyOfAxiom: {"sub_property_id": "p1", "super_property_id": "p2"},
    InversePropertiesAxiom: {"property_a_id": "p1", "property_b_id": "p2"},
    PropertyChainAxiom: {"property_id": "p1", "chain_property_ids": ["p2", "p3"]},
    HasKeyAxiom: {"class_id": "c1", "property_ids": ["p1", "p2"]},
    ImportAxiom: {"target_iri": "http://ex.org/other", "source_ontology_iri": "http://ex.org/this"},
}
