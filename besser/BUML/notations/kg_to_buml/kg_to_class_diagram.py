"""KG → BUML Class Diagram (TBox extraction).

``kg_to_class_diagram(kg)`` turns a :class:`KnowledgeGraph` into a
:class:`DomainModel` enriched with OCL :class:`Constraint`\\ s, implementing the
rule tables of *"Translating OWL 2 Ontologies and SHACL Shapes into UML/OCL
Models"* (KGMDSE 2026).

The work is split across three stages, each independently testable:

1. :mod:`~besser.BUML.notations.kg_to_buml.kg_to_rdf` projects the KG — whether
   it came from a TTL import or was drawn in the editor — into an in-memory
   RDF graph.
2. :mod:`~besser.BUML.notations.kg_to_buml.owl2uml` applies the paper's D, O
   and S rules to that graph, producing a pure-data ``UMLModel``.
3. :mod:`~besser.BUML.notations.kg_to_buml.to_buml` lowers the ``UMLModel``
   onto BUML, absorbing the metamodel's validation rules (association end-name
   uniqueness, the fixed primitive set, multiplicity bounds, OCL expression
   shape).

Both KG entry paths therefore share a single code path, and the OWL/SHACL
semantics live in one place rather than being re-derived from the KG's
node-and-edge representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from besser.BUML.metamodel.kg import KnowledgeGraph
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Property,
)

from besser.BUML.notations.kg_to_buml._common import KGConversionWarning
from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import build_uml_model
from besser.BUML.notations.kg_to_buml.owl2uml import naming
from besser.BUML.notations.kg_to_buml.to_buml import lower_to_buml

if TYPE_CHECKING:  # avoids the import cycle the lazy import below sidesteps
    from besser.BUML.notations.kg_to_buml.resolutions import KGResolution

__all__ = ["ClassConversionResult", "kg_to_class_diagram"]


@dataclass
class ClassConversionResult:
    """Output of :func:`kg_to_class_diagram`."""

    domain_model: DomainModel
    iri_to_class: Dict[str, Class] = field(default_factory=dict)
    property_iri_to_attribute: Dict[str, Property] = field(default_factory=dict)
    property_iri_to_association: Dict[str, BinaryAssociation] = field(default_factory=dict)
    # For each derived BinaryAssociation, the Property end that represents the
    # *source* (KG triple subject) side. Important for self-referential
    # associations where both ends share the same type and would otherwise be
    # indistinguishable.
    assoc_source_end: Dict[int, Property] = field(default_factory=dict)
    warnings: List[KGConversionWarning] = field(default_factory=list)
    # Classifications BUML has nowhere to store — class name → ["aux"],
    # ["external"], ["deprecated"], ["dataType"]. See ``to_buml._report_stereotypes``.
    class_stereotypes: Dict[str, List[str]] = field(default_factory=dict)


def kg_to_class_diagram(
    kg: KnowledgeGraph,
    *,
    model_name: Optional[str] = None,
    resolutions: Optional[List["KGResolution"]] = None,
    emit_ocl: bool = True,
) -> ClassConversionResult:
    """Convert ``kg`` into a BUML class diagram enriched with OCL constraints.

    Args:
        kg: The knowledge graph to convert.
        model_name: Name for the resulting ``DomainModel``; defaults to the
            KG's own name.
        resolutions: Optional preflight resolutions to apply first. They are
            applied to a deep copy, so ``kg`` is never mutated.
        emit_ocl: When ``False``, produce the structure only and skip every
            OCL invariant.
    """
    if resolutions:
        from besser.BUML.notations.kg_to_buml.resolutions import apply_resolutions

        kg = apply_resolutions(kg, resolutions)

    warnings: List[KGConversionWarning] = []
    graph, base = kg_to_rdf(kg, warnings=warnings)
    uml_model = build_uml_model(graph, base, shapes=graph, warnings=warnings)
    lowered = lower_to_buml(
        uml_model,
        model_name=_model_name(kg, model_name),
        emit_ocl=emit_ocl,
        warnings=warnings,
    )

    return ClassConversionResult(
        domain_model=lowered.domain_model,
        iri_to_class=lowered.iri_to_class,
        property_iri_to_attribute=lowered.property_iri_to_attribute,
        property_iri_to_association=lowered.property_iri_to_association,
        assoc_source_end=lowered.assoc_source_end,
        warnings=warnings,
        class_stereotypes=lowered.class_stereotypes,
    )


def _model_name(kg: KnowledgeGraph, override: Optional[str]) -> str:
    raw = override or kg.name or "KGClassDiagram"
    return naming.sanitize(raw.strip()) or "KGClassDiagram"
