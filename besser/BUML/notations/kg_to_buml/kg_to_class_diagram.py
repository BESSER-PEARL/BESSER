"""KG → BUML Class Diagram (TBox extraction).

``kg_to_class_diagram(kg)`` turns a :class:`KnowledgeGraph` into a
:class:`DomainModel` enriched with OCL :class:`Constraint`\\ s, implementing the
rule tables of *"Translating OWL 2 Ontologies and SHACL Shapes into UML/OCL
Models"* (KGMDSE 2026).

The work is split across three stages, each independently testable:

1. :mod:`~besser.BUML.notations.kg_rdf.projection` projects the KG — whether
   it came from a TTL import or was drawn in the editor — into an in-memory
   RDF graph.
2. :mod:`~besser.BUML.notations.kg_to_buml.owl2uml` applies the paper's D, O
   and S rules to that graph, producing a pure-data ``ModelPlan``.
3. :mod:`~besser.BUML.notations.kg_to_buml.to_buml` lowers the plan
   onto BUML, absorbing the metamodel's validation rules (association end-name
   uniqueness, the fixed primitive set, multiplicity bounds, OCL expression
   shape).

Both KG entry paths therefore share a single code path, and the OWL/SHACL
semantics live in one place rather than being re-derived from the KG's
node-and-edge representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from besser.BUML.metamodel.kg import KnowledgeGraph
from besser.BUML.notations.kg_to_buml._common import KGConversionWarning
from besser.BUML.notations.kg_rdf import kg_to_rdf
from besser.BUML.notations.kg_to_buml.owl2uml import plan_domain_model
from besser.BUML.notations.kg_to_buml.owl2uml import naming
from besser.BUML.notations.kg_to_buml.to_buml import ClassConversionResult, build_domain_model

if TYPE_CHECKING:  # avoids the import cycle the lazy import below sidesteps
    from besser.BUML.notations.kg_to_buml.resolutions import KGResolution

__all__ = ["ClassConversionResult", "kg_to_class_diagram"]


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

    Returns:
        ClassConversionResult: The domain model, the IRI maps the object
        diagram needs, and every warning (resolution notes included).
    """
    warnings: List[KGConversionWarning] = []
    if resolutions:
        from besser.BUML.notations.kg_to_buml.resolutions import apply_resolutions

        kg = apply_resolutions(kg, resolutions, warnings=warnings)

    graph, base = kg_to_rdf(kg, warnings=warnings)
    plan = plan_domain_model(graph, base, shapes=graph, warnings=warnings)
    result = build_domain_model(
        plan,
        model_name=_model_name(kg, model_name),
        emit_ocl=emit_ocl,
        warnings=warnings,
    )

    return result


def _model_name(kg: KnowledgeGraph, override: Optional[str]) -> str:
    raw = override or kg.name or "KGClassDiagram"
    return naming.sanitize(raw.strip()) or "KGClassDiagram"
