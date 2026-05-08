"""KG → BUML transformations.

Deterministic, offline (no LLM) converters that turn a ``KnowledgeGraph``
metamodel instance into BUML structural / object models:

* :func:`kg_to_class_diagram` — TBox extraction (KGClass → Class,
  KGProperty → Attribute or BinaryAssociation, ``rdfs:subClassOf`` →
  Generalization).  ABox is also scanned to bump multi-valued literal
  property multiplicities to ``0..*``.
* :func:`kg_to_object_diagram` — ABox extraction (KGIndividual → Object,
  literal-target edges → AttributeLink slots, individual-target edges →
  Links). Blank nodes are skipped.

For ambiguous KGs, run :func:`analyze_kg_for_class_diagram` first to
surface preflight issues; let the user pick :class:`KGResolution`
choices, then pass them via ``resolutions=`` to the converter.
"""

from besser.BUML.notations.kg_to_buml._common import KGConversionWarning
from besser.BUML.notations.kg_to_buml.kg_to_class_diagram import (
    ClassConversionResult,
    kg_to_class_diagram,
)
from besser.BUML.notations.kg_to_buml.kg_to_object_diagram import (
    ObjectConversionResult,
    kg_to_object_diagram,
)
from besser.BUML.notations.kg_to_buml.llm_cleanup import (
    LLM_ALLOWED_KEYS,
    ORPHAN_ALLOWED_KEYS,
    analyze_kg_with_description,
    build_kg_snapshot,
    classify_orphan_nodes_with_llm,
    parse_llm_suggestions,
)
from besser.BUML.notations.kg_to_buml.preflight import (
    KGAction,
    KGIssue,
    KGPreflightReport,
    analyze_kg_for_class_diagram,
    analyze_kg_for_object_diagram,
    kg_signature,
)
from besser.BUML.notations.kg_to_buml.resolutions import (
    DeferredOrphanClassification,
    KGResolution,
    ResolutionError,
    apply_resolutions,
    dispatch_decision,
)

__all__ = [
    "KGConversionWarning",
    "ClassConversionResult",
    "ObjectConversionResult",
    "kg_to_class_diagram",
    "kg_to_object_diagram",
    "KGAction",
    "KGIssue",
    "KGPreflightReport",
    "analyze_kg_for_class_diagram",
    "analyze_kg_for_object_diagram",
    "kg_signature",
    "KGResolution",
    "ResolutionError",
    "DeferredOrphanClassification",
    "apply_resolutions",
    "dispatch_decision",
    "LLM_ALLOWED_KEYS",
    "ORPHAN_ALLOWED_KEYS",
    "analyze_kg_with_description",
    "build_kg_snapshot",
    "classify_orphan_nodes_with_llm",
    "parse_llm_suggestions",
]
