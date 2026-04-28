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

__all__ = [
    "KGConversionWarning",
    "ClassConversionResult",
    "ObjectConversionResult",
    "kg_to_class_diagram",
    "kg_to_object_diagram",
]
