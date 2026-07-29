"""OWL 2 / SHACL → UML+OCL transformation rules (ported from the KG2UML paper).

This subpackage is a near-verbatim port of the reference implementation
accompanying *"Translating OWL 2 Ontologies and SHACL Shapes into UML/OCL
Models"* (Dalle Lucca Tosi, Ul Haq, Cabot — KGMDSE 2026). It implements the
paper's three rule tables:

* **D01–D41** — OWL 2 constructs that map directly onto UML.
* **O01–O30** — OWL 2 constructs that need a UML element *plus* an OCL
  invariant (class expressions, restrictions, property characteristics).
* **S01–S31** — SHACL shapes translated into additional OCL invariants over
  the classes the OWL phase already produced.

It reads an :class:`rdflib.Graph` and produces the pure-data
:class:`~besser.BUML.notations.kg_to_buml.owl2uml.model.UMLModel` — no BESSER
metamodel types are involved. Lowering ``UMLModel`` onto a BUML
``DomainModel`` is the job of
:mod:`besser.BUML.notations.kg_to_buml.to_buml`, and projecting a
:class:`~besser.BUML.metamodel.kg.KnowledgeGraph` into the input graph is the
job of :mod:`besser.BUML.notations.kg_to_buml.kg_to_rdf`.

Deviations from the reference are deliberate and confined to:

1. **Determinism.** The reference iterates rdflib result *sets*, so its output
   varies between runs (measured: three different digests over three runs on
   the same input). Every iteration point is now sorted.
2. **Datatype vocabulary.** ``primitive_for`` resolves to BUML's nine
   primitives (``str``/``int``/``float``/…) via
   :mod:`~besser.BUML.notations.kg_to_buml.datatype_mapping` rather than the
   paper's ``String``/``Integer``/``URI``/``GYear``/… names, so every type
   named in a generated OCL body is a real BUML type.
3. **Diagnostics.** ``sys.stderr`` prints become
   :class:`~besser.BUML.notations.kg_to_buml._common.KGConversionWarning`
   entries on a caller-supplied sink.
4. **Dead options dropped.** The reference's ``annotations``,
   ``emit_external``, ``enum_from_instances`` and ``neg_assertions`` flags were
   accepted but never read; ``emit_ocl`` only gated PlantUML rendering and now
   lives on the public converter entry point.
"""

from __future__ import annotations

from typing import List, Optional

from besser.BUML.notations.kg_to_buml._common import KGConversionWarning
from besser.BUML.notations.kg_to_buml.owl2uml.model import (
    Association,
    AssociationEnd,
    Attribute,
    Class,
    Comment,
    DataType,
    Enumeration,
    Generalization,
    InstanceSpecification,
    Link,
    OCLConstraint,
    Slot,
    UMLModel,
)


def build_uml_model(
    graph,
    base: Optional[str] = None,
    *,
    shapes=None,
    warnings: Optional[List[KGConversionWarning]] = None,
) -> UMLModel:
    """Apply the D/O/S rules to ``graph`` and return the intermediate UML model.

    Args:
        graph: The OWL 2 graph, as an :class:`rdflib.Graph`.
        base: Base namespace of the ontology, used to tell "foreign" classes
            from local ones. ``None`` disables the distinction.
        shapes: Optional SHACL shapes graph. When the shapes live in the same
            graph as the ontology (which is what
            :func:`~besser.BUML.notations.kg_to_buml.kg_to_rdf.kg_to_rdf`
            produces), pass the same graph for both arguments — the SHACL
            phase only looks at ``sh:NodeShape`` subjects.
        warnings: Optional sink for non-fatal diagnostics.
    """
    from besser.BUML.notations.kg_to_buml.owl2uml.mapping import Mapper

    return Mapper(graph, base, shapes=shapes, warnings=warnings).run()


__all__ = [
    "build_uml_model",
    "UMLModel",
    "Class",
    "Attribute",
    "DataType",
    "Enumeration",
    "Association",
    "AssociationEnd",
    "Generalization",
    "OCLConstraint",
    "InstanceSpecification",
    "Slot",
    "Link",
    "Comment",
]
