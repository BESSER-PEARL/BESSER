"""Stable IRI constants used by the KG metamodel.

These constants are referenced by:

- the OWL/SHACL importer when classifying nodes and edges,
- the JSON converters when serialising edges to the frontend,
- the RDF exporter when translating internal edges into vocabulary-appropriate
  predicates (or skipping them, in the case of ``BESSER_KG_NS`` edges, which
  are internal-only and never emitted to RDF).
"""

from __future__ import annotations

__all__ = [
    "BESSER_KG_NS",
    "SH",
    "OWL",
    "RDF",
    "RDFS",
    "XSD",
    "CONSTRAINT_TARGET_CLASS",
    "CONSTRAINT_TARGET_PROPERTY",
    "SH_PROPERTY",
    "SH_TARGET_CLASS",
    "SH_PATH",
    "SH_NODE_SHAPE",
    "SH_PROPERTY_SHAPE",
]


# BESSER's one namespace. It names the vocabulary BESSER owns -- the internal
# edges that link a reified constraint node to its class or property, which are
# never written to RDF -- and it is the base for the IRIs the exporter has to
# *mint* (a node with no IRI of its own, a synthesised shape), which do appear in
# exported files. The frontend declares the same constraint-link IRIs
# (``KG_CONSTRAINT_TARGET_*`` in ``shared/types/project.ts``); keep them in step.
BESSER_KG_NS = "http://besser-pearl.org/kg#"
SH = "http://www.w3.org/ns/shacl#"
OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"


CONSTRAINT_TARGET_CLASS = f"{BESSER_KG_NS}constraintTargetClass"
CONSTRAINT_TARGET_PROPERTY = f"{BESSER_KG_NS}constraintTargetProperty"

SH_PROPERTY = f"{SH}property"
SH_TARGET_CLASS = f"{SH}targetClass"
SH_PATH = f"{SH}path"
SH_NODE_SHAPE = f"{SH}NodeShape"
SH_PROPERTY_SHAPE = f"{SH}PropertyShape"
