"""Knowledge graph <-> RDF.

Everything that reads or writes RDF for a
:class:`~besser.BUML.metamodel.kg.KnowledgeGraph`:

* :mod:`~besser.BUML.notations.kg_rdf.importer`: an OWL / RDF / SHACL file ->
  knowledge graph (:func:`owl_file_to_knowledge_graph`).
* :mod:`~besser.BUML.notations.kg_rdf.exporter`: knowledge graph -> the
  user-facing Turtle / RDF-XML export (:func:`knowledge_graph_to_rdf`,
  :func:`serialize_knowledge_graph`).
* :mod:`~besser.BUML.notations.kg_rdf.projection`: knowledge graph -> the RDF
  graph the KG -> B-UML conversion and the consistency check reason over
  (:func:`kg_to_rdf`).

The exporter and the projection write through the same emitters
(:mod:`~besser.BUML.notations.kg_rdf._emit`). Turning a knowledge graph into a
class diagram is the job of :mod:`besser.BUML.notations.kg_to_buml`.
"""

from besser.BUML.notations.kg_rdf._emit import DEFAULT_NAMESPACE
from besser.BUML.notations.kg_rdf.exporter import (
    VocabChoice,
    knowledge_graph_to_rdf,
    serialize_knowledge_graph,
)
from besser.BUML.notations.kg_rdf.importer import owl_file_to_knowledge_graph
from besser.BUML.notations.kg_rdf.projection import kg_to_rdf

__all__ = [
    "DEFAULT_NAMESPACE",
    "VocabChoice",
    "kg_to_rdf",
    "knowledge_graph_to_rdf",
    "owl_file_to_knowledge_graph",
    "serialize_knowledge_graph",
]
