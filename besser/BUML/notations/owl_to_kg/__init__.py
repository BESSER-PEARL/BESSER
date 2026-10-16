"""OWL / RDF → :class:`~besser.BUML.metamodel.kg.KnowledgeGraph` import.

A file-format parser, so it lives with the other notations. It returns a
knowledge graph, not a B-UML model: turning that graph into a class diagram is
the job of :mod:`besser.BUML.notations.kg_to_buml`.
"""

from besser.BUML.notations.owl_to_kg.owl_to_kg import owl_file_to_knowledge_graph

__all__ = ["owl_file_to_knowledge_graph"]
