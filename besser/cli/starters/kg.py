"""A starter B-UML model: a knowledge graph of authors and the books they wrote.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/knowledge_graph.html
"""

from besser.BUML.metamodel.kg import (
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.constants import (
    RDF,
    RDFS,
    XSD,
)

# Namespace of this graph's own IRIs
EX = "http://example.org/library#"

# Classes: the concepts. Every node has a unique id, a display label and, usually, an IRI.
person = KGClass(id="Person", label="Person", iri=f"{EX}Person")
author = KGClass(id="Author", label="Author", iri=f"{EX}Author")
book = KGClass(id="Book", label="Book", iri=f"{EX}Book")

# Property: a relation that is itself described in the graph (here by its domain and range)
wrote = KGProperty(id="wrote", label="wrote", iri=f"{EX}wrote")

# Individuals: the instances
jane_austen = KGIndividual(id="jane_austen", label="Jane Austen", iri=f"{EX}jane_austen")
emma = KGIndividual(id="emma", label="Emma", iri=f"{EX}emma")

# Literal: a value, with the IRI of its datatype
emma_pages = KGLiteral(id="emma_pages", value="474", datatype=f"{XSD}integer")

# Edges: directed, from source to target; the iri is the predicate
edges = {
    KGEdge(id="author_is_a_person", source=author, target=person, label="subClassOf", iri=f"{RDFS}subClassOf"),
    KGEdge(id="wrote_domain", source=wrote, target=author, label="domain", iri=f"{RDFS}domain"),
    KGEdge(id="wrote_range", source=wrote, target=book, label="range", iri=f"{RDFS}range"),
    KGEdge(id="austen_is_an_author", source=jane_austen, target=author, label="type", iri=f"{RDF}type"),
    KGEdge(id="emma_is_a_book", source=emma, target=book, label="type", iri=f"{RDF}type"),
    KGEdge(id="austen_wrote_emma", source=jane_austen, target=emma, label="wrote", iri=f"{EX}wrote"),
    KGEdge(id="emma_has_pages", source=emma, target=emma_pages, label="pages", iri=f"{EX}pages"),
}

# The model. The besser command reads every model bound to a top-level name of this file.
library_graph = KnowledgeGraph(
    name="LibraryGraph",
    nodes={person, author, book, wrote, jane_austen, emma, emma_pages},
    edges=edges,
)
