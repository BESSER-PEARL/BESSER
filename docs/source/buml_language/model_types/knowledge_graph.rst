Knowledge Graph model
=====================

The Knowledge Graph (KG) sublanguage represents an RDF-style graph: typed nodes
joined by directed, labelled edges. It is the model behind the Web Modeling
Editor's Knowledge Graph diagram, and the input to the deterministic
:doc:`OWL 2 / SHACL to B-UML transformation <../model_building/kg_to_buml>`.

Unlike the other sublanguages, KG identifiers are IRIs, blank-node ids and
literal values rather than Python names. ``KGNode`` therefore extends
``Element`` instead of ``NamedElement``: IRIs and labels are preserved exactly
as written, and safe variable names are derived separately when the model is
exported to Python.

Nodes
-----

Seven node kinds, matching what an OWL/RDF graph can contain:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Kind
     - Represents
   * - ``KGClass``
     - A class or concept (``owl:Class``, ``rdfs:Class``)
   * - ``KGIndividual``
     - An instance (``owl:NamedIndividual``)
   * - ``KGProperty``
     - A property that is itself the subject or object of assertions
   * - ``KGLiteral``
     - A literal value, with an optional datatype IRI
   * - ``KGBlank``
     - An anonymous resource (RDF blank node)
   * - ``KGNodeConstraint``
     - A constraint on a class: an OWL restriction set or a ``sh:NodeShape``
   * - ``KGPropertyConstraint``
     - A constraint on a property: an ``owl:Restriction`` or a ``sh:PropertyShape``

The two constraint kinds reify constraints as first-class, visible nodes so
they can be inspected and edited on the canvas. Their payload lives in
``metadata['constraintSpecs']`` as a list of vocabulary-agnostic
``ConstraintSpec`` dicts, each naming a ``kind`` (``minCardinality``,
``allValuesFrom``, ``pattern``, ``shaclOr``, …) and a value.

Edges and axioms
----------------

``KGEdge`` is a directed relation carrying an optional predicate IRI. Some OWL
constructs relate *sets* or *ordered lists* of elements and so fit on neither a
single node nor a single edge — ``owl:equivalentClass``, ``owl:disjointUnionOf``,
``owl:propertyChainAxiom``, ``owl:hasKey`` and friends. Those are recorded as
``KGAxiom`` dataclasses on ``KnowledgeGraph.axioms``, referencing node ids
rather than node objects so they stay serialisable.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.kg import (
        KnowledgeGraph, KGClass, KGProperty, KGEdge,
    )

    person = KGClass(id="ex:Person", label="Person", iri="http://ex.org/Person")
    knows = KGProperty(id="ex:knows", label="knows", iri="http://ex.org/knows")

    kg = KnowledgeGraph(name="social")
    kg.add_node(person)
    kg.add_node(knows)
    kg.add_edge(KGEdge(
        id="e1", source=knows, target=person,
        iri="http://www.w3.org/2000/01/rdf-schema#domain",
    ))

Building and consuming a KG
---------------------------

* Import an ontology with
  :func:`besser.utilities.owl_to_buml.owl_file_to_knowledge_graph`.
* Export one with
  :func:`besser.utilities.kg_to_owl.serialize_knowledge_graph` (Turtle or
  RDF/XML, with OWL restrictions, SHACL shapes, or both).
* Transform one into a :doc:`structural model <structural>` with OCL
  constraints — see :doc:`../model_building/kg_to_buml`.
