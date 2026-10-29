Web Modeling Editor
===================

The editor lets you draw B-UML models, describe changes to an assistant, and
download generated code. Open `editor.besser-pearl.org
<https://editor.besser-pearl.org>`_ to use the hosted version.

Start using the editor
----------------------

* :doc:`start/browser`: create a project and download your first code.
* `Editor user guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>`_:
  diagram tools, project backups, display settings, and AI features.
* :doc:`guides/choose-generator`: choose an output for your model.
* :doc:`guides/build-with-ai`: review a model and generate an application.

The editor guide is the reference for interface instructions and screenshots.
This site covers the underlying model language, generators, and services.

Models and services
-------------------

Use class diagrams for data, GUI diagrams for screens, state machines and
agent diagrams for behaviour, BPMN for processes, neural-network diagrams
for training architectures, and knowledge-graph diagrams for ontologies. See :doc:`buml_language` for the model catalogue.
Available generation actions depend on the selected diagram and deployment.

The assistant service creates and edits diagrams. The :doc:`Spec-Driven Agent
<spec_driven_agent/index>` handles application files. Agent chat simulation
uses a separate :doc:`simulator service <utilities/agent_simulator>`.

Knowledge-graph diagrams
------------------------

A knowledge-graph diagram models an ontology as a graph of classes, properties,
individuals and literals, plus OWL axioms and SHACL/OWL constraints. It is a
diagram type of its own (``KnowledgeGraphDiagram``) and needs no other diagram.
The metamodel is described in :doc:`buml_language/model_types/knowledge_graph`.

* **Edit.** The graph is drawn on a canvas rather than with the class-diagram
  editor: drag node types from the palette, draw relations by dragging from a
  source node to a target node, edit labels and constraints in the inspector,
  and undo or redo changes. Large ontologies show only the highest-priority
  nodes; the limit is set on the *KG Settings* page.
* **Import.** Load an OWL/RDF ontology (``.owl``, ``.ttl``, ``.rdf``, ``.xml``,
  ``.nt``, ``.n3``, up to 5 MB). Anything the importer had to skip is reported
  as a warning.
* **Export.** Download the graph as Turtle or RDF/XML, choosing whether
  constraints are written as OWL restrictions, SHACL shapes or both, as a
  standalone interactive HTML file, or as B-UML Python code.
* **Refine and convert.** *Refine KG* checks the graph before it becomes a class
  diagram. Its *Automatic* tab lists the elements that have no clean 1-to-1
  B-UML mapping, each with a recommended and a skip action. Its *Consistency*
  tab runs the OWL/SHACL consistency check, and its *AI* tab asks an LLM for
  suggestions using your own provider key, which is sent with the request and
  never logged by the backend. Converting then produces a class diagram with
  OCL constraints; see :doc:`buml_language/model_building/kg_to_buml`.

The endpoints behind these actions are listed under *Knowledge Graph* in
:doc:`web_editor_backend`.

Run or extend the editor
------------------------

The frontend is a Git submodule at
``besser/utilities/web_modeling_editor/frontend``. Backend services live at
``besser/utilities/web_modeling_editor/backend``.

* `Run locally <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/deploy_locally.html>`_.
* `Add a diagram type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.
* `Frontend source <https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor>`_.

.. toctree::
   :maxdepth: 1

   web_editor_backend
