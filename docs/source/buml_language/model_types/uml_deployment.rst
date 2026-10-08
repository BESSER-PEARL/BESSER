UML Deployment model
====================

This metamodel describes **UML 2.5 Deployment diagrams** — the allocation of
software artifacts onto the nodes (hardware or execution environments) that
run them. It is the deployment-side companion of the
:doc:`UML Component model <uml_component>`.

Why two deployment metamodels
-----------------------------

BESSER has two metamodels about deployment, answering different questions:

* The :doc:`Deployment architecture model <deployment>`
  (``besser.BUML.metamodel.deployment``) describes *infrastructure to
  provision*: Kubernetes clusters, deployments, services and containers, cloud
  regions and zones, nodes with IP ranges and resources. It feeds the
  Terraform generator.
* This UML Deployment model (``besser.BUML.metamodel.uml_deployment``)
  describes *what a UML 2.5 Deployment diagram shows*: which artifact runs on
  which execution node, and which Component each artifact manifests. It is
  what the web editor's Deployment diagram serialises to, and it feeds the
  Docker Compose generator.

Folding one into the other would force UML notation onto the infrastructure
model (or Kubernetes concepts onto UML diagrams), so they stay separate and do
not reference each other.

.. warning::

   Both packages define classes named ``Node`` and ``DeploymentModel``, and
   they are different types. Import them through their package path
   (``from besser.BUML.metamodel.uml_deployment import Node``) and **never
   star-import both packages into one namespace**: the second import would
   silently shadow the first. UML's artifact-on-node relationship, called
   ``Deployment`` in UML 2.5, is named ``DeploymentRelation`` here because
   ``deployment.Deployment`` already denotes a Kubernetes Deployment.

The metamodel lives in
``besser.BUML.metamodel.uml_deployment.uml_deployment``.

Metamodel
---------

* ``Node`` — a deployment target. Carries a ``kind`` (``NodeKind``:
  ``GENERIC`` / ``DEVICE`` / ``EXECUTION_ENVIRONMENT``) and a ``locality``.
  Nodes can nest other nodes and artifacts (e.g. an execution environment
  inside a device).
* ``Artifact`` -- the deployable unit. Carries a ``locality``,
  ``manifests`` -- a list of ``Component`` identifiers (a cross-diagram
  reference into a :doc:`UML Component model <uml_component>`) -- and optional
  ``agent_model_ref``, the id of the Agent diagram this artifact deploys.
* ``DeploymentComponent`` -- an ``Artifact`` subclass for the Component an
  artifact manifests, drawn on the Deployment diagram (the editor's
  ``DeploymentComponent`` element). It is a view element: generators never
  turn it into a deployable service.
* ``Interface`` — a provided / required interface on a node or artifact.
* Relationships:

  * ``DeploymentRelation`` — an artifact deployed on a node. Carries a
    ``multiplicity`` (reused from the :doc:`structural metamodel
    <structural>`) for the instance count, e.g. ``[3]`` or ``[1..*]``.
  * ``CommunicationPath`` — a link between two nodes.
  * ``DeploymentDependency`` — a plain dependency between deployment
    elements.
  * ``InterfaceProvided`` / ``InterfaceRequired`` — node/artifact to
    interface.

* ``DeploymentModel`` — the root container (nodes, artifacts, interfaces,
  relationships). ``DeploymentModel.validate()`` returns a
  ``{"success", "errors", "warnings"}`` dictionary.

Node containment cannot form a cycle: nesting a node inside itself or inside
one of its own descendants raises ``ValueError``, and ``validate()`` reports a
cycle wired through private attributes as an error instead of recursing.

``Locality`` (``LOCAL`` / ``EXTERNAL`` / ``HYBRID``) is a BESSER general
profile addition shared with the
:doc:`UML Component model <uml_component>`; see
:ref:`uml-component-locality`.

When a Deployment diagram is generated from a multi-agent system project, WME can
stamp ``agentModelRef`` on an artifact. The JSON converter stores that value as
``Artifact.agent_model_ref``. Project-level deployment generators use it to
resolve the artifact back to the Agent diagram and bake the generated agent
runtime into the deployment output.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.structural import Multiplicity
    from besser.BUML.metamodel.uml_deployment import (
        Artifact, DeploymentModel, DeploymentRelation, Node, NodeKind,
    )

    runtime = Node("AgentRuntime", kind=NodeKind.EXECUTION_ENVIRONMENT)
    artifact = Artifact(
        "advisor", manifests=["component-uuid-1"], agent_model_ref="agent-diagram-uuid-1"
    )
    deployed = DeploymentRelation(artifact, runtime, multiplicity=Multiplicity(1, 3))

    model = DeploymentModel(
        "deployment", nodes={runtime}, artifacts={artifact},
        relationships={deployed},
    )
    result = model.validate()  # {"success": ..., "errors": [...], "warnings": [...]}

Supported notations
-------------------

* :doc:`Coding in Python using the B-UML library <../model_building/buml_core>`
* The :doc:`Web Modeling Editor <../../web_editor>` (``DeploymentDiagram``).
