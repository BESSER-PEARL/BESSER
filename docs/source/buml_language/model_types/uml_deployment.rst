UML Deployment model
====================

This metamodel describes **UML 2.5 Deployment diagrams** — the allocation of
software artifacts onto the nodes (hardware or execution environments) that
run them. It is the deployment-side companion of the
:doc:`UML Component model <uml_component>`.

.. note::

   This is the UML 2.5 Deployment notation (``Node``, ``Artifact``,
   ``DeploymentRelation``). It is **distinct** from the
   :doc:`Deployment architecture model <deployment>`, which models
   cloud-infrastructure concepts (clusters, services, containers) and feeds
   the Terraform generator. The two metamodels live in separate packages and
   do not interact.

The metamodel lives in
``besser.BUML.metamodel.uml_deployment.uml_deployment``.

Metamodel
---------

* ``Node`` — a deployment target. Carries a ``kind`` (``NodeKind``:
  ``GENERIC`` / ``DEVICE`` / ``EXECUTION_ENVIRONMENT``) and a ``locality``.
  Nodes can nest other nodes and artifacts (e.g. an execution environment
  inside a device).
* ``Artifact`` — the deployable unit. Carries a ``locality`` and
  ``manifests`` — a list of ``Component`` identifiers (a cross-diagram
  reference into a :doc:`UML Component model <uml_component>`).
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

``Locality`` (``LOCAL`` / ``EXTERNAL`` / ``HYBRID``) is a BESSER general
profile addition shared with the
:doc:`UML Component model <uml_component>`; see
:ref:`uml-component-locality`.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.structural import Multiplicity
    from besser.BUML.metamodel.uml_deployment import (
        Artifact, DeploymentModel, DeploymentRelation, Node, NodeKind,
    )

    runtime = Node("AgentRuntime", kind=NodeKind.EXECUTION_ENVIRONMENT)
    artifact = Artifact("advisor.whl", manifests=["component-uuid-1"])
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
