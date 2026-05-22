UML Component model
===================

This metamodel describes **UML 2.5 Component diagrams** — the structural view
of a system as a set of components, the interfaces they provide and require,
and the dependencies between them. It is a first-class B-UML model, alongside
the :doc:`structural <structural>`, :doc:`state machine <state_machine>`, and
:doc:`agent <agent>` metamodels.

.. note::

   This is the UML 2.5 Component metamodel. It is unrelated to the
   :doc:`Deployment architecture model <deployment>`, which models
   cloud-infrastructure concepts. The companion UML Deployment notation is
   the :doc:`UML Deployment model <uml_deployment>`.

The metamodel is split into two modules:

* ``besser.BUML.metamodel.uml_component.uml_component`` — the **vanilla
  UML 2.5 base**.
* ``besser.BUML.metamodel.uml_component.agentic`` — an **agentic-swarm
  extension** layered on top.

The package ``__init__`` re-exports both, so
``from besser.BUML.metamodel.uml_component import *`` exposes every name.

Base metamodel
--------------

* ``Component`` — the canonical node of a Component diagram. Carries a
  ``locality`` (see :ref:`uml-component-locality`) and ``realizes`` — a list
  of structural ``Class`` names this component realizes (UML Realization).
* ``Subsystem`` — a ``Component`` that is also a container of child
  components.
* ``Interface`` — a provided / required interface, referenced by interface
  relationships.
* Relationships: ``InterfaceProvided`` and ``InterfaceRequired`` (Component
  to Interface), and ``ComponentDependency`` (a plain dependency between two
  Components).
* ``ComponentModel`` — the root container (components, interfaces,
  relationships). ``ComponentModel.validate()`` returns a
  ``{"success", "errors", "warnings"}`` dictionary.

Agentic extension
-----------------

The agentic extension adds the collaborative-agent-swarm profile used to
model multi-agent systems. It is a pure addition — the base module is
untouched and remains valid vanilla UML 2.5 on its own.

* ``AgenticComponent`` — a ``Component`` that is an *agent*. Carries an
  ``agent_category`` (``AgentCategory``: ``NONE`` / ``SOLUTION`` /
  ``SUPERVISION`` / ``CONSENSUS`` / ``COLLABORATION``), an ``is_human`` flag
  for human-in-the-loop agents, and ``process_model_refs`` — cross-diagram
  references to the BPMN processes the agent participates in.
* ``Skill`` / ``Tool`` — agent capabilities (plain ``Component``
  subclasses).
* ``Permission`` — a named authority (with a ``scope``) carried on an
  agentic edge.
* ``AgenticEdge`` — a typed dependency between agents and / or capabilities.
  The ``AgenticEdgeKind`` covers agent-to-agent edges (``DELEGATES`` /
  ``SUPERVISES`` / ``REVISES`` / ``COLLABORATES``), agent-to-capability
  edges (``HAS`` → Skill, ``USES`` → Tool, ``GRANTED`` → Permission), and
  capability composition (``IMPLEMENTS``, Tool → Skill). Each kind enforces
  its endpoint types at construction.
* ``AgenticComponentModel`` — a ``ComponentModel`` that additionally holds
  the model's ``Permission`` set and runs the agentic validation rules
  (agent-endpoint typing, permission membership, and structural-smell
  warnings).

.. _uml-component-locality:

Locality
--------

``Locality`` (``LOCAL`` / ``EXTERNAL`` / ``HYBRID``) classifies where a
component — or a deployment node / artifact — is hosted. It is a BESSER
general profile addition (not part of UML 2.5.1) and is shared, with the
same semantics, with the :doc:`UML Deployment model <uml_deployment>`.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.uml_component import (
        AgentCategory, AgenticComponent, AgenticComponentModel,
        AgenticEdge, AgenticEdgeKind, Skill,
    )

    advisor = AgenticComponent("CodeAdvisor", agent_category=AgentCategory.SOLUTION)
    search = Skill("CodeSearch")
    has = AgenticEdge(advisor, search, kind=AgenticEdgeKind.HAS)

    model = AgenticComponentModel(
        "swarm", components={advisor, search}, relationships={has},
    )
    result = model.validate()  # {"success": ..., "errors": [...], "warnings": [...]}

Supported notations
-------------------

* :doc:`Coding in Python using the B-UML library <../model_building/buml_core>`
* The :doc:`Web Modeling Editor <../../web_editor>` (``ComponentDiagram``).
