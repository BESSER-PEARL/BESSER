BPMN model
==========

.. _bpmn-metamodel:

BPMN metamodel
--------------

This metamodel allows the definition of BPMN (Business Process Model and Notation)
diagrams — the OMG standard for visualising and specifying business processes. A BPMN
model captures the flow of work across **flow nodes** (tasks, events, gateways)
connected by **sequence flows**, optionally organised into **pools** (one per
participant) and **lanes** (sub-partitions of a pool). The metamodel covers the
WME BPMN editor's element set and follows the BPMN 2.0.2 spec's class structure where
the two diverge.

The top-level container is ``BPMNModel``. A pool-less diagram has a single
``Process``; a pool-ful diagram has a ``Collaboration`` whose ``Participant``\ s each
reference one ``Process``.

Key concepts
^^^^^^^^^^^^

- **Flow nodes** (``FlowNode``): everything that can sit in a process and be the
  source or target of a sequence flow. Three concrete branches:

  - ``Activity`` — work to be performed. Concrete subtypes: ``Task``,
    ``SubProcess``, ``Transaction`` (a ``SubProcess`` subclass per spec), and
    ``CallActivity``. Tasks carry a ``task_type`` (user, service, send, receive,
    manual, business-rule, script, default). All activities carry
    ``loop_characteristics`` (none / loop / parallel / sequential multi-instance).
  - ``Event`` — something that happens. Concrete subtypes: ``StartEvent``,
    ``IntermediateEvent``, ``EndEvent``. See the *Event model* subsection below.
  - ``Gateway`` — branch / merge / join. The kind is set via
    ``gateway_type`` (exclusive, inclusive, parallel, complex, event-based).

- **Connecting objects** (``BPMNConnectingObject``): four concrete subtypes —
  ``SequenceFlow`` (orders flow nodes inside a process / sub-process),
  ``MessageFlow`` (message-eligible nodes across pool boundaries),
  ``Association`` (links an artifact to anything), ``DataAssociation``
  (connects exactly one ``DataElement`` to exactly one ``FlowNode``).
- **Containers**: ``Process`` and ``SubProcess`` hold flow nodes and sequence
  flows (they expose the same ``flow_nodes`` / ``sequence_flows`` API so they're
  duck-type compatible). ``Lane`` partitions a process; ``Participant`` is a
  pool referencing a process; ``Collaboration`` groups participants and the
  message flows between them.
- **Data & artifacts**: ``DataObject`` (process-scoped), ``DataStore``
  (model-scoped, per spec § 10.3 a root-level element), ``TextAnnotation``,
  ``Group``.

Event model
^^^^^^^^^^^

Events split along **two orthogonal axes** rather than the flat string enum the
WME frontend uses:

- ``direction`` (``EventDirection.CATCH`` / ``THROW``) — fixed to ``CATCH`` on
  ``StartEvent`` and ``THROW`` on ``EndEvent``; free on ``IntermediateEvent``.
- ``event_definition`` (``EventDefinitionType.NONE`` / ``MESSAGE`` / ``TIMER`` /
  ``SIGNAL`` / ``ESCALATION`` / ``ERROR`` / ``COMPENSATION`` / ``LINK`` /
  ``CONDITIONAL`` / ``TERMINATE``).

The metamodel enforces a legality table of valid ``(class, direction,
event_definition)`` triples at construction time, so illegal combinations
(e.g. ``StartEvent(event_definition=TERMINATE)``) fail fast rather than later
in ``validate()``.

Sequence-flow defaults
^^^^^^^^^^^^^^^^^^^^^^

``SequenceFlow.is_default`` may be set to ``True`` only when the source is an
``Activity`` or an exclusive / inclusive / complex ``Gateway`` (BPMN 2.0.2
§ 8.3.13). The metamodel guards this in the setter and re-validates in
``BPMNModel.validate()``. The derived ``Activity.default_flow`` /
``Gateway.default_flow`` properties return the single ``is_default=True``
outgoing flow (or ``None``) — a single source of truth so the two views can't
desync.

Example
^^^^^^^

A pool-less process with a start event, a user task, and an end event:

.. code-block:: python

    from besser.BUML.metamodel.bpmn import (
        BPMNModel, Process, Task, StartEvent, EndEvent, SequenceFlow,
        TaskType,
    )

    start = StartEvent(name="received")
    task = Task(name="Review order", task_type=TaskType.USER)
    end = EndEvent(name="done")
    process = Process(
        name="Order Review",
        flow_nodes={start, task, end},
        sequence_flows={SequenceFlow(start, task), SequenceFlow(task, end)},
    )
    model = BPMNModel(name="OrderReview", processes={process})

Validation
^^^^^^^^^^

Call ``BPMNModel.validate()`` to check structural correctness — endpoint
references resolve, sequence flows stay within a single container, message
flows cross pool boundaries, default-flow source rules hold, event-definition
triples are legal, lane membership matches process membership, every
participant references a process in the model:

.. code-block:: python

    result = model.validate(raise_exception=False)
    # result = {"success": True/False, "errors": [...], "warnings": [...]}

Validation collects errors (E1–E11) and warnings (W1–W4) rather than throwing
on the first failure, so a single call reports everything that's wrong.

Round-trip with the Web Modeling Editor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BPMN converters live alongside the others under
``besser.utilities.web_modeling_editor.backend.services.converters``:

- ``process_bpmn_diagram(json)`` — WME BPMN JSON → ``BPMNModel``.
- ``bpmn_object_to_json(model)`` — ``BPMNModel`` → WME BPMN JSON.
- ``bpmn_to_json(content)`` — BUML ``.py`` source string → WME BPMN JSON
  (execs the source and delegates to ``bpmn_object_to_json``).
- ``bpmn_model_to_code(model)`` — ``BPMNModel`` → executable Python that
  reconstructs the model when ``exec()``'d.

Because BPMN names can be empty, whitespace, or repeated, the metamodel
identifies elements **by object** (no ``__eq__`` / ``__hash__`` overrides);
the converters keep the original WME ids in an opaque ``BPMNElement.layout``
side-channel so round-trips remain stable.

.. _bpmn-agentic-extension:

SEAA'25 agentic extension
-------------------------

The BPMN metamodel ships a paper-faithful extension for **human-agentic
collaborative workflows** alongside the vanilla BPMN base. The extension is
defined in Ait et al., *Towards Modeling Human-Agentic Collaborative
Workflows: A BPMN Extension* (SEAA'25), and adds three
stereotype-as-subclass primitives in a sibling module
(``besser/BUML/metamodel/bpmn/agentic.py``); the base ``bpmn.py`` is
unmodified.

- ``AgenticTask(Task)`` — a Task with a ``reflection_mode``
  (``NONE`` / ``SELF`` / ``CROSS`` / ``HUMAN``) and a ``trust_score``
  ∈ ``[0, 100]`` (paper § 4.2, Fig 3b).
- ``AgenticGateway(Gateway)`` — a Gateway with collaboration semantics
  (paper § 4.3). Restricted to ``PARALLEL`` or ``INCLUSIVE`` ``gateway_type``
  (the paper's *AgenticAND* / *AgenticOR*); ``EXCLUSIVE`` / ``COMPLEX`` /
  ``EVENT_BASED`` are rejected by the setter override. Carries a
  ``gateway_role`` (``DIVERGING`` / ``MERGING``), a ``collaboration_mode``
  (``VOTING`` / ``ROLE`` / ``DEBATE`` / ``COMPETITION``), an optional
  ``merging_strategy`` (seven values per paper Table 2: ``MAJORITY`` /
  ``ABSOLUTE_MAJORITY`` / ``MINORITY`` / ``LEADER_DRIVEN`` / ``COMPOSED`` /
  ``FASTEST`` / ``MOST_COMPLETE``), and a ``trust_score``.
- ``AgenticLane(Lane)`` — a Lane with a ``role`` (``WORKER`` / ``MANAGER``)
  and a ``trust_score`` (paper § 4.1, Fig 3a).

Two tri-attribute invariants on ``AgenticGateway`` are auto-maintained by
the setters so the common ergonomic path is single-statement:

- **INV-A** (paper § 4.3): ``merging_strategy is None`` iff
  ``gateway_role == DIVERGING``. The strategy is meaningful only at the
  merging gateway.
- **INV-B**: when set, ``merging_strategy`` must be in the legality table
  for the current ``collaboration_mode`` (paper Table 2; debate borrows
  from voting + role per paper § 4.3 last paragraph).

Flipping ``gateway_role`` from ``DIVERGING`` to ``MERGING`` auto-sets
``merging_strategy`` to the legal default for the current mode; changing
``collaboration_mode`` resets the strategy if its previous value is no
longer legal.

Example
^^^^^^^

An agentic task feeding a merging agentic gateway (voting / majority),
inside a manager-role agentic lane:

.. code-block:: python

    from besser.BUML.metamodel.bpmn import (
        AgenticGateway, AgenticLane, AgenticTask, AgentRole,
        BPMNModel, CollaborationMode, GatewayRole, GatewayType,
        MergingStrategy, Process, ReflectionMode, SequenceFlow, TaskType,
    )

    review = AgenticTask(
        name="Review",
        task_type=TaskType.USER,
        reflection_mode=ReflectionMode.CROSS,
        trust_score=85,
    )
    vote = AgenticGateway(
        name="Vote",
        gateway_type=GatewayType.PARALLEL,
        gateway_role=GatewayRole.MERGING,
        collaboration_mode=CollaborationMode.VOTING,
        merging_strategy=MergingStrategy.MAJORITY,
        trust_score=85,
    )
    lane = AgenticLane(
        name="Reviewers",
        role=AgentRole.MANAGER,
        trust_score=85,
        flow_nodes={review, vote},
    )
    process = Process(
        name="AgenticReview",
        flow_nodes={review, vote},
        sequence_flows={SequenceFlow(review, vote)},
        lanes={lane},
    )
    model = BPMNModel(name="Agentic", processes={process})

Round-trip
^^^^^^^^^^

The same converter entry points (``process_bpmn_diagram`` /
``bpmn_object_to_json`` / ``bpmn_to_json`` / ``bpmn_model_to_code``)
handle the agentic subclasses transparently — import dispatches on the
WME ``isAgentic`` flag to construct the right subclass; export emits the
WME shape. Trust scores are clamped on import (``max(0, min(100, value))``)
to bridge WME's tolerant data into the metamodel's strict ``[0, 100]``;
the metamodel itself raises ``ValueError`` on out-of-range values per
B-UML house style.

The :doc:`../../generators/bpmn` emits the agentic information as
``<bpmn:extensionElements>`` / ``<agentic:agentic .../>`` blocks in the
BPMN 2.0 XML output (see that page for the on-disk shape).
