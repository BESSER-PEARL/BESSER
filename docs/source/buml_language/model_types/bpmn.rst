BPMN model
==========

.. _bpmn-metamodel:

BPMN metamodel
--------------

This metamodel allows the definition of BPMN (Business Process Model and Notation)
diagrams, the OMG standard for visualising and specifying business processes. A BPMN
model captures the flow of work across **flow nodes** (tasks, events, gateways)
connected by **sequence flows**, optionally organised into **pools** (one per
participant) and **lanes** (sub-partitions of a pool). The metamodel covers the
BPMN element set supported by the Web Modeling Editor (WME) and follows the `BPMN
2.0.2`_ class hierarchy.

.. _`BPMN 2.0.2`: https://www.omg.org/spec/BPMN/2.0.2/PDF/

The top-level container is ``BPMNModel``. A diagram without pools has a single
``Process``; a diagram with pools has a ``Collaboration`` whose ``Participant``\s each
reference one ``Process``.

.. image:: ../../img/bpmn_mm.png
  :width: 700
  :alt: BPMN metamodel
  :align: center

.. note::

  The classes highlighted in green originate from the :doc:`structural metamodel <structural>`.
  For the sake of clarity, associations of ``BPMNModel`` and ``Process`` are represented as attributes.

Key concepts
^^^^^^^^^^^^

- **Flow nodes** (``FlowNode``): everything that can sit in a process and be the
  source or target of a sequence flow. Three concrete branches:

  - ``Activity``: work to be performed. Concrete subtypes: ``Task``,
    ``SubProcess``, ``Transaction`` (a ``SubProcess`` subclass per BPMN 2.0.2
    § 10.3), and ``CallActivity``. Tasks carry a ``task_type`` (``USER``,
    ``SERVICE``, ``SEND``, ``RECEIVE``, ``MANUAL``, ``BUSINESS_RULE``,
    ``SCRIPT``, ``DEFAULT``). All activities carry ``loop_characteristics``
    (``NONE`` / ``STANDARD_LOOP`` / ``PARALLEL_MI`` / ``SEQUENTIAL_MI``).
  - ``Event``: something that happens. Concrete subtypes: ``StartEvent``,
    ``IntermediateEvent``, ``EndEvent``. See the *Event model* subsection below.
  - ``Gateway``: branch / merge / join point. The kind is set via
    ``gateway_type`` (``EXCLUSIVE``, ``INCLUSIVE``, ``PARALLEL``, ``COMPLEX``,
    ``EVENT_BASED``).

- **Connecting objects** (``BPMNConnectingObject``): four concrete subtypes:
  ``SequenceFlow`` (orders flow nodes inside a process or sub-process),
  ``MessageFlow`` (across pool boundaries; endpoints may be message-eligible
  nodes (activities or events) or whole ``Participant`` pools per
  BPMN 2.0.2 § 9.3),
  ``Association`` (links an artifact to any element), ``DataAssociation``
  (connects exactly one ``DataElement`` to exactly one ``FlowNode``).
- **Containers**: ``Process`` and ``SubProcess`` both hold flow nodes and sequence
  flows (they expose the same ``flow_nodes`` / ``sequence_flows`` API and are
  duck-type compatible). ``Lane`` partitions a process; ``Participant`` is a pool
  referencing a process; ``Collaboration`` groups participants and the message
  flows between them.
- **Data & artifacts**: ``DataObject`` (process-scoped), ``DataStore``
  (model-scoped; per BPMN 2.0.2 § 10.3 a root-level element shared across
  processes), ``TextAnnotation``, ``Group``.

``BPMNModel`` also exposes three convenience methods: ``all_flow_nodes()`` returns
every flow node across all processes, ``all_sequence_flows()`` returns every sequence
flow including those nested inside sub-processes, and ``all_connecting_objects()``
returns every connecting object across the entire model.

Event model
^^^^^^^^^^^

Events split along **two orthogonal axes**:

- ``direction`` (``EventDirection.CATCH`` / ``THROW``):  ``CATCH`` means the event waits to
  receive a trigger (message arrives, timer fires, signal broadcast from elsewhere); ``THROW``
  means the event sends a trigger as the flow passes through it. Fixed to ``CATCH`` on
  ``StartEvent`` and ``THROW`` on ``EndEvent``; free on ``IntermediateEvent``.
- ``event_definition`` (``EventDefinitionType``): ``NONE``, ``MESSAGE``,
  ``TIMER``, ``SIGNAL``, ``ESCALATION``, ``ERROR``, ``COMPENSATION``, ``LINK``,
  ``CONDITIONAL``, ``TERMINATE``.

The metamodel enforces a legality table of valid ``(class, direction,
event_definition)`` triples at construction time, so illegal combinations
(e.g. ``StartEvent(event_definition=TERMINATE)``) raise ``ValueError``
immediately rather than waiting for ``validate()``.

Sequence-flow defaults
^^^^^^^^^^^^^^^^^^^^^^

``SequenceFlow.is_default`` may be set to ``True`` only when the source is an
``Activity`` or an exclusive / inclusive / complex ``Gateway`` (BPMN 2.0.2
§ 8.3.13). The metamodel guards this in the setter and re-validates in
``BPMNModel.validate()``. The derived ``Activity.default_flow`` /
``Gateway.default_flow`` properties return the single outgoing flow marked
``is_default=True``, or ``None`` if no default is set. Because the flag and the
property read from the same underlying field they can never disagree.

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

Call ``BPMNModel.validate()`` to check structural correctness: endpoint
references resolve, sequence flows stay within a single container, message
flows cross pool boundaries, default-flow source rules hold, event-definition
triples are legal, lane membership matches process membership, and every
participant references a process in the model:

.. code-block:: python

    result = model.validate(raise_exception=False)
    # result = {"success": True/False, "errors": [...], "warnings": [...]}

Validation collects all errors and warnings in a single pass
rather than stopping at the first failure, so one call reports everything that
is wrong.

Round-trip with the Web Modeling Editor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The BPMN converters live alongside the others under
``besser.utilities.web_modeling_editor.backend.services.converters``:

``process_bpmn_diagram(json)``
    WME BPMN JSON → ``BPMNModel``.

``bpmn_object_to_json(model)``
    ``BPMNModel`` → WME BPMN JSON.

``bpmn_buml_to_json(content)``
    B-UML ``.py`` source string → WME BPMN JSON (executes the source and
    delegates to ``bpmn_object_to_json``).

``bpmn_model_to_code(model)``
    ``BPMNModel`` → executable Python that reconstructs the model when
    ``exec()``'d.

Because BPMN names can be empty, whitespace, or repeated, the metamodel
identifies elements **by object** (no ``__eq__`` / ``__hash__`` overrides);
the converters keep the original WME ids in an opaque ``BPMNElement.layout``
side-channel so round-trips remain stable.

.. _bpmn-agentic-extension:

Agentic extension
-----------------

The BPMN metamodel ships BESSER's **Agentic extension** alongside the
vanilla BPMN base. The extension adds stereotype-as-subclass primitives in a
sibling module (``besser/BUML/metamodel/bpmn/agentic.py``); the base
``bpmn.py`` remains valid BPMN on its own.

- ``AgenticTask(Task)`` -- a Task with a ``reflection_mode``
  (``NONE`` / ``SELF`` / ``CROSS`` / ``HUMAN``), a ``trust_score`` in
  ``[0, 100]``, and an optional ``agent_diagram_ref`` for task-level links to
  the Agent diagram that defines the task behaviour.
- ``AgenticGateway(Gateway)`` -- a Gateway restricted to ``PARALLEL`` or
  ``INCLUSIVE`` ``gateway_type``; ``EXCLUSIVE`` / ``COMPLEX`` /
  ``EVENT_BASED`` are rejected by the setter override. It carries a
  ``gateway_role`` (``DIVERGING`` / ``MERGING``), a ``trust_score``, and an
  optional ``governance_dsl`` snippet. Governance DSL is authored externally,
  stored opaquely by BESSER, and round-tripped on the gateway.
- ``AgenticLane(Lane)`` -- a Lane with a ``role`` (``SOLUTION`` /
  ``SUPERVISION`` / ``COLLABORATION`` / ``CONSENSUS``), a ``trust_score``,
  an optional ``agent_diagram_ref`` linking the lane to its Agent diagram,
  and ``multiplicity``: the number of identical agent instances represented
  by that lane. ``multiplicity`` is an integer ``>= 1`` and defaults to ``1``.

Agent-to-agent coordination is represented by ordinary BPMN flow structure, lane/task links
to Agent diagrams, Governance DSL on merge gateways, and A2A tags in the
derived Agent diagrams used by deployment generation.

Example
^^^^^^^

An agentic task feeding a merging agentic gateway, inside a supervision-role
agentic lane:

.. code-block:: python

    from besser.BUML.metamodel.bpmn import (
        AgenticGateway, AgenticLane, AgenticTask, AgentRole,
        BPMNModel, GatewayRole, GatewayType, Process, ReflectionMode,
        SequenceFlow, TaskType,
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
        trust_score=85,
        governance_dsl="policy MajorityPolicy { participants Reviewers }",
    )
    lane = AgenticLane(
        name="Reviewers",
        role=AgentRole.SUPERVISION,
        trust_score=85,
        multiplicity=3,
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
handle the agentic subclasses transparently -- import dispatches on the
WME ``isAgentic`` flag to construct the right subclass; export emits the
WME shape. Trust scores are clamped on import (``max(0, min(100, value))``)
to bridge WME's tolerant data into the metamodel's strict ``[0, 100]``;
lane multiplicity is clamped to ``>= 1``. The metamodel itself raises
``ValueError`` on out-of-range values per B-UML house style. The
cross-diagram and policy fields round-trip too:
``AgenticTask.agent_diagram_ref``, ``AgenticLane.agent_diagram_ref``,
``AgenticLane.multiplicity``, and ``AgenticGateway.governance_dsl`` all
survive the JSON import / export cycle.

The :doc:`../../generators/bpmn` emits the agentic information as
``<bpmn:extensionElements>`` / ``<agentic:agentic .../>`` blocks in the
BPMN 2.0 XML output (see that page for the on-disk shape).
