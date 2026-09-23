BPMN Generator
==============

This code generator produces a vendor-neutral **BPMN 2.0 XML** file from a
:doc:`../buml_language/model_types/bpmn`. The output is plain BPMN 2.0, so it
can be opened by every BPMN-aware tool: Camunda 7/8, Flowable,
bpmn-js, and others.

Engine-specific execution semantics are not emitted; an opened file renders the
process structure (flow nodes, sequence flows, gateways, events, pools / lanes,
data objects / stores, sub-processes) and round-trips through any conformant
modeller.

Usage
-----

Create a ``BPMNGenerator`` object, provide a :ref:`bpmn-metamodel` instance,
and call ``generate``:

.. code-block:: python

    from besser.generators.bpmn import BPMNGenerator
    from besser.BUML.metamodel.bpmn import (
        BPMNModel, Process, Task, StartEvent, EndEvent, SequenceFlow, TaskType,
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

    generator = BPMNGenerator(model=model, output_dir="output")
    generator.generate()

The ``bpmn_diagram.bpmn`` file with the BPMN 2.0 XML representation will be
generated in the ``output/`` folder.

Output structure
----------------

The generator emits a ``<bpmn:definitions>`` root element with the standard
BPMN 2.0 namespace declarations and a ``targetNamespace`` identifying the
producing organisation. Inside it:

- One ``<bpmn:process>`` element per ``Process`` in the model, containing
  ``<bpmn:task>`` / ``<bpmn:userTask>`` / ``<bpmn:serviceTask>`` / … elements
  for each task (the concrete XML tag matches ``task_type``),
  ``<bpmn:startEvent>`` / ``<bpmn:endEvent>`` / ``<bpmn:intermediateCatchEvent>``
  / ``<bpmn:intermediateThrowEvent>`` elements for events (``CATCH`` / ``THROW``
  direction determines the tag), ``<bpmn:exclusiveGateway>`` / ``<bpmn:parallelGateway>``
  / … elements for gateways, ``<bpmn:sequenceFlow>`` elements for sequence flows,
  ``<bpmn:laneSet>`` / ``<bpmn:lane>`` elements for lane sets, and
  ``<bpmn:dataObjectReference>`` / ``<bpmn:dataObject>`` pairs for data objects.
- A single ``<bpmn:collaboration>`` element when the model has a
  ``Collaboration``, containing ``<bpmn:participant>`` elements and
  ``<bpmn:messageFlow>`` elements for cross-pool flows.
- ``<bpmn:dataStore>`` elements at the definitions level for model-scoped data
  stores.

Default sequence flows emit the diagonal-slash marker per BPMN 2.0.2 § 8.3.13
(the ``default`` attribute on the source element plus a ``<bpmn:conditionExpression>``
on the default flow).

Diagram Interchange (DI)
------------------------

Diagram Interchange information — element bounds and edge waypoints — is
emitted only when the input model carries layout data on its elements (via the
opaque ``BPMNElement.layout`` passthrough populated by ``process_bpmn_diagram``
when importing from the Web Modeling Editor). For freshly-built or programmatic
models the DI section is omitted and the importing tool auto-lays out on open.

Identifier handling
-------------------

WME ids are preserved on round-trip when they are
`NCName <https://www.w3.org/TR/REC-xml-names/#NT-NCName>`_-valid; otherwise
fresh ``<Class>_<uuid8>`` ids are minted to guarantee well-formed XML.

Agentic extension elements
--------------------------

Models that use the :ref:`bpmn-agentic-extension` (``AgenticTask`` /
``AgenticGateway`` / ``AgenticLane``) emit additional information per element
through the standard BPMN 2.0 ``<bpmn:extensionElements>`` mechanism. The root
``<bpmn:definitions>`` element gains the
``xmlns:agentic="https://www.besser-pearl.org/bpmn/agentic"`` namespace
declaration, and each agentic element carries one ``<agentic:agentic .../>``
child with flat attributes.

An ``AgenticTask`` emits ``reflectionMode`` and ``trustScore``, plus
``agentDiagramRef`` when the task is linked to an Agent diagram:

.. code-block:: xml

    <bpmn:userTask id="Task_1" name="Review">
      <bpmn:extensionElements>
        <agentic:agentic reflectionMode="cross" trustScore="85"
                         agentDiagramRef="a1b2c3d4-..."/>
      </bpmn:extensionElements>
      <bpmn:incoming>Flow_in</bpmn:incoming>
      <bpmn:outgoing>Flow_out</bpmn:outgoing>
    </bpmn:userTask>

An ``AgenticGateway`` emits ``gatewayRole`` and ``trustScore``. A merging
gateway that carries ``governance_dsl`` additionally emits a sibling
``<agentic:governance>`` child holding the policy snippet:

.. code-block:: xml

    <bpmn:parallelGateway id="Gateway_M" name="Vote">
      <bpmn:extensionElements>
        <agentic:agentic gatewayRole="merging" trustScore="85"/>
        <agentic:governance>policy MajorityPolicy {
        ...governance policy...
        }</agentic:governance>
      </bpmn:extensionElements>
    </bpmn:parallelGateway>

An ``AgenticLane`` emits ``role`` and ``trustScore``. It also emits
``multiplicity`` when the lane represents more than one agent instance, and
``agentDiagramRef`` when the lane is linked to an Agent diagram:

.. code-block:: xml

    <bpmn:lane id="Lane_Reviewer" name="Reviewers">
      <bpmn:extensionElements>
        <agentic:agentic role="supervision" trustScore="85"
                         multiplicity="3" agentDiagramRef="a1b2c3d4-..."/>
      </bpmn:extensionElements>
      <bpmn:flowNodeRef>Task_1</bpmn:flowNodeRef>
    </bpmn:lane>

Attribute presence follows the metamodel defaults. Lane ``multiplicity`` is
omitted when it is ``1`` (absence means one instance), while ``trustScore`` is
emitted on every agentic element. Message flows remain standard BPMN
``<bpmn:messageFlow>`` elements; A2A runtime wiring is carried by Agent diagram
tags during deployment generation, not by agentic message-flow attributes in
BPMN XML.

The ``<bpmn:extensionElements>`` element is always emitted as the **first
child** of its host (``tFlowNode`` / ``tLane`` per the BPMN 2.0 schema). Enum
values use lowercase strings (``"cross"`` / ``"supervision"`` /
``"merging"`` / ...), mirroring the WME editor's wire format. The
``<agentic:governance>`` body is written as escaped text rather than a
``CDATA`` section (Python's standard ``ElementTree`` has no native CDATA);
this is semantically equivalent -- any XML parser yields the same text
content, which WME trims on import.

Vanilla BPMN models (no agentic subclasses) emit no
``<bpmn:extensionElements>`` blocks -- the agentic emission is strictly
additive.

Engine-specific variants (Camunda 7 / 8 / Flowable BPMN XML with execution
attributes) are out of scope for this generator and can be added as separate
generators in the future.
