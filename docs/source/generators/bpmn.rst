BPMN Generator
==============

This code generator produces a vendor-neutral **BPMN 2.0 XML** file from a
:doc:`../buml_language/model_types/bpmn`. The output is plain BPMN 2.0 (no
``<camunda:*>`` / ``<zeebe:*>`` / ``<flowable:*>`` extension attributes), so it
can be opened by every BPMN-aware tool — Camunda 7 / 8, Flowable, jBPM,
bpmn-js, Camunda Modeler, SpiffWorkflow, Operaton, and others.

Engine-specific execution semantics are not emitted; an opened file renders the
process structure (flow nodes, sequence flows, gateways, events, pools / lanes,
data objects / stores, sub-processes) and round-trips through any conformant
modeller.

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

Diagram Interchange (DI) information — element bounds and edge waypoints — is
emitted only when the input model carries layout data on its elements (via the
opaque ``BPMNElement.layout`` passthrough populated by ``process_bpmn_diagram``
when importing from the Web Modeling Editor). For freshly-built / programmatic
models the DI section is omitted and the importing tool auto-lays out on open.

Identifier handling preserves WME ids on round-trip when they are
`NCName <https://www.w3.org/TR/REC-xml-names/#NT-NCName>`_-valid; otherwise
fresh ``<Class>_<uuid8>`` ids are minted.

Agentic extension elements
--------------------------

Models that use the :ref:`bpmn-agentic-extension` (``AgenticTask`` /
``AgenticGateway`` / ``AgenticLane`` / ``AgenticMessageFlow``) emit
additional information per element through the standard BPMN 2.0
``<bpmn:extensionElements>`` mechanism. The root ``<bpmn:definitions>``
element gains the
``xmlns:agentic="https://www.besser-pearl.org/bpmn/agentic"`` namespace
declaration, and each agentic element carries one
``<agentic:agentic .../>`` child with flat attributes. An ``AgenticTask``
emits ``reflectionMode``, ``collaborationMode`` and ``trustScore``, plus
``agentDiagramRef`` when the task is linked to an Agent diagram:

.. code-block:: xml

    <bpmn:userTask id="Task_1" name="Review">
      <bpmn:extensionElements>
        <agentic:agentic reflectionMode="cross" collaborationMode="voting"
                         trustScore="85" agentDiagramRef="a1b2c3d4-..."/>
      </bpmn:extensionElements>
      <bpmn:incoming>Flow_in</bpmn:incoming>
      <bpmn:outgoing>Flow_out</bpmn:outgoing>
    </bpmn:userTask>

    <bpmn:parallelGateway id="Gateway_M" name="Vote">
      <bpmn:extensionElements>
        <agentic:agentic gatewayRole="merging" collaborationMode="voting"
                         mergingStrategy="majority" trustScore="85"/>
        <agentic:governance>Scopes:
        ...governance-policy DSL...
        </agentic:governance>
      </bpmn:extensionElements>
    </bpmn:parallelGateway>

    <bpmn:lane id="Lane_Reviewer" name="Reviewers">
      <bpmn:extensionElements>
        <agentic:agentic role="manager" trustScore="85"/>
      </bpmn:extensionElements>
      <bpmn:flowNodeRef>Task_1</bpmn:flowNodeRef>
    </bpmn:lane>

Attribute presence follows the metamodel's invariants. Per paper § 4.3,
a **diverging** ``AgenticGateway`` does not emit ``mergingStrategy`` (the
strategy is meaningful only at the merging gateway):

.. code-block:: xml

    <bpmn:parallelGateway id="Gateway_D" name="Fork">
      <bpmn:extensionElements>
        <agentic:agentic gatewayRole="diverging" collaborationMode="voting"
                         trustScore="85"/>
      </bpmn:extensionElements>
    </bpmn:parallelGateway>

A merging ``AgenticGateway`` that carries a ``governance_dsl`` additionally
emits a sibling ``<agentic:governance>`` child holding the policy snippet
(shown above). An ``AgenticMessageFlow`` emits its own block on the
``<bpmn:messageFlow>`` element (``collaborationMode`` + ``mergingStrategy`` +
``trustScore``; no ``gatewayRole``):

.. code-block:: xml

    <bpmn:messageFlow id="Flow_M1" sourceRef="Pool_A" targetRef="Pool_B">
      <bpmn:extensionElements>
        <agentic:agentic collaborationMode="role" mergingStrategy="leader-driven"
                         trustScore="40"/>
      </bpmn:extensionElements>
    </bpmn:messageFlow>

The ``<bpmn:extensionElements>`` element is always emitted as the **first
child** of its host (``tFlowNode`` / ``tLane`` / ``tMessageFlow`` per the
BPMN 2.0 schema). Enum values use lowercase strings (``"cross"`` /
``"voting"`` / ``"majority"`` / ``"manager"`` / ``"diverging"`` / …),
mirroring the WME editor's wire format so the ``<agentic:agentic>``
attributes round-trip byte-for-byte with WME's exporter. The
``<agentic:governance>`` body is written as escaped text rather than a
``CDATA`` section (Python's standard ``ElementTree`` has no native CDATA);
this is semantically equivalent — any XML parser yields the same text
content, which WME trims on import.

Vanilla BPMN models (no agentic subclasses) emit no
``<bpmn:extensionElements>`` blocks — the agentic emission is strictly
additive.

Engine-specific variants (Camunda 7 / 8 / Flowable BPMN XML with execution
attributes) are out of scope for this generator and can be added as separate
generators in the future.
