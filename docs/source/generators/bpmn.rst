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

Engine-specific variants (Camunda 7/8 or Flowable BPMN XML with execution
attributes) are out of scope for this generator.
