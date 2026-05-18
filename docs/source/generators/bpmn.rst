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

Engine-specific variants (Camunda 7 / 8 / Flowable BPMN XML with execution
attributes) are out of scope for this generator and can be added as separate
generators in the future.
