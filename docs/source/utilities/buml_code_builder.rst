B-UML Code Builder
==================

The B-UML code builder provides functions to generate Python code from B-UML models.
This output can be used to recreate models programmatically or as a starting point
for custom generators.

Available entry points
----------------------

* ``domain_model_to_code`` - structural (class diagram) models
* ``agent_model_to_code`` - agent models
* ``gui_model_to_code`` - GUI models
* ``state_machine_to_code`` - state machine models
* ``quantum_model_to_code`` - quantum circuit models
* ``nn_model_to_code`` - neural network models
* ``kg_model_to_code`` - knowledge graph models
* ``component_model_to_code`` - UML Component models (base ``ComponentModel``
  and agentic ``AgenticComponentModel``)
* ``deployment_model_to_code`` - UML Deployment models
* ``bpmn_model_to_code`` - BPMN process models
* ``trace_model_to_code`` - traceability models (see :doc:`../buml_language/model_types/traceability`)
* ``project_to_code`` - full projects that bundle multiple diagrams, optionally with their trace

Domain model generation
-----------------------

To generate Python code from a B-UML domain model, use ``domain_model_to_code()``:

.. code-block:: python

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities import buml_code_builder

    model: DomainModel = DomainModel(name="MyModel")
    # ... model definition ...

    buml_code_builder.domain_model_to_code(model=model, file_path="output/generated_model.py")

The generated code will include:

* All model enumerations with their literals
* All classes with their attributes and methods
* All relationships (associations and generalizations)
* A complete domain model instance that can be used to recreate the model

An element whose ``timestamp`` was given explicitly is recreated with that
timestamp, passed as a constructor argument, so generators that order members
by timestamp see the same order in the recreated model:

.. code-block:: python

    import datetime

    Library = Class(name="Library", timestamp=datetime.datetime.fromisoformat("2024-03-01T09:30:06.123456"))

A timestamp is explicit when it was passed to the constructor, assigned
later through the ``timestamp`` setter, or read from editor JSON
(``Element.has_explicit_timestamp``). A timestamp left to its default, the
creation time, is not emitted: writing it would make the same model, for
example the same diagram exported twice, produce different files. A model
built in Python without explicit timestamps therefore gets fresh timestamps
when the generated file runs.

The rule covers the domain model, classes, enumerations and their literals,
attributes, methods and their parameters, associations and their ends,
association classes, generalizations and OCL constraints. A class named
``datetime`` is emitted as ``datetime_`` so that it does not shadow the
imported module.

An element's ``id`` (``Element.id``, such as the editor element id) can be
written the same way, as an ``id="..."`` constructor argument, by passing
``emit_ids=True`` to ``domain_model_to_code`` or ``project_to_code``. It is off
by default, because BESSER 8.2.1 and older cannot load such a file (the editor
backend turns it on with ``BESSER_BUML_EMIT_IDS=1``, see
:doc:`../web_editor_backend`). The loader always accepts ``id=``. Elements
without an id never get one, so the file is then the same as before.

Agent / GUI / State Machine / Quantum / Project generation
-----------------------------------------------------------

Use the specialized helpers in the same way:

.. code-block:: python

    from besser.utilities import buml_code_builder

    buml_code_builder.agent_model_to_code(agent_model, file_path="output/agent.py")
    buml_code_builder.gui_model_to_code(gui_model, file_path="output/gui.py")
    buml_code_builder.state_machine_to_code(state_machine, file_path="output/state_machine.py")
    buml_code_builder.quantum_model_to_code(quantum_circuit, file_path="output/quantum.py")
    buml_code_builder.kg_model_to_code(knowledge_graph, file_path="output/kg.py")
    buml_code_builder.trace_model_to_code(trace_model, file_path="output/trace.py")
    buml_code_builder.project_to_code(project_model, file_path="output/project.py")

``project_to_code`` also takes ``trace=``, a ``TraceModel``: it is written as a
``TRACEABILITY`` section bound to ``trace_model``, after the models and outside
``Project(models=[...])``, since a trace describes the models rather than being
one of them. The section references the structural elements it is about
(``ModelArtifact(element=Library)``) by the variables the structural sections bind
them to; ``structural_variables`` computes them with the structural builder's own
naming helpers (``safe_class_name``, ``attribute_var_name``, ``method_var_name``).

All builder functions accept an optional ``model_var_name`` parameter to customize the variable
name used in the generated code (defaults vary by builder type, e.g., ``"domain_model"``,
``"agent"``, ``"gui_model"``, ``"sm"``, ``"qc"``, ``"kg_model"``).

Each generated file starts with a ``# Generated by BESSER <version>`` comment
(one per file, also for a project). It is an ordinary comment, so the file
still loads back into the editor and runs with ``exec()`` as before.

.. note::

    For a detailed description of the code builder APIs, see
    :doc:`API documentation <../api/utilities/api_buml_code_builder>`.
