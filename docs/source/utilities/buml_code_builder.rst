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
* ``project_to_code`` - full projects that bundle multiple diagrams

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

Agent / GUI / Project generation
--------------------------------

Use the specialized helpers in the same way:

.. code-block:: python

    from besser.utilities import buml_code_builder

    buml_code_builder.agent_model_to_code(agent_model, file_path="output/agent.py")
    buml_code_builder.gui_model_to_code(gui_model, file_path="output/gui.py")
    buml_code_builder.project_to_code(project_model, file_path="output/project.py")

.. note::

    For a detailed description of the code builder APIs, see
    :doc:`API documentation <../api/utilities/api_buml_code_builder>`.
