JSON Schema Generator
=====================

The JSON Schema generator produces a JSON Schema definition based on a given B-UML model. This schema can then be used to validate JSON objects against the model's structure and constraints.

Let's generate the JSON Schema for our :doc:`../examples/library_example`. You should create a ``JSONSchemaGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.json import JSONSchemaGenerator

    generator: JSONSchemaGenerator = JSONSchemaGenerator(model=library_model)
    generator.generate()

The ``json_schema.json`` file containing the JSON Schema will be generated in the ``<<current_directory>>/output``
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/json_schema.json
   :language: json
   :linenos:


This schema can now be used by any JSON Schema validator to ensure that the JSON objects conform to the model's structure and constraints.

