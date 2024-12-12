B-UML Code Builder
=====================

The B-UML code builder component provides functionality to generate Python code from B-UML domain models. This generated code
can be used to recreate the model programmatically.

Code Generation
------------------

To generate Python code from a B-UML domain model, you can use the ``domain_model_to_code()`` function as follows:

.. code-block:: python

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities import buml_code_builder

    # Assuming you have a domain model instance
    model: DomainModel = DomainModel(name="MyModel")
    # ... model definition ...

    # Generate Python code
    buml_code_builder.domain_model_to_code(model=model, file_path="output/generated_model.py")

The generated code will include:

* All model enumerations with their literals
* All classes with their attributes and methods
* All relationships (associations and generalizations)
* A complete domain model instance that can be used to recreate the model

The generated file structure follows this pattern:

1. Required imports
2. Enumeration definitions
3. Class definitions
4. Class members (attributes and methods)
5. Relationship definitions
6. Domain model instantiation

.. note::
    
    For a detailed description of the code builder please refer to the :doc:`API documentation <../api/api_utilities>`.
