Pydantic Classes Generator
============================

This code generator produces Pydantic classes, which represent the entities and relationships of a B-UML model.
These Pydantic classes can be utilized by other code generators to generate code that uses Pydantic classes, 
such as :doc:`rest_api` and :doc:`backend`.

Let's generate the code for the Pydantic classes of our :doc:`../examples/library_example` B-UML model example. 
You should create a ``PydanticGenerator`` object, provide the B-UML model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.Pydantic_classes import PydanticGenerator
    
    generator: PydanticGenerator = Pydantic_Generator(model=library_model)
    generator.generate()

Upon executing this code, a ``pydantic_classes.py`` file containing the Pydantic models will be generated.  in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output_backend/pydantic_classes.py
   :language: Python
   :linenos: