Python Classes Generator
========================

This code generator produces the Python domain model, i.e. the set of Python classes that represent the entities and 
relationships of a :doc:`../buml_language/model_types/structural`.

Let's generate the code for the Python domain model of our :doc:`../examples/library_example` structural model example. 
You should create a ``PythonGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use 
the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.python_classes import PythonGenerator
    
    generator: PythonGenerator = PythonGenerator(model=library_model)
    generator.generate()

The ``classes.py`` file with the Python domain model (i.e., the set of classes) will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/classes.py
   :language: python
   :linenos: