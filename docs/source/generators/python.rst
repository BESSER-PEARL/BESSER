Python Classes Generator
========================

This code generator produces the Python domain model, i.e. the set of Python classes that represent the entities and 
relationships of the B-UML model.

Let's generate the code for the Python domain model of our :doc:`../examples/library_example` B-UML model example. 
You should create a ``PythonGenerator`` object, provide the B-UML model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.python_classes import PythonGenerator
    
    generator: Python_Generator = PythonGenerator(model=library_model)
    generator.generate()

The ``classes.py`` file with the Django domain model (i.e., the set of classes) will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/library_test/output/classes.py
   :language: python
   :linenos: