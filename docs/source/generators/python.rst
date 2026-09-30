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

.. versionchanged:: 8.0.0
   A ``default_value`` is coerced to the attribute's type and emitted as a
   literal. A default that cannot be coerced (for example ``"abc"`` for an
   ``int``), an empty default for a non-string type, or an enumeration default
   that names none of its literals raises ``InvalidDefaultValueError`` at
   generation time.

   A method without an implementation gets a body only when one can be
   inferred from the class (getters, setters, boolean predicates,
   ``__str__``); any other such method is generated with a body that raises
   ``NotImplementedError`` instead of silently doing nothing. See the :doc:`release notes </releases/v8/v8.0.0>`.
