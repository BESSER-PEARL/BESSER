Django Generator
================

BESSER provides a code generator for `Django models <https://docs.djangoproject.com/en/4.2/topics/db/models/>`_.
These models represent the classes and relationships specified in a :doc:`../buml_language/model_types/structural`.

Let's generate the code for the Django models of our :doc:`../examples/library_example` structural model example. 
You should create a ``DjangoGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use 
the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.django import DjangoGenerator
    
    generator: DjangoGenerator = DjangoGenerator(model=library_model)
    generator.generate()

The ``models.py`` file with the Django models defined will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/models.py
   :language: python
   :linenos: