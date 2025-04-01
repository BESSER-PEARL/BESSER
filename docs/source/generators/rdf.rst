RDF Generator
=============

This code generator produces the vocabulary specification (in `RDF turtle format <https://en.wikipedia.org/wiki/Turtle_(syntax)>`_) 
that represent the entities and relationships of a :doc:`../buml_language/model_types/structural`.

Let's generate the code for the vocabulary of our :doc:`../examples/library_example` structural model example. 
You should create a ``RDFGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use 
the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.rdf import RDFGenerator
    
    generator: RDFGenerator = RDFGenerator(model=library_model)
    generator.generate()

The ``vocabulary.ttl`` file with the vocabulary specification in turtle forma will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/vocabulary.ttl
   :language: console
   :linenos: