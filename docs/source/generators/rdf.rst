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

The ``vocabulary.ttl`` file with the vocabulary specification in turtle format will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/vocabulary.ttl
   :language: console
   :linenos:

Trace Links
-----------

The generator records which lines of ``vocabulary.ttl`` come from which model element: after ``generate()``,
``generator.traced_files["vocabulary.ttl"].regions`` lists, for every class, generalization, attribute, association,
enumeration and enumeration literal that has an ``id`` (as every element drawn in the web editor does), the line
ranges rendered from it and the template block that rendered them (``class``, ``generalization``, ``attribute``,
``association``, ``enumeration``, ``literal``). A generalization's lines lie within its subclass's. Recording them
does not change the generated file. ``build_generation_trace`` in ``besser.utilities.generation_trace`` turns them
into a :doc:`traceability model <../buml_language/model_types/traceability>`; the web editor shows them on its
*Traceability* page.
