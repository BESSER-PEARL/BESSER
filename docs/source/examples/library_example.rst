Library
================

This example present the Python code of a basic model (designed with the :doc:`../buml_language`) that describes the typical 
domain of libraries, books and authors. The UML diagram is depicted in the following figure.

.. image:: ../img/library_uml_model.jpg
  :width: 600
  :alt: Library model
  :align: center

The Python code to specify the B-UML model, including its classes, attributes, and relationships, is presented in the following
code (lines 1-44). Additionally, the ``Python_Generator``, ``DjangoGenerator``, ``SQLAlchemyGenerator``, ``SQLGenerator``, ``RESTAPIGenerator`` and ``Pydanti_cGenerator``
code generators are implemented in this example (lines 50-61). Running this script will generate the ``output/`` folder with the
``classes.py``, ``models.py``, ``sql_alchemy.py``, ``tables.sql``, ``rest_api.py`` and ``pydantic_classes.py`` files produced by each of the Generators respectively.

.. literalinclude:: ../../../tests/library_test/library.py
   :language: python
   :linenos:

.. note::
    
    A B-UML model can also be created :doc:`from a model designed with PlantUML <../buml_language/plantuml>` or even 
    :doc:`from an image <../buml_language/image_to_buml>`.