SQLAlchemy Generator
====================

BESSER provides a code generator to produce the module-level constructors that will form the structure of 
a database managed with `SQLAlchemy <https://www.sqlalchemy.org/>`_. This structure (known as 
`Declarative Mapping <https://docs.sqlalchemy.org/en/20/orm/mapping_styles.html#orm-declarative-mapping>`_) 
defines the database metadata (using a Python object model) that will represent the input B-UML model.

Now, let's generate the code for SQLAlchemy of our :doc:`../examples/library_example` B-UML model example. 
For this, you should create the generator, provide the B-UML model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from generators.sql_alchemy.sql_alchemy_generator import SQLAlchemyGenerator
    
    generator: SQLAlchemyGenerator = SQLAlchemyGenerator(model=library_model)
    generator.generate()

The ``sql_alchemy.py`` file with the Declarative Mapping of the database will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../examples/library/output/sql_alchemy.py
   :language: python
   :linenos: