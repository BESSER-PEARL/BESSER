SQL Schema Generator
=====================

The SQL generator produces the code or set of SQL statements used to define and modify the structure of the tables 
in a database.

Let's generate the SQL code of our :doc:`../examples/library_example` structural model example. You should create a 
``SQLGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.sql import SQLGenerator
    
    generator: SQLGenerator = SQLGenerator(model=library_model)
    generator.generate()

The ``tables.sql`` file with the SQL statements will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/structural/library/output/tables.sql
   :language: sql
   :linenos:

If you want to use this generator for technology-specific SQL commands such as PostgreSQL, you can use the ``sql_dialects`` 
parameter in the ``SQLGenerator``. Currently only ``postgres`` and ``mysql`` are valid values. For example:

.. code-block:: python
    
    from besser.generators.sql import SQLGenerator
    
    generator: SQLGenerator = SQLGenerator(model=library_model, sql_dialects="postgres")
    generator.generate()