SQL Schema Generator
=====================

The SQL generator produces the code or set of SQL statements used to define and modify the structure of the tables 
in a database.

Let's generate the SQL code for the :doc:`../examples/library_example`. You should create a 
``SQLGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.sql import SQLGenerator
    
    generator: SQLGenerator = SQLGenerator(model=library_model, sql_dialects="sqlite")
    generator.generate()

The ``model`` parameter specifies the input B-UML structural model, while the ``sql_dialects`` parameter specifies the target SQL 
dialect for the generated statements.
In this example, we use ``sqlite``, but you can also specify ``postgres``, ``mysql``, ``mssql``, ``mariadb`` or ``oracle``.

Output
------
The generated SQL script, ``tables_sqlite.sql``, will be saved in the ``output/`` folder inside your current working directory.
You can customize the output directory by setting the ``output_dir`` parameter in the generator
(see the :doc:`API docs <../api/generators/api_sql>` for details).
The generated output for this example is shown below.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/tables_sqlite.sql
   :language: sql
   :linenos:
