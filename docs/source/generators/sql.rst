SQL Schema Generator
=====================

The SQL generator produces the code or set of SQL statements used to define and modify the structure of the tables 
in a database.

Let's generate the SQL code for the :doc:`../examples/library_example`. You should create a 
``SQLGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.sql import SQLGenerator
    
    generator: SQLGenerator = SQLGenerator(model=library_model, sql_dialect="sqlite")
    generator.generate()

Parameters
----------

- ``model``: The input B-UML structural model.
- ``sql_dialect``: The target SQL dialect for the generated statements (default: ``"sqlite"``).
- ``output_dir``: Optional output directory (default: ``output/`` in the current directory).

Supported Dialects
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dialect
     - Description
   * - ``sqlite``
     - Default. Generates SQLite-compatible DDL statements.
   * - ``postgresql``
     - PostgreSQL DDL with ``CREATE TYPE`` for enumerations.
   * - ``mysql``
     - MySQL DDL with ``ENUM()`` column types.
   * - ``mssql``
     - Microsoft SQL Server DDL.
   * - ``mariadb``
     - MariaDB DDL (similar to MySQL).
   * - ``oracle``
     - Oracle DDL with ``CHECK`` constraints for enumeration values.

The generator handles enumeration types differently depending on the dialect: PostgreSQL
uses ``CREATE TYPE ... AS ENUM``, MySQL/MariaDB use inline ``ENUM()`` column types, and
Oracle uses ``CHECK`` constraints.

Output
------
The generated SQL script, ``tables_sqlite.sql``, will be saved in the ``output/`` folder inside your current working directory.
You can customize the output directory by setting the ``output_dir`` parameter in the generator
(see the :doc:`API docs <../api/generators/api_sql>` for details).
The generated output for this example is shown below.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/tables_sqlite.sql
   :language: sql
   :linenos:
