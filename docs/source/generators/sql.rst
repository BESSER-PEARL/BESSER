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
- ``ocl_check_constraints``: Translate simple OCL invariants into ``CHECK`` constraints
  (default: ``True``). See :ref:`sql-ocl-check-constraints`.

No database driver is needed: the statements are compiled against the SQLAlchemy dialect
only, so generating, for example, ``mssql`` DDL does not require ``pyodbc``. SQL Server DDL
targets a current server version (``DATE`` and ``TIME`` columns, not the pre-2008 ``DATETIME``
fallback). An unknown dialect raises ``ValueError``; if the DDL cannot be produced, a
``RuntimeError`` carrying the underlying error is raised.

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

.. _sql-ocl-check-constraints:

OCL Constraints as CHECK Constraints
------------------------------------

Simple OCL invariants of the model are emitted as table-level ``CHECK`` constraints on the
table of their context class. An invariant is translated when its body is a comparison
between an attribute of the context class and a literal, or several such comparisons joined
by ``and``:

.. code-block:: text

    context Book inv minPages: self.pages > 0
    context Book inv pagesRange: self.pages >= 1 and self.pages <= 5000
    context Book inv hasTitle: self.title <> ''
    context Book inv inPrint: self.inPrint = true

.. code-block:: sql

    CONSTRAINT ck_book_minpages CHECK (pages > 0),
    CONSTRAINT ck_book_pagesrange CHECK (pages >= 1 AND pages <= 5000),
    CONSTRAINT ck_book_hastitle CHECK (title != ''),
    CONSTRAINT ck_book_inprint CHECK ("inPrint" = true)

The rules are deliberately conservative:

- the operators are ``=``, ``<>``, ``<``, ``<=``, ``>`` and ``>=``; the literal may be on either side;
- ``Integer`` and ``Real`` attributes compare with numeric literals, ``String`` attributes with string
  literals (``=`` and ``<>`` only, since string ordering depends on the database collation), and
  ``Boolean`` attributes with ``true`` / ``false`` (``=`` and ``<>`` only);
- the attribute must be declared on the context class itself (not inherited), single-valued and
  not derived, and the context class must have its own table.

Anything else (``or``, ``not``, ``implies``, navigation through associations, collection operations
such as ``size()``, enumeration literals, comparisons between two attributes, ...) is skipped
without an error; it is still enforced by the application-level validators of the other
generators. Constraint names are ``ck_<class>_<constraint name>``, lower-cased and kept to
30 characters. Column quoting and literal rendering follow the dialect (for example
``[inPrint] = 1`` on SQL Server). Pass ``ocl_check_constraints=False`` to leave them out.

Output
------
The generated SQL script, ``tables_sqlite.sql``, will be saved in the ``output/`` folder inside your current working directory.
You can customize the output directory by setting the ``output_dir`` parameter in the generator
(see the :doc:`API docs <../api/generators/api_sql>` for details).
The generated output for this example is shown below.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/tables_sqlite.sql
   :language: sql
   :linenos:
