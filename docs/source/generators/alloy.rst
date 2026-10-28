Alloy Specifications Generator
==============================

This code generator produces an `Alloy specification <https://alloytools.org/>`_ from a B-UML
:doc:`../buml_language/model_types/structural` and its OCL invariants. The
:doc:`object diagram generator <../utilities/object_diagram_generator>` analyzes this specification to check the
satisfiability of the model and to generate object diagrams. For a complete walkthrough, see the
:doc:`../examples/semantic_check_example`.

Let's generate the Alloy specification for our :doc:`../examples/library_example`.
You must create an ``AlloyGenerator`` object, provide the :doc:`../buml_language/model_types/structural`,
and use the ``generate`` method as follows:

.. code-block:: python

    from besser.generators.alloy import AlloyGenerator

    generator = AlloyGenerator(model=library_model)
    generator.generate(scope=3)

The ``model.als`` file will be generated in the ``<<current_directory>>/output`` folder, and it will look as
follows:

.. code-block:: alloy

    module model
    open date
    open strings

    sig Author {
      Author_email: Str,
      Author_name: Str,
      Author_publishes: set Book
    }

    sig Book {
      Book_pages: Int,
      Book_release: Date,
      Book_title: Str,
      Book_writtenBy: set Author,
      Book_locatedIn: one Library
    }

    sig Library {
      Library_address: Str,
      Library_name: Str,
      Library_has: set Book
    }

    one sig Str_str0 extends Str {
    }

    one sig Str_str1 extends Str {
    }

    one sig Str_str2 extends Str {
    }

    one sig Date_2017_04_12 extends Date {
    }

    one sig Date_2033_12_31 extends Date {
    }

    one sig Date_2037_11_17 extends Date {
    }

    fact { all a: Book | #(a.Book_writtenBy) >= 1 }

    fact { Author_publishes = ~Book_writtenBy }

    fact { Library_has = ~Book_locatedIn }

    fact { dateLt[Date_2017_04_12, Date_2033_12_31] and dateLt[Date_2033_12_31, Date_2037_11_17] }

    pred instance_model {
      some Author
      some Book
      some Library
    }

    run instance_model for 3 Author, 3 Book, 3 Library, 3 Str, 3 Date, 3 Int

Each class becomes a signature, with a field per attribute and association end. The ``Str`` and ``Date``
constants are the strings and dates that instances can use. The facts state the multiplicity constraints, that the
two ends of bidirectional associations are inverses of each other, and the order of the dates. The ``run``
command searches for an instance with up to 3 objects per class, 3 strings, 3 dates, and integers with
bitwidth 3 (i.e., integers in [-4, 3]).

Configuration Parameters
------------------------

At its creation, the ``AlloyGenerator`` class can be configured with the following parameters:

- ``model``: The structural model to be used for generating the Alloy specification.
- ``output_dir``: (Optional) The directory where the generated Alloy specification will be saved.

The ``generate`` method takes:

- ``scope``: (Optional, default ``5``) The number of objects of each class to search for.
- ``seed``: (Optional, default ``0``) The seed of the random dates in the specification.

After ``generate``, ``get_specification_path()`` returns the path of ``model.als``, and
``get_scopes()`` describes the bounds, e.g. "5 objects per class, integers in [-16, 15] and 5 strings".

OCL Invariants
--------------

For each OCL invariant, the generator adds a ``fact`` to the Alloy specification. For example, in our
:doc:`../examples/library_example` we can define an OCL invariant on ``Book`` to enforce that the
number of pages on any book is greater than 10:

.. code-block:: python

    from besser.BUML.metamodel.structural import Constraint

    inv1: Constraint = Constraint(
        name="inv1",
        context=book,
        expression="context Book inv inv1: self.pages > 10",
        language="OCL"
    )

    library_model.constraints = {inv1}

The specification then includes the following fact:

.. code-block:: alloy

    fact { all self:this/Book|(self.Book_pages > 10) }

Web Editor
----------

In the web editor, the Alloy specification of a class diagram can be downloaded from
**Generate > Formal Methods > Alloy Specification**.

Unsupported Features
--------------------

The Alloy generator is a new feature of BESSER and is still under development. When a model cannot be translated
faithfully, the generator raises an ``AlloyTranslationError`` instead of approximating it. This avoids invalid
satisfiability results and the generation of invalid instances. We plan to add support for new features
periodically.

Supported attribute types are ``bool``, enumerations, integers with a relatively small bitwidth, and ``date`` and
``str``, which are uninterpreted: they can only be compared with ``=`` and ``<>``.

The following features are not supported yet:

- Association classes.
- Multiple inheritance.
- Associations with more than two ends.
- Attributes with several values (a multiplicity upper bound above 1).
- Attributes of type ``float``, ``datetime``, ``time``, ``timedelta`` or ``any``, and reals in OCL.
- OCL operations on strings and dates other than ``=`` and ``<>``.
- Enumeration literals in OCL (``Enum::Literal``).
- OCL type operations (``oclIsKindOf``, ``oclIsTypeOf``, ``oclAsType``) on collections.
- OCL iterators without iterator variables (except ``closure``), and ``select``, ``reject`` or ``collect`` with
  several.
- ``size()``, ``=`` and ``<>`` on OCL Bags: use ``asSet()`` first.
- ``includes``, ``excludes`` and ``collect`` of arithmetic on optional values.
- Negations of comparisons on optional values: a missing value makes them true.
