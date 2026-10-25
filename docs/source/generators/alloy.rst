Alloy Specifications Generator
==============================

This code generator produces an `Alloy specification <https://alloytools.org/>`_ from a B-UML
:doc:`../buml_language/model_types/structural` and its OCL invariants. The
:doc:`object diagram generator <./object_diagram>` analyzes this specification to check the
satisfiability of the model and to generate object diagrams. For a complete walkthrough, see the
:doc:`../examples/semantic_check_example`.

Let's generate the Alloy specification for our :doc:`../examples/library_example`.
You must create an ``AlloyGenerator`` object, provide the :doc:`../buml_language/model_types/structural`,
and use the ``generate`` method as follows:

.. code-block:: python

    from besser.generators.alloy import AlloyGenerator

    generator = AlloyGenerator(model=library_model)
    generator.generate(scope=5)

The ``model.als`` file will be generated in the ``<<current_directory>>/output`` folder, together
with the Alloy modules it uses (e.g. ``strings.als``). Each class becomes a signature, with a field
per attribute and association end:

.. code-block:: alloy

    module model
    open date
    open strings

    sig Book {
      Book_pages: Int,
      Book_release: Date,
      Book_title: Str,
      Book_writtenBy: set Author,
      Book_locatedIn: one Library
    }

    ...

    pred instance_model {
      some Author
      some Book
      some Library
    }

    run instance_model for 5 Author, 5 Book, 5 Library, 5 Str, 5 Date, 5 Int

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

The ``closure`` iterator is also supported, e.g. ``self.parents->closure(parents)->excludes(self)``.

Unsupported Models
------------------

The generator raises an ``AlloyTranslationError`` when a model cannot be translated faithfully:

- Models with association classes, multiple inheritance, n-ary associations, or attributes of type
  ``float``, ``datetime``, ``time``, ``timedelta`` or ``any``.
- OCL with real numbers, arithmetic on collections, ``size()`` or ``=`` on Bags, enumeration
  literals, or string and date operations other than ``=`` and ``<>``.

In the web editor, the Alloy specification of a class diagram can be downloaded from
**Generate > Formal Methods > Alloy Specification**.
