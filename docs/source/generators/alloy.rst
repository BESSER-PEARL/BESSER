Alloy Specifications Generator
==============================

This code generator produces an `Alloy specification <https://alloytools.org/>`_ from a B-UML
:doc:`../buml_language/model_types/structural` and its OCL invariants. The specification is what
the :doc:`object diagram generator <./object_diagram>` analyzes, with the Alloy Analyzer, to check
the semantic consistency of the model and to generate object diagrams of it.

Let's generate the Alloy specification for our :doc:`../examples/library_example`.
You must create an ``AlloyGenerator`` object, provide the :doc:`../buml_language/model_types/structural`,
and use the ``generate`` method as follows:

.. code-block:: python

    from besser.generators.alloy import AlloyGenerator

    generator = AlloyGenerator(model=library_model)
    generator.generate(scope=5)

The generator writes the specification to ``model.als`` in the ``<<current_directory>>/output``
folder, together with the library modules it opens: ``strings.als`` when the model has strings,
``date.als`` when it has dates, and ``boolean.als`` when it has Booleans. For the library model,
``model.als`` declares a signature per class, with a field per attribute and navigable association
end:

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

    fact { Library_has = ~Book_locatedIn }

    pred instance_model {
      some Author
      some Book
      some Library
    }

    run instance_model for 5 Author, 5 Book, 5 Library, 5 Str, 5 Date, 5 Int

The generator is deterministic: the same model, scope and seed give the same specification.

Configuration Parameters
------------------------

At its creation, the ``AlloyGenerator`` class can be configured with the following parameters:

- ``model``: The structural model to be used for generating the Alloy specification.
- ``output_dir``: (Optional) The directory where the generated Alloy specification will be saved.

The ``generate`` method takes:

- ``scope``: (Optional, default ``5``) The number of objects of each class that the ``run``
  command searches. It also bounds the strings and dates, and sets the integer bit width, which
  is large enough for the integer literals of the OCL invariants.
- ``seed``: (Optional, default ``0``) The seed of the random dates that fill the scope of ``Date``.

After ``generate``, ``get_specification_path()`` returns the path of ``model.als`` and
``get_scopes()`` returns what the ``run`` command searches, e.g. "5 objects per class, integers
in [-16, 15], 5 strings and 5 dates".

OCL Invariants
--------------

For each OCL invariant, the generator adds a ``fact`` to the Alloy specification, so every
instance of the specification satisfies the invariants. For example, in our
:doc:`../examples/library_example` we can define an OCL invariant on ``Book`` to enforce that
the number of pages on any book is greater than 10:

.. code-block:: python

    from besser.BUML.metamodel.structural import Constraint

    inv1: Constraint = Constraint(
        name="inv1",
        context=book,
        expression="context Book inv inv1: self.pages > 10",
        language="OCL"
    )

    library_model.constraints = {inv1}

The specification then includes the fact:

.. code-block:: alloy

    fact { all self:this/Book|(self.Book_pages > 10) }

The ``closure`` iterator becomes the reflexive transitive closure ``*`` of Alloy, so invariants
about reachability can be checked, e.g. that nobody is their own ancestor:
``self.parents->closure(parents)->excludes(self)``. The search is bounded, so the closure is exact
within the scope.

Unsupported Models
------------------

The generator raises ``AlloyTranslationError`` (a ``ValueError``) rather than approximating a
model it cannot translate faithfully:

- association classes, and attributes of type ``datetime``, ``time``, ``timedelta`` or ``any``;
- OCL that is invalid, or has no faithful Alloy translation yet, such as real numbers and real
  division, arithmetic on collections, ``size()`` or ``=`` on Bags, string operations other than
  ``=`` and ``<>``, date operations other than ``=`` and ``<>``, and enumeration literals.

In the web editor, the Alloy specification of a class diagram can be downloaded from
**Generate > Formal Methods > Alloy Specification**, as a ZIP with ``model.als`` and the library
modules it opens.
