Object Diagram Generator
========================

The object diagram generator uses the `Alloy Analyzer <https://alloytools.org/>`_ to check the
satisfiability of a B-UML :doc:`../buml_language/model_types/structural` and its OCL invariants, and
to generate object diagrams that satisfy them. It translates the B-UML model into an Alloy specification
with the :doc:`Alloy generator <./alloy>`, and analyzes that specification.

Note that the analysis is bounded: finding no instance means that **no instance exists within the
bounds**. It does not prove that the model is inconsistent, since a larger instance might exist. As SAT
solving is computationally expensive, the analysis might also time out before reaching an answer (see
the ``timeout`` parameter below). For a complete walkthrough, see the :doc:`../examples/semantic_check_example`.

Manual Installation (no Docker)
-------------------------------

This generator depends on the following software:

- Java 17 or greater.
- The `Alloy Analyzer 6.2 <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>`_ .jar.

Afterwards, you must set the following environment variables:

- ``JAVA_HOME``: The path to the Java installation.
- ``BESSER_ALLOY_JAR``: The path to the downloaded Alloy Analyzer file.

.. note::
   Alloy Analyzer 6.2 is distributed under the MIT License, Copyright (c) the Alloy
   project (see
   `LICENSE <https://github.com/AlloyTools/org.alloytools.alloy/blob/v6.2.0/LICENSE>`_).
   The BESSER backend Docker image bundles the Alloy jar and a Java runtime (see ``Dockerfile``); a
   manual installation as above does not. See the project's ``THIRD_PARTY_LICENSES.md`` for
   details.

Consistency Checks
------------------

For example, in our :doc:`../examples/library_example` we can define an OCL invariant
on ``Book`` to enforce that the number of pages on any book is greater than 10:

.. code-block:: python

    from besser.BUML.metamodel.structural import Constraint

    inv1: Constraint = Constraint(
        name="inv1",
        context=book,
        expression="context Book inv inv1: self.pages > 10",
        language="OCL"
    )

    library_model.constraints = {inv1}

The object diagram generator is implemented in the ``AlloySolver`` class. Now we can check whether
the model is satisfiable by calling the ``check_consistency`` method:

.. code-block:: python

    from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloySolver

    solver = AlloySolver(library_model)
    result = solver.check_consistency(scope=5)

The result will be one of the following:

- ``SAT``: An instance was found.
- ``UNSAT``: No instance exists within the bounds.
- ``TIMEOUT``: The Alloy Analyzer timed out.
- ``SCOPES_TOO_LARGE``: The bounds are too large for the Alloy Analyzer. This is a limitation of the
  Alloy Analyzer, not of BESSER: try a smaller scope.
- ``ERROR``: The Alloy Analyzer failed. This is unexpected: please file a bug report with the model.

The bounds grow to fit the literals of the OCL invariants: the integer range includes the integer
literals, and there are at least as many strings and dates as string and date literals.
``solver.get_scopes()`` describes the bounds of the last analysis: "5 objects per class, integers
in [-16, 15], 5 strings and 5 dates" in this case.

Object Diagram Generation
-------------------------

To generate B-UML object diagrams for our example, invoke the ``generate_object_models`` method:

.. code-block:: python

    solver = AlloySolver(library_model)
    result, object_models = solver.generate_object_models(scope=5, num_models=4)

It returns the result of the analysis, as before, and a list with the ``ObjectModel`` of each
instance found.

Generation of a B-UML Project (Class + Object Diagrams)
-------------------------------------------------------

To generate a project combining the class model and its object diagrams, call the
``generate_project_file`` method:

.. code-block:: python

    solver = AlloySolver(library_model, output_dir="output")
    result = solver.generate_project_file(scope=5)

If an instance is found, an ``alloy_project.py`` file will be created in the output directory. It can
be loaded into the web editor with **Import Project**. To get the ``Project`` object instead of the
file, use ``generate_project``.

Configuration Parameters
------------------------

At its creation, the ``AlloySolver`` class can be configured with the following parameters:

- ``model``: The structural model to analyze.
- ``output_dir``: (Optional) The directory for the specification and the Alloy Analyzer's output.
  Without it, a temporary directory is used.

Its methods take:

- ``scope``: (Optional, default ``5``) The bounds of the search: up to ``scope`` objects of each class,
  ``scope`` strings and dates, and integers with bitwidth ``scope`` (e.g. [-16, 15] for 5). The bounds grow
  to fit the literals of the OCL invariants (see above) and the multiplicity bounds of the associations.
- ``timeout``: (Optional, default ``30``) The seconds the Alloy Analyzer may run before the result is
  ``TIMEOUT``.
- ``seed``: (Optional, default ``0``) The seed of the random dates in the specification.
- ``num_models``: (Optional, default ``1``) The number of object diagrams to generate.

Supported Data Types
--------------------

- ``int``: bounded integers, e.g. ``[-16, 15]`` with scope ``5``.
- ``bool``: ``true`` and ``false``.
- ``str``: uninterpreted strings, compared only with ``=`` and ``<>``.
- ``date``: uninterpreted dates, compared only with ``=`` and ``<>``.
- Enumerations.

The object diagram generator is a new feature of BESSER and is still under development, so some models
and OCL constructs are not supported yet. The :doc:`Alloy generator <./alloy>` lists what cannot be
analyzed, and the solver reports errors when constructs are not supported.

Web Editor
----------

In the web editor, **Quality Check > Semantic check** checks the satisfiability of a class diagram,
and **Semantic generation** generates an object diagram (using ``check_consistency`` and
``generate_object_models`` above, respectively). Both search with increasingly large scopes of 5, 7
and 9. See :doc:`../web_editor_backend` for the endpoints.
