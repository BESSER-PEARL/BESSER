Object Diagram Generator
========================

The object diagram generator uses the `Alloy Analyzer <https://alloytools.org/>`_ to check the
satisfiability of a B-UML :doc:`../buml_language/model_types/structural` and its OCL invariants, and
to generate object diagrams that satisfy them. It translates the model with the
:doc:`Alloy generator <./alloy>`.

Note that the analysis is bounded: finding no instance means that **no instance exists within the
bounds**, not that the model is inconsistent. For a complete walkthrough, see the
:doc:`../examples/semantic_check_example`. For more uses of its API, refer to
``tests/utilities/web_modeling_editor/backend/services/validators/alloy/run_alloy_solver_object_generation_API.py``.

Manual Installation (no Docker)
-------------------------------

This generator depends on the following software:

- Java 17 or greater must be installed on your system.
- Download the `Alloy Analyzer 6.2 <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>`_ .jar.

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
the model has an instance by calling the ``check_consistency`` method:

.. code-block:: python

    from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloySolver

    solver = AlloySolver(library_model)
    result = solver.check_consistency(scope=5)

The result will be one of the following:

- ``SAT``: An instance was found.
- ``UNSAT``: No instance exists within the bounds.
- ``TIMEOUT``: The Alloy Analyzer timed out.
- ``SCOPES_TOO_LARGE``: The bounds are too large for the Alloy Analyzer.
- ``ERROR``: The Alloy Analyzer failed.

``solver.get_scopes()`` describes the bounds searched, e.g. "5 objects per class, integers in
[-16, 15] and 5 strings".

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

- ``scope``: (Optional, default ``5``) The number of objects of each class to search for.
- ``timeout``: (Optional, default ``30``) The seconds the Alloy Analyzer may run.
- ``seed``: (Optional, default ``0``) The seed of the random dates in the specification.
- ``num_models``: (Optional, default ``1``) The number of object diagrams to generate.

Supported Data Types
--------------------

- ``int``: bounded integers, e.g. ``[-16, 15]`` with scope ``5``.
- ``bool``: ``true`` and ``false``.
- ``str``: uninterpreted strings, compared only with ``=`` and ``<>``.
- ``date``: uninterpreted dates, compared only with ``=`` and ``<>``.
- Enumerations.

Optional attributes may have no value, and id attributes are unique. The
:doc:`Alloy generator <./alloy>` lists what cannot be translated.

Web Editor
----------

In the web editor, **Quality Check > Semantic check** checks a class diagram, and **Semantic
generation** generates an object diagram. Both search with 5, 7 and 9 objects per class, in turn.
See :doc:`../web_editor_backend` for the endpoints.
