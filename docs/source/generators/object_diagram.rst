Object Diagram Generator
========================

The object diagram generator analyzes a B-UML :doc:`../buml_language/model_types/structural` and
its OCL invariants with the `Alloy Analyzer <https://alloytools.org/>`_. It translates the model
with the :doc:`Alloy generator <./alloy>` and searches for instances of it within bounds: a number
of objects per class, a range of integers, and a number of strings and dates. It can:

- check the semantic consistency of the model, i.e. whether its invariants can all hold
  together in some instance within the bounds;
- generate B-UML object diagrams that satisfy the invariants;
- generate a B-UML project with the model and those object diagrams, which the web editor can
  import.

The search is bounded, so finding no instance means that **no instance exists within the
bounds**, not that the model is inconsistent: a larger instance might still exist.

The generator is implemented in the ``AlloySolver`` class, in
``besser/utilities/web_modeling_editor/backend/services/validators/alloy``. For a complete
example of its API, refer to
``tests/utilities/web_modeling_editor/backend/services/validators/alloy/run_alloy_solver_object_generation_API.py``.

Manual Installation (no Docker)
-------------------------------

This generator depends on the following software:

- Java 17 or greater must be installed on your system.
- Download the `Alloy Analyzer 6.2 <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>`_ .jar.

Afterwards, you must set the following environment variables:

- ``JAVA_HOME``: The path to the Java installation.
- ``BESSER_ALLOY_JAR``: The path to the downloaded Alloy Analyzer file.

``AlloySolver`` raises a ``RuntimeError`` when it cannot find them.

.. note::
   Alloy Analyzer 6.2 is distributed under the MIT License, Copyright (c) the Alloy
   project (see
   `LICENSE <https://github.com/AlloyTools/org.alloytools.alloy/blob/v6.2.0/LICENSE>`_).
   The BESSER Docker images bundle the Alloy jar and a Java runtime (see ``Dockerfile``); a
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

Now we can check whether the model has an instance within the bounds of scope 5, by calling the
``check_consistency`` method:

.. code-block:: python

    from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloySolver

    solver = AlloySolver(library_model)
    result = solver.check_consistency(scope=5)
    print(result, solver.get_scopes())

The result is an ``AlloyResult``:

- ``SAT``: an instance was found, so the invariants can all hold together.
- ``UNSAT``: no instance exists within the bounds.
- ``TIMEOUT``: the Alloy Analyzer did not finish within ``timeout`` seconds.

``get_scopes()`` describes the bounds searched, e.g. "5 objects per class, integers in
[-16, 15], 5 strings and 5 dates". ``generate_specification(scope)`` writes the specification of
a scope without analyzing it, and returns the same description.

Object Diagram Generation
-------------------------

To generate B-UML object diagrams of the model, invoke the ``generate_object_models`` method:

.. code-block:: python

    solver = AlloySolver(library_model)
    result, object_models = solver.generate_object_models(scope=5, num_models=4)

It returns the result of the analysis, as before, and a list with an ``ObjectModel`` per instance
found, empty unless the result is ``SAT``.

Generation of a B-UML Project (Class + Object Diagrams)
-------------------------------------------------------

``generate_project`` returns, besides the result, a B-UML ``Project`` whose models are the domain
model followed by the generated object models. ``generate_project_file`` writes that project's
B-UML code to ``alloy_project.py`` in the output directory, when instances were found:

.. code-block:: python

    solver = AlloySolver(library_model, output_dir="output")
    result = solver.generate_project_file(scope=5)

The ``alloy_project.py`` file can be loaded into the web editor with **Import Project**.

Configuration Parameters
------------------------

At its creation, the ``AlloySolver`` class can be configured with the following parameters:

- ``model``: The structural model to analyze.
- ``output_dir``: (Optional) The directory for the specification and the Alloy Analyzer's
  output. Without it, the solver uses a temporary directory of its own.

Its analyses take:

- ``scope``: (Optional, default ``5``) The number of objects of each class to search.
- ``timeout``: (Optional, default ``30``) The seconds the Alloy Analyzer may run.
- ``seed``: (Optional, default ``0``) The seed of the random dates of the specification.
- ``num_models``: (Optional, default ``1``) The number of instances to generate.

Supported Data Types
--------------------

The generator supports the following B-UML data types:

- ``int``: bounded integers in two's complement. With scope ``5``, integers are in
  ``[-16, 15]``; larger scopes, and larger integer literals in the invariants, widen the range.
- ``float``: represented as ``int`` values.
- ``bool``: ``true`` and ``false``.
- ``str``: uninterpreted strings, compared with ``=`` and ``<>`` only. Generated objects get the
  string literals of the invariants or placeholders such as ``str0``.
- ``date``: uninterpreted dates, compared with ``=`` and ``<>`` only. The scope is filled with
  random dates.
- enumerations, whose values are their literals.

The :doc:`Alloy generator <./alloy>` lists the models and OCL constructs that cannot be translated.

Web Editor
----------

The web editor runs these analyses for class diagrams through two endpoints, which search the
scopes 5, 7, 9 and 11 in turn, each for at most 30 seconds, and stream their progress:
**Quality Check > Semantic check** (``/semantic-consistency-check``) and **Generate from
constraints** in object diagrams (``/generate-object-diagram``). See :doc:`../web_editor_backend`.
