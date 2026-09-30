Test Case Generator
====================

This code generator produces an automated test suite for the Python domain model
of a :doc:`../buml_language/model_types/structural`. The generated suite combines
plain `pytest <https://docs.pytest.org/>`_ structural checks with
`Hypothesis <https://hypothesis.readthedocs.io/>`_ property-based tests.

The generated ``test_hypothesis.py`` contains three sections:

- **Structural tests** — assert that each class is concrete, exposes the expected
  constructor parameters, and declares each attribute as a property; enumerations
  are checked for their literals.
- **Property-based tests** — build instances with ``hypothesis.strategies.builds``
  and assert instantiation, attribute setter round-trips, and association
  multiplicity contracts.
- **OCL post-condition tests** — for each method post-condition, a test calls the
  operation with sample arguments and asserts the post-condition, translated from
  OCL to Python (``@pre`` capture, collection operations such as ``->size()`` /
  ``->includes()``, and ``=`` to ``==``).

You should create a ``TestCaseGenerator`` object, provide the
:doc:`../buml_language/model_types/structural`, and use the ``generate`` method as
follows:

.. code-block:: python

    from besser.generators.testgen import TestCaseGenerator

    generator: TestCaseGenerator = TestCaseGenerator(model=library_model)
    generator.generate()

The ``test_hypothesis.py`` file will be generated in the
``<<current_directory>>/output`` folder (or in ``output_dir`` if you provide one).

The generated suite imports the domain classes with ``from classes import ...``,
matching the default output module of the :doc:`python` generator. If your
domain model lives in a different module, set the ``module_name`` argument:

.. code-block:: python

    from besser.generators.testgen import TestCaseGenerator

    generator = TestCaseGenerator(model=library_model, module_name="my_domain")
    generator.generate()

This changes the import line of the generated file to ``from my_domain import ...``.

.. note::
   Running the generated suite requires ``pytest`` and ``hypothesis`` to be
   installed, and the domain model classes to be importable from the configured
   ``module_name`` module (e.g. the ``classes.py`` produced by the
   :doc:`python` generator).
