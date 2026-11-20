Guide: Create a New Generator
=============================

Use this playbook when you want to add support for a new technology stack (e.g., FastAPI, Spring Boot, Vue) in BESSER.

1. Define the Scope
-------------------

* Identify the target runtime, deployment model, and tooling (framework versions, package managers, Docker images).
* Document the mapping rules between B-UML constructs and generated artifacts inside ``docs/source/generators`` so the
  community can review the expected behavior.

2. Bootstrap the Module
-----------------------

* Copy the minimal scaffold from an existing generator (folder structure, setup module, test layout) and rename all
  packages, entry points, and CLI hooks.
* Keep reusable helpers inside ``besser/utilities``. Only generator-specific code should live under your new folder in
  ``besser/generators``.

Your generator directory should follow this structure:

.. code-block:: text

   besser/generators/my_generator/
   ├── __init__.py              # Exports MyGenerator
   ├── my_generator.py          # Main generator class
   └── templates/
       └── my_template.py.j2    # Jinja2 template(s)

The generator class must inherit from ``GeneratorInterface``. Build the Jinja environment with
``traced_environment`` and write each render with ``_write_traced``, so the editor can trace generated lines back
to model elements (see :doc:`../generators/traceable_generator`):

.. code-block:: python

   import os

   from besser.generators import GeneratorInterface
   from besser.utilities.generation_trace import traced_environment

   TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


   class MyGenerator(GeneratorInterface):
       def __init__(self, model, output_dir=None):
           super().__init__(model, output_dir)

       def generate(self):
           env = traced_environment(TEMPLATES_DIR, trim_blocks=True, lstrip_blocks=True)
           template = env.get_template("my_template.py.j2")
           # Sort what you iterate over: the output must be the same for the same model.
           for cls in sorted(self.model.get_classes(), key=lambda c: c.name):
               self._write_traced(f"{cls.name}.py", template.render(cls=cls))

Register it with one ``GeneratorSpec`` in ``besser/generators/registry.py``, add its id to the pinned lists and
snapshot of ``tests/generators/test_generator_registry.py``, and, to offer it to the Spec-Driven Agent, follow
step 5 of "Adding a deterministic generator" in the repository's ``CLAUDE.md``.

3. Implement Transformations
----------------------------

* Build a clear pipeline: BUML model -> intermediate representation -> rendered files.
* Respect the generator service interface so the component can be triggered from both the CLI and the web modeling
  editor backend.
* Store templates separately (Jinja2, f-strings, etc.) and favor small, composable functions to simplify testing.

4. Add Regression Tests
-----------------------

* Place unit and integration tests under ``tests/generators/<your-generator>``.
* Use small BUML fixture models to test mappings in isolation and full-generation scenarios.
* Validate both structure (e.g., class names, endpoints) and content (e.g., business logic snippets, configuration files).

Example test structure:

.. code-block:: python

   # tests/generators/my_generator/test_my_generator.py
   import pytest
   from besser.generators.my_generator import MyGenerator

   def test_generates_output(library_book_author_model):
       """Use shared fixture from tests/conftest.py."""
       generator = MyGenerator(model=library_book_author_model)
       generator.generate()
       # Assert output files exist and contain expected content

Every registered generator must also be covered for :doc:`regeneration <../generators/regenerating>`:
either set ``regenerable = True`` and add a builder to ``tests/generators/regenerable_builders.py``, or
list it with a reason in ``NOT_REGENERABLE`` in ``tests/generators/test_regenerable_generators.py``.

5. Document and Demo
--------------------

* Add ``docs/source/generators/<name>.rst``: what the generator produces from which model, a usage snippet and the
  output. Add it to a toctree in ``docs/source/generators.rst`` **and** add a row to that page's "Choosing a
  Generator" table. ``python docs/check_docs.py`` fails on a page missing from every toctree.
* Optionally extend ``docs/source/examples.rst`` with a walkthrough that starts from a BUML model and ends with the
  generated artifact running locally.
* List prerequisites (installed toolchains, environment variables, Docker images) and troubleshooting hints.
* Attach screenshots, logs, or GIFs if they make reviewer validation easier.

.. seealso::

   :doc:`../generators/build_generator`
      Annotated source code example of the Python generator and instructions for
      registering your generator with one entry in ``besser/generators/registry.py``.
