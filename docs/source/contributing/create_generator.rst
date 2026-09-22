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

The generator class must inherit from ``GeneratorInterface``:

.. code-block:: python

   from besser.generators import GeneratorInterface

   class MyGenerator(GeneratorInterface):
       def __init__(self, model, output_dir=None):
           super().__init__(model, output_dir)

       def generate(self):
           # 1. Traverse the model
           for cls in self.model.get_classes():
               # 2. Render templates
               output = self.render_template('my_template.py.j2', cls=cls)
               # 3. Write output files
               self.write_file(f'{cls.name}.py', output)

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

5. Document and Demo
--------------------

* Extend ``docs/source/examples.rst`` (or add a new section) with a walkthrough that starts from a BUML model and ends
  with the generated artifact running locally.
* List prerequisites (installed toolchains, environment variables, Docker images) and troubleshooting hints.
* Attach screenshots, logs, or GIFs if they make reviewer validation easier.

.. seealso::

   :doc:`../generators/build_generator`
      Annotated source code example of the Python generator and instructions for
      registering your generator in the web editor via ``GeneratorInfo``.
