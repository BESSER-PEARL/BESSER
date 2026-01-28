Django App with Admin Panel & UI Components (deprecated)
========================================================

.. warning::

   This code generator is **deprecated** and may be discontinued in future versions.

**B-UML Models required**

- :doc:`../../buml_language/model_types/structural`: This model defines the data structure (classes, relationships, and attributes) that will be used to generate the Django application.

- :doc:`../../buml_language/model_types/gui`: Specifies the user interface components (forms, layouts, navigation) that will be generated as part of the application.

**Getting Started**

To generate a Django web app, including templates and UI components, you need to use the BESSER Python API.
Follow the steps below to get started. You can use our :doc:`../../examples/library_example`
and :doc:`../../examples/mobile_app_example` as input examples to test the generator.

.. code-block:: python

    from besser.generators.django import DjangoGenerator

    generator: DjangoGenerator = DjangoGenerator(model=library_model,
                                                project_name="my_django_app",
                                                app_name="library_app",
                                                gui_model=library_gui_model,
                                                containerization=False)
    generator.generate()

**Configuration Parameters**

- ``model``: The structural model to be used for generating the Django application.
- ``project_name``: The name of the Django project to be created.
- ``app_name``: The name of the Django app to be created within the project.
- ``gui_model`` The GUI model to be used for generating the Django application.
- ``containerization``: A boolean flag indicating whether to generate containerization files to deploy the app using containers.

**Output**

In addition to the files generated for a :ref:`Django app with Admin Panel <basic_app>`, this approach also includes:

- `views.py`, `urls.py`, and `forms.py` for handling user interactions.
- Predefined HTML templates for the application's UI.

Once the application is generated, follow the steps in :ref:`deploy` to set it up.