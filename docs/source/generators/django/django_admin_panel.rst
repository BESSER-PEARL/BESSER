Django App with Admin Panel
==============================
This approach creates a Django application with database models and basic CRUD functionality. It does not generate custom UI
components beyond Django’s built-in admin panel.

**B-UML Model required**

- :doc:`../../buml_language/model_types/structural`: This model defines the data structure (classes, relationships, and attributes) that will be used to generate the Django application.

**Getting started**

Let's generate a Django web app taking as input our :doc:`../../examples/library_example`.
Below is an example of how to implement the code generator with Python (alternatively,
you can use the :doc:`../../web_editor` to generate the code):

.. code-block:: python

    from besser.generators.django import DjangoGenerator

    generator: DjangoGenerator = DjangoGenerator(model=library_model,
                                                project_name="my_django_app",
                                                app_name="library_app",
                                                containerization=False)
    generator.generate()

**Configuration Parameters**

- ``model``: The structural model to be used for generating the Django application.
- ``project_name``: The name of the Django project to be created.
- ``app_name``: The name of the Django app to be created within the project.
- ``containerization``: A boolean flag to enable/disable containerization for deployment.

.. _basic_app:


**Output**

After running the generator, the following files will be created:

- A project folder containing essential Django files such as `settings.py`, `urls.py`, etc.
- An application folder including `models.py` and `admin.py`.
- `manage.py` and `requirements.txt` for managing the application.

If `containerization=True`, the following files will also be generated for Docker deployment:

- `docker-compose.yml`
- `Dockerfile`
- `entrypoint.sh`

To run the application, follow the steps in :ref:`deploy`.