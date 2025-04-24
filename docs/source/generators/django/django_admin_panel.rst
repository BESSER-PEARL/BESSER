Django App with Admin Panel
==============================

**B-UML Model required**

- :doc:`../../buml_language/model_types/structural`: This model defines the data structure (classes, relationships, and attributes) that will be used to generate the Django application.

**Getting started**

To generate a Django App, you can either use the :doc:`../../web_editor` to create the 
structural model and generate the code directly, or use the BESSER Python API as shown below.

In this example, we’ll generate a Django web app using the :doc:`../../examples/library_example`
as input. Here's how to implement the code generator with Python:

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