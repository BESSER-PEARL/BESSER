Django Generator
================

BESSER provides a code generator for `Django web applications <https://www.djangoproject.com/>`_.
The generator supports two different approaches:

1. **Structural Model Only:** Generates a Django web application with CRUD functionality for models, suitable for testing with the Django admin panel.
2. **Structural + GUI Model:** Uses both the Structural model and a GUI model to generate a Django application with predefined user interfaces.


Using Only the Structural Model
-------------------------------
This method generates a Django application with models and CRUD functionality.
It does not generate UI components beyond what is available in Django's admin panel.

**Getting Started**

Let's generate a Django web app taking as input our :doc:`../examples/library_example`.
Below is an example of how to implement the code generator with Python (alternatively,
you can use the :doc:`../web_editor` to generate the code):

.. code-block:: python

    from besser.generators.django import DjangoGenerator

    generator: DjangoGenerator = DjangoGenerator(model=library_model,
                                                project_name="my_django_app",
                                                app_name="library_app",
                                                containerization=False)
    generator.generate()

The configuration parameters for the `DjangoGenerator` are as follows:

- **model**: The structural model to be used for generating the Django application.
- **project_name**: The name of the Django project to be created.
- **app_name**: The name of the Django app to be created within the project.
- **containerization**: A boolean flag indicating whether to generate containerization files to deploy the app using containers.

Different files will be generated in the folder ``<<current_directory>>/<<project_name>>`` including `models.py`,
`settings.py`, `admin.py`, etc.


Using Structural + GUI Model
----------------------------
This method generates a complete Django web application with predefined UI components based on the provided GUI model.

**Getting Started**

+ Create both:
    1. A Structural model (you can use our :doc:`../examples/library_example`)
    2. A GUI model (you can use our :doc:`../examples/mobile_app_example`)

+ Run the generator with:

.. code-block:: python

    from besser.generators.django import DjangoGenerator

    generator: DjangoGenerator = DjangoGenerator(model=library_model,
                                                project_name="my_django_app",
                                                app_name="library_app",
                                                gui_model=library_gui_model,
                                                containerization=False)
    generator.generate()

The configuration parameters for the `DjangoGenerator` are as follows:

- **model**: The structural model to be used for generating the Django application.
- **project_name**: The name of the Django project to be created.
- **app_name**: The name of the Django app to be created within the project.
- **gui_model** The GUI model to be used for generating the Django application.
- **containerization**: A boolean flag indicating whether to generate containerization files to deploy the app using containers.

Different files will be generated in the folder ``<<current_directory>>/<<project_name>>`` including `models.py`, `forms.py`, `views.py`
`urls.py`, and required HTML template files.


How to Run the Web Application
------------------------------

There are two ways to excecute the web application, depending on how the `containerization` parameter was configured in the code
generator. Follow the steps below based on your setup:

1. If containerization is set to `False`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommendation**: Use Python 3.12 or higher for optimal performance.

Enter the project folder and run the following commands:

.. code-block:: bash

    # Install the dependencies
    pip install -r requirements.txt

    # Prepare the database
    python manage.py makemigrations
    python manage.py migrate

    # Create a superuser account
    python manage.py createsuperuser

    # Start the development server
    python manage.py runserver


2. If containerization is set to `True`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requirement**: `Docker Compose <https://docs.docker.com/compose/>`_

Enter the project folder and run this command:

.. code-block:: bash

    # Run docker-compose
    docker-compose up

If you generate the project using the Structural Model, follow these steps to access the Admin panel:

1. Open a web browser and navigate to:

**http://localhost:8000/admin**

2. Login Credentials:

    + **If containerized**: The default username and password are both ``admin``.
    + **If not containerized**: Use the username and password you set during step 1 (``createsuperuser``).

.. image:: ../img/django-lib.png
   :alt: Application screenshot
   :align: center

If you generate the project using the Structural and GUI Model, follow these steps to run the application:

1. Open a web browser and navigate to:

**http://127.0.0.1:8000/**


.. image:: ../img/django_book_page.png
   :alt: Django Book page screenshot
   :align: center


.. image:: ../img/django_book_form_page.png
   :alt: Django Book form page screenshot
   :align: center

