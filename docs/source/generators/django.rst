Django Generator
================

BESSER provides a code generator for `Django web applications <https://www.djangoproject.com/>`_.
You can create the application in two ways:

1. :doc:`Django App with Admin Panel </generators/django/django_admin_panel>` — A Django application with database models and CRUD functionality, accessible via Django’s admin panel.

2. :doc:`Django App with Admin Panel & UI Components </generators/django/django_ui_components>` — Includes everything from the Admin Panel version, plus predefined user interfaces such as forms and templates.

.. note::

   The :doc:`../web_editor` supports only the generation of Django apps with the Admin Panel. To generate a Django app with UI Components,
   you must use the Python API.




.. toctree::
  :maxdepth: 1

  django/django_admin_panel
  django/django_ui_components



.. _deploy:

How to Run the Web Application
---------------------------------

You can run the application in two ways, depending on whether ``containerization`` is enabled or not.

1. Running without containerization (``containerization = False``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

2. Running with containerization (``containerization = True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requirement**: `Docker Compose <https://docs.docker.com/compose/>`_

Enter the project folder and run this command:

.. code-block:: bash

    # Run docker-compose
    docker-compose up

Access the Web Application
--------------------------

**Admin panel**

To access the admin panel of your web app, open your browser and navigate to:

`http://localhost:8000/admin <http://localhost:8000/admin>`_

Login Credentials:
    + *If containerized*: The default username and password are both ``admin``.
    + *If not containerized*: Use the username and password you set in Section 3.1 (``createsuperuser``).

The following is the admin panel for a Django web app generated using this :doc:`../buml_language/model_types/structural`:

.. image:: ../img/django-lib.png
   :alt: Application screenshot
   :align: center


**Home page**

On the other hand, if you generate the full web app, you can check the home page and different forms at:

`http://localhost:8000 <http://localhost:8000>`_

The following is an screenshoot of the application generated using the :doc:`../buml_language/model_types/structural`
and the GUI model from :doc:`../examples/mobile_app_example`:

.. image:: ../img/django_book_page.png
   :alt: Django Book page screenshot
   :align: center


.. image:: ../img/django_book_form_page.png
   :alt: Django Book form page screenshot
   :align: center

