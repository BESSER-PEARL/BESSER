Django Generator
================

BESSER provides a code generator for `Django web applications <https://www.djangoproject.com/>`_.
You can create the application in two ways:

1. :doc:`Django App with Admin Panel </generators/django/django_admin_panel>` — A Django application with database models and CRUD functionality, accessible via Django’s admin panel.

2. :doc:`Django App with Admin Panel & UI Components </generators/django/django_ui_components>` (**deprecated**) — Includes everything from the Admin Panel version, plus predefined user interfaces such as forms and templates.

.. note::

   The :doc:`../web_editor` supports only the generation of Django apps with the Admin Panel. To generate a Django app with UI Components,
   you must use the Python API.

Check the guidelines below to learn how to generate a Django app depending on your needs:

.. toctree::
  :maxdepth: 1

  django/django_admin_panel
  django/django_ui_components


.. _deploy:

How to Run the Web Application
---------------------------------

After generating the code, you can run the web application locally (either with or without containers) depending on how you configured the generator.
Keep in mind that the ``containerization`` parameter in the code generator determines whether files for container-based deployment are included.

.. _no_containerization:

A. Running without containerization (``containerization = False``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommendation**: Use Python 3.12 or higher for optimal performance.

In this case, you will get a Django project configured to deploy with a 
`SQLite database <https://www.sqlite.org/>`_. 
Go to the project folder and run the following commands to deploy the web app.

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

B. Running with containerization (``containerization = True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Requirement**: `Docker Compose <https://docs.docker.com/compose/>`_

In this case, you will get a Django project configured to deploy using software containers: 
one for the Django app server and another for the 
`PostgreSQL database <https://www.postgresql.org/>`_.
Go to the project folder and run the following command.

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
    + *If not containerized*: Use the username and password you set in :ref:`Step A <no_containerization>` (``Create a superuser account``).

**Home page**

If you have generated the full web app, you can also check the home page and different forms at:

`http://localhost:8000 <http://localhost:8000>`_

Example
-------
* :doc:`../examples/dpp`