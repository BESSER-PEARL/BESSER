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

OCL Constraint Validation
--------------------------

The Django generator automatically generates a ``clean()`` method from OCL (Object Constraint
Language) invariant constraints defined in your B-UML model, the same way the
:doc:`../pydantic` generates field validators.

Defining OCL Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^

You can define OCL constraints on your domain model classes:

.. code-block:: python

    from besser.BUML.metamodel.structural import Class, Constraint

    Player = Class(name="Player", attributes={...})

    age_constraint = Constraint(
        name="min_age",
        context=Player,
        expression="context Player inv: self.age > 10",
        language="OCL"
    )

    domain_model.constraints = {age_constraint}

The same OCL comparison operators (``>``, ``<``, ``>=``, ``<=``, ``=``, ``<>``) documented for the
:doc:`../pydantic` are supported here.

Generated Validation
^^^^^^^^^^^^^^^^^^^^^

For each constrained class, the generator adds a ``clean()`` method that checks every OCL
constraint defined on it and raises a Django ``ValidationError`` (keyed by field name) when one
fails:

.. code-block:: python

    from django.core.exceptions import ValidationError

    class Player(models.Model):
        age = models.IntegerField()
        name = models.CharField(max_length=255)

        def clean(self):
            super().clean()
            errors = {}
            if not (self.age > 10):
                errors.setdefault('age', []).append('age must be > 10')
            if errors:
                raise ValidationError(errors)

.. important::

   Unlike the Pydantic validators, Django does **not** call ``clean()`` automatically on
   ``save()``. It runs as part of ``full_clean()``, which Django's ``ModelForm`` and the admin
   panel already call for you — so OCL constraints are enforced automatically when creating or
   editing entities through the admin panel. If you save model instances directly (e.g. in a
   script or a custom view), call ``full_clean()`` yourself before ``save()`` to get the same
   validation.