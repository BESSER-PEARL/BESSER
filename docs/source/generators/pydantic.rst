Pydantic Classes Generator
============================

This code generator produces Pydantic classes, which represent the entities and relationships of a B-UML model.
These Pydantic classes can be utilized by other code generators to generate code that uses Pydantic classes, 
such as :doc:`rest_api` and :doc:`backend`.

Let's generate the code for the Pydantic classes of our :doc:`../examples/library_example` B-UML model example. 
You should create a ``PydanticGenerator`` object, provide the B-UML model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.pydantic_classes import PydanticGenerator
    
    generator: PydanticGenerator = PydanticGenerator(model=library_model)
    generator.generate()

Upon executing this code, a ``pydantic_classes.py`` file containing the Pydantic models will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output_backend/pydantic_classes.py
   :language: Python
   :linenos:

OCL Constraint Validation
-------------------------

The Pydantic generator automatically generates field validators from OCL (Object Constraint Language) invariant 
constraints defined in your B-UML model. This provides automatic validation of data at the API level.

Defining OCL Constraints
^^^^^^^^^^^^^^^^^^^^^^^^

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

Supported Operators
^^^^^^^^^^^^^^^^^^^

The following OCL comparison operators are supported:

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - OCL Operator
     - Python Equivalent
     - Example
   * - ``>``
     - ``>``
     - ``self.age > 18``
   * - ``<``
     - ``<``
     - ``self.age < 65``
   * - ``>=``
     - ``>=``
     - ``self.score >= 0``
   * - ``<=``
     - ``<=``
     - ``self.price <= 100``
   * - ``=``
     - ``==``
     - ``self.status = 'active'``
   * - ``<>``
     - ``!=``
     - ``self.name <> ''``

Generated Validators
^^^^^^^^^^^^^^^^^^^^

For each OCL constraint, the generator produces a Pydantic ``field_validator``:

.. code-block:: python

    class PlayerCreate(BaseModel):
        age: int
        name: str
        
        @field_validator('age')
        @classmethod
        def validate_age_1(cls, v):
            """OCL Constraint: min_age"""
            if not (v > 10):
                raise ValueError('age must be > 10')
            return v

These validators automatically enforce constraints when creating or updating entities via the REST API, 
and the error messages are displayed in the frontend web application.