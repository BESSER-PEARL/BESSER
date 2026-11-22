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

.. versionchanged:: 8.0.0
   With ``backend=True``, the ``<Class>Create`` schemas no longer contain
   fields the server owns: an attribute named ``id`` that is not a declared
   primary key, the ``createdAt`` / ``updatedAt`` timestamps, and attributes
   marked ``is_derived``. A declared primary key (``is_id=True``) is
   client-supplied and always stays in the schema. On the non-owning side of a
   one-to-one association the Create schema has no field for the link, and
   relationship fields are typed after the referenced primary key rather than
   always ``int``.

   A ``default_value`` is now coerced to the attribute's type and emitted as a
   literal. A default that cannot be coerced (for example ``"abc"`` for an
   ``int``), an empty default for a non-string type, or an enumeration default
   that names none of its literals raises ``InvalidDefaultValueError`` at
   generation time. See the :doc:`release notes </releases/v8/v8.0.0>`.

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
OCL constraint support details
------------------------------

- ``self.<attr>.matches('<regex>')`` becomes a ``re.fullmatch(...)`` field validator
  (the module imports ``re`` automatically).
- Constraints comparing two attributes of the same class (e.g.
  ``self.check_in <= self.check_out``) become ``@model_validator(mode='after')``
  validators.
- Constraints that involve collections or relationships (``->size()``,
  ``->collect()``, ``->sum()``, …) cannot be enforced on a Create payload; the
  generator emits an explanatory ``# NOTE:`` comment instead of broken code, and
  every emitted validator is syntax-checked before it is written.
