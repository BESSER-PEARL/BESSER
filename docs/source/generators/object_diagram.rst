Object Diagram Generator
========================

The code generator automatically generates object diagrams from a B-UML/OCL 
:doc:`../buml_language/model_types/structural`. It employs the :doc:`Alloy generator <./alloy>` 
to create an Alloy specification and it uses the `Alloy Analyzer <https://alloytools.org/>`_ 
as a backend for object generation. 

It can also be employed to perform consistency checks of B-UML/OCL models 
(i.e., it can answer whether the model has at least one satisfying instance).

For a complete example of using the Object Diagram Generator's API to
perform different kinds of analyses on a B-UML/OCL model, refer to:
``tests/generators/alloy/run_object_diagram_generator_API.py``.

Installation
------------

This generator depends on the following software:

- Java JDK 17 or greater must be installed on your system.
- Download the `Alloy Analyzer 6.2 <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>`_ .jar.

Afterwards, you must set the following environment variables:

- ``JAVA_HOME``: The path to the Java JDK installation.
- ``BESSER_ALLOY_JAR``: The path to the downloaded Alloy Analyzer .jar file.

Now we are ready to automatically generate object diagrams from B-UML/OCL models.

Consistency Checks
------------------

For example, in our :doc:`../examples/library_example` we can define an OCL invariant 
on ``Book`` to enforce that the number of pages on any book is greater than 10:

.. code-block:: python

    from besser.BUML.metamodel.structural import Constraint

    # OCL Constraints
    inv1: Constraint = Constraint(
        name="inv1",
        context=book,
        expression="context Book inv inv1: self.pages> 10",
        language="OCL"
    )

    library_model.constraints = {inv1}

The object diagram generator is implemented in the ``AlloySolver`` class.
Now, we can ask the generator whether the model is consistent by calling
the ``check_consistency`` method as follows:

.. code-block:: python

    from besser.generators.alloy.instance_generator import AlloySolver

    solver = AlloySolver(library_model)
    result = solver.check_consistency()

The result will be one of the following:

- ``SAT``: The model is consistent (has at least one satisfying instance).
- ``UNSAT``: The model has no satisfying instances.
- ``TIMEOUT``: The Alloy Analysis timed out before finding a solution.

Object Diagram Generation
-------------------------

Let's now generate B-UML object diagrams for our :doc:`../examples/library_example` using Alloy. 
You must create an ``AlloySolver`` object, provide the
:doc:`../buml_language/model_types/structural`, and invoke the 
``generate_object_diagrams`` method:

.. code-block:: python
    
    from besser.generators.alloy.instance_generator import AlloySolver

    solver = AlloySolver(library_model)
    (res, buml_instances) = solver.generate_object_diagrams(num_instances=4)

The result if a tuple with the result of the Alloy analysis (i.e., ``res`` is ``SAT``, 
``UNSAT``, or ``TIMEOUT`` as before) and a list of generated object diagrams in BUML 
format (``buml_instances``) 

Generation of a BUML Project (Class + Object Diagram)
-----------------------------------------------------

Let's now generate a project combining a class model and object diagram in BUML format 
for our example. To achieve this, we must call the ``generate_class_and_object_model`` 
method of the ``AlloySolver`` as follows:  

.. code-block:: python

    from besser.generators.alloy.instance_generator import AlloySolver

    solver = AlloySolver(library_model)
    solver.generate_class_and_object_model()

As before, the method returns ``SAT``, ``UNSAT``, or ``TIMEOUT``.
If the model is satisfiable, a ``buml_class_object_model.py`` file will 
be created in the output directory, containing the combined BUML classes and constraints 
(e.g., for ``library_model`` in this example), and an automatically generated
object diagram in BUML format. 

The ``buml_class_object_model.py`` file can be loaded directly into the ``BESSER`` 
frontend using the ``Import Project`` feature.

Configuration Parameters
------------------------

At its creation, the ``AlloySolver`` class can be configured with the following parameters:

- ``model``: The structural model to be used for generating object diagrams.
- ``output_dir``: (Optional) The directory where the generated results will be saved.
- ``scope``: (Optional) The scope for the Alloy analysis.

Supported Data Types
--------------------

Currently, the object diagram generator works with B-UML specifications annotated with OCL
invariants. It supports the following B-UML data types:

- `int`: Represented as bounded integer values using a two's complement representation.
  E.g., for scope of ``5``, the available integers are in the range
  ``[-16, +15]`` ([``-2^4``, ``+2^4 - 1``]).
- `float`: Currently represented as ``int`` values.
- `str`: Represented as sequence of characters of bounded length (e.g., sequences of size 5
  for a scope of 5). Currently, supported characters are in the range ``[a-z]``.
- `date`: Dates and their operations are currently represented as uninterpreted values and functions.
  Dates are ordered, i.e., the operator is less than (``<``) is supported.

We plan to improve handling of the data types above and add support for additional
data types in future releases.
