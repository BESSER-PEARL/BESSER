Alloy Specifications Generator
==============================

This code generator produces an `Alloy specification <https://alloytools.org/>`_ from a B-UML/OCL 
:doc:`../buml_language/model_types/structural`. The Alloy specification can be employed afterwards 
for semantic consistency checks and automated :doc:`object diagram generation <./object_diagram>` 
for the B-UML/OCL model.

Let's generate the Alloy specification for our :doc:`../examples/library_example`. 
You must create an ``AlloyGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, 
and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.alloy_generator import AlloyGenerator    

    generator = AlloyGenerator(model=library_model)
    generator.generate()

The ``model.als`` file with the Alloy specification will be generated in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/model.als
   :language: alloy
   :linenos:


Configuration Parameters
------------------------

At its creation, the ``AlloyGenerator`` class can be configured with the following parameters:

- ``model``: The structural model to be used for generating the Alloy specification.
- ``output_dir``: (Optional) The directory where the generated Alloy specification will be saved.
- ``scope``: (Optional) The scope for the Alloy analysis.

OCL Invariants Validation
-------------------------

The Alloy generator incorporates ``facts`` in the Alloy specification to ensure that 
instances created from the specification satisfy the OCL (Object Constraint
Language) invariants defined in your B-UML models. It also employs the OCL invariants 
for consistency checks of the B-UML model.

Defining OCL Invariants 
^^^^^^^^^^^^^^^^^^^^^^^

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

For each OCL invariant the generator adds a ``fact`` to the Alloy specification. 
In this case, the specification will include the following fact:

.. code-block:: alloy 

    fact inv1 { all self : this/Book | self.book_pages > 10 }

Hence, when using the :doc:`automated object diagram generator <./object_diagram>`, 
all generated object diagrams will satisfy the OCL invariants. The 
:doc:`object diagram generator <./object_diagram>` can also be used to 
verify the consistency of the constrained B-UML/OCL model.
