Alloy Specifications Generator
==============================

This code generator produces an `Alloy specification <https://alloytools.org/>`_ from a B-UML/OCL 
:doc:`../buml_language/model_types/structural`. The Alloy specification can be employed afterwards 
for semantic consistency checks and automated object diagram generation for the B-UML/OCL model.

Let's generate the Alloy specification for our :doc:`../examples/library_example`. 
You should create an ``AlloyGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, 
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

- ``model``: The structural model to be used for generating the Alloy specification.
- ``output_dir``: (Optional) The directory where the generated Alloy specification will be saved.
- ``scope``: (Optional) The scope for the Alloy analysis.

OCL Constraint Validation
--------------------------

The Alloy generator incorporates ``facts`` in the Alloy specification to ensure that 
instances created from the specification satisfy the OCL (Object Constraint
Language) invariant constraints defined in your B-UML models.

Defining OCL Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^

For exampple, in our :doc:`../examples/library_example` we can define an OCL constraint 
on the ``Book`` class to ensure that the number of pages is greater than 10:

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

For each OCL invariant the generator adds a ``fact`` to the Alloy specification:

.. code-block:: alloy 

    fact inv1 { all self : this/Book | self.book_pages > 10 }

