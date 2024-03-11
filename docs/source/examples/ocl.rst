Object Constraint Language (OCL)
==============================

This example present defining the OCL constraints on the model.
Lets take an example of model containing libraries, books and authors. The UML diagram is depicted in the following figure.

.. image:: ../img/library_uml_model.jpg
  :width: 600
  :alt: Library model
  :align: center

We can define the constraint that 'the number of pages for all instances of the Book class should be greater than zero' using the following constraint:

.. code-block:: python

    context Book: self.pages >0