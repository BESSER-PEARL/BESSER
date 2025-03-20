Object Constraint Language (OCL)
================================

This example present defining the OCL constraints on the model.
Lets take an example of model containing libraries, books and authors. The UML diagram is depicted in the following figure.

.. image:: ../img/library_uml_model.png
  :width: 600
  :alt: Library model
  :align: center

We can set a rule that says *every book must have more than zero pages*. To do this, we use the following constraint:

.. code-block:: python

    context Book: self.pages > 0

.. note::

  B-OCL Interpreter is available at https://github.com/BESSER-PEARL/B-OCL-Interpreter. With this interpreter you can validate your OCL constraints defined on B-UML models.
