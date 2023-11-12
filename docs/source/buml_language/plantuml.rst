From PlantUML to B-UML
======================

A B-UML model can also be generated from a class model built with `PlantUML <https://plantuml.com/>`_ .
All you need is to provide the textual model (PlantUML) and a T2M transformation will produce the B-UML based model.

Let's see an example with the classic library model. The textual model writed in PlantUML is shown below.

.. literalinclude:: ../../../examples/library/library.plantuml
   :language: console
   :linenos:


And the diagram produced by PlantUML is as follows.

.. image:: ../../img/library_plantuml.png
  :width: 120
  :alt: Library model
  :align: center

Save the PlantUML model code in a file with extension ``.plantuml``, e.g. ``library.plantuml``

Then, load and process the model using our grammar and apply the transformation to obtain the B-UML based model.

.. code-block:: python

    # Import methods and classes
    from BUML.notations.plantUML.plantuml_to_buml import plantuml_to_buml
    from BUML.metamodel.structural.structural import DomainModel

    # PlantUML to B-UML model
    library_buml: DomainModel = plantuml_to_buml(model_path='library.plantuml')

.. note::
    
    The ``model_path`` parameter contains the path and name of the ``.plantuml`` model to be transformed

``library_buml`` is the model containing the domain specification. You can look up the classes, attributes, relationships, 
etc. For example, the following is the way to get the class names.

.. code-block:: python

    # Print class names
    for cls in library_buml.get_classes():
        print(cls.name)

You should get output like this:

.. code-block:: console

    Library
    Book
    Author